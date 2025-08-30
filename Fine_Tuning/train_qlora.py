import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import numpy as np

# ======================= Config =======================
BASE_MODEL_DIR = r"D:\huggingface\hub\Qwen2.5-7B-Instruct-bnb-4bit"

# Train on ONE file only (your QA set)
DATA_FILE = "outputs/qa_generated_for_fine_tuning.json"

# We only use stored refs; needed to reconstruct full notifications
EMBEDDINGS_PATH = "embeddings_cache/nuclear_notifications_embeddings.pkl"

OUTPUT_DIR = "Fine_Tuning/checkpoints/qwen2_5_7b_qlora_ctx_full"
SYSTEM_PROMPT = "You are a helpful domain expert assistant for nuclear plant notifications."

# We do NOT truncate. If a sample is too long, we skip (or error).
MAX_SEQ_TOKENS = 10000
OVERFLOW_POLICY = "error"  # "error" or "skip"
OVERFLOW_REPORT = "Fine_Tuning/too_long_examples.json"

# Limit how many notifications/refs we include per example, independent of token budget
REF_TOP_K = 4  # set to 3 or 4 as desired
DEDUP_BY_NOTIFICATION = True  # collapse multiple chunks from the same notification id

# Training knobs (safer for 2x20GB with long contexts)
BATCH_SIZE = 1
GRAD_ACCUM = 32
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
LR_SCHED = "cosine"
WARMUP_RATIO = 0.03
SAVE_STEPS = 1000
SAVE_TOTAL_LIMIT = 3
LOGGING_STEPS = 25
PACKING = False
SEED = 42

# Use both GPUs for capacity (model sharded)
GPU_MAX_MEMORY_GIB = 19  # per GPU cap

# ======================= Helpers =======================
def assert_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")
    print(f"[GPU] Visible devices: {torch.cuda.device_count()} | 0={torch.cuda.get_device_name(0)}")

def load_embeddings_map(path: str) -> Dict[int, Dict[str, Any]]:
    import pickle
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Embeddings not found at {p}")
    with open(p, "rb") as f:
        data = pickle.load(f)
    chunks = data["chunks"]
    by_row = {int(ch.get("row_index", i)): ch for i, ch in enumerate(chunks)}
    return by_row

def load_results(file_path: str) -> List[Dict[str, Any]]:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", data)
    if not isinstance(results, list):
        raise ValueError("Expected a list under 'results'.")
    return results

def build_blocks_from_refs(refs: List[Dict[str, Any]], by_row: Dict[int, Dict[str, Any]]) -> List[str]:
    blocks: List[str] = []
    for i, r in enumerate(refs or [], start=1):
        ri = int(r.get("row_index", -1))
        ch = by_row.get(ri)
        if not ch:
            label = r.get("reference_label", f"row_{ri}")
            blocks.append(f"[REF {i}] {label}\n(Missing reference text)")
            continue
        label = ch.get("reference_label") or f"row_{ch.get('row_index', ri)}"
        text = (ch.get("text") or "")
        blocks.append(f"[REF {i}] {label}\n{text}")
    return blocks

def _extract_notification_id(label: str) -> str:
    if not isinstance(label, str):
        return ""
    m = re.match(r"^\s*(\d+)", label)
    return m.group(1) if m else label

def limit_refs(refs: List[Dict[str, Any]], top_k: int, dedup_by_notification: bool = True) -> List[Dict[str, Any]]:
    """
    Limit the list of refined refs to at most top_k, optionally deduplicating
    by notification id parsed from reference_label. Preserves original order.
    """
    if not refs:
        return []
    out: List[Dict[str, Any]] = refs
    if dedup_by_notification:
        seen = set()
        dedup: List[Dict[str, Any]] = []
        for r in refs:
            nid = _extract_notification_id(r.get("reference_label", ""))
            if nid in seen:
                continue
            seen.add(nid)
            dedup.append(r)
        out = dedup
    if isinstance(top_k, int) and top_k > 0:
        out = out[:top_k]
    return out

def ensure_assistant_mask_capable_template(tok):
    """
    Ensure tokenizer chat template supports assistant-only loss masking
    by providing {% generation %} markers. If missing, install a safe ChatML template.
    """
    def _has_generation_blocks(tpl: Any) -> bool:
        return isinstance(tpl, str) and "{% generation %}" in tpl

    # If current template lacks generation markers, install a ChatML template with them.
    if not _has_generation_blocks(getattr(tok, "chat_template", None)):
        tok.chat_template = """{{- bos_token if bos_token else '' -}}
{%- for message in messages -%}
{%- if message['role'] == 'system' -%}
<|im_start|>system
{{ message['content'] }}<|im_end|>
{%- elif message['role'] == 'user' -%}
<|im_start|>user
{{ message['content'] }}<|im_end|>
{%- elif message['role'] == 'assistant' -%}
<|im_start|>assistant
{% generation %}{{ message['content'] }}{% endgeneration %}<|im_end|>
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
<|im_start|>assistant
{% generation %}{% endgeneration %}
{%- endif -%}
"""
        if tok.eos_token is None or tok.eos_token != "<|im_end|>":
            tok.add_special_tokens({"eos_token": "<|im_end|>"})

    # Probe to verify mask-capability; support both newer and older Transformers.
    probe = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    try:
        tok.apply_chat_template(
            probe,
            tokenize=True,
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
            return_dict=True,
        )
    except TypeError:
        # Older Transformers may not support return_dict; try without it.
        try:
            tok.apply_chat_template(
                probe,
                tokenize=True,
                add_generation_prompt=False,
                return_assistant_tokens_mask=True,
            )
        except Exception:
            pass
    except Exception:
        # If anything else goes wrong, we still return the tokenizer after template injection.
        pass
    return tok

# ======================= Data → messages (ALWAYS WITH CONTEXT) =======================
def build_messages_examples(results: List[Dict[str, Any]],
                            tok,
                            by_row: Dict[int, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build TRL 'messages' with FULL notifications from stored refined_refs.
    No truncation: if tokenized example > MAX_SEQ_TOKENS, we skip or error.
    """
    examples: List[Dict[str, Any]] = []
    too_long: List[Dict[str, Any]] = []

    for r in results:
        q = (r.get("final_question") or r.get("question") or r.get("draft_question") or "").strip()
        a = (r.get("final_answer") or r.get("answer") or "").strip()
        refs = r.get("refined_refs") or []
        # Apply top-k and optional de-duplication by notification id
        refs = limit_refs(refs, REF_TOP_K, DEDUP_BY_NOTIFICATION)
        if not q or not a or not refs:
            # Always-with-context policy: skip samples without stored refined_refs
            continue

        blocks = build_blocks_from_refs(refs, by_row)
        context = "\n\n".join(blocks)

        user = (
            f"Context:\n{context}\n\n"
            f"Question: {q}\n"
            f"Answer concisely using only the context and include citations like [REF 1], [REF 2]."
        )
        sample = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": a},
            ]
        }

        rendered = tok.apply_chat_template(sample["messages"], tokenize=False, add_generation_prompt=False)
        n_tok = len(tok.encode(rendered, add_special_tokens=False))
        if n_tok > MAX_SEQ_TOKENS:
            too_long.append({
                "question": q,
                "num_tokens": n_tok,
                "max_seq_tokens": MAX_SEQ_TOKENS,
                "approx_chars": len(rendered),
                "num_refs": len(refs),
            })
        else:
            examples.append(sample)

    return examples, too_long

# ======================= Model loading (QLoRA) =======================
def load_tokenizer_and_model(model_dir: str):
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,  # fp16 compute saves VRAM
    )
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # Map the quantized model to a single GPU to avoid Accelerate relocating it
    curr = torch.cuda.current_device() if torch.cuda.is_available() else 0
    device_map = {"": curr}
    max_memory = None

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quant_cfg,
        device_map=device_map,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="sdpa",  # stable on Windows; no extra installs
    )
    model.config.use_cache = False  # needed with gradient checkpointing

    # QLoRA adapters
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    return tok, model

# ======================= Train =======================
def train():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cuda.matmul.allow_tf32 = True
    assert_cuda()

    # 1) Model + tokenizer (and ensure template supports assistant-only loss)
    tok, model = load_tokenizer_and_model(BASE_MODEL_DIR)
    tok = ensure_assistant_mask_capable_template(tok)

    # 2) Data
    results = load_results(DATA_FILE)
    by_row = load_embeddings_map(EMBEDDINGS_PATH)
    examples, too_long = build_messages_examples(results, tok, by_row)

    if too_long:
        Path(Path(OVERFLOW_REPORT).parent).mkdir(parents=True, exist_ok=True)
        with open(OVERFLOW_REPORT, "w", encoding="utf-8") as f:
            json.dump(too_long, f, indent=2)
        msg = f"{len(too_long)} samples exceed MAX_SEQ_TOKENS={MAX_SEQ_TOKENS}. See {OVERFLOW_REPORT}."
        if OVERFLOW_POLICY == "error":
            raise RuntimeError(msg)
        else:
            print("[WARN]", msg)

    if not examples:
        raise RuntimeError("No usable training examples under current token budget (and context-only policy).")

    ds = Dataset.from_list(examples)  # TRL will apply chat template for 'messages'

    # 3) Trainer config
    sft_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHED,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        bf16=False,  # stick to fp16 path for consistency with 4-bit compute
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        report_to=None,
        seed=SEED,

        # Important: TRL 0.21.0 uses `max_length` (not `max_seq_length`).
        max_length=MAX_SEQ_TOKENS,
        packing=PACKING,
        remove_unused_columns=True,
        assistant_only_loss=True,   # now safe because we fixed the chat template
        eos_token="<|im_end|>",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=ds,
        processing_class=tok,
    )

    trainer.train()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    print(f"QLoRA training complete. Adapters saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    train()
