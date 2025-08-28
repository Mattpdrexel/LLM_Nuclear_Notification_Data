# train_qlora_context_full.py
# QLoRA fine-tune with FULL notification context (no truncation). If a sample
# exceeds MAX_LENGTH, we DO NOT trim — we report and (by default) ERROR so you
# can manually adjust TOP_K or edit the long samples.

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from sentence_transformers import SentenceTransformer
import numpy as np


# ======================= Config =======================
BASE_MODEL_DIR = r"D:\huggingface\hub\Qwen2.5-7B-Instruct-bnb-4bit"  # 4-bit base for QLoRA

DATA_FILES: List[str] = [
    "outputs/qa_generated_for_fine_tuning.json",
    "outputs/idk_generated_for_fine_tuning.json",
]

EMBEDDINGS_PATH = "embeddings_cache/nuclear_notifications_embeddings.pkl"  # to rebuild contexts if refs missing

OUTPUT_DIR = "Fine_Tuning/checkpoints/qwen2_5_7b_qlora_ctx_full"
MAX_LENGTH = 5000

BATCH_SIZE = 1
GRAD_ACCUM = 32
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
LR_SCHED = "cosine"
WARMUP_RATIO = 0.03
SAVE_STEPS = 1000
SAVE_TOTAL_LIMIT = 3
LOGGING_STEPS = 25
PACKING = False  # keep off for simplicity with long contexts
SEED = 42

SYSTEM_PROMPT = "You are a helpful domain expert assistant for nuclear plant notifications."

# IMPORTANT: We keep FULL notifications (no truncation). If too long, we error/skip.
TOP_K = 4                       # number of notifications to retrieve per example (adjust manually)
OVERFLOW_POLICY = "error"       # "error" or "skip"
OVERFLOW_REPORT = "Fine_Tuning/too_long_examples.json"

# Treat these as "unanswerable" signals to use EMPTY context
REFUSAL_MATCH = (
    "do not contain enough information",
    "insufficient information",
    "cannot answer with the provided context",
)

# GPU usage
MULTI_GPU = True                # set True to shard the model across all visible GPUs
GPU_MAX_MEMORY_GIB = 19         # per-GPU memory cap for sharding


# ======================= Utils =======================
def assert_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. No torch.cuda device found.")
    print(f"[GPU] Using device: {torch.cuda.get_device_name(0)}")


def load_embeddings_matrix(path: str) -> Tuple[np.ndarray, List[Dict[str, Any]], str]:
    import pickle
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Embeddings not found at {p}")
    with open(p, "rb") as f:
        data = pickle.load(f)
    emb = np.asarray(data["embeddings"], dtype=np.float32)
    chunks = data["chunks"]
    model_name = data["model_name"]
    return emb, chunks, model_name


def cosine_topk(query_vec: np.ndarray, matrix: np.ndarray, k: int) -> List[int]:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    m_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    m = matrix / m_norms
    sims = (m @ q.reshape(-1, 1)).ravel()
    if k >= len(sims):
        return list(np.argsort(-sims))
    idx = np.argpartition(-sims, k)[:k]
    return list(idx[np.argsort(-sims[idx])])


def build_blocks_from_refs(refs: List[Dict[str, Any]],
                           by_row: Dict[int, Dict[str, Any]]) -> List[str]:
    """Turn refined_refs (row_index, reference_label) into [REF i] blocks with FULL text."""
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


def build_blocks_fresh(CHUNKS: List[Dict[str, Any]], indices: List[int]) -> List[str]:
    """Build FULL text blocks using fresh retrieval indices."""
    blocks: List[str] = []
    for i, idx in enumerate(indices, start=1):
        ch = CHUNKS[int(idx)]
        label = ch.get("reference_label") or f"row_{ch.get('row_index')}"
        text = (ch.get("text") or "")
        blocks.append(f"[REF {i}] {label}\n{text}")
    return blocks


def load_generated_runs(paths: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in paths:
        fp = Path(p)
        if not fp.exists():
            continue
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", data)
        if isinstance(results, list):
            rows.extend(results)
    return rows


def is_refusal(answer: str) -> bool:
    low = answer.strip().lower()
    return any(key in low for key in REFUSAL_MATCH)


# ======================= Data -> messages (FULL context) =======================
def build_messages_examples(results: List[Dict[str, Any]],
                            tok,
                            CHUNKS: List[Dict[str, Any]],
                            EMB_MATRIX: np.ndarray,
                            RETRIEVER: SentenceTransformer) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build TRL 'messages' with FULL notifications. If a sample exceeds MAX_LENGTH,
    we DO NOT trim. We collect it and either skip or error later.
    Returns: (usable_examples, too_long_examples_report)
    """
    examples: List[Dict[str, Any]] = []
    too_long: List[Dict[str, Any]] = []

    by_row = {int(ch.get("row_index", i)): ch for i, ch in enumerate(CHUNKS)}

    for r in results:
        q = (r.get("final_question") or r.get("draft_question") or r.get("question") or "").strip()
        a = (r.get("final_answer") or r.get("answer") or "").strip()
        if not q or not a:
            continue

        # If refusal, force EMPTY context so the behavior is trained clearly
        if is_refusal(a):
            blocks: List[str] = []
        else:
            # Prefer stored refined_refs to preserve citation alignment
            refs = r.get("refined_refs")
            if refs:
                blocks = build_blocks_from_refs(refs, by_row)
            else:
                # Fallback: fresh retrieval with TOP_K (FULL text)
                q_vec = np.asarray(RETRIEVER.encode([q], convert_to_tensor=False), dtype=np.float32)[0]
                top_idx = cosine_topk(q_vec, EMB_MATRIX, TOP_K)
                blocks = build_blocks_fresh(CHUNKS, top_idx)

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

        # Token length check (FULL prompt incl. assistant) — no truncation here
        rendered = tok.apply_chat_template(sample["messages"], tokenize=False, add_generation_prompt=False)
        n_tok = len(tok.encode(rendered, add_special_tokens=False))

        if n_tok > MAX_LENGTH:
            too_long.append({
                "question": q,
                "num_tokens": n_tok,
                "max_length": MAX_LENGTH,
                "has_refs": bool(r.get("refined_refs")),
                "top_k": TOP_K,
                "approx_chars": len(rendered),
            })
            # Do not include in training set
        else:
            examples.append(sample)

    return examples, too_long


# ======================= Model loading (QLoRA) =======================
# --- in load_tokenizer_and_model() ---

def load_tokenizer_and_model(model_dir: str):
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,  # fp16 compute to save RAM
    )
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # >>> shard across both GPUs with headroom on GPU0
    n_gpus = torch.cuda.device_count()
    if n_gpus >= 2:
        device_map = "balanced"  # better headroom for logits & inputs on GPU0
        max_memory = {i: "19GiB" for i in range(n_gpus)}
    else:
        device_map = {"": 0}
        max_memory = None

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quant_cfg,
        device_map=device_map,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="sdpa",  # if available, flash_attention_2 is better
    )
    model.config.use_cache = False  # needed with grad checkpointing

    # QLoRA prep + adapters
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, lora_config)

    # tiny extra: keep output head in fp16
    try:
        model.lm_head = model.lm_head.to(torch.float16)
    except Exception:
        pass

    return tok, model



# ======================= Train =======================
def train():
    # allocator hint (Windows may print a harmless warning if unsupported)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    
    if not MULTI_GPU:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    torch.backends.cuda.matmul.allow_tf32 = True
    assert_cuda()

    # 1) Model
    tok, model = load_tokenizer_and_model(BASE_MODEL_DIR)

    # 2) Data: load runs + embeddings for possible fresh retrieval
    results = load_generated_runs(DATA_FILES)
    if not results:
        raise RuntimeError("No results found in DATA_FILES. Generate QA/IDK first.")

    EMB_MATRIX, CHUNKS, emb_model_name = load_embeddings_matrix(EMBEDDINGS_PATH)
    RETRIEVER = SentenceTransformer(emb_model_name, device="cpu")

    examples, too_long = build_messages_examples(results, tok, CHUNKS, EMB_MATRIX, RETRIEVER)

    # Report long samples
    if too_long:
        Path(Path(OVERFLOW_REPORT).parent).mkdir(parents=True, exist_ok=True)
        with open(OVERFLOW_REPORT, "w", encoding="utf-8") as f:
            json.dump(too_long, f, indent=2)
        msg = (f"{len(too_long)} samples exceed MAX_LENGTH={MAX_LENGTH} with FULL notifications. "
               f"Details written to {OVERFLOW_REPORT}. "
               f"Adjust TOP_K or edit the long samples and re-run.")
        if OVERFLOW_POLICY == "error":
            raise RuntimeError(msg)
        else:
            print("[WARN]", msg)

    if not examples:
        raise RuntimeError("No usable training examples under the current MAX_LENGTH and TOP_K.")

    ds = Dataset.from_list(examples)  # TRL will apply chat template for 'messages'

    # 3) Trainer config (SFTConfig)
    # --- SFTConfig: use max_seq_length and the modern checkpointing kwarg ---

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
        bf16=False,                         # keep fp16 path consistent
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # ↓ peak activations
        optim="paged_adamw_8bit",
        report_to=None,
        seed=SEED,

        # IMPORTANT: TRL uses max_seq_length (not max_length)
        packing=False,                      # long-context safety; flip ON later if stable
        remove_unused_columns=True,
        assistant_only_loss=False,          # keep false unless you add a gen-marked chat template
    )


    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=ds,
        processing_class=tok,  # tokenizer for chat template
    )

    trainer.train()

    # 4) Save LoRA adapters + tokenizer
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    print(f"QLoRA training complete. Adapters saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
