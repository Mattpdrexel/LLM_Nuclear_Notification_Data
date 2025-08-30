import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


CONFIG = {
    "BASE_MODEL_DIR": r"D:\huggingface\hub\Qwen2.5-7B-Instruct-bnb-4bit",
    "ADAPTER_DIR": "Fine_Tuning/checkpoints/qwen2_5_7b_qlora_ctx_full",
    "EMBEDDINGS_PATH": "embeddings_cache/nuclear_notifications_embeddings.pkl",
    "TOP_K": 4,
    "MAX_INPUT_TOKENS": 10000,
    "MAX_NEW_TOKENS": 1024,
    "TEMPERATURE": 0.2,
    "TOP_P": 0.9,
}


def load_embeddings(path: str) -> Tuple[np.ndarray, List[Dict[str, Any]], str]:
    import pickle
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Embeddings not found at {p}")
    with open(p, "rb") as f:
        data = pickle.load(f)
    emb = np.asarray(data["embeddings"], dtype=np.float32)
    return emb, data["chunks"], data["model_name"]


def cosine_topk(query_vec: np.ndarray, matrix: np.ndarray, k: int) -> List[int]:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    m_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    m = matrix / m_norms
    sims = (m @ q.reshape(-1, 1)).ravel()
    if k >= len(sims):
        return list(np.argsort(-sims))
    idx = np.argpartition(-sims, k)[:k]
    return list(idx[np.argsort(-sims[idx])])


def build_blocks(chunks: List[Dict[str, Any]], indices: List[int]) -> Tuple[List[str], List[Dict[str, Any]]]:
    blocks: List[str] = []
    refs: List[Dict[str, Any]] = []
    for i, idx in enumerate(indices, start=1):
        ch = chunks[int(idx)]
        label = ch.get("reference_label") or f"row_{ch.get('row_index')}"
        text = (ch.get("text") or "")
        blocks.append(f"[REF {i}] {label}\n{text}")
        refs.append({
            "ref_id": f"REF {i}",
            "reference_label": label,
            "row_index": int(ch.get("row_index", -1)),
        })
    return blocks, refs


def truncate_prompt(prompt: str, tok, max_tokens: int) -> str:
    ids = tok.encode(prompt, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return prompt
    ids = ids[:max_tokens]
    return tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)


def load_model_and_tokenizer(base_dir: str, adapter_dir: str):
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    curr = torch.cuda.current_device() if torch.cuda.is_available() else 0
    device_map = {"": curr}
    model = AutoModelForCausalLM.from_pretrained(
        base_dir,
        quantization_config=quant_cfg,
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return tok, model


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cfg = CONFIG

    emb, chunks, emb_model = load_embeddings(cfg["EMBEDDINGS_PATH"])
    retriever = SentenceTransformer(emb_model, device="cpu")

    tok, model = load_model_and_tokenizer(cfg["BASE_MODEL_DIR"], cfg["ADAPTER_DIR"])

    print("Type your question (or blank to exit):")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break

        # retrieve context
        q_vec = retriever.encode([q], convert_to_tensor=False)
        q_vec = np.asarray(q_vec, dtype=np.float32)[0]
        top_idx = cosine_topk(q_vec, emb, cfg["TOP_K"])
        blocks, refs = build_blocks(chunks, top_idx)
        context = "\n\n".join(blocks)

        system = "You are a domain expert assistant. Use ONLY the provided notifications."
        user = (
            f"# ─── Context Notifications ─────────────────────────────\n{context}\n\n"
            f"# ─── Your Task ─────────────────────────────────────────\n"
            f"Answer the user's question concisely using only the context and include citations like [REF 1], [REF 2].\n"
            f"Question: {q}"
        )

        if hasattr(tok, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = system + "\n\n" + user

        prompt = truncate_prompt(prompt, tok, cfg["MAX_INPUT_TOKENS"])
        inputs = tok(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=cfg["MAX_NEW_TOKENS"],
                temperature=cfg["TEMPERATURE"],
                top_p=cfg["TOP_P"],
                eos_token_id=tok.eos_token_id,
                pad_token_id=getattr(tok, "pad_token_id", tok.eos_token_id),
            )
        text = tok.decode(out_ids[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

        print("\nAnswer:\n" + text + "\n")
        print("References:")
        for r in refs:
            print(f"- {r['ref_id']}: {r['reference_label']}")
        print()


if __name__ == "__main__":
    main()
