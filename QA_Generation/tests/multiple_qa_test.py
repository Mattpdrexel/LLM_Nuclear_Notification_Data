import os
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc


CONFIG = {
    "EMBEDDINGS_PATH": "embeddings_cache/nuclear_notifications_embeddings.pkl",

    # Either set a single QUESTION or a list in QUESTIONS. If both exist, QUESTIONS wins.
    "QUESTION": "What are likely causes for high circ. pump vibrations?",
    "QUESTIONS": [
        "What are likely causes for high circ. pump vibrations?",
        "What is the most likely cause for elevated circ. water motor stator temps?",
        "What typically causes low screenwash header pressure?",
    ],

    "TOP_K": 3,
    "MODEL_DIR": r"D:\huggingface\hub\Qwen2.5-32B-Instruct-bnb-4bit",
    "MAX_NEW_TOKENS": 1000,
    "TEMPERATURE": 0.2,
    "TOP_P": 0.9,
    "DEVICE_MAP": "auto",          # "balanced_low_0" if GPU0 runs tight
    "GPU_MAX_MEMORY_GIB": 19,
    "CTX_SNIPPET_CHARS": 5000,
}


# ----------------------------- utils -----------------------------

def clear_cuda():
    """Free as much VRAM as possible between generations."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def load_embeddings(path: str) -> Dict[str, Any]:
    import pickle
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Embeddings not found at {p}")
    with open(p, "rb") as f:
        data = pickle.load(f)
    for k in ("embeddings", "chunks", "model_name"):
        if k not in data:
            raise ValueError(f"Embeddings file missing key: {k}")
    return data


def cosine_topk(query_vec: np.ndarray, matrix: np.ndarray, k: int) -> List[int]:
    # Normalize
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    m_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    m = matrix / m_norms
    sims = (m @ q.reshape(-1, 1)).ravel()
    if k >= len(sims):
        return list(np.argsort(-sims))
    idx = np.argpartition(-sims, k)[:k]
    return list(idx[np.argsort(-sims[idx])])


def build_context(chunks: List[Dict[str, Any]], indices: List[int], max_chars: int) -> str:
    lines: List[str] = []
    for i, idx in enumerate(indices, start=1):
        ch = chunks[int(idx)]
        label = ch.get("reference_label") or f"row_{ch.get('row_index')}"
        text = (ch.get("text") or "")[:max_chars]
        lines.append(f"[REF {i}] {label}\n{text}")
    return "\n\n".join(lines)


def load_llm(model_dir: str, device_map: str, per_gpu_gib: int):
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    max_mem = (
        {i: f"{per_gpu_gib}GiB" for i in range(torch.cuda.device_count())}
        if torch.cuda.is_available() else None
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quant_cfg,
        device_map=device_map,
        max_memory=max_mem,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    return tok, model


@torch.inference_mode()
def ask_llm(question: str, context: str, tok, model, max_new_tokens: int, temperature: float, top_p: float) -> str:
    system = "You are a domain expert assistant. Use ONLY the provided notifications. If uncertain, say so."
    user = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer (cite refs like [REF 1], [REF 2]):"
    )
    if hasattr(tok, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = system + "\n\n" + user

    inputs = tok(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        # use_cache=False,  # keeps memory from ballooning across long generations
        eos_token_id=tok.eos_token_id,
        pad_token_id=getattr(tok, "pad_token_id", tok.eos_token_id),
    )

    # Decode only the generated tail
    tail = out_ids[0, inputs["input_ids"].shape[-1]:]
    text = tok.decode(tail, skip_special_tokens=True).strip()

    # Clean up *this* call's big tensors before returning
    del inputs, out_ids, tail
    clear_cuda()
    return text


# ----------------------------- main -----------------------------

def main():
    cfg = CONFIG

    # 1) Load embeddings and chunks (CPU)
    data = load_embeddings(cfg["EMBEDDINGS_PATH"])
    emb = np.asarray(data["embeddings"], dtype=np.float32)
    chunks = data["chunks"]
    model_name = data["model_name"]

    # 2) Retriever on CPU (saves VRAM)
    retriever = SentenceTransformer(model_name, device="cpu")

    # 3) Load LLM once
    tok, model = load_llm(cfg["MODEL_DIR"], cfg["DEVICE_MAP"], cfg["GPU_MAX_MEMORY_GIB"])

    # 4) Determine questions list
    questions: List[str] = []
    if isinstance(cfg.get("QUESTIONS"), list) and cfg["QUESTIONS"]:
        questions = cfg["QUESTIONS"]
    elif cfg.get("QUESTION"):
        questions = [cfg["QUESTION"]]
    else:
        raise ValueError("Provide CONFIG['QUESTION'] or CONFIG['QUESTIONS'].")

    # 5) Answer each question; clear memory between LLM calls
    for i, q in enumerate(questions, start=1):
        q_vec = retriever.encode([q], convert_to_tensor=False)
        q_vec = np.asarray(q_vec, dtype=np.float32)[0]

        top_idx = cosine_topk(q_vec, emb, cfg["TOP_K"])
        context = build_context(chunks, top_idx, cfg["CTX_SNIPPET_CHARS"])

        print("=" * 80)
        print(f"[{i}] Question: {q}\n")
        # print("Context (top notifications):\n")
        # print(context, "\n")

        answer = ask_llm(q, context, tok, model, cfg["MAX_NEW_TOKENS"], cfg["TEMPERATURE"], cfg["TOP_P"])
        print("Answer:\n", answer, "\n")

        # Per-iteration cleanup (CPU objects)
        del q_vec, context, top_idx, answer
        clear_cuda()

    # 6) Final cleanup
    del model, tok, retriever, emb, chunks, data
    clear_cuda()


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
