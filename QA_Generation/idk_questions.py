import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
from datetime import datetime

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


CONFIG = {
    "EMBEDDINGS_PATH": "embeddings_cache/nuclear_notifications_embeddings.pkl",
    "OUTPUTS_DIR": "outputs",

    # Generation target
    "TARGET_COUNT": 200,            # tunable number of unanswerable examples
    "MAX_ATTEMPTS_MULTIPLIER": 5,   # safety cap: attempts <= TARGET_COUNT * this

    # Sampling
    "SNIPPET_NUM_CHUNKS": 2,        # how many complete notifications to seed draft question
    "CTX_SNIPPET_CHARS": 4000,
    "TOP_K": 4,
    "NO_TRUNCATE_CONTEXT": True,    # always use full notification text when building context blocks

    # LLM
    "MODEL_DIR": r"D:\huggingface\hub\Qwen2.5-32B-Instruct-bnb-4bit",
    "DEVICE_MAP": "balanced",
    "GPU_MAX_MEMORY_GIB": 19,
    "MAX_INPUT_TOKENS": 4000,
    "MAX_NEW_TOKENS": 600,
    "TEMPERATURE": 0.2,
    "TOP_P": 0.9,
}


# ----------------------------- utils -----------------------------

def ensure_outputs_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _json_default(o: Any):
    try:
        import numpy as _np
        if isinstance(o, (_np.integer,)):
            return int(o)
        if isinstance(o, (_np.floating,)):
            return float(o)
        if isinstance(o, (_np.ndarray,)):
            return o.tolist()
    except Exception:
        pass
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


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
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    m_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    m = matrix / m_norms
    sims = (m @ q.reshape(-1, 1)).ravel()
    if k >= len(sims):
        return list(np.argsort(-sims))
    idx = np.argpartition(-sims, k)[:k]
    return list(idx[np.argsort(-sims[idx])])


def build_blocks(chunks: List[Dict[str, Any]], indices: List[int], max_chars: int, no_trunc: bool) -> List[str]:
    blocks: List[str] = []
    for i, idx in enumerate(indices, start=1):
        ch = chunks[int(idx)]
        label = ch.get("reference_label") or f"row_{ch.get('row_index')}"
        full_text = (ch.get("text") or "")
        text = full_text if no_trunc else (full_text[:max_chars] if max_chars and max_chars > 0 else full_text)
        blocks.append(f"[REF {i}] {label}\n{text}")
    return blocks


def build_refs(chunks: List[Dict[str, Any]], indices: List[int]) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    for i, idx in enumerate(indices, start=1):
        ch = chunks[int(idx)]
        refs.append({
            "ref_id": f"REF {i}",
            "reference_label": ch.get("reference_label") or f"row_{ch.get('row_index')}",
            "row_index": int(ch.get("row_index", -1)),
        })
    return refs


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


def truncate_prompt_to_tokens(prompt: str, tok, max_tokens: int) -> str:
    ids = tok.encode(prompt, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return prompt
    ids = ids[:max_tokens]
    return tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)


# ----------------------------- LLM calls -----------------------------

def ask_for_draft_question(blocks: List[str], tok, model,
                           max_input_tokens: int, max_new_tokens: int,
                           temperature: float, top_p: float) -> str:
    context = "\n\n".join(blocks)
    system = "You are a domain expert assistant."
    user = (
        f"# ─── Context Notifications ─────────────────────────────\n{context}\n\n"
        "# ─── Your Task ─────────────────────────────────────────\n"
        "Propose ONE single-sentence question that is clearly related to the context, BUT cannot be answered using ONLY the notifications above.\n"
        "Requirements:\n"
        "- It must be grounded in the same systems/components mentioned in the context.\n"
        "- It must REQUIRE a specific detail that is ABSENT from the context (e.g., exact numeric values not shown, official cause codes, vendor report conclusions, calibration records, acceptance criteria thresholds, external system states, exact dates/times, drawing or part numbers, or completion results not present).\n"
        "- It must NOT include any [REF x] tags or exact reference labels/IDs from the context.\n"
        "- It must NOT be answerable by reasonable inference or domain knowledge; it should demand explicit evidence that is not provided.\n"
        "Return ONLY the question text with no extra words or formatting."
    )
    if hasattr(tok, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = system + "\n\n" + user

    prompt = truncate_prompt_to_tokens(prompt, tok, max_input_tokens)
    inputs = tok(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tok.eos_token_id,
            pad_token_id=getattr(tok, "pad_token_id", tok.eos_token_id),
        )
    text = tok.decode(out_ids[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    return text.splitlines()[0].strip()


def ask_answerability_judge(question: str, blocks: List[str], tok, model,
                            max_input_tokens: int, max_new_tokens: int,
                            temperature: float, top_p: float) -> bool:
    context = "\n\n".join(blocks)
    system = "You are a strict evaluator. Use ONLY the provided notifications."
    user = (
        f"# ─── Question ───────────────────────────────────────────\n{question}\n\n"
        f"# ─── Context Notifications ─────────────────────────────\n{context}\n\n"
        "# ─── Your Task ─────────────────────────────────────────\n"
        "Decide if the question is directly and unambiguously answerable using ONLY the context text.\n"
        "Answerable = true ONLY if the notifications explicitly contain the exact details needed.\n"
        "Set answerable = false if the question seeks details that are missing (e.g., exact numeric values not present, official cause codes, vendor report conclusions, calibration or test records, acceptance criteria thresholds, external system states, exact dates/times, drawing/part numbers, work order completion results) OR if any inference/speculation beyond the text would be required.\n"
        "If partial hints exist but not enough for a precise answer, set answerable = false.\n"
        "Return strict JSON: {\"answerable\": true|false}."
    )
    if hasattr(tok, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = system + "\n\n" + user

    prompt = truncate_prompt_to_tokens(prompt, tok, max_input_tokens)
    inputs = tok(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tok.eos_token_id,
            pad_token_id=getattr(tok, "pad_token_id", tok.eos_token_id),
        )
    text = tok.decode(out_ids[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    try:
        obj = json.loads(text)
    except Exception:
        import re
        m = re.search(r"\{[\s\S]*\}", text)
        obj = json.loads(m.group(0)) if m else {"answerable": None}

    return bool(obj.get("answerable") is True)


# ----------------------------- sampling -----------------------------

def sample_random_notifications_with_refs(chunks: List[Dict[str, Any]], num_items: int, ctx_chars: int, no_trunc: bool) -> Tuple[List[str], List[Dict[str, Any]]]:
    import random
    if not chunks:
        return [], []
    row_to_idx = {}
    for idx, ch in enumerate(chunks):
        ri = int(ch.get("row_index", idx))
        if ri not in row_to_idx:
            row_to_idx[ri] = idx
    candidate_indices = list(row_to_idx.values())
    if not candidate_indices:
        candidate_indices = list(range(len(chunks)))
    take = min(num_items, len(candidate_indices))
    chosen = random.sample(candidate_indices, k=take)
    blocks = build_blocks(chunks, chosen, ctx_chars, no_trunc)
    refs = build_refs(chunks, chosen)
    return blocks, refs


# ----------------------------- main -----------------------------

def main():
    cfg = CONFIG

    # Load embeddings and chunks
    data = load_embeddings(cfg["EMBEDDINGS_PATH"])
    emb = np.asarray(data["embeddings"], dtype=np.float32)
    chunks = data["chunks"]
    model_name = data["model_name"]

    # Retriever (CPU)
    retriever = SentenceTransformer(model_name, device="cpu")

    # LLM once
    tok, model = load_llm(cfg["MODEL_DIR"], cfg["DEVICE_MAP"], cfg["GPU_MAX_MEMORY_GIB"])

    results: List[Dict[str, Any]] = []
    seen_questions: Set[str] = set()

    out_dir = ensure_outputs_dir(cfg["OUTPUTS_DIR"])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file_json = out_dir / f"idk_generated_{ts}.json"

    target = int(cfg.get("TARGET_COUNT", 200))
    max_attempts = target * int(cfg.get("MAX_ATTEMPTS_MULTIPLIER", 5))

    attempts = 0
    while len(results) < target and attempts < max_attempts:
        attempts += 1
        seed_blocks: List[str] = []
        seed_refs: List[Dict[str, Any]] = []
        draft_q: str = ""
        refined_refs: List[Dict[str, Any]] = []
        record: Dict[str, Any] = {}

        try:
            # Step 1: Draft question from random complete notifications
            seed_blocks, seed_refs = sample_random_notifications_with_refs(
                chunks,
                int(cfg.get("SNIPPET_NUM_CHUNKS", 2)),
                int(cfg.get("CTX_SNIPPET_CHARS", 4000)),
                bool(cfg.get("NO_TRUNCATE_CONTEXT", True)),
            )
            if not seed_blocks:
                raise RuntimeError("No seed blocks available")

            draft_q = ask_for_draft_question(
                seed_blocks,
                tok,
                model,
                int(cfg["MAX_INPUT_TOKENS"]),
                int(cfg["MAX_NEW_TOKENS"]),
                float(cfg["TEMPERATURE"]),
                float(cfg["TOP_P"]),
            )
            if not draft_q or draft_q in seen_questions:
                raise RuntimeError("Invalid or duplicate draft question")

            # Step 2: RAG around draft question
            q_vec = retriever.encode([draft_q], convert_to_tensor=False)
            q_vec = np.asarray(q_vec, dtype=np.float32)[0]
            top_idx = cosine_topk(q_vec, emb, int(cfg["TOP_K"]))
            refined_blocks = build_blocks(chunks, top_idx, int(cfg["CTX_SNIPPET_CHARS"]), bool(cfg.get("NO_TRUNCATE_CONTEXT", True)))
            refined_refs = build_refs(chunks, top_idx)

            # Step 3: Judge answerability
            is_answerable = ask_answerability_judge(
                draft_q,
                refined_blocks,
                tok,
                model,
                int(cfg["MAX_INPUT_TOKENS"]),
                int(cfg["MAX_NEW_TOKENS"]),
                float(cfg["TEMPERATURE"]),
                float(cfg["TOP_P"]),
            )

            if is_answerable:
                # Skip; we only keep unanswerable examples
                continue

            # Keep as unanswerable: standardized answer and empty references
            record = {
                "question": draft_q,
                "answer": "Insufficient information to answer this question.",
                "references": [],
                "draft_refs": seed_refs,
                "refined_refs": refined_refs,
            }
            results.append(record)
            seen_questions.add(draft_q)
        except Exception as e:
            # non-fatal; skip and continue
            pass
        finally:
            # incremental save
            try:
                with open(out_file_json, "w", encoding="utf-8") as f:
                    json.dump({"config": cfg, "results": results, "generated_at": datetime.now().isoformat()}, f, indent=2, ensure_ascii=False, default=_json_default)
            except Exception:
                pass

    print(f"Saved {len(results)} unanswerable examples to: {out_file_json}")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
