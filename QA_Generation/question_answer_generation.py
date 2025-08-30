import os
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


CONFIG = {
    "EMBEDDINGS_PATH": "embeddings_cache/nuclear_notifications_embeddings.pkl",
    "SEEDS_PATH": "raw_data/seeds.json",
    "OUTPUTS_DIR": "outputs",

    # Dynamic sampling-driven generation
    "SAMPLES_PER_RUN": 200, # 1000 for training, 200 for testing
    "SNIPPET_NUM_CHUNKS": 2,
    "SNIPPET_CHAR_BUDGET": 3000,

    # Retrieval around identified subject
    "TOP_K": 4,
    "CTX_SNIPPET_CHARS": 4000,
    "NO_TRUNCATE_CONTEXT": True,  # if True, always include full notification text in context blocks

    # LLM
    "MODEL_DIR": r"D:\huggingface\hub\Qwen2.5-32B-Instruct-bnb-4bit",
    "DEVICE_MAP": "balanced",
    "GPU_MAX_MEMORY_GIB": 19,
    "MAX_INPUT_TOKENS": 4000,
    "MAX_NEW_TOKENS": 1000,
    "TEMPERATURE": 0.2,
    "TOP_P": 0.9,
}


# ----------------------------- utils -----------------------------

def ensure_outputs_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


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


def load_seeds(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def cosine_topk(query_vec: np.ndarray, matrix: np.ndarray, k: int) -> List[int]:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    m_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    m = matrix / m_norms
    sims = (m @ q.reshape(-1, 1)).ravel()
    if k >= len(sims):
        return list(np.argsort(-sims))
    idx = np.argpartition(-sims, k)[:k]
    return list(idx[np.argsort(-sims[idx])])


def build_blocks(chunks: List[Dict[str, Any]], indices: List[int], max_chars: int) -> List[str]:
    blocks: List[str] = []
    no_trunc = CONFIG.get("NO_TRUNCATE_CONTEXT", False)
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


def build_examples_from_seeds(seeds: List[Dict[str, Any]], n: int = 2) -> str:
    if not seeds:
        return ""
    picked = seeds[:n]
    blocks = []
    for i, s in enumerate(picked, start=1):
        q = s.get("question", "").strip()
        a = s.get("answer", "").strip()
        if q:
            blocks.append(f"Example {i}:\nQ: {q}\nA: {a}")
    return "\n\n".join(blocks)


def ask_for_qa_json(blocks: List[str], seeds_examples: str, tok, model,
                    max_input_tokens: int, max_new_tokens: int, temperature: float, top_p: float) -> Dict[str, Any]:
    background = "# ─── Examples ─────────────────────────────────────────\n" + (seeds_examples or "(no examples)")
    context = "\n\n".join(blocks)
    system = "You are a domain expert assistant. Use ONLY the provided notifications."
    user = (
        f"{background}\n\n"
        f"# ─── Context Notifications ─────────────────────────────\n{context}\n\n"
        "# ─── Your Task ─────────────────────────────────────────\n"
        "Write one highly insightful, grounded question about the context, then answer it concisely.\n"
        "Don't make question overly specific to one detail of a single notification.\n"
        "Question should be specific to the systems or components in the context.\n"
        "Question should not be focused on only a single notification.\n"
        "Question should be related to diagnosis of certain symptoms (like what could be cause of high temps or vibrations, etc.), or other reasonable questions that can be answered with the context.\n"
        "Sometimes the question can be more creative, as long as it is grounded in the context.\n"
        "Answer should use the context as supporting evidence to answer the question.\n"
        "Think about the types of questions an engineer would want to know to help with new issues by understand notifications from the past.\n"
        "Ensure references are unique and not repeated.\n"
        "Return strictly valid JSON with keys \"question\", \"answer\", and \"references\" (list of [REF x] strings you used)."
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
            # use_cache=False,
            eos_token_id=tok.eos_token_id,
            pad_token_id=getattr(tok, "pad_token_id", tok.eos_token_id),
        )
    text = tok.decode(out_ids[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    try:
        obj = json.loads(text)
    except Exception:
        import re
        m = re.search(r"\{[\s\S]*\}", text)
        obj = json.loads(m.group(0)) if m else {"question": None, "answer": None, "references": []}
    return obj


def ask_for_draft_question(blocks: List[str], seeds_examples: str, tok, model,
                           max_input_tokens: int, max_new_tokens: int,
                           temperature: float, top_p: float) -> str:
    context = "\n\n".join(blocks)
    system = "You are a domain expert assistant."
    user = (
        f"# ─── Context Notifications ─────────────────────────────\n{context}\n\n"
        "# ─── Your Task ─────────────────────────────────────────\n"
        "Write one single-sentence, highly insightful and grounded question that a domain engineer would ask based on this context. \n"
        "Question should be specific to the systems or components in the context.\n"
        "Note this is a draft question, and will be used to retrieve more context to refine and answer the question.\n"
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
            # use_cache=False,
            eos_token_id=tok.eos_token_id,
            pad_token_id=getattr(tok, "pad_token_id", tok.eos_token_id),
        )
    text = tok.decode(out_ids[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    return text.splitlines()[0].strip()


# ----------------------------- sampling + subject identification -----------------------------

def sample_random_notifications(chunks: List[Dict[str, Any]], num_items: int) -> List[str]:
    import random
    if not chunks:
        return []
    # Build a map of unique notifications by row_index
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
    # Use build_blocks to format full notifications (NO_TRUNCATE_CONTEXT controls truncation)
    return build_blocks(chunks, chosen, CONFIG.get("CTX_SNIPPET_CHARS", 4000))


def sample_random_notifications_with_refs(chunks: List[Dict[str, Any]], num_items: int) -> (List[str], List[Dict[str, Any]]):
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
    blocks = build_blocks(chunks, chosen, CONFIG.get("CTX_SNIPPET_CHARS", 4000))
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

    # Examples from seeds
    seeds = load_seeds(cfg["SEEDS_PATH"])
    examples = build_examples_from_seeds(seeds, n=2)

    results: List[Dict[str, Any]] = []

    # Prepare output file upfront
    out_dir = ensure_outputs_dir(cfg["OUTPUTS_DIR"])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file_json = out_dir / f"qa_generated_{ts}.json"

    for sample_idx in range(cfg.get("SAMPLES_PER_RUN", 3)):
        seed_blocks = None
        seed_refs = []
        draft_q = None
        refined_refs = []
        record: Dict[str, Any] = {}
        try:
            # Step 1: Draft question from random complete notifications
            seed_blocks, seed_refs = sample_random_notifications_with_refs(chunks, cfg.get("SNIPPET_NUM_CHUNKS", 6))
            draft_q = ask_for_draft_question(
                seed_blocks,
                examples,
                tok,
                model,
                cfg["MAX_INPUT_TOKENS"],
                cfg["MAX_NEW_TOKENS"],
                cfg["TEMPERATURE"],
                cfg["TOP_P"],
            )

            # Step 2: RAG around draft question
            q_vec = retriever.encode([draft_q], convert_to_tensor=False)
            q_vec = np.asarray(q_vec, dtype=np.float32)[0]
            top_idx = cosine_topk(q_vec, emb, cfg["TOP_K"])
            refined_blocks = build_blocks(chunks, top_idx, cfg["CTX_SNIPPET_CHARS"])
            refined_refs = build_refs(chunks, top_idx)

            # Step 3: Final Q/A JSON using refined context
            qa_obj = ask_for_qa_json(
                refined_blocks,
                examples,
                tok,
                model,
                cfg["MAX_INPUT_TOKENS"],
                cfg["MAX_NEW_TOKENS"],
                cfg["TEMPERATURE"],
                cfg["TOP_P"],
            )

            record = {
                "idx": sample_idx,
                "draft_question": draft_q,
                "final_question": qa_obj.get("question"),
                "final_answer": qa_obj.get("answer"),
                "final_references": qa_obj.get("references", []),
                "refined_refs": refined_refs,
                "draft_refs": seed_refs,
            }
        except Exception as e:
            # Capture the error but continue the run
            record = {
                "idx": sample_idx,
                "error": str(e),
                "draft_question": draft_q,
                "final_question": None,
                "final_answer": None,
                "final_references": [],
                "refined_refs": refined_refs,
                "draft_refs": seed_refs,
            }
        results.append(record)

        # Incremental checkpointing: write partial JSON after each iteration regardless of success
        try:
            with open(out_file_json, "w", encoding="utf-8") as f:
                json.dump({"config": cfg, "results": results, "generated_at": datetime.now().isoformat()}, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    # Final message
    print(f"Saved results to: {out_file_json}")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
