
import os, torch, gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Runtime configuration (edit these variables as needed)
CONFIG = {
    "TOP_K": 8,
    "ROUNDS": 2,
    "USE_LLM": True,
    "MODEL_DIR": r"D:\huggingface\hub\Qwen2.5-32B-Instruct-bnb-4bit",
    "MAX_NEW_TOKENS": 5000,
    "TEMPERATURE": 0.2,
    "TOP_P": 0.55,
    "GENERATE_SEEDS": True,
    "SEEDS_LIMIT": 3,
    "EMBEDDINGS_CACHE_DIR": "embeddings_cache",
    "EMBEDDINGS_FILENAME": "nuclear_notifications_embeddings.pkl",
    "TRAINING_TXT_PATH": "raw_data/cw_training.txt",
    "SEEDS_PATH": "raw_data/seeds.json",
    "OUTPUTS_DIR": "outputs",
    # Multi-GPU settings
    "DEVICE_MAP": "balanced",          # or "balanced_low_0"
    "GPU_MAX_MEMORY_GIB": 19,           # per GPU budget
    "LLM_CONTEXT_CHAR_BUDGET": 10000,    # trim total context text for stability
    # Dynamic pipeline sampling
    "SNIPPET_NUM_CHUNKS": 12,
    "SNIPPET_CHAR_BUDGET": 4000,
    "RANDOM_SEED": 42,
    # Token budgets for safety
    "MAX_INPUT_TOKENS": 2048,
    "MAX_NEW_TOKENS_CAP": 1024,
}

# Encourage less fragmentation if not set by environment
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    torch.set_float32_matmul_precision("high")
    # Prefer memory-efficient SDPA
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
        torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
except Exception:
    pass


# -----------------------------
# Token budgeting helpers
# -----------------------------

def _approx_token_len(text: str, avg_chars_per_token: int = 4) -> int:
    return max(1, len(text) // max(1, avg_chars_per_token))


def _truncate_prompt_to_token_budget(prompt: str, max_tokens: int, tok) -> str:
    est = _approx_token_len(prompt)
    if est <= max_tokens:
        return prompt
    trim_ratio = max(0.1, float(max_tokens) / float(est))
    cut = int(len(prompt) * trim_ratio)
    return prompt[: max(1, cut)]

# -----------------------------
# Data loading utilities
# -----------------------------

def load_embeddings(embeddings_cache_dir: str = "embeddings_cache",
                    filename: str = "nuclear_notifications_embeddings.pkl") -> Dict[str, Any]:
    import pickle
    cache_path = Path(embeddings_cache_dir) / filename
    if not cache_path.exists():
        raise FileNotFoundError(f"Embeddings not found at {cache_path}. Run embeddings pipeline first.")
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    # Expect keys: embeddings (np.ndarray), chunks (List[Dict]), model_name (str)
    if not {"embeddings", "chunks", "model_name"}.issubset(data.keys()):
        raise ValueError("Embeddings file missing required keys: embeddings, chunks, model_name")
    logger.info(f"Loaded embeddings: {data['embeddings'].shape} using model {data['model_name']}")
    return data


def load_seeds(seeds_path: str = "raw_data/seeds.json") -> List[Dict[str, Any]]:
    p = Path(seeds_path)
    if not p.exists():
        logger.warning(f"Seeds file not found at {p}; proceeding without seeds.")
        return []
    with open(p, 'r', encoding='utf-8') as f:
        seeds = json.load(f)
    return seeds


def load_training_text(training_path: str = "raw_data/cw_training.txt") -> str:
    p = Path(training_path)
    if not p.exists():
        logger.warning(f"Training text not found at {p}; proceeding without it.")
        return ""
    with open(p, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


# -----------------------------
# Text chunking for ad-hoc corpora
# -----------------------------

def chunk_text(text: str, max_chars: int = 1000, overlap: int = 100) -> List[str]:
    if not text:
        return []
    # Normalize whitespace a bit
    normalized = "\n".join(line.rstrip() for line in text.splitlines())
    chunks: List[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + max_chars, len(normalized))
        chunk = normalized[start:end]
        chunks.append(chunk)
        if end == len(normalized):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


# -----------------------------
# Embedding and retrieval
# -----------------------------

def ensure_model(model_name: str) -> SentenceTransformer:
    m = SentenceTransformer(model_name)
    return m


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: (n, d), b: (m, d) -> (n, m)
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


def build_augmented_corpus(base_embeddings: np.ndarray,
                           base_chunks: List[Dict[str, Any]],
                           model: SentenceTransformer,
                           training_text: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    if not training_text:
        return base_embeddings, base_chunks
    extra_chunks_text = chunk_text(training_text, max_chars=900, overlap=100)
    if not extra_chunks_text:
        return base_embeddings, base_chunks
    logger.info(f"Embedding {len(extra_chunks_text)} cw_training chunks for augmentation…")
    extra_embeddings = model.encode(extra_chunks_text, batch_size=32, convert_to_tensor=False, show_progress_bar=False)
    extra_embeddings = np.asarray(extra_embeddings, dtype=np.float32)

    augmented_embeddings = np.concatenate([base_embeddings, extra_embeddings], axis=0)
    augmented_chunks = list(base_chunks)
    for i, txt in enumerate(extra_chunks_text):
        augmented_chunks.append({
            "id": f"cw_training_chunk_{i}",
            "text": txt,
            "row_index": None,
            "columns": ["cw_training"],
            "chunk_index": i,
            "source": "cw_training.txt"
        })
    logger.info(f"Augmented corpus size: {augmented_embeddings.shape[0]} chunks")
    return augmented_embeddings, augmented_chunks


def multi_query_variants(query: str) -> List[str]:
    q = query.strip()
    variants = [q]
    # heuristic variants
    variants.append(q + " root causes")
    variants.append("causes of " + q)
    variants.append(q + " troubleshooting")
    variants.append("common issues related to " + q)
    # deduplicate while preserving order
    seen = set()
    unique = []
    for v in variants:
        if v not in seen:
            unique.append(v)
            seen.add(v)
    return unique


def retrieve_iterative(query: str,
                       model: SentenceTransformer,
                       corpus_embeddings: np.ndarray,
                       chunks: List[Dict[str, Any]],
                       k: int = 8,
                       rounds: int = 2) -> List[Tuple[int, float]]:
    # Returns list of (index, score)
    scores_accum: Dict[int, float] = {}

    def add_scores(q_text: str, weight: float = 1.0) -> None:
        q_emb = model.encode([q_text], convert_to_tensor=False)
        q_emb = np.asarray(q_emb, dtype=np.float32)
        sims = cosine_sim(q_emb, corpus_embeddings)[0]  # (N,)
        top_idx = np.argpartition(-sims, kth=min(k, len(sims)-1))[:k]
        for idx in top_idx:
            scores_accum[idx] = scores_accum.get(idx, 0.0) + float(sims[idx]) * weight

    # Round 1: multi-query expansion
    for v in multi_query_variants(query):
        add_scores(v, weight=1.0)

    # Round 2+: build synthesized query from top texts
    for _ in range(max(0, rounds - 1)):
        if not scores_accum:
            break
        top_sorted = sorted(scores_accum.items(), key=lambda x: x[1], reverse=True)[:k]
        seed_text = "\n".join(chunks[i]["text"][:400] for i, _ in top_sorted)
        refined_q = f"{query}\nContext:\n{seed_text}"
        add_scores(refined_q, weight=0.5)

    ranked = sorted(scores_accum.items(), key=lambda x: x[1], reverse=True)[:k]
    return ranked


def synthesize_answer_from_context(query: str, contexts: List[str]) -> str:
    # Simple extractive heuristic: pick a few high-signal sentences containing keywords
    import re
    joined = "\n".join(contexts)
    sentences = re.split(r"(?<=[.!?])\s+", joined)
    key_terms = [w for w in query.lower().split() if len(w) > 3]
    scored: List[Tuple[float, str]] = []
    for s in sentences:
        s_l = s.lower()
        score = sum(1.0 for t in key_terms if t in s_l)
        if score > 0:
            scored.append((score, s.strip()))
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [s for _, s in scored[:5]]
    if not selected:
        selected = sentences[:3]
    answer = " ".join(selected)
    return answer.strip()


# -----------------------------
# Optional LLM (Qwen2.5) generation over retrieved contexts
# -----------------------------

_LLM_CACHE: Dict[str, Any] = {}


def _build_max_memory_dict(per_gpu_gib: int) -> Dict[int, str]:
    n = torch.cuda.device_count()
    if n <= 0:
        return {}
    budget = f"{per_gpu_gib}GiB"
    return {i: budget for i in range(n)}


def _trim_contexts_by_char_budget(contexts: List[str], budget: int) -> List[str]:
    if budget <= 0:
        return contexts
    trimmed: List[str] = []
    used = 0
    for c in contexts:
        if used >= budget:
            break
        space_left = budget - used
        piece = c if len(c) <= space_left else c[:space_left]
        trimmed.append(piece)
        used += len(piece)
    return trimmed


def load_llm_and_tokenizer(model_dir: str, device_map: str, per_gpu_gib: int):
    cache_key = f"{model_dir}|{device_map}|{per_gpu_gib}"
    if cache_key in _LLM_CACHE:
        return _LLM_CACHE[cache_key]

    quant_cfg = None
    if "bnb-4bit" in model_dir.lower():
        quant_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    max_memory = _build_max_memory_dict(per_gpu_gib)

    model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config=quant_cfg,
            device_map=device_map,
            max_memory=max_memory if len(max_memory) > 0 else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
)

    logger.info("Loaded LLM with device map sample: %s", list(getattr(model, "hf_device_map", {}).items())[:5])
    _LLM_CACHE[cache_key] = (tok, model)
    return tok, model


def build_llm_prompt(query: str, contexts: List[str]) -> str:
    context_block = "\n\n".join(f"[CTX {i+1}]\n{c}" for i, c in enumerate(contexts))
    return (
        "You are a domain expert assistant. Use ONLY the provided context to answer.\n"
        "If the answer is uncertain, say so.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )


def generate_llm_answer(query: str, contexts: List[str], model_dir: str,
                        max_new_tokens: int = 256, temperature: float = 0.2, top_p: float = 0.95,
                        device_map: str = "balanced", per_gpu_gib: int = 19) -> str:
    # Trim contexts to reduce prompt size and OOM chance
    contexts = _trim_contexts_by_char_budget(contexts, CONFIG.get("LLM_CONTEXT_CHAR_BUDGET", 3500))

    tok, model = load_llm_and_tokenizer(model_dir, device_map=device_map, per_gpu_gib=per_gpu_gib)
    if hasattr(tok, "apply_chat_template"):
        messages = [
                    {"role": "system", "content": "You are a domain expert assistant. Be concise and cite context IDs if obvious."},
                    {"role": "user", "content": build_llm_prompt(query, contexts)},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = build_llm_prompt(query, contexts)

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
    gen = tok.decode(out_ids[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return gen.strip()


# -----------------------------
# Main generation flow
# -----------------------------

def run_rag(query: str,
            embeddings_cache_dir: str = "embeddings_cache",
            embeddings_filename: str = "nuclear_notifications_embeddings.pkl",
            training_txt_path: str = "raw_data/cw_training.txt",
            top_k: int = 8,
            rounds: int = 2) -> Dict[str, Any]:
    data = load_embeddings(embeddings_cache_dir, embeddings_filename)
    model_name = data["model_name"]
    model = ensure_model(model_name)

    base_embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    base_chunks = data["chunks"]

    training_text = load_training_text(training_txt_path)
    corpus_embeddings, corpus_chunks = build_augmented_corpus(base_embeddings, base_chunks, model, training_text)

    ranked = retrieve_iterative(query, model, corpus_embeddings, corpus_chunks, k=top_k, rounds=rounds)
    contexts = [corpus_chunks[int(i)]["text"] for i, _ in ranked]
    answer = synthesize_answer_from_context(query, contexts)

    return {
        "query": query,
        "top_indices": [int(i) for i, _ in ranked],
        "top_scores": [float(s) for _, s in ranked],
        "contexts": contexts,
        "answer": answer,
        "model_name": model_name,
    }


def generate_synthetic_qa_from_seeds(seeds_path: str = "raw_data/seeds.json",
                                     limit: int = 3,
                                     use_llm: bool = False,
                                     model_dir: str = r"D:\huggingface\hub\Qwen2.5-32B-Instruct-bnb-4bit",
                                     max_new_tokens: int = 256,
                                     temperature: float = 0.2,
                                     top_p: float = 0.95) -> List[Dict[str, Any]]:
    seeds = load_seeds(seeds_path)
    if not seeds:
        logger.warning("No seeds available; returning empty synthetic set.")
        return []
    out: List[Dict[str, Any]] = []
    for item in seeds[:limit]:
        q = item.get("question", "").strip()
        if not q:
            continue
        rag = run_rag(q,
                      embeddings_cache_dir=CONFIG["EMBEDDINGS_CACHE_DIR"],
                      embeddings_filename=CONFIG["EMBEDDINGS_FILENAME"],
                      training_txt_path=CONFIG["TRAINING_TXT_PATH"],
                      top_k=CONFIG["TOP_K"],
                      rounds=CONFIG["ROUNDS"],)
        llm_ans = None
        if use_llm:
            llm_ans = generate_llm_answer(
                query=q,
                contexts=rag["contexts"],
                model_dir=model_dir,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                device_map=CONFIG.get("DEVICE_MAP", "balanced"),
                per_gpu_gib=CONFIG.get("GPU_MAX_MEMORY_GIB", 19),
            )
        out.append({
            "question": q,
            "seed_answer": item.get("answer", ""),
            "rag_answer": rag["answer"],
            "llm_answer": llm_ans,
            "contexts": rag["contexts"],
        })
    return out


def ensure_outputs_dir(path: str) -> Path:
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _json_default(o):
    # Ensure numpy types serialize cleanly
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    # Last resort string
    return str(o)


# Prompt templates for multi-step pipeline
COMPONENT_ID_PROMPT_TEMPLATE = """
# ─── Background ─────────────────────────────────────────
{CW_TRAINING}

Here are several notification entries:

{snippet}

Identify the single primary component these notifications relate to.
Your answer must:
- Include a specific equipment identifier and descriptor if appropriate (e.g., "11A CWP", "12A Travel Screen", "13B Screenwash Pump").
- Alternatively, can include class of component (e.g., "CWP", "Travel Screen", "Screenwash Pump").
- Not be a generic term like "pump", "lube oil", "motor", etc.

Return only the exact component string.
"""

QUESTION_GEN_PROMPT_TEMPLATE = """
# ─── Background ─────────────────────────────────────────
{background}

# ─── Examples ───────────────────────────────────────────
Example 1:
Q: {seed1_question}
A: {seed1_answer}

Example 2:
Q: {seed2_question}
A: {seed2_answer}

# ─── Context Notifications ─────────────────────────────
{component_context}

# ─── Your Task ─────────────────────────────────────────
Using the notifications above and focusing on component "{component}", craft one single-sentence question that is both insightful and grounded. Return only the question.
"""

COMBINED_QA_PROMPT_TEMPLATE = """
# ─── Background ─────────────────────────────────────────
{background}

# ─── Examples ───────────────────────────────────────────
Example 1:
Q: {seed1_question}
A: {seed1_answer}

Example 2:
Q: {seed2_question}
A: {seed2_answer}

# ─── Context Notifications ─────────────────────────────
{question_context}

# ─── Draft Question ─────────────────────────────────────
{draft_question}

# ─── Your Task ─────────────────────────────────────────
Using ONLY the notifications above (all of which mention {component}), refine the draft question if needed to be highly insightful and grounded, then provide a concise answer.
The answer MUST be directly supported from the reference notifications. The references MUST be directly related to the component "{component}" (for example, if issue is related to 11A CWP, don't reference notifications related to 12A CWP).
Return a JSON object with keys "question", "answer", and "references" where:
  - "references" is the list of notifications cited (e.g., "Not. Notification – ShortText (YYYY-MM-DD)").
  - If the provided notifications do not contain sufficient information to directly answer the question, set "answer" to "Insufficient information to answer this question." and "references" to [].
We will use these Q&A pairs to fine-tune a later model, so please return strictly valid JSON.
"""


def _llm_generate_text(prompt_text: str,
                       system_prompt: str = "You are a domain expert assistant. Be concise and follow instructions strictly.",
                       max_new_tokens: int = None,
                       temperature: float = None,
                       top_p: float = None) -> str:
    cfg = CONFIG
    tok, model = load_llm_and_tokenizer(
        cfg["MODEL_DIR"],
        device_map=cfg.get("DEVICE_MAP", "balanced"),
        per_gpu_gib=cfg.get("GPU_MAX_MEMORY_GIB", 19),
    )
    if hasattr(tok, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = prompt_text

    # Enforce input token budget
    prompt = _truncate_prompt_to_token_budget(prompt, cfg.get("MAX_INPUT_TOKENS", 2048), tok)

    def _try_generate(curr_max_new: int) -> str:
        inputs = tok(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=curr_max_new,
                temperature=(temperature if temperature is not None else cfg["TEMPERATURE"]),
                top_p=(top_p if top_p is not None else cfg["TOP_P"]),
                eos_token_id=tok.eos_token_id,
                pad_token_id=getattr(tok, "pad_token_id", tok.eos_token_id),
            )
        gen = tok.decode(out_ids[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return gen.strip()

    # Try with configured tokens; if OOM, back off
    target_new = min(cfg.get("MAX_NEW_TOKENS", 512), cfg.get("MAX_NEW_TOKENS_CAP", 1024))
    try:
        return _try_generate(target_new)
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or "cuda out of memory" in msg:
            torch.cuda.empty_cache()
            gc.collect()
            # Back off both input and output
            prompt_backoff = _truncate_prompt_to_token_budget(prompt, max(256, cfg.get("MAX_INPUT_TOKENS", 2048)//2), tok)
            prompt = prompt_backoff
            try:
                return _try_generate(max(128, target_new // 2))
            except Exception:
                torch.cuda.empty_cache()
                gc.collect()
                return _try_generate(64)
        raise


def _safe_json_parse(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        import re
        match = re.search(r"\{[\s\S]*\}", s)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return {"question": None, "answer": None, "references": []}


def _join_contexts(contexts: List[str], max_chars: int) -> str:
    out, used = [], 0
    for c in contexts:
        if used >= max_chars:
            break
        take = c[: max_chars - used]
        out.append(take)
        used += len(take)
    return "\n\n".join(out)


def identify_primary_component_from_contexts(cw_training: str, contexts: List[str]) -> str:
    snippet = _join_contexts(contexts, CONFIG.get("LLM_CONTEXT_CHAR_BUDGET", 8000))
    prompt = COMPONENT_ID_PROMPT_TEMPLATE.format(CW_TRAINING=cw_training, snippet=snippet)
    comp = _llm_generate_text(prompt_text=prompt, max_new_tokens=128)
    return comp.strip().strip('"')


def generate_question_for_component(component: str, cw_training: str, seeds: List[Dict[str, Any]], contexts: List[str]) -> str:
    seed1_q = seeds[0].get("question", "") if len(seeds) > 0 else ""
    seed1_a = seeds[0].get("answer", "") if len(seeds) > 0 else ""
    seed2_q = seeds[1].get("question", "") if len(seeds) > 1 else ""
    seed2_a = seeds[1].get("answer", "") if len(seeds) > 1 else ""
    ctx_text = _join_contexts(contexts, CONFIG.get("LLM_CONTEXT_CHAR_BUDGET", 8000))
    prompt = QUESTION_GEN_PROMPT_TEMPLATE.format(
        background=cw_training,
        seed1_question=seed1_q,
        seed1_answer=seed1_a,
        seed2_question=seed2_q,
        seed2_answer=seed2_a,
        component_context=ctx_text,
        component=component,
    )
    q = _llm_generate_text(prompt_text=prompt, max_new_tokens=128)
    return q.strip().rstrip('?') + '?'


def generate_combined_qa(component: str, draft_question: str, cw_training: str, seeds: List[Dict[str, Any]], contexts: List[str]) -> Dict[str, Any]:
    seed1_q = seeds[0].get("question", "") if len(seeds) > 0 else ""
    seed1_a = seeds[0].get("answer", "") if len(seeds) > 0 else ""
    seed2_q = seeds[1].get("question", "") if len(seeds) > 1 else ""
    seed2_a = seeds[1].get("answer", "") if len(seeds) > 1 else ""
    ctx_text = _join_contexts(contexts, CONFIG.get("LLM_CONTEXT_CHAR_BUDGET", 8000))
    prompt = COMBINED_QA_PROMPT_TEMPLATE.format(
        background=cw_training,
        seed1_question=seed1_q,
        seed1_answer=seed1_a,
        seed2_question=seed2_q,
        seed2_answer=seed2_a,
        question_context=ctx_text,
        draft_question=draft_question,
        component=component,
    )
    raw = _llm_generate_text(prompt_text=prompt, max_new_tokens=CONFIG.get("MAX_NEW_TOKENS", 1024))
    obj = _safe_json_parse(raw)
    # Ensure keys exist
    return {
        "question": obj.get("question"),
        "answer": obj.get("answer"),
        "references": obj.get("references", []),
        "raw": raw,
    }


def pipeline_identify_question_answer(base_query: str) -> Dict[str, Any]:
    # Step 0: initial retrieval to get dataset notifications
    rag0 = run_rag(
        query=base_query,
        embeddings_cache_dir=CONFIG["EMBEDDINGS_CACHE_DIR"],
        embeddings_filename=CONFIG["EMBEDDINGS_FILENAME"],
        training_txt_path=CONFIG["TRAINING_TXT_PATH"],
        top_k=CONFIG["TOP_K"],
        rounds=CONFIG["ROUNDS"],
    )

    cw_training = load_training_text(CONFIG["TRAINING_TXT_PATH"])
    seeds = load_seeds(CONFIG["SEEDS_PATH"])

    # Step 1: identify component
    component = identify_primary_component_from_contexts(cw_training, rag0["contexts"]) or "Circulating Water System"

    # Step 2: retrieve contexts specific to the component
    component_query = f"{component} issues notifications"
    rag_comp = run_rag(
        query=component_query,
        embeddings_cache_dir=CONFIG["EMBEDDINGS_CACHE_DIR"],
        embeddings_filename=CONFIG["EMBEDDINGS_FILENAME"],
        training_txt_path=CONFIG["TRAINING_TXT_PATH"],
        top_k=CONFIG["TOP_K"],
        rounds=CONFIG["ROUNDS"],
    )

    # Step 3: draft question
    draft_q = generate_question_for_component(component, cw_training, seeds, rag_comp["contexts"]) if CONFIG.get("USE_LLM", True) else "What are the recurring issues?"

    # Step 4: combined QA
    qa = generate_combined_qa(component, draft_q, cw_training, seeds, rag_comp["contexts"]) if CONFIG.get("USE_LLM", True) else {"question": draft_q, "answer": rag_comp["answer"], "references": []}

    return {
        "base_query": base_query,
        "identified_component": component,
        "draft_question": draft_q,
        "qa": qa,
        "rag_initial": {"top_scores": rag0["top_scores"], "top_indices": rag0["top_indices"]},
        "rag_component": {"top_scores": rag_comp["top_scores"], "top_indices": rag_comp["top_indices"]},
    }


def _sample_random_contexts_from_corpus(num_chunks: int, char_budget: int) -> Dict[str, Any]:
    data = load_embeddings(CONFIG["EMBEDDINGS_CACHE_DIR"], CONFIG["EMBEDDINGS_FILENAME"])
    chunks = data["chunks"]
    n = len(chunks)
    if n == 0:
        return {"indices": [], "contexts": []}
    rng = np.random.RandomState(CONFIG.get("RANDOM_SEED", 42))
    take = min(num_chunks, n)
    indices = rng.choice(n, size=take, replace=False).tolist()
    texts = [chunks[int(i)]["text"] for i in indices]
    # Respect char budget by trimming join
    joined, used, contexts = [], 0, []
    for t in texts:
        if used >= char_budget:
            break
        can = t[: max(0, char_budget - used)]
        contexts.append(can)
        used += len(can)
    return {"indices": [int(i) for i in indices], "contexts": contexts}


def pipeline_dynamic_identify_question_answer() -> Dict[str, Any]:
    # Step 0: sample notifications to bootstrap without a fixed query
    sample = _sample_random_contexts_from_corpus(
        CONFIG.get("SNIPPET_NUM_CHUNKS", 12), CONFIG.get("SNIPPET_CHAR_BUDGET", 4000)
    )
    cw_training = load_training_text(CONFIG["TRAINING_TXT_PATH"])  # background
    seeds = load_seeds(CONFIG["SEEDS_PATH"])  # examples

    # Step 1: identify a primary component from sampled notifications
    component = identify_primary_component_from_contexts(cw_training, sample["contexts"]) or "Circulating Water System"

    # Step 2: retrieve contexts specific to that component
    rag_comp = run_rag(
        query=f"{component} notifications",
        embeddings_cache_dir=CONFIG["EMBEDDINGS_CACHE_DIR"],
        embeddings_filename=CONFIG["EMBEDDINGS_FILENAME"],
        training_txt_path=CONFIG["TRAINING_TXT_PATH"],
        top_k=CONFIG["TOP_K"],
        rounds=CONFIG["ROUNDS"],
    )

    # Step 3: generate a question about that component
    draft_q = generate_question_for_component(component, cw_training, seeds, rag_comp["contexts"]) if CONFIG.get("USE_LLM", True) else "What are the recurring issues?"

    # Step 4: answer with references using the retrieved contexts
    qa = generate_combined_qa(component, draft_q, cw_training, seeds, rag_comp["contexts"]) if CONFIG.get("USE_LLM", True) else {"question": draft_q, "answer": rag_comp["answer"], "references": []}

    return {
        "sample_indices": sample["indices"],
        "identified_component": component,
        "draft_question": draft_q,
        "qa": qa,
        "rag_component": {"top_scores": rag_comp["top_scores"], "top_indices": rag_comp["top_indices"]},
    }


def main():
    # Read config
    cfg = CONFIG

    # Run dynamic pipeline (no hardcoded query)
    pipeline = pipeline_dynamic_identify_question_answer()

    # Compose output payload
    payload = {
        "config": cfg,
        "pipeline": pipeline,
        "generated_at": datetime.now().isoformat(),
    }

    # Save to outputs directory
    out_dir = ensure_outputs_dir(cfg["OUTPUTS_DIR"])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"rag_pipeline_{ts}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)

    print(f"Saved results to: {out_file}")


if __name__ == "__main__":
    main()
