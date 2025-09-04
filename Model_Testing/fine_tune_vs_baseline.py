# Fine_Tuning/assess_ft_vs_baseline.py
# Compare baseline (RAG + base model) vs fine-tuned (RAG + base + LoRA) on a held-out test set.
# Offline, Windows-friendly, no extra pip deps.

import os
import re
import gc
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer

# =========================== Config ===========================
CONFIG = {
    # Models
    "BASE_MODEL_DIR": r"D:\huggingface\hub\Qwen2.5-7B-Instruct-bnb-4bit",
    "ADAPTER_DIR": r"Fine_Tuning/checkpoints/qwen2_5_7b_qlora_ctx_full",  # fine-tuned LoRA; leave None to skip FT

    # Data
    "EVAL_FILE": r"outputs/test_qa_generated_for_fine_tuning.json",
    "EMBEDDINGS_PATH": r"embeddings_cache/nuclear_notifications_embeddings.pkl",

    # Context policy: ALWAYS use context from stored refined_refs; no truncation
    "REF_TOP_K": 4,                 # cap number of refs per example (preserves order)
    "DEDUP_BY_NOTIFICATION": True,  # collapse multiple chunks for same notification id
    "MAX_INPUT_TOKENS": 10000,      # hard cap for input prompt; skip if exceeded

    # Generation
    "MAX_NEW_TOKENS": 1024,
    "TEMPERATURE": 0.2,
    "TOP_P": 0.9,

    # Runtime
    "USE_SDPA": True,               # good on Windows; no flash-attn install needed
    "DEVICE": 0,                    # default single-GPU device id
    "AUTO_MULTI_GPU": True,         # auto shard across both GPUs when available
    "GPU_MAX_MEMORY_GIB": 19,       # per-GPU cap when sharding
    "OUTPUT_DIR": r"outputs/eval_reports",
    "SYSTEM_PROMPT": "You are a domain expert assistant. Use ONLY the provided notifications.",

    # Summarization of latest eval report (side-by-side Q/A)
    "SUMMARY_OUTPUT_CSV": r"outputs/eval_reports/latest_qa_comparison.csv",
    "SUMMARY_MAX_ROWS": 50,  # preview rows to print in console (set 0 to skip printing)

    # Resume behavior
    "RESUME_LATEST": True,          # if a prior eval_report_*.json exists, continue into it
}

# ======================= Utilities: IO & retrieval =======================
def load_eval_results(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Eval file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", data)
    if not isinstance(results, list):
        raise ValueError("Expected a list under 'results'.")
    return results

def load_embeddings_map(path: str) -> Dict[int, Dict[str, Any]]:
    import pickle
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Embeddings not found at {p}")
    with open(p, "rb") as f:
        obj = pickle.load(f)
    chunks = obj["chunks"]
    return {int(ch.get("row_index", i)): ch for i, ch in enumerate(chunks)}

def _extract_notification_id(label: str) -> str:
    if not isinstance(label, str):
        return ""
    m = re.match(r"^\s*(\d+)", label)
    return m.group(1) if m else label

def limit_refs(refs: List[Dict[str, Any]], top_k: int, dedup_by_notification: bool = True) -> List[Dict[str, Any]]:
    if not refs:
        return []
    out = refs
    if dedup_by_notification:
        seen = set()
        dedup = []
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

def build_blocks_from_refs(refs: List[Dict[str, Any]], by_row: Dict[int, Dict[str, Any]]) -> Tuple[str, List[str]]:
    """
    Build FULL [REF i] blocks from stored refined_refs (no truncation).
    Returns (context_text, allowed_ref_ids) where allowed_ref_ids are like ["[REF 1]", "[REF 2]", ...]
    """
    blocks, allowed = [], []
    for i, r in enumerate(refs or [], start=1):
        ri = int(r.get("row_index", -1))
        ch = by_row.get(ri)
        label = r.get("reference_label", f"row_{ri}")
        if not ch:
            blocks.append(f"[REF {i}] {label}\n(Missing reference text)")
        else:
            text = (ch.get("text") or "")
            blocks.append(f"[REF {i}] {label}\n{text}")
        allowed.append(f"[REF {i}]")
    return "\n\n".join(blocks), allowed

# ======================= Token / metrics helpers =======================
def count_tokens(text: str, tok) -> int:
    return len(tok.encode(text, add_special_tokens=False))

def tokenize_simple(s: str) -> List[str]:
    return re.findall(r"\w+", s.lower())

def f1_token_overlap(pred: str, ref: str) -> float:
    """
    Token-level F1 (precision/recall harmonic mean) on whitespace/word chars.
    """
    p = tokenize_simple(pred)
    r = tokenize_simple(ref)
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    from collections import Counter
    pc, rc = Counter(p), Counter(r)
    common = sum((pc & rc).values())
    if common == 0:
        return 0.0
    precision = common / max(1, sum(pc.values()))
    recall = common / max(1, sum(rc.values()))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def extract_citations(text: str) -> List[str]:
    return re.findall(r"\[REF\s*\d+\]", text)

def citation_metrics(pred: str, allowed_refs: List[str], expected_refs: Optional[List[str]]) -> Dict[str, float]:
    """
    allowed_refs: [ "[REF 1]", "[REF 2]", ... ]  present in built context
    expected_refs: list like from file's 'final_references' (e.g., ["[REF 1]", "[REF 4]"]) or None
    """
    cited = extract_citations(pred)
    allowed_set = set(allowed_refs)
    # precision: how many cited are actually in allowed set
    correct_cites = sum(1 for c in cited if c in allowed_set)
    precision = (correct_cites / len(cited)) if cited else 0.0

    # recall vs expected_refs (if provided)
    recall = 0.0
    if expected_refs:
        exp = set(expected_refs)
        hit = sum(1 for e in exp if e in cited)
        recall = hit / max(1, len(exp))
    return {
        "citation_precision": float(precision),
        "citation_recall_vs_expected": float(recall),
        "num_citations_in_pred": int(len(cited)),
    }

# ======================= Model loading & generation =======================
def load_model_and_tokenizer(base_dir: str, adapter_dir: Optional[str], device: int = 0):
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # Device mapping: shard over multiple GPUs if available and requested
    n = torch.cuda.device_count()
    if CONFIG.get("AUTO_MULTI_GPU", True) and n >= 2:
        device_map = "balanced_low_0"
        max_memory = {i: f"{CONFIG.get('GPU_MAX_MEMORY_GIB', 19)}GiB" for i in range(n)}
    else:
        device_map = {"": device}
        max_memory = None
    model = AutoModelForCausalLM.from_pretrained(
        base_dir,
        quantization_config=quant_cfg,
        device_map=device_map,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation=("sdpa" if CONFIG["USE_SDPA"] else None),
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)

    # Reduce KV growth a bit for long contexts
    model.config.use_cache = False
    model.eval()
    return tok, model

def build_prompt(system_prompt: str, context: str, question: str, tok) -> str:
    user = (
        f"# ─── Context Notifications ─────────────────────────────\n{context}\n\n"
        f"# ─── Your Task ─────────────────────────────────────────\n"
        f"Answer the user's question concisely using only the context and include citations like [REF 1], [REF 2].\n"
        f"Question: {question}"
    )
    if hasattr(tok, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ]
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        return system_prompt + "\n\n" + user

def generate_one(prompt: str, tok, model, max_new_tokens: int, temperature: float, top_p: float) -> str:
    inputs = tok(prompt, return_tensors="pt")
    dev = next(model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    # Track peak memory for this single call
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(dev)

    t0 = time.perf_counter()
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tok.eos_token_id,
            pad_token_id=getattr(tok, "pad_token_id", tok.eos_token_id),
            use_cache=False,
        )
    latency = time.perf_counter() - t0

    text = tok.decode(out_ids[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    peak_mem_gb = 0.0
    if torch.cuda.is_available():
        peak_mem_gb = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)

    # Light cleanup per-sample to avoid creep
    del inputs, out_ids
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return text, latency, peak_mem_gb

# ======================= Main eval loop =======================
def run_system(
    name: str,
    base_dir: str,
    adapter_dir: Optional[str],
    eval_items: List[Dict[str, Any]],
    by_row: Dict[int, Dict[str, Any]],
    retriever: SentenceTransformer,  # not used (we rely on stored refs), but kept for flexibility
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    print(f"\n=== Loading system: {name} ===")
    tok, model = load_model_and_tokenizer(base_dir, adapter_dir, device=cfg["DEVICE"])

    per_sample: List[Dict[str, Any]] = []
    skipped = []

    for i, r in enumerate(eval_items, start=1):
        q = (r.get("final_question") or r.get("question") or r.get("draft_question") or "").strip()
        ref_answer = (r.get("final_answer") or r.get("answer") or "").strip()
        refs = r.get("refined_refs") or []
        refs = limit_refs(refs, cfg["REF_TOP_K"], cfg["DEDUP_BY_NOTIFICATION"])
        if not q or not ref_answer or not refs:
            skipped.append({"idx": i - 1, "reason": "missing question/answer/refs"})
            continue

        context, allowed_refs = build_blocks_from_refs(refs, by_row)
        prompt = build_prompt(cfg["SYSTEM_PROMPT"], context, q, tok)

        # Hard budget: do NOT truncate; skip if too long
        n_in = count_tokens(prompt, tok)
        if n_in > cfg["MAX_INPUT_TOKENS"]:
            skipped.append({"idx": i - 1, "reason": f"input_tokens {n_in} > MAX_INPUT_TOKENS"})
            continue

        pred, latency_s, peak_mem_gb = generate_one(
            prompt, tok, model, cfg["MAX_NEW_TOKENS"], cfg["TEMPERATURE"], cfg["TOP_P"]
        )

        # Metrics
        f1 = f1_token_overlap(pred, ref_answer)
        cm = citation_metrics(pred, allowed_refs, r.get("final_references"))

        record = {
            "idx": i - 1,
            "question": q,
            "reference_answer": ref_answer,
            "predicted_answer": pred,
            "latency_sec": latency_s,
            "peak_mem_gb": peak_mem_gb,
            "token_f1": f1,
            **cm,
            "allowed_refs": allowed_refs,
            "expected_refs": r.get("final_references"),
        }
        per_sample.append(record)

        # Streaming write/append to checkpoint file for this system
        try:
            ckpt_path = Path(CONFIG["OUTPUT_DIR"]) / f"_stream_{name}.jsonl"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(ckpt_path, "a", encoding="utf-8") as f:
                import json as _json
                f.write(_json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

        if i % 10 == 0:
            print(f"  [{name}] processed {i}/{len(eval_items)} ...")

    # Aggregate
    def _avg(key, default=0.0):
        vals = [x[key] for x in per_sample if isinstance(x.get(key), (int, float))]
        return float(sum(vals) / max(1, len(vals))) if vals else default

    report = {
        "system_name": name,
        "num_items": len(eval_items),
        "num_evaluated": len(per_sample),
        "num_skipped": len(skipped),
        "metrics": {
            "avg_token_f1": _avg("token_f1"),
            "avg_latency_sec": _avg("latency_sec"),
            "avg_peak_mem_gb": _avg("peak_mem_gb"),
            "avg_citation_precision": _avg("citation_precision"),
            "avg_citation_recall_vs_expected": _avg("citation_recall_vs_expected"),
        },
        "details": per_sample,
        "skipped": skipped,
    }

    # Free the model before loading the next system
    del tok, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return report

# ======================= Report summarization =======================
def _latest_eval_report(dir_path: str) -> Path:
    p = Path(dir_path)
    cands = sorted(p.glob("eval_report_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No eval_report_*.json found in {p}")
    return cands[0]

def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def summarize_report(report_path: Path, out_csv: Path, max_rows: int = 50) -> None:
    obj = _read_json(report_path)
    reports = obj.get("reports", [])
    if len(reports) < 2:
        print("[WARN] Expected two systems in report (baseline and fine-tuned). Skipping summary.")
        return

    sysA, sysB = reports[0], reports[1]
    nameA = sysA.get("system_name", "system_A")
    nameB = sysB.get("system_name", "system_B")
    A = {d.get("idx"): d for d in sysA.get("details", [])}
    B = {d.get("idx"): d for d in sysB.get("details", [])}

    rows: List[Dict[str, Any]] = []
    idxs = sorted(set(A.keys()) & set(B.keys()))
    for i in idxs:
        a, b = A[i], B[i]
        rows.append({
            "idx": i,
            "question": a.get("question") or b.get("question"),
            "gold_answer": a.get("reference_answer"),
            f"{nameA}__answer": a.get("predicted_answer"),
            f"{nameB}__answer": b.get("predicted_answer"),
            f"{nameA}__token_f1": a.get("token_f1"),
            f"{nameB}__token_f1": b.get("token_f1"),
            f"{nameA}__latency_sec": a.get("latency_sec"),
            f"{nameB}__latency_sec": b.get("latency_sec"),
            f"{nameA}__cit_prec": a.get("citation_precision"),
            f"{nameB}__cit_prec": b.get("citation_precision"),
        })

    # Write CSV (no extra deps)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    try:
        import csv
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                w.writeheader()
                w.writerows(rows)
        print(f"Wrote comparison CSV: {out_csv}")
    except Exception as e:
        print(f"[WARN] Could not write CSV: {e}")

    # Console preview
    if max_rows and max_rows > 0:
        preview = rows[:max_rows]
        for r in preview:
            q = (r.get("question") or "").strip()
            print("\n=== idx:", r["idx"], "===")
            print("Q:", (q[:240] + ("…" if len(q) > 240 else "")))
            print(f"{nameA}:", (str(r.get(f"{nameA}__answer") or "").strip()[:240] + ("…" if len(str(r.get(f"{nameA}__answer") or "")) > 240 else "")))
            print(f"{nameB}:", (str(r.get(f"{nameB}__answer") or "").strip()[:240] + ("…" if len(str(r.get(f"{nameB}__answer") or "")) > 240 else "")))

def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cuda.matmul.allow_tf32 = True

    cfg = CONFIG
    Path(cfg["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)

    # 0) Determine resume mode
    latest_report_path = None
    if cfg.get("RESUME_LATEST", True):
        try:
            latest_report_path = _latest_eval_report(cfg["OUTPUT_DIR"])
            print(f"[resume] Found latest report: {latest_report_path}")
        except Exception:
            latest_report_path = None

    # 1) Load eval set + embeddings map (for full-context blocks)
    eval_items = load_eval_results(cfg["EVAL_FILE"])
    by_row = load_embeddings_map(cfg["EMBEDDINGS_PATH"])

    # We keep the retriever available if you ever want to fall back to fresh retrieval.
    # For fairness here, we ONLY use stored refined_refs to build context.
    # (Avoids drift; keeps [REF i] alignment with test file.)
    with open(cfg["EMBEDDINGS_PATH"], "rb") as f:
        import pickle
        emb_obj = pickle.load(f)
    retriever = SentenceTransformer(emb_obj["model_name"], device="cpu")

    systems = [
        {"name": "baseline_qwen2.5_7b_instruct_4bit", "adapter": None},
        {"name": "finetuned_qwen2.5_7b_lora", "adapter": cfg["ADAPTER_DIR"]},
    ]

    all_reports = []

    # If resuming and a combined report exists, prefer continuing by skipping idx that already exist
    skip_idxs_by_system = {}
    if latest_report_path is not None:
        try:
            prev = _read_json(latest_report_path)
            for sys_rep in prev.get("reports", []):
                name = sys_rep.get("system_name")
                seen = {int(d.get("idx")) for d in sys_rep.get("details", []) if d.get("idx") is not None}
                skip_idxs_by_system[name] = seen
            print({k: len(v) for k, v in skip_idxs_by_system.items()})
        except Exception:
            skip_idxs_by_system = {}

    for s in systems:
        name = s["name"]
        if name in skip_idxs_by_system and skip_idxs_by_system[name]:
            # Filter eval_items to only those not yet processed
            remaining = []
            for i, r in enumerate(eval_items):
                if i not in skip_idxs_by_system[name]:
                    remaining.append(r)
            print(f"[{name}] Resuming: {len(remaining)} remaining out of {len(eval_items)}")
            items = remaining
        else:
            items = eval_items

        rep = run_system(
            name,
            cfg["BASE_MODEL_DIR"],
            s["adapter"],
            items,
            by_row,
            retriever,
            cfg,
        )
        print(f"\n[{name}] Aggregate metrics:")
        for k, v in rep["metrics"].items():
            print(f"  {k}: {v:.4f}")
        all_reports.append(rep)

    # 3) Save combined report
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(cfg["OUTPUT_DIR"]) / f"eval_report_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "reports": all_reports}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved evaluation report to: {out_path}")

    # 4) Summarize latest report (including just-created one)
    try:
        latest = _latest_eval_report(cfg["OUTPUT_DIR"])
        summarize_report(latest, Path(cfg["SUMMARY_OUTPUT_CSV"]), cfg.get("SUMMARY_MAX_ROWS", 50))
    except Exception as e:
        print(f"[WARN] Summary step skipped: {e}")

if __name__ == "__main__":
    main()
