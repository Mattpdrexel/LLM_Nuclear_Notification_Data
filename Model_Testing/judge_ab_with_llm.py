# judge_ab_with_llm.py
# Offline A/B judging: baseline vs. fine-tuned using a local "judge" LLM.
# - Loads latest combined eval_report_*.json produced by assess_ft_vs_baseline.py
# - Rebuilds FULL context from test file refined_refs (no truncation in prompt)
# - Uses Qwen2.5-32B-Instruct-bnb-4bit as judge to score both answers
# - Writes an Excel sheet with judge scores + placeholders for human review

import os
import re
import gc
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ============================ CONFIG ============================

CONFIG = {
    # ---- Inputs ----
    # Test file used in evaluation (we need refined_refs & gold answers)
    "TEST_FILE": r"outputs/test_qa_generated_for_fine_tuning.json",
    # Embeddings cache to rebuild full notifications from row_index
    "EMBEDDINGS_PATH": r"embeddings_cache/nuclear_notifications_embeddings.pkl",
    # Directory containing eval_report_*.json (created by assess_ft_vs_baseline.py)
    "EVAL_REPORT_DIR": r"outputs/eval_reports",
    # If you want to point to a specific eval report, set this; otherwise latest will be used
    "EVAL_REPORT_FILE": None,  # e.g., r"outputs/eval_reports/eval_report_20250830_120101.json"

    # ---- Judge model (local, offline) ----
    "JUDGE_MODEL_DIR": r"D:\huggingface\hub\Qwen2.5-32B-Instruct-bnb-4bit",
    "JUDGE_DEVICE_MAP": "balanced_low_0",  # shard across 2×20GB
    "JUDGE_MAX_MEMORY_GB": 19,             # per GPU cap
    "JUDGE_MAX_NEW_TOKENS": 384,
    "JUDGE_TEMPERATURE": 0.0,
    "JUDGE_TOP_P": 0.9,
    "JUDGE_SYSTEM_PROMPT": (
        "You are a meticulous evaluation assistant. Judge the two answers strictly using ONLY the provided context "
        "and the provided reference answer. Do not rely on outside knowledge. Penalize unsupported claims or missing citations."
    ),

    # ---- Prompting policy ----
    "ALWAYS_USE_FULL_CONTEXT": True,  # we rebuild full [REF i] blocks; no truncation in prompt
    "REF_TOP_K": 4,                   # optional cap and dedup to keep context sane (preserves order)
    "DEDUP_BY_NOTIFICATION": True,

    # ---- Output ----
    "OUTPUT_XLSX": r"outputs/eval_reports/llm_judge_report.xlsx",
    "ALSO_WRITE_CSV": True,           # fallback CSV for very large cells
}

# ============================ IO utils ============================

def _latest_eval_report(dir_path: str) -> Path:
    p = Path(dir_path)
    cands = sorted(p.glob("eval_report_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No eval_report_*.json found in {p}")
    return cands[0]

def load_eval_report(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_test_results(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    results = obj.get("results", obj)
    if not isinstance(results, list):
        raise ValueError("Expected list under 'results'.")
    return results

def load_embeddings_map(path: str) -> Dict[int, Dict[str, Any]]:
    import pickle
    with open(path, "rb") as f:
        data = pickle.load(f)
    chunks = data["chunks"]
    return {int(ch.get("row_index", i)): ch for i, ch in enumerate(chunks)}

# ============================ context utils ============================

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

def build_full_blocks_from_refs(refs: List[Dict[str, Any]], by_row: Dict[int, Dict[str, Any]]) -> Tuple[str, List[str], List[str]]:
    """
    Returns:
      context_text: concatenated [REF i] blocks (FULL text)
      allowed_ref_ids: ["[REF 1]", ...]
      allowed_labels:  ["20751312 – LEAK: ...", ...]
    """
    blocks, allowed_ids, labels = [], [], []
    for i, r in enumerate(refs or [], start=1):
        ri = int(r.get("row_index", -1))
        ch = by_row.get(ri)
        label = r.get("reference_label", f"row_{ri}")
        labels.append(label)
        if not ch:
            blocks.append(f"[REF {i}] {label}\n(Missing reference text)")
        else:
            blocks.append(f"[REF {i}] {label}\n{ch.get('text','')}")
        allowed_ids.append(f"[REF {i}]")
    return "\n\n".join(blocks), allowed_ids, labels

# ============================ judge model ============================

def load_judge(model_dir: str, device_map: str, max_mem_gb: int):
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # shard across both GPUs for capacity
    max_memory = {i: f"{max_mem_gb}GiB" for i in range(torch.cuda.device_count())} if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quant_cfg,
        device_map=device_map,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="sdpa",  # robust on Windows
    )
    model.config.use_cache = False
    model.eval()
    return tok, model

def judge_prompt(system: str, question: str, context: str, gold: str,
                 ansA: str, ansB: str, expected_refs: Optional[List[str]], tok) -> str:
    exp = f"Expected citations: {', '.join(expected_refs)}" if expected_refs else "Expected citations: (not provided)"
    user = (
        "You will evaluate two candidate answers (A and B) to a user question.\n"
        "Use ONLY the provided Context and the provided Gold answer. Do NOT use outside knowledge.\n\n"
        f"Question:\n{question}\n\n"
        f"Context (FULL notifications):\n{context}\n\n"
        f"Gold reference answer:\n{gold}\n\n"
        f"{exp}\n\n"
        "Candidate A:\n"
        f"{ansA}\n\n"
        "Candidate B:\n"
        f"{ansB}\n\n"
        "Scoring rules:\n"
        "1) Factual accuracy & grounding (0–1): claims must be supported by the Context and align with Gold.\n"
        "2) Completeness (0–1): covers key facts needed to answer.\n"
        "3) Clarity (0–1): understandable and well-structured.\n"
        "4) Citation validity (0–1): only cite [REF i] that exist in Context; higher if correctly cites relevant refs.\n"
        "Return STRICT JSON only, with keys:\n"
        "{\n"
        '  "A": {"accuracy": float, "completeness": float, "clarity": float, "citations": float, "hallucination": bool},\n'
        '  "B": {"accuracy": float, "completeness": float, "clarity": float, "citations": float, "hallucination": bool},\n'
        '  "winner": "A" | "B" | "tie",\n'
        '  "rationale": "brief explanation"\n'
        "}\n"
        "Do not include any extra text before or after the JSON."
    )
    if hasattr(tok, "apply_chat_template"):
        msgs = [
            {"role": "system", "content": system},
            {"role": "user",    "content": user},
        ]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return system + "\n\n" + user

def _safe_json_extract(text: str) -> Optional[Dict[str, Any]]:
    # Extract the first JSON object from text
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    s = m.group(0)
    try:
        return json.loads(s)
    except Exception:
        # try to fix common trailing commas etc. if needed
        try:
            s = re.sub(r",\s*}", "}", s)
            s = re.sub(r",\s*]", "]", s)
            return json.loads(s)
        except Exception:
            return None

def judge_once(prompt: str, tok, model, max_new_tokens: int, temperature: float, top_p: float) -> Dict[str, Any]:
    dev = next(model.parameters()).device
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}
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
    text = tok.decode(out_ids[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    del inputs, out_ids
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    data = _safe_json_extract(text)
    if data is None:
        # hard fallback if judge responded oddly
        data = {"A": {"accuracy": 0.0, "completeness": 0.0, "clarity": 0.0, "citations": 0.0, "hallucination": True},
                "B": {"accuracy": 0.0, "completeness": 0.0, "clarity": 0.0, "citations": 0.0, "hallucination": True},
                "winner": "tie", "rationale": "parse_error"}
    return data

# ============================ main ============================

def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cuda.matmul.allow_tf32 = True

    cfg = CONFIG
    Path(Path(cfg["OUTPUT_XLSX"]).parent).mkdir(parents=True, exist_ok=True)

    # 1) Locate & load eval report (two systems present)
    report_path = cfg["EVAL_REPORT_FILE"] or str(_latest_eval_report(cfg["EVAL_REPORT_DIR"]))
    eval_report = load_eval_report(report_path)
    reports = eval_report.get("reports", [])
    if len(reports) < 2:
        raise RuntimeError("Expected a combined report with two systems (baseline and fine-tuned).")

    # Map details by idx for each system
    sysA = reports[0]  # assume baseline first
    sysB = reports[1]  # fine-tuned second
    A_name = sysA.get("system_name", "system_A")
    B_name = sysB.get("system_name", "system_B")
    A = {d["idx"]: d for d in sysA.get("details", [])}
    B = {d["idx"]: d for d in sysB.get("details", [])}

    # 2) Load test items + embeddings map for full context blocks
    test_items = load_test_results(cfg["TEST_FILE"])
    by_row = load_embeddings_map(cfg["EMBEDDINGS_PATH"])

    # 3) Load judge LLM
    tok_j, judge = load_judge(cfg["JUDGE_MODEL_DIR"], cfg["JUDGE_DEVICE_MAP"], cfg["JUDGE_MAX_MEMORY_GB"])

    rows = []
    for idx, item in enumerate(test_items):
        if idx not in A or idx not in B:
            # was likely skipped by one of the systems; ignore in judge pass
            continue

        q = (item.get("final_question") or item.get("question") or item.get("draft_question") or "").strip()
        gold = (item.get("final_answer") or item.get("answer") or "").strip()
        refs = item.get("refined_refs") or []
        if not q or not gold or not refs:
            continue

        # Keep ordering, optional dedup + top-k
        refs_limited = limit_refs(refs, cfg["REF_TOP_K"], cfg["DEDUP_BY_NOTIFICATION"])
        context, allowed_ids, labels = build_full_blocks_from_refs(refs_limited, by_row)

        # Candidate answers from prior eval run
        ansA = A[idx]["predicted_answer"]
        ansB = B[idx]["predicted_answer"]

        prompt = judge_prompt(
            cfg["JUDGE_SYSTEM_PROMPT"], q, context, gold, ansA, ansB, item.get("final_references"), tok_j
        )
        result = judge_once(
            prompt, tok_j, judge, cfg["JUDGE_MAX_NEW_TOKENS"], cfg["JUDGE_TEMPERATURE"], cfg["JUDGE_TOP_P"]
        )

        # Assemble row for Excel (plus placeholders for human review)
        rows.append({
            "idx": idx,
            "question": q,
            "gold_answer": gold,
            "ref_ids_in_context": ", ".join(allowed_ids),
            "ref_labels_in_context": " | ".join(labels),

            f"{A_name}__answer": ansA,
            f"{B_name}__answer": ansB,

            f"{A_name}__latency_sec": A[idx].get("latency_sec"),
            f"{B_name}__latency_sec": B[idx].get("latency_sec"),
            f"{A_name}__peak_mem_gb": A[idx].get("peak_mem_gb"),
            f"{B_name}__peak_mem_gb": B[idx].get("peak_mem_gb"),

            f"{A_name}__token_f1_vs_gold": A[idx].get("token_f1"),
            f"{B_name}__token_f1_vs_gold": B[idx].get("token_f1"),
            f"{A_name}__cit_prec": A[idx].get("citation_precision"),
            f"{B_name}__cit_prec": B[idx].get("citation_precision"),
            f"{A_name}__cit_recall_vs_expected": A[idx].get("citation_recall_vs_expected"),
            f"{B_name}__cit_recall_vs_expected": B[idx].get("citation_recall_vs_expected"),

            "judge_A_accuracy": result.get("A", {}).get("accuracy"),
            "judge_A_completeness": result.get("A", {}).get("completeness"),
            "judge_A_clarity": result.get("A", {}).get("clarity"),
            "judge_A_citations": result.get("A", {}).get("citations"),
            "judge_A_hallucination": result.get("A", {}).get("hallucination"),

            "judge_B_accuracy": result.get("B", {}).get("accuracy"),
            "judge_B_completeness": result.get("B", {}).get("completeness"),
            "judge_B_clarity": result.get("B", {}).get("clarity"),
            "judge_B_citations": result.get("B", {}).get("citations"),
            "judge_B_hallucination": result.get("B", {}).get("hallucination"),

            "judge_winner": result.get("winner"),
            "judge_rationale": result.get("rationale"),

            # ---- Human-in-the-loop placeholders ----
            "human_correct_A": "",         # Y/N
            "human_correct_B": "",         # Y/N
            "human_preference": "",        # baseline/finetuned/tie
            "human_notes": "",
        })

        if (idx + 1) % 10 == 0:
            print(f"Judged {idx + 1}/{len(test_items)} samples...")

    # 4) Write Excel (with CSV fallback)
    df = pd.DataFrame(rows)
    out_xlsx = Path(CONFIG["OUTPUT_XLSX"])
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_excel(out_xlsx, index=False)
        print(f"Saved LLM judge report to: {out_xlsx}")
    except Exception as e:
        print(f"[WARN] Could not write Excel ({e}). Writing CSV fallback.")
        out_csv = out_xlsx.with_suffix(".csv")
        df.to_csv(out_csv, index=False)
        print(f"Saved CSV to: {out_csv}")

    # Optional also CSV
    if CONFIG["ALSO_WRITE_CSV"]:
        out_csv = out_xlsx.with_suffix(".csv")
        try:
            df.to_csv(out_csv, index=False)
            print(f"(Also) Saved CSV to: {out_csv}")
        except Exception as e:
            print(f"[WARN] CSV write failed: {e}")

    # Clean up judge
    del tok_j, judge
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
