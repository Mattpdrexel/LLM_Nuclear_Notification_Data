import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_XLSX = Path("outputs/eval_reports/llm_judge_report.xlsx")
PLOTS_DIR = Path("outputs/eval_reports/plots")
EVAL_DIR = Path("outputs/eval_reports")


def _infer_system_names(df: pd.DataFrame) -> Tuple[str, str, str, str]:
    """Return (prefixA, prefixB, labelA, labelB).
    prefix* are the column name prefixes, label* are short display names.
    """
    # Look for columns ending with "__answer" to infer system name prefixes
    cand = [c for c in df.columns if c.endswith("__answer")]
    if len(cand) >= 2:
        # keep order of appearance
        a = cand[0].split("__")[0]
        b = cand[1].split("__")[0]
    else:
        a, b = "system_A", "system_B"

    # Short display labels
    la = "baseline"
    lb = "qlora"
    al = a.lower()
    bl = b.lower()
    if ("finetuned" in al or "lora" in al) and ("baseline" in bl or "base" in bl):
        la, lb = "qlora", "baseline"
    elif ("finetuned" in bl or "lora" in bl) and ("baseline" in al or "base" in al):
        la, lb = "baseline", "qlora"
    # otherwise keep default first->baseline, second->qlora
    return a, b, la, lb


def _has_cols(df: pd.DataFrame, cols: List[str]) -> bool:
    return all(c in df.columns for c in cols)


def plot_judge_scores(df: pd.DataFrame, prefixA: str, prefixB: str, labelA: str, labelB: str, out_dir: Path):
    metrics = ["accuracy", "completeness", "clarity", "citations"]
    meansA, meansB = [], []
    for m in metrics:
        ca = f"judge_A_{m}"
        cb = f"judge_B_{m}"
        if ca in df.columns and cb in df.columns:
            meansA.append(df[ca].dropna().astype(float).mean())
            meansB.append(df[cb].dropna().astype(float).mean())
        else:
            meansA.append(float("nan"))
            meansB.append(float("nan"))

    x = range(len(metrics))
    width = 0.35
    plt.figure(figsize=(8, 4.5))
    plt.bar([i - width/2 for i in x], meansA, width=width, label=labelA)
    plt.bar([i + width/2 for i in x], meansB, width=width, label=labelB)
    plt.xticks(list(x), [m.title() for m in metrics])
    plt.ylim(0, 1)
    plt.ylabel("Average score")
    plt.title(f"Judge scores (avg): {labelA} vs {labelB}")
    plt.legend()
    out = out_dir / "judge_scores_bar.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


def plot_winner_counts(df: pd.DataFrame, labelA: str, labelB: str, out_dir: Path):
    if "judge_winner" not in df.columns:
        return
    counts = df["judge_winner"].fillna("").str.lower().value_counts()
    a = int(counts.get("a", 0))
    b = int(counts.get("b", 0))
    tie = int(counts.get("tie", 0))
    labels = [labelA, labelB, "tie"]
    values = [a, b, tie]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=["tab:blue", "tab:orange", "tab:gray"])
    plt.title(f"Judge winner counts: {labelA} vs {labelB}")
    plt.ylabel("Count")
    out = out_dir / "judge_winners_bar.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


def plot_token_f1(df: pd.DataFrame, prefixA: str, prefixB: str, labelA: str, labelB: str, out_dir: Path):
    colA = f"{prefixA}__token_f1"
    colB = f"{prefixB}__token_f1"
    # Some versions label token_f1 as token_f1_vs_gold
    if colA not in df.columns:
        altA = f"{prefixA}__token_f1_vs_gold"
        if altA in df.columns:
            colA = altA
    if colB not in df.columns:
        altB = f"{prefixB}__token_f1_vs_gold"
        if altB in df.columns:
            colB = altB
    if colA not in df.columns or colB not in df.columns:
        return

    data = [df[colA].dropna().astype(float), df[colB].dropna().astype(float)]
    plt.figure(figsize=(6.5, 4))
    plt.boxplot(data, labels=[labelA, labelB], showmeans=True)
    plt.ylabel("Token F1 vs gold")
    plt.title(f"Token F1 distribution: {labelA} vs {labelB}")
    out = out_dir / "token_f1_box.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


def plot_latency_vs_f1(df: pd.DataFrame, prefixA: str, prefixB: str, labelA: str, labelB: str, out_dir: Path):
    latA, latB = f"{prefixA}__latency_sec", f"{prefixB}__latency_sec"
    f1A, f1B = f"{prefixA}__token_f1", f"{prefixB}__token_f1"
    if f1A not in df.columns:
        altA = f"{prefixA}__token_f1_vs_gold"
        if altA in df.columns:
            f1A = altA
    if f1B not in df.columns:
        altB = f"{prefixB}__token_f1_vs_gold"
        if altB in df.columns:
            f1B = altB
    need = [latA, f1A, latB, f1B]
    if not _has_cols(df, [c for c in need if c]):
        return

    plt.figure(figsize=(7, 4.5))
    if latA in df.columns and f1A in df.columns:
        plt.scatter(df[latA], df[f1A], alpha=0.5, label=labelA)
    if latB in df.columns and f1B in df.columns:
        plt.scatter(df[latB], df[f1B], alpha=0.5, label=labelB)
    plt.xlabel("Latency (s)")
    plt.ylabel("Token F1 vs gold")
    plt.title(f"Latency vs F1: {labelA} vs {labelB}")
    plt.legend()
    out = out_dir / "latency_vs_f1_scatter.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


def main():
    # --- Visualize LLM judge XLSX ---
    xlsx = DEFAULT_XLSX
    if xlsx.exists():
        print(f"Loading judge report: {xlsx}")
        df = pd.read_excel(xlsx)
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)

        prefixA, prefixB, labelA, labelB = _infer_system_names(df)
        print(f"Detected systems (prefixes): {prefixA} vs {prefixB}")
        print(f"Display labels: {labelA} vs {labelB}")

        plot_judge_scores(df, prefixA, prefixB, labelA, labelB, PLOTS_DIR)
        plot_winner_counts(df, labelA, labelB, PLOTS_DIR)
        plot_token_f1(df, prefixA, prefixB, labelA, labelB, PLOTS_DIR)
        plot_latency_vs_f1(df, prefixA, prefixB, labelA, labelB, PLOTS_DIR)

        # Text summary report
        summary_path = PLOTS_DIR / "analysis_summary.txt"
        try:
            lines: List[str] = []
            # Winner counts
            if "judge_winner" in df.columns:
                counts = df["judge_winner"].fillna("").str.lower().value_counts()
                w_a, w_b, w_t = int(counts.get("a", 0)), int(counts.get("b", 0)), int(counts.get("tie", 0))
                lines.append(f"Winners: {labelA}={w_a}, {labelB}={w_b}, tie={w_t}")
            # Judge means
            def _mean(col: str) -> float:
                return float(df[col].dropna().astype(float).mean()) if col in df.columns else float('nan')
            j_metrics = ["accuracy", "completeness", "clarity", "citations"]
            for m in j_metrics:
                a, b = _mean(f"judge_A_{m}"), _mean(f"judge_B_{m}")
                lines.append(f"Judge {m}: {labelA}={a:.3f}, {labelB}={b:.3f}")
            # Token F1 & latency means
            def _col_any(prefix: str, base: str, alt: str) -> str:
                c = f"{prefix}__{base}"
                if c in df.columns:
                    return c
                c2 = f"{prefix}__{alt}"
                return c2 if c2 in df.columns else ""
            f1A = _col_any(prefixA, "token_f1", "token_f1_vs_gold")
            f1B = _col_any(prefixB, "token_f1", "token_f1_vs_gold")
            latA, latB = f"{prefixA}__latency_sec", f"{prefixB}__latency_sec"
            if f1A and f1B:
                lines.append(f"Token F1 mean: {labelA}={_mean(f1A):.3f}, {labelB}={_mean(f1B):.3f}")
            if latA in df.columns and latB in df.columns:
                lines.append(f"Latency mean (s): {labelA}={_mean(latA):.2f}, {labelB}={_mean(latB):.2f}")

            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            print("\n".join(lines))
            print(f"Saved summary: {summary_path}")
        except Exception as e:
            print(f"[WARN] Could not write summary: {e}")
    else:
        print(f"[WARN] Judge report not found: {xlsx}")

    # --- Visualize baseline vs finetuned eval report JSON (from fine_tune_vs_baseline.py) ---
    try:
        latest_json = sorted(EVAL_DIR.glob("eval_report_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[0]
    except Exception:
        latest_json = None
    if latest_json is None:
        print("[WARN] No eval_report_*.json found for additional metrics.")
        return

    print(f"Loading eval report: {latest_json}")
    # Robust JSON load (avoid pandas read_json ordering issues)
    with open(latest_json, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    reports = raw.get('reports', [])
    if len(reports) < 2:
        print("[WARN] eval report has fewer than 2 systems.")
        return
    sysA, sysB = reports[0], reports[1]
    nameA = sysA.get('system_name', 'baseline')
    nameB = sysB.get('system_name', 'qlora')
    labelA = 'baseline' if 'finetuned' not in nameA.lower() else 'qlora'
    labelB = 'qlora' if 'finetuned' in nameB.lower() else 'baseline'

    dfA = pd.DataFrame(sysA.get('details', []))
    dfB = pd.DataFrame(sysB.get('details', []))

    def _boxplot_metric(metric: str, title: str, fname: str):
        if metric not in dfA.columns or metric not in dfB.columns:
            return
        plt.figure(figsize=(6.5, 4))
        data = [pd.to_numeric(dfA[metric], errors='coerce').dropna(), pd.to_numeric(dfB[metric], errors='coerce').dropna()]
        plt.boxplot(data, labels=[labelA, labelB], showmeans=True)
        plt.title(title)
        out = PLOTS_DIR / fname
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        print(f"Saved: {out}")

    _boxplot_metric('token_f1', 'Token F1 (eval report)', 'token_f1_box_eval.png')
    _boxplot_metric('latency_sec', 'Latency (s) (eval report)', 'latency_box_eval.png')
    _boxplot_metric('peak_mem_gb', 'Peak GPU memory (GiB) (eval report)', 'peak_mem_box_eval.png')
    _boxplot_metric('citation_precision', 'Citation precision (eval report)', 'cit_prec_box_eval.png')
    _boxplot_metric('citation_recall_vs_expected', 'Citation recall vs expected (eval report)', 'cit_recall_box_eval.png')

    # Additional summary for eval report
    eval_summary = PLOTS_DIR / 'analysis_summary_eval.txt'
    try:
        def _m(s: pd.Series) -> float:
            return float(pd.to_numeric(s, errors='coerce').dropna().mean())
        lines = [
            f"Eval means — {labelA} vs {labelB}",
            f"token_f1: {_m(dfA.get('token_f1', pd.Series())):.3f} vs {_m(dfB.get('token_f1', pd.Series())):.3f}",
            f"latency_sec: {_m(dfA.get('latency_sec', pd.Series())):.2f} vs {_m(dfB.get('latency_sec', pd.Series())):.2f}",
            f"peak_mem_gb: {_m(dfA.get('peak_mem_gb', pd.Series())):.2f} vs {_m(dfB.get('peak_mem_gb', pd.Series())):.2f}",
            f"citation_precision: {_m(dfA.get('citation_precision', pd.Series())):.3f} vs {_m(dfB.get('citation_precision', pd.Series())):.3f}",
            f"citation_recall_vs_expected: {_m(dfA.get('citation_recall_vs_expected', pd.Series())):.3f} vs {_m(dfB.get('citation_recall_vs_expected', pd.Series())):.3f}",
        ]
        with open(eval_summary, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines))
        print(f"Saved summary: {eval_summary}")
    except Exception as e:
        print(f"[WARN] Could not write eval summary: {e}")

    print(f"Plots written to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()


