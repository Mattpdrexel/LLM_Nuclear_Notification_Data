## LLM Nuclear Notification Data – RAG + QLoRA Fine-Tuning

This repo builds a Retrieval-Augmented Generation (RAG) system over plant notification data, generates synthetic Q/A datasets, fine-tunes a compact model with QLoRA, and provides interactive inference and A/B evaluation utilities.

### Project structure
- `QA_Generation/embeddings.py`: Load Excel notifications and create one embedding per full notification.
- `QA_Generation/question_answer_generation.py`: Generate answerable Q/A pairs with RAG using a larger model.
- `QA_Generation/idk_questions.py`: Generate “unanswerable” questions to discourage hallucinations.
- `Fine_Tuning/train_qlora.py`: QLoRA fine-tuning (4-bit) of `Qwen2.5-7B-Instruct` using the generated datasets.
- `Inference/chat.py`: Interactive RAG chat against the fine-tuned adapters.
- `Model_Testing/`: Offline comparisons of baseline vs fine-tuned and LLM-judge evaluation.

### Setup
- Python: 3.11.x recommended
- Create venv and install deps:
  ```bash
  python -m venv .venv
  .\.venv\Scripts\pip install -r requirements.txt --no-input --disable-pip-version-check
  ```
- GPU notes:
  - PyTorch should match your CUDA runtime (this repo used CUDA 12.6 builds: `+cu126`).
  - Large models can be sharded across multiple GPUs for inference/eval; training (QLoRA) uses one GPU by default.

### 1) Build the embedding index
- Source: Excel notifications (each DataFrame row becomes one full-text chunk).
- Script: `QA_Generation/embeddings.py`
- Defaults: reads `raw_data/salem_cw_data.xlsx`, writes `embeddings_cache/nuclear_notifications_embeddings.pkl`.
  ```bash
  .\.venv\Scripts\python.exe QA_Generation\embeddings.py
  ```
- Notes:
  - Uses `BAAI/bge-m3` sentence embeddings.
  - Each notification is kept intact (no splitting) to enable precise citation later.

### 2) Generate Q/A datasets (training + test)
- Goal: Create realistic, context-grounded Q/A pairs for fine-tuning and hold-out evaluation.
- Retrieval is done over the embeddings index; a larger model (e.g., `Qwen2.5-32B-Instruct-bnb-4bit`) provides drafting and final answers.

Steps:
1. Answerable Q/A generation
   - Script: `QA_Generation/question_answer_generation.py`
   - Output: `outputs/qa_generated_YYYYMMDD_HHMMSS.json`
   ```bash
   .\.venv\Scripts\python.exe QA_Generation\question_answer_generation.py
   ```
2. Unanswerable Q generation
   - Script: `QA_Generation/idk_questions.py`
   - Output: `outputs/idk_generated_YYYYMMDD_HHMMSS.json`
   ```bash
   .\.venv\Scripts\python.exe QA_Generation\idk_questions.py
   ```
3. Split into train/test
   - We used a 1000 / 200 split for fine-tuning vs. testing. Consolidate or rename generated files to:
     - Training: `outputs/qa_generated_for_fine_tuning.json` (+ optionally `outputs/idk_generated_for_fine_tuning.json`)
     - Testing:  `outputs/test_qa_generated_for_fine_tuning.json`

Design notes:
- Context policy uses FULL notification text in prompts (no truncation), preserving `[REF i]` blocks and labels for citations.
- The generator enforces incremental checkpointing to recover progress on long runs.

### 3) Fine-tune with QLoRA
- Script: `Fine_Tuning/train_qlora.py`
- Model: `Qwen2.5-7B-Instruct-bnb-4bit` with 4-bit quantization and LoRA adapters.
- Key settings:
  - LoRA rank (`r`): 16
  - Quantization: nf4, double-quant, compute fp16
  - Assistant-only loss enabled via a ChatML template that marks assistant spans
  - Always-with-context training: builds prompts from stored refined_refs (FULL notifications)
  - Token budget gate to skip overly long samples (no hidden truncation)
```bash
.\.venv\Scripts\python.exe Fine_Tuning\train_qlora.py
```
Output adapters are saved to `Fine_Tuning/checkpoints/qwen2_5_7b_qlora_ctx_full`.

### 4) Interactive inference (RAG + fine-tuned adapters)
- Script: `Inference/chat.py`
- What it does:
  - Loads base `Qwen2.5-7B-Instruct-bnb-4bit` + LoRA adapters
  - Retrieves TOP_K full notifications and answers with citations
  - Single-GPU mapping by default (`device_map = {"": current_device}`)
```bash
.\.venv\Scripts\python.exe Inference\chat.py
```

### 5) Compare baseline vs fine-tuned
Two offline utilities in `Model_Testing/`:

- `fine_tune_vs_baseline.py`
  - Rebuilds full `[REF i]` blocks from `refined_refs` for each test item
  - Runs both systems (baseline vs fine-tuned) with identical context and budgets
  - Records latency, peak memory, citation metrics, and token-level F1 vs reference answers
  - Supports auto multi‑GPU mapping (`device_map="balanced_low_0"`) when multiple GPUs are available
  - Streams per‑sample results to `outputs/eval_reports/_stream_<system>.jsonl`
  - Resumable: on restart, skips already processed indices when `RESUME_LATEST=True`
  ```bash
  .\.venv\Scripts\python.exe Model_Testing\fine_tune_vs_baseline.py
  ```

- `judge_ab_with_llm.py`
  - Uses an offline judge LLM (`Qwen2.5-32B-Instruct-bnb-4bit`) to score A/B answers
  - Multi‑GPU judge: uses `device_map` across available GPUs with per‑GPU memory caps
  - Streams judged rows to `outputs/eval_reports/llm_judge_stream.csv` for live progress
  - Resumable: reads the stream CSV and continues from the next unjudged index
  - Writes a final Excel report to `outputs/eval_reports/llm_judge_report.xlsx`
  ```bash
  .\.venv\Scripts\python.exe Model_Testing\judge_ab_with_llm.py
  ```

### 6) Plot and summarize results
- Script: `Model_Testing/plot_results.py`
- Inputs:
  - `outputs/eval_reports/llm_judge_report.xlsx` (judge scores and winners)
  - Latest `outputs/eval_reports/eval_report_*.json` from `fine_tune_vs_baseline.py`
- Outputs (to `outputs/eval_reports/plots/`):
  - `judge_scores_bar.png`, `judge_winners_bar.png`, `token_f1_box.png`, `latency_vs_f1_scatter.png`
  - `token_f1_box_eval.png`, `latency_box_eval.png`, `peak_mem_box_eval.png`, `cit_prec_box_eval.png`, `cit_recall_box_eval.png`
  - Text summaries: `analysis_summary.txt` (judge) and `analysis_summary_eval.txt` (baseline vs qlora eval)
```bash
.\.venv\Scripts\python.exe Model_Testing\plot_results.py
```

### Implementation notes & policies
- Embeddings: one vector per notification; no mixing training text into the same index.
- RAG: cosine similarity over normalized embeddings, with `[REF i]` blocks preserved for citations.
- Context: scripts default to using full notification bodies and will deduplicate or cap refs where appropriate during evaluation.
- Robustness: generation scripts incrementally checkpoint JSON outputs to survive interruptions.
  - Evaluation scripts stream results and can resume from prior outputs.

### Data and repo hygiene
- Raw materials like `raw_data/seeds.json` and `raw_data/cw_training.txt` are ignored by Git.
- Long-sample audit logs (e.g., `Fine_Tuning/too_long_examples.json`) are also ignored.

### Troubleshooting
- CUDA OOM: reduce `TOP_K`, lower `MAX_NEW_TOKENS`, or shard larger models (where supported). Ensure drivers match PyTorch CUDA build.
- Token limits: training scripts skip samples that exceed the configured token budget; adjust `MAX_SEQ_TOKENS` if needed.


