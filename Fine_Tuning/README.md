# Fine-Tuning (QLoRA) for Qwen2.5-7B-Instruct (4-bit)

This project fine-tunes `Qwen2.5-7B-Instruct-bnb-4bit` using QLoRA on your generated Q/A data.

## Data
- Default input: `outputs/qa_generated_20250826_041027.json`.
- The script expects a JSON with a top-level `results` array with fields `final_question` and `final_answer` per item (falls back to `draft_question` if needed).

## Quickstart
1. Ensure the base model exists locally:
   - `BASE_MODEL_DIR` in the script defaults to `D:\huggingface\hub\Qwen2.5-7B-Instruct-bnb-4bit`.
2. (Optional) Install extra training deps:
   - `pip install -r Fine_Tuning/requirements.txt`
3. Run training:
   - `python Fine_Tuning/train_qlora.py`

Artifacts (LoRA adapters and checkpoints) are saved under `Fine_Tuning/checkpoints/` by default.

## Notes
- Uses QLoRA via PEFT and TRL SFTTrainer.
- Loads base model in 4-bit with bitsandbytes.
- Packs multiple samples into sequences up to `MAX_SEQ_LEN`.
- Adjust training config at the top of `train_qlora.py`.
