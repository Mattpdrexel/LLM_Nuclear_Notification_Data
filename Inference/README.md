# Inference for Qwen2.5-7B-Instruct (QLoRA adapters)

This tool loads the base model and the LoRA adapters produced by `Fine_Tuning/train_qlora.py`, then provides an interactive RAG chat over your nuclear notifications.

## Requirements
- Same virtualenv as training
- Access to:
  - Base model dir (default: `D:\huggingface\hub\Qwen2.5-7B-Instruct-bnb-4bit`)
  - LoRA adapters (default: `Fine_Tuning/checkpoints/qwen2_5_7b_qlora_ctx_full`)
  - Embeddings cache (default: `embeddings_cache/nuclear_notifications_embeddings.pkl`)

## Run
```
.\.venv\Scripts\python.exe Inference\chat.py
```

- Type your question and press Enter to get an answer grounded on retrieved notifications.
- The script prints the context references used.
