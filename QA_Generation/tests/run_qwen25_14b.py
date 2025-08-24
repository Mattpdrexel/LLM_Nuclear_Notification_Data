# run_gpt_oss20b_mxfp4.py
from transformers import pipeline

MODEL_DIR = r"D:\huggingface\hub\Qwen2.5-14B-Instruct-bnb-4bit"

pipe = pipeline(
    "text-generation",
    model=MODEL_DIR,
    tokenizer=MODEL_DIR,
    torch_dtype="auto",
    device_map="auto",            # will place on your GPU
    trust_remote_code=True,
)

messages = [
    {"role": "user", "content": "In one sentence, what is MXFP4 and why does gpt-oss-20b use it?"}
]
out = pipe(messages, max_new_tokens=120, temperature=0.2)
print(out[0]["generated_text"][-1])
