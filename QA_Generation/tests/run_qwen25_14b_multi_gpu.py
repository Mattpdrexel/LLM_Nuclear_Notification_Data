# run_gpt_oss20b_multi_gpu.py
import torch
import gc
from transformers import pipeline

def clear_gpu_memory():
    """Clear memory on all available GPUs"""
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
    gc.collect()

def print_gpu_usage():
    """Print current GPU memory usage"""
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} ({torch.cuda.get_device_name(i)}):")
        print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
        print(f"  Total: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")

# Clear memory before loading
clear_gpu_memory()

# MODEL_DIR = r"D:\huggingface\hub\Qwen2.5-14B-Instruct-bnb-4bit"
MODEL_DIR = r"D:\huggingface\hub\Qwen2.5-14B-Instruct-1M"

print("Loading model across both GPUs...")

# Load model and tokenizer separately to avoid parameter conflicts
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="balanced",
    trust_remote_code=True,
    max_memory={
        0: "18GB",
        1: "18GB", 
    },
    low_cpu_mem_usage=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True
)

# Create pipeline from loaded components (avoids parameter passing issues)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

print("Model loaded! GPU usage:")
print_gpu_usage()

messages = [
    {"role": "user", "content": "Explain why nuclear plants use steam turbines in detail."}
]

print("\nGenerating response...")
out = pipe(messages, max_new_tokens=120, temperature=0.2)
print("\nResponse:")
# Handle different response formats
try:
    # Try chat format first
    if isinstance(out[0]["generated_text"], list):
        response = out[0]["generated_text"][-1]["content"]
    else:
        # Fallback to string format
        response = out[0]["generated_text"]
    print(response)
except (KeyError, TypeError, IndexError) as e:
    print(f"Response format: {out[0]['generated_text']}")
    print("Raw output structure might be different than expected")

print("\nFinal GPU usage:")
print_gpu_usage()