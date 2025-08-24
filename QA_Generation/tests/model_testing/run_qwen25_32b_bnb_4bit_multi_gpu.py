# run_qwen25_72b_bnb4bit_dual_gpu.py
import os, torch, gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# (Windows often ignores this, but harmless:)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_DIR = r"D:\huggingface\hub\Qwen2.5-32B-Instruct-bnb-4bit"

def clear():
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
    gc.collect()

clear()

# Allow ~19 GiB on each GPU; IMPORTANT: no "cpu" key here
max_memory = {0: "19GiB", 1: "19GiB"}

# If the repo already includes a bnb-4bit config, this may be ignored (that’s OK).
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # or torch.float16
)

print("Loading tokenizer…")
tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

print("Loading model sharded over 2 GPUs…")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    quantization_config=bnb_cfg,
    device_map="balanced",        # or "balanced_low_0"
    max_memory=max_memory,        # make sure total >= ~39–41 GiB
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

print("Loaded. Device map sample:", list(model.hf_device_map.items())[:5])

# quick test
messages = [
    {"role": "user", "content": "Explain why nuclear plants use steam turbines in detail."}
]
if hasattr(tok, "apply_chat_template"):
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
else:
    prompt = "User: Explain why nuclear plants use steam turbines in detail.\nAssistant:"

inputs = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)

with torch.no_grad():
    out_ids = model.generate(**inputs, max_new_tokens=200, temperature=0.2, top_p=0.95,
                             eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)

gen = tok.decode(out_ids[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print("\n=== Response ===\n", gen.strip())
