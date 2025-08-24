# gpu_speed_comparison.py
import torch
import gc
import time
import statistics
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

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

def load_model(device_map_strategy, max_memory=None):
    """Load model with specified device mapping strategy"""
    # MODEL_DIR = r"D:\huggingface\hub\Qwen2.5-14B-Instruct-1M"
    MODEL_DIR = r"D:\huggingface\hub\Qwen2.5-14B-Instruct-bnb-4bit"
    
    clear_gpu_memory()
    print(f"\nLoading model with device_map='{device_map_strategy}'...")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map=device_map_strategy,
        trust_remote_code=True,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    
    print_gpu_usage()
    return pipe

def benchmark_inference(pipe, test_queries, num_runs=3):
    """Benchmark inference speed with multiple queries"""
    print(f"\nRunning benchmark with {num_runs} runs per query...")
    
    all_times = []
    total_tokens = 0
    
    for i, query in enumerate(test_queries):
        print(f"\nQuery {i+1}: {query[:50]}...")
        query_times = []
        
        messages = [{"role": "user", "content": query}]
        
        # Warmup run (not timed)
        pipe(messages, max_new_tokens=100, temperature=0.2, do_sample=False)
        
        # Timed runs
        for run in range(num_runs):
            torch.cuda.synchronize()  # Ensure GPU operations complete
            start_time = time.time()
            
            output = pipe(messages, max_new_tokens=100, temperature=0.2, do_sample=False)
            
            torch.cuda.synchronize()  # Ensure GPU operations complete
            end_time = time.time()
            
            inference_time = end_time - start_time
            query_times.append(inference_time)
            
            # Count tokens generated
            if isinstance(output[0]["generated_text"], list):
                response = output[0]["generated_text"][-1]["content"]
            else:
                response = output[0]["generated_text"]
            
            tokens_generated = len(response.split())  # Rough token count
            total_tokens += tokens_generated
            
            print(f"  Run {run+1}: {inference_time:.2f}s ({tokens_generated} tokens)")
        
        avg_time = statistics.mean(query_times)
        std_time = statistics.stdev(query_times) if len(query_times) > 1 else 0
        print(f"  Average: {avg_time:.2f}s ± {std_time:.2f}s")
        
        all_times.extend(query_times)
    
    # Overall statistics
    overall_avg = statistics.mean(all_times)
    overall_std = statistics.stdev(all_times) if len(all_times) > 1 else 0
    tokens_per_second = total_tokens / sum(all_times)
    
    print(f"\n--- Overall Results ---")
    print(f"Average time: {overall_avg:.2f}s ± {overall_std:.2f}s")
    print(f"Total tokens: {total_tokens}")
    print(f"Tokens/second: {tokens_per_second:.1f}")
    
    return {
        'avg_time': overall_avg,
        'std_time': overall_std,
        'tokens_per_second': tokens_per_second,
        'all_times': all_times
    }

def main():
    # Test queries of varying complexity
    test_queries = [
        "Explain why nuclear plants use steam turbines in detail.",
        "What are the main safety systems in a nuclear reactor?",
        "Compare PWR and BWR reactor designs briefly.",
        "Describe the nuclear fuel cycle from mining to disposal.",
        "How does nuclear fission generate electricity step by step?"
    ]
    
    print("=" * 60)
    print("GPU INFERENCE SPEED COMPARISON")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Single GPU (cuda:0)
    print("\n" + "="*20 + " SINGLE GPU TEST " + "="*20)
    pipe_single = load_model("cuda:0")
    results['single_gpu'] = benchmark_inference(pipe_single, test_queries)
    
    # Clean up
    del pipe_single
    clear_gpu_memory()
    
    # Test 2: Dual GPU (balanced)
    print("\n" + "="*20 + " DUAL GPU TEST " + "="*20)
    pipe_dual = load_model("balanced", max_memory={0: "18GB", 1: "18GB"})
    results['dual_gpu'] = benchmark_inference(pipe_dual, test_queries)
    
    # Comparison
    print("\n" + "="*20 + " COMPARISON " + "="*20)
    single_avg = results['single_gpu']['avg_time']
    dual_avg = results['dual_gpu']['avg_time']
    single_tps = results['single_gpu']['tokens_per_second']
    dual_tps = results['dual_gpu']['tokens_per_second']
    
    speedup = single_avg / dual_avg
    tps_improvement = dual_tps / single_tps
    
    print(f"Single GPU: {single_avg:.2f}s avg, {single_tps:.1f} tokens/sec")
    print(f"Dual GPU:   {dual_avg:.2f}s avg, {dual_tps:.1f} tokens/sec")
    print(f"Speedup:    {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
    print(f"Throughput: {tps_improvement:.2f}x {'better' if tps_improvement > 1 else 'worse'}")
    
    if speedup > 1:
        print(f"✅ Dual GPU is {speedup:.2f}x faster!")
    else:
        print(f"⚠️  Single GPU is {1/speedup:.2f}x faster (communication overhead?)")

if __name__ == "__main__":
    main()