#!/usr/bin/env python3
"""
Comprehensive Gemma 3n E4B Performance Benchmark
Test different configurations to identify bottlenecks
"""

import time
import torch
import logging
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def benchmark_baseline_cpu():
    """Benchmark baseline CPU performance"""
    logger.info("🔥 BENCHMARK 1: Baseline CPU Performance")
    logger.info("=" * 50)
    
    start_time = time.time()
    
    # Load model
    logger.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("./models/gemma-3n-e4b-it", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "./models/gemma-3n-e4b-it",
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    load_time = time.time() - start_time
    logger.info(f"✅ Model loaded in {load_time:.1f}s")
    
    # Test prompts of different lengths
    test_prompts = [
        "Hello",
        "Hello, how are you today?",
        "Hello, I'm Aaron. Please tell me about yourself and your capabilities.",
        "Write a detailed explanation of how neural processing units work and their advantages over traditional CPUs."
    ]
    
    results = []
    
    for i, prompt in enumerate(test_prompts):
        logger.info(f"\n🔍 Test {i+1}: '{prompt[:30]}...' ({len(tokenizer.encode(prompt))} input tokens)")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        input_length = len(inputs["input_ids"][0])
        
        # Generate
        start_gen = time.time()
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        gen_time = time.time() - start_gen
        
        # Calculate metrics
        generated_tokens = len(outputs[0]) - input_length
        tps = generated_tokens / gen_time if gen_time > 0 else 0
        
        # Decode response
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        logger.info(f"  Generated: {generated_tokens} tokens")
        logger.info(f"  Time: {gen_time:.2f}s")
        logger.info(f"  TPS: {tps:.1f}")
        logger.info(f"  Response: {response[:50]}...")
        
        results.append({
            'prompt_length': input_length,
            'generated_tokens': generated_tokens,
            'time': gen_time,
            'tps': tps
        })
    
    # Summary
    avg_tps = sum(r['tps'] for r in results) / len(results)
    logger.info(f"\n📊 BASELINE CPU RESULTS:")
    logger.info(f"   Average TPS: {avg_tps:.1f}")
    logger.info(f"   Best TPS: {max(r['tps'] for r in results):.1f}")
    logger.info(f"   Worst TPS: {min(r['tps'] for r in results):.1f}")
    
    return results

def benchmark_optimized_cpu():
    """Benchmark optimized CPU performance"""
    logger.info("\n🔥 BENCHMARK 2: Optimized CPU Performance")
    logger.info("=" * 50)
    
    start_time = time.time()
    
    # Load model with optimizations
    logger.info("Loading model with optimizations...")
    tokenizer = AutoTokenizer.from_pretrained("./models/gemma-3n-e4b-it", trust_remote_code=True)
    
    # Enable optimizations
    torch.set_num_threads(psutil.cpu_count())
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    model = AutoModelForCausalLM.from_pretrained(
        "./models/gemma-3n-e4b-it",
        torch_dtype=torch.bfloat16,  # Use bfloat16 instead of float32
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager"  # Use optimized attention
    )
    
    # Try to compile the model for faster inference
    logger.info("Optimizing model...")
    # model = torch.compile(model, mode="max-autotune")  # This might take a while
    
    load_time = time.time() - start_time
    logger.info(f"✅ Optimized model loaded in {load_time:.1f}s")
    logger.info(f"   Using {torch.get_num_threads()} CPU threads")
    logger.info(f"   Model dtype: {next(model.parameters()).dtype}")
    
    # Quick test
    prompt = "Hello, I'm Aaron. Please tell me about yourself."
    inputs = tokenizer(prompt, return_tensors="pt")
    input_length = len(inputs["input_ids"][0])
    
    logger.info(f"\n🔍 Optimized test: '{prompt[:30]}...'")
    
    start_gen = time.time()
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True  # Enable KV caching
        )
    gen_time = time.time() - start_gen
    
    generated_tokens = len(outputs[0]) - input_length
    tps = generated_tokens / gen_time if gen_time > 0 else 0
    
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    logger.info(f"  Generated: {generated_tokens} tokens")
    logger.info(f"  Time: {gen_time:.2f}s")
    logger.info(f"  TPS: {tps:.1f}")
    logger.info(f"  Response: {response[:50]}...")
    
    logger.info(f"\n📊 OPTIMIZED CPU RESULTS:")
    logger.info(f"   TPS: {tps:.1f}")
    
    return tps

def benchmark_accelerated():
    """Benchmark with our acceleration framework"""
    logger.info("\n🔥 BENCHMARK 3: NPU+Vulkan Acceleration")
    logger.info("=" * 50)
    
    try:
        from gemma3n_e4b_simple_acceleration import SimpleGemma3nE4BAcceleratedModel
        
        start_time = time.time()
        model = SimpleGemma3nE4BAcceleratedModel()
        load_time = time.time() - start_time
        
        logger.info(f"✅ Accelerated model loaded in {load_time:.1f}s")
        
        # Quick test
        prompt = "Hello, I'm Aaron. Please tell me about yourself."
        
        logger.info(f"\n🔍 Acceleration test: '{prompt[:30]}...'")
        
        result = model.accelerated_generate(prompt, max_tokens=50)
        
        logger.info(f"  Generated: {result['tokens_generated']} tokens")
        logger.info(f"  Time: {result['inference_time']:.2f}s") 
        logger.info(f"  TPS: {result['tokens_per_second']:.1f}")
        logger.info(f"  NPU available: {result['acceleration_ready']['npu']}")
        logger.info(f"  Vulkan available: {result['acceleration_ready']['vulkan']}")
        logger.info(f"  Response: {result['generated_text'][:50]}...")
        
        logger.info(f"\n📊 ACCELERATION RESULTS:")
        logger.info(f"   TPS: {result['tokens_per_second']:.1f}")
        
        return result['tokens_per_second']
        
    except Exception as e:
        logger.error(f"❌ Acceleration benchmark failed: {e}")
        return 0

def system_info():
    """Show system information"""
    logger.info("\n💻 SYSTEM INFORMATION:")
    logger.info("=" * 30)
    logger.info(f"   CPU: {psutil.cpu_count()} cores")
    logger.info(f"   RAM: {psutil.virtual_memory().total / 1024**3:.1f}GB")
    logger.info(f"   PyTorch: {torch.__version__}")
    logger.info(f"   CUDA available: {torch.cuda.is_available()}")
    logger.info(f"   MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")

def main():
    """Run comprehensive benchmark"""
    logger.info("🦄 Gemma 3n E4B Performance Benchmark")
    logger.info("=" * 60)
    
    system_info()
    
    # Run benchmarks
    try:
        baseline_results = benchmark_baseline_cpu()
        baseline_avg = sum(r['tps'] for r in baseline_results) / len(baseline_results)
    except Exception as e:
        logger.error(f"❌ Baseline benchmark failed: {e}")
        baseline_avg = 0
    
    try:
        optimized_tps = benchmark_optimized_cpu()
    except Exception as e:
        logger.error(f"❌ Optimized benchmark failed: {e}")
        optimized_tps = 0
    
    try:
        accelerated_tps = benchmark_accelerated()
    except Exception as e:
        logger.error(f"❌ Accelerated benchmark failed: {e}")
        accelerated_tps = 0
    
    # Final comparison
    logger.info("\n🏆 FINAL PERFORMANCE COMPARISON:")
    logger.info("=" * 40)
    logger.info(f"   Baseline CPU:     {baseline_avg:.1f} TPS")
    logger.info(f"   Optimized CPU:    {optimized_tps:.1f} TPS") 
    logger.info(f"   NPU+Vulkan:       {accelerated_tps:.1f} TPS")
    logger.info("=" * 40)
    
    # Analysis
    if accelerated_tps > optimized_tps:
        improvement = (accelerated_tps / optimized_tps - 1) * 100
        logger.info(f"🚀 Acceleration working: {improvement:.1f}% improvement")
    else:
        logger.warning("⚠️  Acceleration not working properly - bottlenecks detected")
        
        if accelerated_tps < baseline_avg:
            logger.warning("❌ Acceleration is actually SLOWER than baseline")
            logger.warning("   Likely issues:")
            logger.warning("   - Hardware acceleration not actually being used")
            logger.warning("   - Overhead from acceleration framework")
            logger.warning("   - Layer replacement not working")
        
    # Recommendations
    logger.info("\n💡 RECOMMENDATIONS:")
    if accelerated_tps < 10:
        logger.info("   - Check if NPU/Vulkan are actually processing layers")
        logger.info("   - Verify layer replacement is working")
        logger.info("   - Profile memory usage and tensor operations")
        logger.info("   - Consider quantization (INT8/INT4)")
    
    if optimized_tps < 5:
        logger.info("   - CPU performance unexpectedly low")
        logger.info("   - Check CPU threading and BLAS libraries")
        logger.info("   - Consider model precision (bfloat16 vs float32)")

if __name__ == "__main__":
    main()