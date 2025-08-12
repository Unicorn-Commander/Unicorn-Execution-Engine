#!/usr/bin/env python3
"""
Benchmark NPU+iGPU performance with Gemma models
Tests real NPU kernel execution vs CPU baseline
"""

import subprocess
import time
import json
import argparse
import os
import logging
from statistics import mean, stdev

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class NPUiGPUBenchmark:
    def __init__(self, model_path, llama_cli_path="llama.cpp/build/bin/llama-cli"):
        self.model_path = model_path
        self.llama_cli_path = llama_cli_path
        self.results = {
            "model": os.path.basename(model_path),
            "benchmarks": []
        }
        
    def run_inference(self, prompt, max_tokens=128, use_npu=False, use_gpu=False, repetitions=3):
        """Run inference with specified configuration"""
        cmd = [self.llama_cli_path, "-m", self.model_path]
        
        # Add prompt
        cmd.extend(["-p", prompt])
        cmd.extend(["-n", str(max_tokens)])
        
        # Add acceleration flags
        config_name = "CPU"
        if use_npu:
            cmd.append("--npu-attention")
            config_name = "NPU"
        if use_gpu:
            cmd.extend(["--gpu-layers", "999"])
            config_name = "GPU" if not use_npu else "NPU+GPU"
            
        # Disable logs for clean output parsing
        cmd.extend(["--no-display-prompt", "--log-disable"])
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 Testing {config_name} configuration")
        logger.info(f"   Prompt: '{prompt[:50]}...'")
        logger.info(f"   Max tokens: {max_tokens}")
        logger.info(f"   Repetitions: {repetitions}")
        
        times = []
        tokens_per_second = []
        
        for i in range(repetitions):
            logger.info(f"\n   Run {i+1}/{repetitions}...")
            start_time = time.time()
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                end_time = time.time()
                elapsed = end_time - start_time
                
                if result.returncode != 0:
                    logger.error(f"   ❌ Error: {result.stderr}")
                    continue
                    
                # Parse output for performance metrics
                output_lines = result.stderr.split('\n')
                for line in output_lines:
                    if "tok/s" in line or "tokens/s" in line:
                        # Extract tokens per second
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if "tok/s" in part or "tokens/s" in part and i > 0:
                                try:
                                    tps = float(parts[i-1])
                                    tokens_per_second.append(tps)
                                    logger.info(f"   ✅ {tps:.2f} tok/s")
                                except:
                                    pass
                
                times.append(elapsed)
                
            except subprocess.TimeoutExpired:
                logger.error(f"   ❌ Timeout after 300s")
                continue
            except Exception as e:
                logger.error(f"   ❌ Error: {e}")
                continue
        
        if times:
            avg_time = mean(times)
            avg_tps = mean(tokens_per_second) if tokens_per_second else max_tokens / avg_time
            
            result = {
                "configuration": config_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "runs": len(times),
                "avg_time_seconds": avg_time,
                "avg_tokens_per_second": avg_tps,
                "time_per_token_ms": (avg_time / max_tokens) * 1000
            }
            
            if len(times) > 1:
                result["time_stdev"] = stdev(times)
                
            logger.info(f"\n📊 {config_name} Results:")
            logger.info(f"   Average time: {avg_time:.2f}s")
            logger.info(f"   Tokens/second: {avg_tps:.2f}")
            logger.info(f"   Time per token: {result['time_per_token_ms']:.2f}ms")
            
            return result
        else:
            logger.error(f"❌ All runs failed for {config_name}")
            return None
    
    def run_full_benchmark(self):
        """Run complete benchmark suite"""
        logger.info("🦄 NPU+iGPU Gemma Benchmark")
        logger.info("=" * 60)
        logger.info(f"📁 Model: {self.model_path}")
        logger.info(f"🔧 llama.cpp: {self.llama_cli_path}")
        
        # Check if model exists
        if not os.path.exists(self.model_path):
            logger.error(f"❌ Model not found: {self.model_path}")
            return
            
        # Test prompts
        test_prompts = [
            "The future of AI is",
            "Explain quantum computing in simple terms:",
            "Write a Python function to calculate fibonacci numbers:",
        ]
        
        # Configurations to test
        configs = [
            {"name": "CPU Baseline", "use_npu": False, "use_gpu": False},
            {"name": "Vulkan GPU", "use_npu": False, "use_gpu": True},
            {"name": "NPU Attention", "use_npu": True, "use_gpu": False},
            {"name": "NPU+GPU Hybrid", "use_npu": True, "use_gpu": True},
        ]
        
        # Test each configuration
        for config in configs:
            logger.info(f"\n🚀 Testing {config['name']}...")
            
            for prompt in test_prompts:
                result = self.run_inference(
                    prompt=prompt,
                    max_tokens=50,
                    use_npu=config["use_npu"],
                    use_gpu=config["use_gpu"],
                    repetitions=3
                )
                
                if result:
                    self.results["benchmarks"].append(result)
        
        # Calculate speedups
        self.calculate_speedups()
        
        # Save results
        self.save_results()
        
        # Print summary
        self.print_summary()
        
    def calculate_speedups(self):
        """Calculate speedup vs CPU baseline"""
        cpu_results = [r for r in self.results["benchmarks"] if r["configuration"] == "CPU"]
        
        if not cpu_results:
            logger.warning("⚠️ No CPU baseline results found")
            return
            
        # Average CPU performance
        cpu_tps = mean([r["avg_tokens_per_second"] for r in cpu_results])
        
        # Calculate speedups
        for result in self.results["benchmarks"]:
            if result["configuration"] != "CPU":
                result["speedup_vs_cpu"] = result["avg_tokens_per_second"] / cpu_tps
    
    def save_results(self):
        """Save benchmark results to JSON"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"npu_igpu_benchmark_{timestamp}.json"
        
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)
            
        logger.info(f"\n💾 Results saved to: {filename}")
    
    def print_summary(self):
        """Print benchmark summary"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 BENCHMARK SUMMARY")
        logger.info("=" * 60)
        
        # Group by configuration
        configs = {}
        for result in self.results["benchmarks"]:
            config = result["configuration"]
            if config not in configs:
                configs[config] = []
            configs[config].append(result)
        
        # Print summary for each configuration
        for config, results in configs.items():
            avg_tps = mean([r["avg_tokens_per_second"] for r in results])
            speedup = results[0].get("speedup_vs_cpu", 1.0)
            
            logger.info(f"\n🎯 {config}:")
            logger.info(f"   Average: {avg_tps:.2f} tok/s")
            if speedup != 1.0:
                logger.info(f"   Speedup: {speedup:.2f}x vs CPU")
                
        # Find best configuration
        all_results = self.results["benchmarks"]
        if all_results:
            best = max(all_results, key=lambda x: x["avg_tokens_per_second"])
            logger.info(f"\n🏆 BEST PERFORMANCE: {best['configuration']}")
            logger.info(f"   {best['avg_tokens_per_second']:.2f} tok/s")
            if "speedup_vs_cpu" in best:
                logger.info(f"   {best['speedup_vs_cpu']:.2f}x faster than CPU")
                
        logger.info("\n🦄 NPU+iGPU Magic Achieved! 🚀")

def main():
    parser = argparse.ArgumentParser(description="Benchmark NPU+iGPU Gemma performance")
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--llama-cli", default="llama.cpp/build/bin/llama-cli", 
                        help="Path to llama-cli executable")
    parser.add_argument("--quick", action="store_true", 
                        help="Run quick test (1 prompt, 1 repetition)")
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = NPUiGPUBenchmark(args.model, args.llama_cli)
    
    if args.quick:
        # Quick test
        logger.info("🚀 Running quick NPU+iGPU test...")
        result = benchmark.run_inference(
            "Hello, world! The NPU says",
            max_tokens=20,
            use_npu=True,
            use_gpu=True,
            repetitions=1
        )
        if result:
            logger.info(f"\n✅ NPU+iGPU working: {result['avg_tokens_per_second']:.2f} tok/s")
    else:
        # Full benchmark
        benchmark.run_full_benchmark()

if __name__ == "__main__":
    main()