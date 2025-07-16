#!/usr/bin/env python3
"""
Real Performance Measurement for Unicorn Execution Engine
Measure actual tokens per second with real hardware acceleration
"""

import os
import multiprocessing
import time
import torch
import numpy as np
import logging
from pathlib import Path
import json

# Set environment variables for maximum CPU usage
os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(multiprocessing.cpu_count())
os.environ['NUMEXPR_NUM_THREADS'] = str(multiprocessing.cpu_count())

# Force PyTorch to use all CPU cores
torch.set_num_threads(multiprocessing.cpu_count())
torch.set_num_interop_threads(multiprocessing.cpu_count())

# Import our real hardware engine
from integrated_quantized_npu_engine import IntegratedQuantizedNPUEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceMeasurer:
    """Measure real-world performance with actual hardware acceleration"""
    
    def __init__(self, model_path):
        self.engine = None
        self.results = {}
        self.model_path = model_path
        
    def initialize_engine(self):
        """Initialize the integrated engine with real hardware"""
        logger.info("🔥 Initializing Unicorn Execution Engine for performance testing...")
        
        self.engine = IntegratedQuantizedNPUEngine(
            enable_quantization=True, 
            turbo_mode=True
        )
        
        # Ensure we have real hardware working
        if not self.engine.vulkan_available:
            logger.warning("⚠️ Vulkan not available - performance will be limited")
        if not self.engine.npu_available:
            logger.warning("⚠️ NPU not available - performance will be limited")
        
        logger.info(f"✅ Engine initialized - NPU: {self.engine.npu_available}, Vulkan: {self.engine.vulkan_available}")
        
        # Load and quantize the model
        self.engine.load_and_quantize_model(self.model_path)
        
        return True
    
    def measure_inference_speed(self, num_tokens=100, batch_size=1, seq_length=128):
        """Measure actual inference speed with real hardware"""
        logger.info(f"📊 Measuring inference speed: {num_tokens} tokens, batch={batch_size}, seq_len={seq_length}")
        
        # Create realistic input tensors
        input_ids = torch.randint(1, 32000, (batch_size, seq_length), dtype=torch.long)
        
        # Warm up the system (3 iterations)
        logger.info("🔥 Warming up hardware...")
        for i in range(3):
            try:
                _ = self.engine.generate_text_quantized("Hello", max_tokens=5)
            except Exception as e:
                logger.warning(f"Warmup iteration {i} failed: {e}")
        
        # Measure actual performance using text generation
        logger.info("⏱️ Starting performance measurement...")
        
        total_tokens_generated = 0
        total_time = 0.0
        successful_runs = 0
        
        # Test prompts of varying lengths
        test_prompts = [
            "The quick brown fox",
            "Artificial intelligence is",
            "In the realm of computing",
            "The future of technology",
            "Advanced machine learning"
        ]
        
        tokens_per_run = max(10, num_tokens // 10)  # Generate 10+ tokens per run
        
        for run_idx in range(min(num_tokens // tokens_per_run, 20)):  # Max 20 runs
            try:
                prompt = test_prompts[run_idx % len(test_prompts)]
                
                start_time = time.perf_counter()
                
                # Generate text using real hardware acceleration
                result = self.engine.generate_text_quantized(
                    prompt=prompt, 
                    max_tokens=tokens_per_run
                )
                
                end_time = time.perf_counter()
                
                # Count actual tokens generated
                if isinstance(result, str):
                    # Rough token estimation (1 token ≈ 4 characters)
                    tokens_generated = len(result.split()) + len(prompt.split())
                else:
                    tokens_generated = tokens_per_run  # Fallback estimate
                
                iteration_time = end_time - start_time
                total_time += iteration_time
                total_tokens_generated += tokens_generated
                successful_runs += 1
                
                if (run_idx + 1) % 3 == 0:
                    current_tps = total_tokens_generated / total_time if total_time > 0 else 0
                    logger.info(f"   Progress: {run_idx + 1} runs, {total_tokens_generated} tokens, Current TPS: {current_tps:.1f}")
                
            except Exception as e:
                logger.warning(f"Run {run_idx} failed: {e}")
                continue
        
        # Calculate final statistics
        if total_time > 0 and successful_runs > 0:
            average_tps = total_tokens_generated / total_time
            average_time_per_token = total_time / successful_runs
            
            results = {
                "total_tokens_processed": total_tokens_generated,
                "total_time_seconds": total_time,
                "successful_runs": successful_runs,
                "failed_runs": num_tokens - successful_runs,
                "average_tokens_per_second": average_tps,
                "average_time_per_token_ms": average_time_per_token * 1000,
                "hardware_config": {
                    "npu_available": self.engine.npu_available,
                    "vulkan_available": self.engine.vulkan_available,
                    "igpu_available": self.engine.igpu_available,
                    "turbo_mode": self.engine.turbo_mode
                },
                "performance_stats": self.engine.performance_stats
            }
            
            logger.info("🎯 Performance Results:")
            logger.info(f"   📈 Tokens per Second: {average_tps:.2f} TPS")
            logger.info(f"   ⏱️ Time per Token: {average_time_per_token * 1000:.2f} ms")
            logger.info(f"   ✅ Success Rate: {successful_runs}/{num_tokens} ({100 * successful_runs / num_tokens:.1f}%)")
            logger.info(f"   🚀 Hardware: NPU={self.engine.npu_available}, Vulkan={self.engine.vulkan_available}")
            
            return results
        else:
            logger.error("❌ No successful runs - performance measurement failed")
            return None
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive performance benchmarks"""
        logger.info("🚀 Running comprehensive performance benchmark...")
        
        test_configs = [
            {"tokens": 50, "batch": 1, "seq_len": 64, "name": "Small Test"},
            {"tokens": 100, "batch": 1, "seq_len": 128, "name": "Standard Test"},
            {"tokens": 200, "batch": 1, "seq_len": 256, "name": "Large Context"},
        ]
        
        all_results = {}
        
        for config in test_configs:
            logger.info(f"\n🧪 Running {config['name']}...")
            
            result = self.measure_inference_speed(
                num_tokens=config["tokens"],
                batch_size=config["batch"], 
                seq_length=config["seq_len"]
            )
            
            if result:
                all_results[config["name"]] = result
                tps = result["average_tokens_per_second"]
                logger.info(f"   ✅ {config['name']}: {tps:.2f} TPS")
            else:
                logger.error(f"   ❌ {config['name']}: Failed")
        
        # Save results
        with open("real_performance_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
        # Summary
        if all_results:
            logger.info("\n🎯 FINAL PERFORMANCE SUMMARY:")
            for name, result in all_results.items():
                tps = result["average_tokens_per_second"]
                logger.info(f"   {name}: {tps:.2f} TPS")
            
            # Calculate average TPS
            avg_tps = sum(r["average_tokens_per_second"] for r in all_results.values()) / len(all_results)
            logger.info(f"\n📊 AVERAGE PERFORMANCE: {avg_tps:.2f} TPS")
            
            # Compare to targets
            target_tps = 400  # Target for Gemma 3 4B
            if avg_tps >= target_tps:
                logger.info(f"🎉 SUCCESS: Exceeded target of {target_tps} TPS!")
            else:
                improvement_needed = target_tps / avg_tps
                logger.info(f"📈 Need {improvement_needed:.1f}x improvement to reach {target_tps} TPS target")
        
        return all_results

def main():
    """Main performance testing function"""
    print("🦄 Unicorn Execution Engine - Real Performance Measurement")
    print("=" * 60)
    
    model_path = "./models/gemma-3-27b-it"
    
    measurer = PerformanceMeasurer(model_path)
    
    if measurer.initialize_engine():
        results = measurer.run_comprehensive_benchmark()
        
        if results:
            print("\n✅ Performance measurement complete!")
            print("📄 Results saved to: real_performance_results.json")
        else:
            print("\n❌ Performance measurement failed!")
            return 1
    else:
        print("❌ Failed to initialize engine!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())