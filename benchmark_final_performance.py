#!/usr/bin/env python3
"""
Comprehensive benchmark for the fully optimized Unicorn Execution Engine.
Measures final TPS with all optimizations enabled:
- Persistent buffers
- RDNA3-optimized shaders
- INT4 quantization
"""

import numpy as np
import time
import logging
from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_benchmark():
    """Run the full performance benchmark."""
    logger.info("🚀 Starting comprehensive performance benchmark...")
    logger.info("======================================================================")

    # Initialize the fully optimized pipeline
    pipeline = PureHardwarePipelineFixed()
    if not pipeline.initialize(model_path="/home/ucadmin/Development/Unicorn-Execution-Engine/quantized_models/gemma-3-27b-it-layer-by-layer"):
        logger.error("❌ Failed to initialize the pipeline. Aborting benchmark.")
        return

    # Benchmark parameters
    warmup_tokens = 10
    benchmark_tokens = 100
    input_ids = [1, 2, 3, 4, 5]  # Example input

    # Warm-up run
    logger.info(f"🔥 Running warm-up with {warmup_tokens} tokens...")
    _ = pipeline.generate_tokens(input_ids, max_tokens=warmup_tokens)
    logger.info("✅ Warm-up complete.")

    # Benchmark run
    logger.info(f"📊 Running benchmark with {benchmark_tokens} tokens...")
    start_time = time.time()
    generated_ids = pipeline.generate_tokens(input_ids, max_tokens=benchmark_tokens)
    end_time = time.time()

    # Calculate and report results
    elapsed_time = end_time - start_time
    tps = benchmark_tokens / elapsed_time if elapsed_time > 0 else float('inf')

    logger.info("======================================================================")
    logger.info("📊 FINAL PERFORMANCE BENCHMARK RESULTS 📊")
    logger.info("======================================================================")
    logger.info(f"✅ Generated {len(generated_ids)} tokens in {elapsed_time:.2f} seconds.")
    logger.info(f"🚀 Final TPS: {tps:.2f} tokens per second")
    logger.info("======================================================================")

    # Compare to original performance
    original_tps = 0.033
    improvement_factor = tps / original_tps
    logger.info(f"📈 Performance Improvement: {improvement_factor:.1f}x over original ({original_tps} TPS)")

    if tps >= 1000:
        logger.info("🎉 CONGRATULATIONS! Target of 1000+ TPS has been achieved! 🎉")
    else:
        logger.warning("⚠️ Target of 1000+ TPS not yet reached. Further optimization may be needed.")

    # Clean up resources
    pipeline.cleanup()

if __name__ == "__main__":
    run_benchmark()
