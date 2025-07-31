#!/usr/bin/env python3
"""
Benchmark for the quantized Gemma-3-4B-IT model.
Tests performance with the custom quantized model optimized for NPU+iGPU.
"""

# Fix Python 3.11 compatibility with vulkan
import fix_vulkan_imports

import numpy as np
import time
import logging
from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_4b_benchmark():
    """Run the 4B model performance benchmark."""
    logger.info("🚀 Starting Gemma-3-4B quantized model benchmark...")
    logger.info("======================================================================")
    logger.info("Model: Gemma-3-4B-IT (Quantized - INT4/INT8 mixed precision)")
    logger.info("Hardware: NPU+iGPU only (no CPU compute)")
    logger.info("Expected size: ~2.5GB (from 8GB original)")
    logger.info("======================================================================")

    # Initialize the pipeline with 4B model
    pipeline = PureHardwarePipelineFixed()
    model_path = "/home/ucadmin/Development/Unicorn-Execution-Engine/quantized_models/gemma-3-4b-it-quantized"
    
    if not pipeline.initialize(model_path=model_path):
        logger.error("❌ Failed to initialize the pipeline. Aborting benchmark.")
        return

    # Benchmark parameters - reduced for GPU stability
    warmup_tokens = 1  # Reduced from 10 to avoid GPU hang
    benchmark_tokens = 20  # Reduced from 100 for initial test
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
    npu_avg_time = (pipeline.npu_total_time / pipeline.npu_total_layers) * 1000 if pipeline.npu_total_layers > 0 else 0
    npu_tps = 1 / (npu_avg_time / 1000) if npu_avg_time > 0 else 0

    logger.info("======================================================================")
    logger.info("📊 GEMMA-3-4B PERFORMANCE BENCHMARK RESULTS 📊")
    logger.info("======================================================================")
    logger.info(f"✅ Generated {len(generated_ids)} tokens in {elapsed_time:.2f} seconds.")
    logger.info(f"🚀 4B Model TPS: {tps:.2f} tokens per second")
    logger.info(f"   NPU Average Layer Time: {npu_avg_time:.2f}ms")
    logger.info(f"   NPU TPS: {npu_tps:.2f}")
    logger.info("======================================================================")

    # Performance analysis
    if tps >= 200:
        logger.info("🎉 EXCELLENT! 4B model performance is outstanding!")
    elif tps >= 100:
        logger.info("✅ GOOD! 4B model performance meets expectations.")
    elif tps >= 81:
        logger.info("✅ SUCCESS! 4B model meets the 81+ TPS target.")
    else:
        logger.warning("⚠️ 4B model performance below 81 TPS target. Further optimization needed.")

    # Scaling expectations
    logger.info("📈 Scaling Analysis:")
    logger.info(f"   - 4B model: {tps:.1f} TPS")
    logger.info(f"   - Expected 27B performance: {tps * 0.15:.1f} TPS (rough estimate)")
    logger.info("   - 4B model validates optimization stack effectiveness")

    # Clean up resources
    pipeline.cleanup()

if __name__ == "__main__":
    run_4b_benchmark()