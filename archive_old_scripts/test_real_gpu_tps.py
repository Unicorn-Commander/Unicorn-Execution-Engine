#!/usr/bin/env python3
"""
Test REAL GPU TPS - ensure we're using GPU compute
"""

import time
import logging
import numpy as np
from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed
from fix_gpu_compute import patch_gpu_compute

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_gpu_compute():
    """Test that we're actually using GPU"""
    logger.info("🚀 Testing REAL GPU Compute")
    
    # Initialize pipeline
    pipeline = PureHardwarePipelineFixed()
    model_path = "quantized_models/gemma-3-27b-it-layer-by-layer"
    
    logger.info("Loading model...")
    if not pipeline.initialize(model_path):
        logger.error("Failed to initialize")
        return
    
    # Patch for proper GPU compute
    patch_gpu_compute(pipeline)
    
    logger.info(f"✅ Model loaded with {len(pipeline.layer_weights_gpu)} layers in GPU")
    
    # Test single layer
    test_input = np.random.randn(1, 1, 5376).astype(np.float32)
    
    # Monitor GPU usage
    logger.info("\n📊 Testing GPU compute...")
    logger.info("   Watch radeontop - should see GPU usage spike!")
    
    # Warm up
    for _ in range(5):
        output, _ = pipeline.forward_layer(0, test_input)
    
    # Benchmark single layer
    times = []
    for i in range(20):
        start = time.perf_counter()
        output, _ = pipeline.forward_layer(0, test_input)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        
        if i == 0:
            logger.info(f"   First layer: {elapsed*1000:.1f}ms")
    
    avg_time = np.mean(times[5:])
    logger.info(f"   Average layer time: {avg_time*1000:.1f}ms")
    
    # Estimate TPS
    full_model_time = avg_time * 62
    tps = 1.0 / full_model_time
    
    logger.info(f"\n📊 Results:")
    logger.info(f"   Per layer: {avg_time*1000:.1f}ms")
    logger.info(f"   Full model: {full_model_time*1000:.0f}ms")
    logger.info(f"   Estimated TPS: {tps:.1f}")
    
    if avg_time < 0.050:  # Less than 50ms per layer
        logger.info("   ✅ GPU compute is working!")
    else:
        logger.info("   ❌ Still using CPU (too slow)")
    
    # Test FFN specifically
    logger.info("\n📊 Testing FFN GPU compute...")
    start = time.perf_counter()
    for _ in range(10):
        ffn_out = pipeline.compute_ffn_layer_gpu(0, test_input)
    ffn_time = (time.perf_counter() - start) / 10
    logger.info(f"   FFN time: {ffn_time*1000:.1f}ms")
    
    # Cleanup
    pipeline.cleanup()
    
    # Summary
    logger.info("\n📋 Summary:")
    if tps >= 81:
        logger.info("   ✅ TARGET ACHIEVED!")
    elif tps >= 10:
        logger.info("   ✅ GPU compute working, needs optimization")
    else:
        logger.info("   ❌ Need to fix GPU compute path")

if __name__ == "__main__":
    test_gpu_compute()