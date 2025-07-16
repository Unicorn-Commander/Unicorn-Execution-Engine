#!/usr/bin/env python3
"""Test the performance improvement from persistent buffer implementation"""

import time
import logging
import numpy as np
from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_persistent_performance():
    """Test the performance with persistent buffers"""
    logger.info("🚀 Testing Persistent Buffer Performance")
    logger.info("=" * 80)
    
    # Initialize pipeline
    pipeline = PureHardwarePipelineFixed()
    
    # Initialize with model
    logger.info("Initializing pipeline...")
    start = time.time()
    success = pipeline.initialize("./quantized_models/gemma-3-27b-it-layer-by-layer")
    init_time = time.time() - start
    
    if not success:
        logger.error("❌ Pipeline initialization failed!")
        return
        
    logger.info(f"✅ Pipeline initialized in {init_time:.1f}s")
    logger.info(f"   Persistent buffers pre-created: ~434 buffers")
    
    # Test single layer performance
    logger.info("\n📊 Testing single layer performance...")
    
    # Create dummy input
    batch_size = 1
    seq_len = 256
    hidden_size = 5376
    hidden_states = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
    
    # Warm up
    logger.info("Warming up...")
    for _ in range(3):
        try:
            output, _ = pipeline.forward_layer(0, hidden_states)
        except Exception as e:
            logger.warning(f"Warmup error: {e}")
    
    # Benchmark
    logger.info("Benchmarking...")
    num_iterations = 10
    layer_times = []
    
    for i in range(num_iterations):
        start = time.time()
        try:
            output, _ = pipeline.forward_layer(0, hidden_states)
            layer_time = time.time() - start
            layer_times.append(layer_time)
            logger.info(f"   Layer 0 forward pass {i+1}: {layer_time*1000:.2f}ms")
        except Exception as e:
            logger.error(f"   Error in iteration {i+1}: {e}")
    
    if layer_times:
        avg_layer_time = np.mean(layer_times)
        logger.info(f"\n📈 Results:")
        logger.info(f"   Average layer time: {avg_layer_time*1000:.2f}ms")
        logger.info(f"   Operations per layer: 7 (Q/K/V/O + gate/up/down)")
        logger.info(f"   Time per operation: {avg_layer_time*1000/7:.2f}ms")
        
        # Calculate TPS
        total_layers = 62
        total_time_per_token = avg_layer_time * total_layers
        tps = 1.0 / total_time_per_token if total_time_per_token > 0 else 0
        
        logger.info(f"\n🎯 Performance Metrics:")
        logger.info(f"   Total time per token: {total_time_per_token*1000:.2f}ms")
        logger.info(f"   Tokens per second: {tps:.1f} TPS")
        logger.info(f"   Target: 81 TPS")
        logger.info(f"   Theoretical max: 1,556 TPS")
        
        if tps > 81:
            logger.info(f"   ✅ TARGET ACHIEVED! {tps/81:.1f}x target performance!")
        else:
            logger.info(f"   ⚠️ Below target. Need {81/tps:.1f}x improvement")
    
    logger.info("=" * 80)

if __name__ == "__main__":
    test_persistent_performance()