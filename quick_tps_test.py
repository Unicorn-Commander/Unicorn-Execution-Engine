#!/usr/bin/env python3
"""Quick TPS test - measure performance potential"""

import time
import logging
import numpy as np
from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def quick_benchmark(pipeline):
    """Quick benchmark of key operations"""
    # Test data
    batch_size = 1
    seq_len = 1
    hidden_size = 5376
    
    # Create test input
    test_input = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
    
    logger.info("⏱️ Benchmarking single layer...")
    
    # Warm up
    for _ in range(5):
        output, _ = pipeline.forward_layer(0, test_input)
    
    # Time 10 forward passes
    times = []
    for i in range(10):
        start = time.perf_counter()
        output, _ = pipeline.forward_layer(0, test_input)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        
        if i == 0:
            logger.info(f"   First layer time: {elapsed*1000:.1f}ms")
    
    avg_layer_time = np.mean(times[1:])  # Skip first
    logger.info(f"   Avg layer time: {avg_layer_time*1000:.1f}ms")
    
    # Estimate full model time (62 layers)
    full_model_time = avg_layer_time * 62
    est_tps = 1.0 / full_model_time
    
    logger.info(f"\n📊 Performance Estimate:")
    logger.info(f"   Per layer: {avg_layer_time*1000:.1f}ms")
    logger.info(f"   Full model (62 layers): {full_model_time*1000:.1f}ms")
    logger.info(f"   Estimated TPS: {est_tps:.1f}")
    
    return est_tps

def main():
    """Quick TPS test"""
    logger.info("🚀 Quick TPS Test")
    
    # Initialize pipeline
    pipeline = PureHardwarePipelineFixed()
    model_path = "quantized_models/gemma-3-27b-it-layer-by-layer"
    
    # Only initialize, don't load all layers for quick test
    logger.info("Initializing pipeline...")
    start = time.time()
    
    # Initialize components
    logger.info("🚀 Initializing FIXED Pure Hardware Pipeline (Direct GPU Loading)")
    from real_vulkan_matrix_compute import VulkanMatrixCompute
    from vulkan_int8_support import add_int8_support
    
    add_int8_support(VulkanMatrixCompute)
    pipeline.vulkan_engine = VulkanMatrixCompute()
    if not pipeline.vulkan_engine.initialize():
        logger.error("Failed to init Vulkan")
        return
        
    # Load just one layer for testing
    from pure_mmap_loader import PureMemoryMappedLoader
    pipeline.loader = PureMemoryMappedLoader(model_path)
    
    # Get one layer's weights
    layer_tensors = pipeline.loader._get_layer_tensors(0)
    logger.info(f"Layer 0 has {len(layer_tensors)} tensors")
    
    # Quick load to GPU
    pipeline.gpu_buffers = {}
    pipeline.layer_weights_gpu = {0: {}}
    
    for name, info in layer_tensors.items():
        if 'weight' in name and 'scale' not in name:
            # Get actual tensor
            tensor = pipeline.loader.get_tensor(info)
            # Allocate to GPU
            gpu_buffer = pipeline.vulkan_engine._allocate_gpu_memory(tensor)
            pipeline.gpu_buffers[name] = {
                'buffer_info': gpu_buffer,
                'shape': info['shape'],
                'dtype': info['dtype']
            }
            pipeline.layer_weights_gpu[0][name] = name
    
    elapsed = time.time() - start
    logger.info(f"✅ Quick setup done in {elapsed:.1f}s")
    
    # Run benchmark
    est_tps = quick_benchmark(pipeline)
    
    # Compare to target
    if est_tps >= 81:
        logger.info("\n✅ PERFORMANCE TARGET ACHIEVABLE!")
    else:
        logger.info(f"\n❌ Need {(81/est_tps - 1)*100:.0f}% speedup to reach 81 TPS")
    
    # Cleanup
    if hasattr(pipeline, 'vulkan_engine'):
        pipeline.vulkan_engine.cleanup()

if __name__ == "__main__":
    main()