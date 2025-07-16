#!/usr/bin/env python3
"""
Test Vulkan GPU memory allocation and performance
"""

import numpy as np
import time
import logging
from real_vulkan_matrix_compute_fixed import VulkanMatrixComputeFixed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_vulkan_performance():
    """Test Vulkan compute performance with VRAM allocation"""
    
    logger.info("🚀 Testing Vulkan GPU Performance with VRAM Allocation")
    logger.info("=" * 60)
    
    # Initialize Vulkan engine
    vulkan_engine = VulkanMatrixComputeFixed()
    if not vulkan_engine.initialize():
        logger.error("Failed to initialize Vulkan engine")
        return
    
    # Test different matrix sizes
    test_sizes = [
        (1, 4096, 4096),      # Single token
        (32, 4096, 4096),     # Small batch
        (64, 4096, 4096),     # Medium batch
        (128, 4096, 4096),    # Large batch
        (256, 4096, 14336),   # FFN size
    ]
    
    for batch, m, n in test_sizes:
        logger.info(f"\n📊 Testing {batch}x{m} @ {m}x{n} matrix multiplication")
        
        # Create test matrices
        matrix_a = np.random.randn(batch, m).astype(np.float32)
        matrix_b = np.random.randn(m, n).astype(np.float32)
        
        # Warmup
        for _ in range(3):
            _ = vulkan_engine.matrix_multiply(matrix_a, matrix_b)
        
        # Measure performance
        num_iterations = 10
        start_time = time.time()
        
        for _ in range(num_iterations):
            result = vulkan_engine.matrix_multiply(matrix_a, matrix_b)
        
        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / num_iterations
        
        # Calculate GFLOPS
        flops = 2 * batch * m * n  # 2 ops per multiply-add
        gflops = (flops / avg_time) / 1e9
        
        # Calculate theoretical tokens/sec for this operation
        # Assuming this is one layer of a 62-layer model
        time_per_token = avg_time * 62  # 62 layers
        tokens_per_second = batch / time_per_token
        
        logger.info(f"   ⏱️  Average time: {avg_time*1000:.2f}ms")
        logger.info(f"   ⚡ Performance: {gflops:.1f} GFLOPS")
        logger.info(f"   🔄 Theoretical TPS (62 layers): {tokens_per_second:.2f}")
    
    # Show memory stats
    stats = vulkan_engine.get_memory_stats()
    logger.info(f"\n📊 GPU Memory Usage:")
    logger.info(f"   VRAM allocated: {stats['vram_allocated_mb']:.1f}MB")
    logger.info(f"   GTT allocated: {stats['gtt_allocated_mb']:.1f}MB")
    
    # Test with real layer sizes from Gemma 27B
    logger.info(f"\n🦄 Testing Gemma 27B Layer Sizes")
    
    # Attention computation (Q, K, V projections)
    qkv_size = (1, 5376, 5376)
    matrix_a = np.random.randn(*qkv_size[:2]).astype(np.float32)
    matrix_b = np.random.randn(*qkv_size[1:]).astype(np.float32)
    
    start_time = time.time()
    for _ in range(10):
        _ = vulkan_engine.matrix_multiply(matrix_a, matrix_b)
    qkv_time = (time.time() - start_time) / 10
    
    # FFN computation
    ffn_up_size = (1, 5376, 14336)
    ffn_down_size = (1, 14336, 5376)
    
    matrix_up = np.random.randn(ffn_up_size[0], ffn_up_size[1]).astype(np.float32)
    weight_up = np.random.randn(ffn_up_size[1], ffn_up_size[2]).astype(np.float32)
    
    start_time = time.time()
    for _ in range(10):
        intermediate = vulkan_engine.matrix_multiply(matrix_up, weight_up)
    ffn_up_time = (time.time() - start_time) / 10
    
    matrix_down = np.random.randn(ffn_down_size[0], ffn_down_size[1]).astype(np.float32)
    weight_down = np.random.randn(ffn_down_size[1], ffn_down_size[2]).astype(np.float32)
    
    start_time = time.time()
    for _ in range(10):
        output = vulkan_engine.matrix_multiply(matrix_down, weight_down)
    ffn_down_time = (time.time() - start_time) / 10
    
    # Calculate layer time
    layer_time = (qkv_time * 3) + ffn_up_time + ffn_down_time
    model_time = layer_time * 62
    tokens_per_second = 1 / model_time
    
    logger.info(f"\n🎯 Gemma 27B Performance Estimate:")
    logger.info(f"   Q/K/V projection: {qkv_time*1000:.2f}ms each")
    logger.info(f"   FFN up projection: {ffn_up_time*1000:.2f}ms")
    logger.info(f"   FFN down projection: {ffn_down_time*1000:.2f}ms")
    logger.info(f"   Total per layer: {layer_time*1000:.2f}ms")
    logger.info(f"   Total per token (62 layers): {model_time:.2f}s")
    logger.info(f"   ⚡ Estimated TPS: {tokens_per_second:.3f}")
    
    # Cleanup
    vulkan_engine.cleanup()
    
    logger.info("\n✅ Vulkan performance test complete!")

if __name__ == "__main__":
    test_vulkan_performance()