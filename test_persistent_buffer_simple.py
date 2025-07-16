#!/usr/bin/env python3
"""Simple test for persistent buffer performance without full model loading"""

import time
import numpy as np
import logging
from real_vulkan_matrix_compute import VulkanMatrixCompute
from vulkan_int8_support import add_int8_support

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_persistent_buffer_simple():
    """Test persistent buffer performance improvement"""
    logger.info("🚀 Testing Persistent Buffer Performance (Simple)")
    logger.info("=" * 80)
    
    # Initialize Vulkan engine
    add_int8_support(VulkanMatrixCompute)
    vulkan = VulkanMatrixCompute()
    if not vulkan.initialize():
        logger.error("Failed to initialize Vulkan")
        return
    
    # Test parameters
    batch_size = 1
    seq_len = 256
    hidden_size = 5376
    intermediate_size = 21504
    
    # Create test matrices
    logger.info(f"Creating test matrices...")
    input_data = np.random.randn(batch_size * seq_len, hidden_size).astype(np.float32)
    weight_data = np.random.randn(intermediate_size, hidden_size).astype(np.float32)
    
    # Test 1: Regular compute_matrix_multiply (with overhead)
    logger.info("\n📊 Test 1: Regular compute_matrix_multiply")
    regular_times = []
    
    for i in range(10):
        start = time.time()
        result = vulkan.compute_matrix_multiply(input_data, weight_data.T)
        end = time.time()
        regular_times.append(end - start)
        logger.info(f"   Iteration {i+1}: {(end-start)*1000:.2f}ms")
    
    avg_regular = np.mean(regular_times[2:])  # Skip warmup
    logger.info(f"   Average: {avg_regular*1000:.2f}ms")
    
    # Test 2: Persistent buffer (no overhead)
    logger.info("\n📊 Test 2: Persistent buffer compute_matrix_multiply")
    
    # Create persistent buffer
    logger.info("   Creating persistent buffer...")
    persistent_buffer = vulkan.create_persistent_buffer(weight_data.T)
    
    persistent_times = []
    for i in range(10):
        start = time.time()
        result = vulkan.compute_matrix_multiply_persistent(input_data, persistent_buffer, weight_data.T.shape)
        end = time.time()
        persistent_times.append(end - start)
        logger.info(f"   Iteration {i+1}: {(end-start)*1000:.2f}ms")
    
    avg_persistent = np.mean(persistent_times[2:])  # Skip warmup
    logger.info(f"   Average: {avg_persistent*1000:.2f}ms")
    
    # Results
    logger.info("\n🎯 Performance Comparison:")
    logger.info(f"   Regular:    {avg_regular*1000:.2f}ms per operation")
    logger.info(f"   Persistent: {avg_persistent*1000:.2f}ms per operation")
    logger.info(f"   Speedup:    {avg_regular/avg_persistent:.1f}x")
    logger.info(f"   Overhead eliminated: {(avg_regular - avg_persistent)*1000:.2f}ms")
    
    # Calculate TPS impact
    ops_per_layer = 7  # Q/K/V/O + gate/up/down
    layers = 62
    
    regular_token_time = avg_regular * ops_per_layer * layers
    persistent_token_time = avg_persistent * ops_per_layer * layers
    
    regular_tps = 1.0 / regular_token_time if regular_token_time > 0 else 0
    persistent_tps = 1.0 / persistent_token_time if persistent_token_time > 0 else 0
    
    logger.info("\n📈 Projected Performance:")
    logger.info(f"   Time per token (regular):    {regular_token_time:.3f}s ({regular_token_time*1000:.1f}ms)")
    logger.info(f"   Time per token (persistent): {persistent_token_time:.3f}s ({persistent_token_time*1000:.1f}ms)")
    logger.info(f"   Regular approach:    {regular_tps:.1f} TPS")
    logger.info(f"   Persistent buffers:  {persistent_tps:.1f} TPS")
    if regular_tps > 0:
        logger.info(f"   Improvement:         {persistent_tps/regular_tps:.1f}x")
    
    if persistent_tps > 81:
        logger.info(f"   ✅ EXCEEDS TARGET! {persistent_tps/81:.1f}x the 81 TPS target!")
    
    logger.info("=" * 80)
    
    # Cleanup
    vulkan.cleanup()

if __name__ == "__main__":
    test_persistent_buffer_simple()