#!/usr/bin/env python3
"""
Direct Vulkan speed test
Tests raw GPU performance without model loading
"""

import numpy as np
import logging
import time
from real_vulkan_matrix_compute import VulkanMatrixCompute

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_matrix_multiply_speed():
    """Test raw matrix multiply performance"""
    logger.info("🚀 Testing Vulkan Matrix Multiply Speed")
    logger.info("=" * 50)
    
    # Initialize Vulkan
    vulkan = VulkanMatrixCompute()
    vulkan.initialize(use_fp16=False)
    vulkan.initialize(use_fp16=False)
    
    # Test different matrix sizes
    sizes = [
        (10, 5376, 4096),      # 10 tokens, attention projection
        (50, 5376, 4096),      # 50 tokens
        (100, 5376, 4096),     # 100 tokens
        (10, 5376, 18432),     # FFN gate/up projection
        (10, 18432, 5376),     # FFN down projection
    ]
    
    for batch_seq, m, n in sizes:
        logger.info(f"\n📊 Testing [{batch_seq} x {m}] × [{m} x {n}]")
        
        # Create test matrices
        a = np.random.randn(batch_seq, m).astype(np.float32)
        b = np.random.randn(m, n).astype(np.float32)
        
        # Warmup (b is already transposed shape)
        _ = vulkan.matrix_multiply(a, b)
        
        # Time multiple runs
        num_runs = 10
        start = time.time()
        
        for _ in range(num_runs):
            result = vulkan.matrix_multiply(a, b)
        
        total_time = time.time() - start
        avg_time = total_time / num_runs
        
        # Calculate GFLOPS
        flops = 2 * batch_seq * m * n  # multiply-add
        gflops = (flops / 1e9) / avg_time
        
        logger.info(f"   Average time: {avg_time*1000:.2f}ms")
        logger.info(f"   Performance: {gflops:.1f} GFLOPS")
        
        # Calculate theoretical TPS for this operation
        if batch_seq <= 100:  # reasonable sequence length
            ms_per_token = (avg_time * 1000) / batch_seq
            logger.info(f"   Per token: {ms_per_token:.2f}ms")

def test_full_layer_speed():
    """Test full transformer layer speed"""
    logger.info("\n\n🧠 Testing Full Layer Speed (Attention + FFN)")
    logger.info("=" * 50)
    
    vulkan = VulkanMatrixCompute()
    vulkan.initialize(use_fp16=False)
    
    # Gemma 27B dimensions
    batch_size = 1
    seq_len = 50
    hidden_dim = 5376
    num_heads = 32
    head_dim = 128
    intermediate_dim = 18432
    
    # Create layer weights (transposed for matrix multiply)
    weights = {
        'q_proj': np.random.randn(hidden_dim, 4096).astype(np.float32),
        'k_proj': np.random.randn(hidden_dim, 2048).astype(np.float32),
        'v_proj': np.random.randn(hidden_dim, 2048).astype(np.float32),
        'o_proj': np.random.randn(4096, hidden_dim).astype(np.float32),
        'gate_proj': np.random.randn(hidden_dim, intermediate_dim).astype(np.float32),
        'up_proj': np.random.randn(hidden_dim, intermediate_dim).astype(np.float32),
        'down_proj': np.random.randn(intermediate_dim, hidden_dim).astype(np.float32),
    }
    
    # Input
    hidden_states = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
    hidden_flat = hidden_states.reshape(-1, hidden_dim)
    
    # Time full layer
    start = time.time()
    
    # Attention projections (transpose weights to match expected dimensions)
    q = vulkan.matrix_multiply(hidden_flat, weights['q_proj'])
    k = vulkan.matrix_multiply(hidden_flat, weights['k_proj'])
    v = vulkan.matrix_multiply(hidden_flat, weights['v_proj'])
    
    # Simplified attention (just measure projection cost)
    # For now, use Q as attention output since full attention is complex
    attn_output = vulkan.matrix_multiply(q.reshape(-1, 4096), weights['o_proj'])
    
    # FFN
    gate = vulkan.matrix_multiply(hidden_flat, weights['gate_proj'])
    up = vulkan.matrix_multiply(hidden_flat, weights['up_proj'])
    
    # SiLU activation (CPU for now)
    gate_activated = gate * (1.0 / (1.0 + np.exp(-gate)))
    intermediate = gate_activated * up
    
    ffn_output = vulkan.matrix_multiply(intermediate, weights['down_proj'])
    
    layer_time = time.time() - start
    
    # Results
    tokens_processed = batch_size * seq_len
    ms_per_token = (layer_time * 1000) / tokens_processed
    tps_per_layer = tokens_processed / layer_time
    
    logger.info(f"\n📊 Layer Performance:")
    logger.info(f"   Total time: {layer_time*1000:.1f}ms for {tokens_processed} tokens")
    logger.info(f"   Per token: {ms_per_token:.2f}ms")
    logger.info(f"   TPS (single layer): {tps_per_layer:.1f}")
    
    # Full model estimate (62 layers)
    full_model_time = layer_time * 62
    full_model_tps = tokens_processed / full_model_time
    
    logger.info(f"\n🎯 Full Model Estimate (62 layers):")
    logger.info(f"   Total time: {full_model_time*1000:.1f}ms")
    logger.info(f"   TPS: {full_model_tps:.1f}")
    
    if full_model_tps >= 81:
        logger.info("   🎉 TARGET ACHIEVED! 81+ TPS possible!")
    elif full_model_tps >= 50:
        logger.info("   ✅ Good performance!")
    else:
        logger.info("   ⚠️ Need optimization")
    
    return full_model_tps

def test_batched_performance():
    """Test performance with different batch sizes"""
    logger.info("\n\n📦 Testing Batched Performance")
    logger.info("=" * 50)
    
    vulkan = VulkanMatrixCompute()
    vulkan.initialize(use_fp16=False)
    
    batch_sizes = [1, 4, 8, 16, 32]
    seq_len = 50
    
    results = []
    
    for batch_size in batch_sizes:
        tokens = batch_size * seq_len
        
        # Test matrix multiply
        a = np.random.randn(tokens, 5376).astype(np.float32)
        b = np.random.randn(5376, 4096).astype(np.float32)
        
        # Time it
        start = time.time()
        for _ in range(5):
            _ = vulkan.matrix_multiply(a, b)
        
        avg_time = (time.time() - start) / 5
        tps = tokens / avg_time
        
        results.append((batch_size, tokens, tps))
        logger.info(f"   Batch {batch_size}: {tokens} tokens, {tps:.1f} TPS")
    
    # Find optimal batch size
    best_batch, _, best_tps = max(results, key=lambda x: x[2])
    logger.info(f"\n🏆 Best batch size: {best_batch} ({best_tps:.1f} TPS)")

def main():
    logger.info("🦄 Vulkan GPU Speed Test")
    logger.info("Testing raw GPU performance for 81 TPS target\n")
    
    # Test raw matrix multiply
    test_matrix_multiply_speed()
    
    # Test full layer
    estimated_tps = test_full_layer_speed()
    
    # Test batching
    test_batched_performance()
    
    logger.info(f"\n\n🎯 FINAL ESTIMATE: {estimated_tps:.1f} TPS")
    logger.info("Note: This is theoretical GPU-only performance.")
    logger.info("Actual performance depends on:")
    logger.info("  - Memory bandwidth")
    logger.info("  - CPU-GPU synchronization")
    logger.info("  - KV cache management")
    logger.info("  - Token generation overhead")

if __name__ == "__main__":
    main()