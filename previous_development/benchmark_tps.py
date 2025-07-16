#!/usr/bin/env python3
"""
Benchmark TPS with the pure hardware pipeline
"""

import numpy as np
import time
import logging
from real_vulkan_matrix_compute_fixed import VulkanMatrixComputeFixed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def benchmark_single_layer():
    """Benchmark performance of a single transformer layer"""
    
    logger.info("🚀 Benchmarking Pure Hardware Pipeline Performance")
    logger.info("=" * 60)
    
    # Initialize Vulkan engine
    vulkan_engine = VulkanMatrixComputeFixed()
    if not vulkan_engine.initialize():
        logger.error("Failed to initialize Vulkan")
        return
    
    # Model dimensions (Gemma 27B)
    batch_size = 8  # Process 8 tokens at once
    seq_len = 1     # Single position (for generation)
    hidden_size = 5376
    num_heads = 32
    head_dim = 168
    ffn_intermediate = 14336
    
    logger.info(f"\n📊 Configuration:")
    logger.info(f"   Batch size: {batch_size} tokens")
    logger.info(f"   Hidden size: {hidden_size}")
    logger.info(f"   FFN intermediate: {ffn_intermediate}")
    
    # Create test data
    hidden_states = np.random.randn(batch_size, hidden_size).astype(np.float32)
    
    # Attention weights (Q, K, V, O projections)
    q_weight = np.random.randn(hidden_size, hidden_size).astype(np.float32)
    k_weight = np.random.randn(hidden_size, hidden_size // 2).astype(np.float32)  # GQA
    v_weight = np.random.randn(hidden_size, hidden_size // 2).astype(np.float32)  # GQA
    o_weight = np.random.randn(hidden_size, hidden_size).astype(np.float32)
    
    # FFN weights
    gate_weight = np.random.randn(hidden_size, ffn_intermediate).astype(np.float32)
    up_weight = np.random.randn(hidden_size, ffn_intermediate).astype(np.float32)
    down_weight = np.random.randn(ffn_intermediate, hidden_size).astype(np.float32)
    
    # Warmup
    logger.info("\n🔥 Warming up...")
    for _ in range(5):
        # Attention projections
        q = vulkan_engine.matrix_multiply(hidden_states, q_weight)
        k = vulkan_engine.matrix_multiply(hidden_states, k_weight)
        v = vulkan_engine.matrix_multiply(hidden_states, v_weight)
        
        # FFN
        gate = vulkan_engine.matrix_multiply(hidden_states, gate_weight)
        up = vulkan_engine.matrix_multiply(hidden_states, up_weight)
        # Simplified - skip activation
        intermediate = gate * up  # Would be SiLU(gate) * up
        output = vulkan_engine.matrix_multiply(intermediate, down_weight)
    
    # Benchmark
    logger.info("\n⚡ Benchmarking...")
    num_iterations = 50
    start_time = time.time()
    
    for _ in range(num_iterations):
        # Attention projections
        q = vulkan_engine.matrix_multiply(hidden_states, q_weight)
        k = vulkan_engine.matrix_multiply(hidden_states, k_weight)
        v = vulkan_engine.matrix_multiply(hidden_states, v_weight)
        
        # Simplified attention (skip actual attention computation)
        attn_output = hidden_states  # Placeholder
        
        # Output projection
        attn_output = vulkan_engine.matrix_multiply(attn_output, o_weight)
        
        # Residual
        hidden_states_2 = hidden_states + attn_output
        
        # FFN
        gate = vulkan_engine.matrix_multiply(hidden_states_2, gate_weight)
        up = vulkan_engine.matrix_multiply(hidden_states_2, up_weight)
        intermediate = gate * up  # Simplified
        ffn_output = vulkan_engine.matrix_multiply(intermediate, down_weight)
        
        # Residual
        hidden_states = hidden_states_2 + ffn_output
    
    elapsed_time = time.time() - start_time
    
    # Calculate metrics
    time_per_iteration = elapsed_time / num_iterations
    time_per_layer = time_per_iteration
    time_per_model = time_per_layer * 62  # 62 layers
    tokens_per_second = batch_size / time_per_model
    
    logger.info(f"\n📊 Results:")
    logger.info(f"   Iterations: {num_iterations}")
    logger.info(f"   Total time: {elapsed_time:.2f}s")
    logger.info(f"   Time per layer: {time_per_layer*1000:.2f}ms")
    logger.info(f"   Time per model (62 layers): {time_per_model*1000:.1f}ms")
    logger.info(f"   ⚡ Tokens per second: {tokens_per_second:.2f} TPS")
    
    # Memory stats
    stats = vulkan_engine.get_memory_stats()
    logger.info(f"\n💾 GPU Memory:")
    logger.info(f"   VRAM allocated: {stats['vram_allocated_mb']:.1f}MB")
    logger.info(f"   GTT allocated: {stats['gtt_allocated_mb']:.1f}MB")
    
    # Performance analysis
    logger.info(f"\n🎯 Analysis:")
    if tokens_per_second >= 8:
        logger.info(f"   ✅ Target achieved! ({tokens_per_second:.1f} >= 8 TPS)")
    else:
        logger.info(f"   ⚠️ Below target ({tokens_per_second:.1f} < 8 TPS)")
        speedup_needed = 8 / tokens_per_second
        logger.info(f"   Need {speedup_needed:.1f}x speedup")
    
    vulkan_engine.cleanup()

if __name__ == "__main__":
    benchmark_single_layer()