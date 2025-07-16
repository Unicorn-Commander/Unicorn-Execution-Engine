#!/usr/bin/env python3
"""
Quick test of GPU pipeline without full model load
Tests the basic inference path
"""

import logging
import time
import numpy as np
from gpu_pipeline_working import GPUPipelineWorking

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_single_layer():
    """Test single layer forward pass"""
    logger.info("🧪 Testing single layer GPU computation...")
    
    # Create pipeline
    pipeline = GPUPipelineWorking()
    
    # Initialize with dummy path (won't actually load model)
    logger.info("Initializing pipeline...")
    # This should initialize vulkan_engine
    initialized = pipeline.initialize("dummy_path")
    
    # Check if vulkan engine is available
    if not hasattr(pipeline, 'vulkan_engine') or pipeline.vulkan_engine is None:
        logger.error("Vulkan engine not initialized!")
        return 0
    
    # Create fake layer weights for testing
    logger.info("Creating test weights...")
    
    # Attention weights (matching Gemma 27B dimensions)
    q_weight = np.random.randn(4096, 5376).astype(np.float32) * 0.02
    k_weight = np.random.randn(2048, 5376).astype(np.float32) * 0.02
    v_weight = np.random.randn(2048, 5376).astype(np.float32) * 0.02
    o_weight = np.random.randn(5376, 4096).astype(np.float32) * 0.02
    
    # FFN weights
    gate_weight = np.random.randn(18432, 5376).astype(np.float32) * 0.02
    up_weight = np.random.randn(18432, 5376).astype(np.float32) * 0.02
    down_weight = np.random.randn(5376, 18432).astype(np.float32) * 0.02
    
    # Test input
    batch_size = 1
    seq_len = 10
    hidden_states = np.random.randn(batch_size, seq_len, 5376).astype(np.float32)
    
    logger.info(f"Input shape: {hidden_states.shape}")
    
    # Test attention
    logger.info("\n🧠 Testing Attention Computation...")
    start = time.time()
    
    # Compute Q, K, V projections
    q = pipeline.vulkan_engine.matrix_multiply(
        hidden_states.reshape(-1, 5376), 
        q_weight.T
    ).reshape(batch_size, seq_len, 4096)
    
    k = pipeline.vulkan_engine.matrix_multiply(
        hidden_states.reshape(-1, 5376), 
        k_weight.T
    ).reshape(batch_size, seq_len, 2048)
    
    v = pipeline.vulkan_engine.matrix_multiply(
        hidden_states.reshape(-1, 5376), 
        v_weight.T
    ).reshape(batch_size, seq_len, 2048)
    
    attention_time = time.time() - start
    logger.info(f"✅ Attention projections: {attention_time*1000:.1f}ms")
    
    # Test FFN
    logger.info("\n🔥 Testing FFN Computation...")
    start = time.time()
    
    # Gate and up projection
    gate_proj = pipeline.vulkan_engine.matrix_multiply(
        hidden_states.reshape(-1, 5376),
        gate_weight.T
    )
    
    up_proj = pipeline.vulkan_engine.matrix_multiply(
        hidden_states.reshape(-1, 5376),
        up_weight.T
    )
    
    # SiLU activation on gate
    gate_activated = gate_proj * (1.0 / (1.0 + np.exp(-gate_proj)))
    
    # Multiply with up projection
    intermediate = gate_activated * up_proj
    
    # Down projection
    ffn_output = pipeline.vulkan_engine.matrix_multiply(
        intermediate,
        down_weight.T
    ).reshape(batch_size, seq_len, 5376)
    
    ffn_time = time.time() - start
    logger.info(f"✅ FFN computation: {ffn_time*1000:.1f}ms")
    
    # Calculate theoretical TPS
    total_time = attention_time + ffn_time
    ms_per_token = (total_time * 1000) / seq_len
    theoretical_tps = 1000 / ms_per_token
    
    logger.info(f"\n📊 Performance Summary:")
    logger.info(f"   Total time: {total_time*1000:.1f}ms for {seq_len} tokens")
    logger.info(f"   Per token: {ms_per_token:.1f}ms")
    logger.info(f"   Theoretical TPS (single layer): {theoretical_tps:.1f}")
    logger.info(f"   Full model (62 layers) estimate: {theoretical_tps/62:.1f} TPS")
    
    # Cleanup
    del pipeline
    
    return theoretical_tps / 62

def test_with_batching():
    """Test with different batch sizes"""
    logger.info("\n🚀 Testing Batch Performance...")
    
    batch_sizes = [1, 4, 8, 16]
    
    for batch_size in batch_sizes:
        logger.info(f"\n📦 Batch size: {batch_size}")
        
        pipeline = GPUPipelineWorking()
        if not hasattr(pipeline, 'vulkan_engine') or pipeline.vulkan_engine is None:
            logger.error("Vulkan engine not initialized!")
            continue
        
        # Create batched input
        seq_len = 50
        hidden_states = np.random.randn(batch_size, seq_len, 5376).astype(np.float32)
        
        # Simple forward computation
        start = time.time()
        
        # Simulate one matrix multiply
        weight = np.random.randn(5376, 5376).astype(np.float32) * 0.02
        output = pipeline.vulkan_engine.matrix_multiply(
            hidden_states.reshape(-1, 5376),
            weight.T
        )
        
        compute_time = time.time() - start
        
        total_tokens = batch_size * seq_len
        tps = total_tokens / compute_time
        
        logger.info(f"   Tokens: {total_tokens}, Time: {compute_time:.3f}s")
        logger.info(f"   TPS: {tps:.1f}")
        
        del pipeline

def main():
    logger.info("🦄 GPU Pipeline Quick Test")
    logger.info("=" * 50)
    
    # Test single layer
    estimated_tps = test_single_layer()
    
    # Test batching
    test_with_batching()
    
    logger.info(f"\n🎯 Estimated full model TPS: {estimated_tps:.1f}")
    
    if estimated_tps >= 81:
        logger.info("🎉 TARGET ACHIEVED! 81+ TPS possible!")
    elif estimated_tps >= 50:
        logger.info("✅ Good performance! Close to target.")
    else:
        logger.info("⚠️ Performance below target. Need optimization.")

if __name__ == "__main__":
    main()