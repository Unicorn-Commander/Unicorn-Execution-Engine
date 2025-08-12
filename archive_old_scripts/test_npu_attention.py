#!/usr/bin/env python3
"""
Quick test for NPU attention implementation
"""

import numpy as np
import logging
from npu_attention_kernel_optimized import NPUAttentionKernelOptimized

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_npu_attention():
    """Test NPU attention with simple synthetic data"""
    logger.info("🧠 Testing NPU Flash Attention implementation...")
    
    # Initialize NPU kernel
    npu_kernel = NPUAttentionKernelOptimized(seq_length=32, d_model=5376, num_heads=32)
    
    if not npu_kernel.initialize():
        logger.error("❌ Failed to initialize NPU kernel")
        return False
    
    # Create synthetic inputs
    batch_size = 1
    seq_len = 32
    hidden_size = 5376
    
    # Hidden states
    hidden_states = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
    
    # Weights for GQA (32 Q heads, 16 K/V heads)
    q_proj_weight = np.random.randn(4096, hidden_size).astype(np.float32)  # 32 heads * 128 dim
    k_proj_weight = np.random.randn(2048, hidden_size).astype(np.float32)  # 16 heads * 128 dim
    v_proj_weight = np.random.randn(2048, hidden_size).astype(np.float32)  # 16 heads * 128 dim
    o_proj_weight = np.random.randn(hidden_size, 4096).astype(np.float32)  # output projection
    
    logger.info(f"Input shapes:")
    logger.info(f"  Hidden states: {hidden_states.shape}")
    logger.info(f"  Q weight: {q_proj_weight.shape}")
    logger.info(f"  K weight: {k_proj_weight.shape}")
    logger.info(f"  V weight: {v_proj_weight.shape}")
    logger.info(f"  O weight: {o_proj_weight.shape}")
    
    try:
        # Test Flash Attention
        output, k_cache, v_cache = npu_kernel.compute_flash_attention(
            hidden_states, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, None
        )
        
        logger.info(f"✅ Flash Attention completed successfully!")
        logger.info(f"  Output shape: {output.shape}")
        logger.info(f"  K cache shape: {k_cache.shape}")
        logger.info(f"  V cache shape: {v_cache.shape}")
        
        # Test with KV cache
        logger.info("🔄 Testing with KV cache...")
        output2, k_cache2, v_cache2 = npu_kernel.compute_flash_attention(
            hidden_states, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, (k_cache, v_cache)
        )
        
        logger.info(f"✅ Flash Attention with KV cache completed!")
        logger.info(f"  Output shape: {output2.shape}")
        logger.info(f"  K cache shape: {k_cache2.shape}")
        logger.info(f"  V cache shape: {v_cache2.shape}")
        
        # Basic sanity checks
        assert output.shape == (batch_size, seq_len, hidden_size), f"Output shape mismatch: {output.shape}"
        assert k_cache.shape == (batch_size, seq_len, 2048), f"K cache shape mismatch: {k_cache.shape}"
        assert v_cache.shape == (batch_size, seq_len, 2048), f"V cache shape mismatch: {v_cache.shape}"
        
        logger.info("✅ All NPU attention tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ NPU attention test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    finally:
        npu_kernel.cleanup()

if __name__ == "__main__":
    success = test_npu_attention()
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}")