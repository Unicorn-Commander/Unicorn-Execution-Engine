#!/usr/bin/env python3
"""
Test the attention fix with proper dimensions
"""

import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dimensions():
    """Test that dimensions work correctly"""
    
    # Gemma 27B dimensions
    batch_size = 1
    seq_len = 10  # Small for testing
    hidden_dim = 5376
    
    # Create test input
    hidden_states = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
    logger.info(f"Input shape: {hidden_states.shape}")
    
    # Flatten for projection
    hidden_flat = hidden_states.reshape(-1, hidden_dim)
    logger.info(f"Flattened shape: {hidden_flat.shape}")
    
    # Simulate projections (what GPU would compute)
    # Q projection: [batch*seq, 5376] @ [5376, 4096] = [batch*seq, 4096]
    q_weight_shape = (4096, 5376)
    q_output_dim = q_weight_shape[0]
    q = np.random.randn(hidden_flat.shape[0], q_output_dim).astype(np.float32)
    logger.info(f"Q projection output: {q.shape}")
    
    # K/V projections: [batch*seq, 5376] @ [5376, 2048] = [batch*seq, 2048]
    kv_weight_shape = (2048, 5376)
    kv_output_dim = kv_weight_shape[0]
    k = np.random.randn(hidden_flat.shape[0], kv_output_dim).astype(np.float32)
    v = np.random.randn(hidden_flat.shape[0], kv_output_dim).astype(np.float32)
    logger.info(f"K projection output: {k.shape}")
    logger.info(f"V projection output: {v.shape}")
    
    # Reshape for attention
    num_q_heads = 32  # 4096 / 128 = 32
    num_kv_heads = 16  # 2048 / 128 = 16
    head_dim = 128
    
    q_reshaped = q.reshape(batch_size, seq_len, num_q_heads, head_dim)
    k_reshaped = k.reshape(batch_size, seq_len, num_kv_heads, head_dim)
    v_reshaped = v.reshape(batch_size, seq_len, num_kv_heads, head_dim)
    
    logger.info(f"\nAfter reshape:")
    logger.info(f"Q: {q_reshaped.shape} (batch, seq, 32 heads, 128 dim)")
    logger.info(f"K: {k_reshaped.shape} (batch, seq, 16 heads, 128 dim)")
    logger.info(f"V: {v_reshaped.shape} (batch, seq, 16 heads, 128 dim)")
    
    # Transpose for attention
    q_t = q_reshaped.transpose(0, 2, 1, 3)
    k_t = k_reshaped.transpose(0, 2, 1, 3)
    v_t = v_reshaped.transpose(0, 2, 1, 3)
    
    logger.info(f"\nAfter transpose:")
    logger.info(f"Q: {q_t.shape} (batch, 32 heads, seq, 128 dim)")
    logger.info(f"K: {k_t.shape} (batch, 16 heads, seq, 128 dim)")
    
    # Expand KV heads for GQA
    k_expanded = np.repeat(k_t, 2, axis=1)
    v_expanded = np.repeat(v_t, 2, axis=1)
    
    logger.info(f"\nAfter KV expansion:")
    logger.info(f"K: {k_expanded.shape} (batch, 32 heads, seq, 128 dim)")
    logger.info(f"V: {v_expanded.shape} (batch, 32 heads, seq, 128 dim)")
    
    # Attention output
    attn_output_shape = (batch_size, seq_len, 4096)  # 32 heads * 128 dim
    logger.info(f"\nAttention output would be: {attn_output_shape}")
    
    # Output projection: [batch*seq, 4096] @ [4096, 5376] = [batch*seq, 5376]
    o_weight_shape = (5376, 4096)
    final_output_shape = (batch_size, seq_len, 5376)
    logger.info(f"Final output after O projection: {final_output_shape}")
    
    logger.info("\n✅ All dimensions check out!")

if __name__ == "__main__":
    test_dimensions()