#!/usr/bin/env python3
"""
Fix GPU attention to use proper dimensions and GPU compute
"""

import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def compute_attention_layer_gpu_fixed(self, layer_idx: int, hidden_states: np.ndarray, 
                                    kv_cache: Optional[Tuple] = None) -> Tuple[np.ndarray, Optional[Tuple]]:
    """
    FIXED: Compute attention using GPU buffers with correct dimensions
    """
    
    # Get buffer keys with layer prefix
    q_key = f'layer_{layer_idx}_language_model.model.layers.{layer_idx}.self_attn.q_proj.weight'
    k_key = f'layer_{layer_idx}_language_model.model.layers.{layer_idx}.self_attn.k_proj.weight'
    v_key = f'layer_{layer_idx}_language_model.model.layers.{layer_idx}.self_attn.v_proj.weight'
    o_key = f'layer_{layer_idx}_language_model.model.layers.{layer_idx}.self_attn.o_proj.weight'
    
    # Check if buffers exist
    if q_key not in self.gpu_buffers:
        logger.warning(f"Attention weights not in GPU for layer {layer_idx}")
        return hidden_states, kv_cache
    
    try:
        # Get GPU buffer handles and shapes
        q_buffer_info, q_shape = self._get_gpu_buffer_with_shape(q_key)
        k_buffer_info, k_shape = self._get_gpu_buffer_with_shape(k_key)
        v_buffer_info, v_shape = self._get_gpu_buffer_with_shape(v_key)
        o_buffer_info, o_shape = self._get_gpu_buffer_with_shape(o_key)
        
        # Shapes are [out_features, in_features] for weights
        # Q: [4096, 5376], K/V: [2048, 5376], O: [5376, 4096]
        
        # Get dimensions
        batch_size = hidden_states.shape[0] if hidden_states.ndim == 3 else 1
        seq_len = hidden_states.shape[1] if hidden_states.ndim == 3 else hidden_states.shape[0]
        hidden_dim = hidden_states.shape[-1]  # Should be 5376
        
        # Flatten for matrix multiply
        hidden_flat = hidden_states.reshape(-1, hidden_dim)
        
        # Ensure float32
        if hidden_flat.dtype != np.float32:
            hidden_flat = hidden_flat.astype(np.float32)
        
        logger.debug(f"GPU Attention: input {hidden_flat.shape}, Q weight {q_shape}")
        
        # Compute Q, K, V projections on GPU
        # Matrix multiply: [batch*seq, 5376] @ [5376, 4096]^T = [batch*seq, 4096]
        q = self.vulkan_engine.compute_matrix_multiply_persistent(
            hidden_flat, q_buffer_info, q_shape, flags=0)  # Output: [batch*seq, 4096]
        
        # [batch*seq, 5376] @ [5376, 2048]^T = [batch*seq, 2048]
        k = self.vulkan_engine.compute_matrix_multiply_persistent(
            hidden_flat, k_buffer_info, k_shape, flags=0)  # Output: [batch*seq, 2048]
        v = self.vulkan_engine.compute_matrix_multiply_persistent(
            hidden_flat, v_buffer_info, v_shape, flags=0)  # Output: [batch*seq, 2048]
        
        # Reshape for multi-head attention
        num_q_heads = 32  # 4096 / 128 = 32 heads
        num_kv_heads = 16  # 2048 / 128 = 16 heads (GQA)
        head_dim = 128  # Standard head dimension
        
        # Reshape: [batch*seq, features] -> [batch, seq, num_heads, head_dim]
        q = q.reshape(batch_size, seq_len, num_q_heads, head_dim)
        k = k.reshape(batch_size, seq_len, num_kv_heads, head_dim)
        v = v.reshape(batch_size, seq_len, num_kv_heads, head_dim)
        
        # Transpose for attention: [batch, num_heads, seq, head_dim]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        # Grouped Query Attention - expand k,v to match q heads
        k = np.repeat(k, num_q_heads // num_kv_heads, axis=1)  # 16 -> 32 heads
        v = np.repeat(v, num_q_heads // num_kv_heads, axis=1)
        
        # TODO: Replace these CPU operations with GPU kernels!
        # For now, we'll keep the CPU operations but note what needs fixing:
        
        # CPU OPERATION 1: Scaled dot-product attention
        scale = 1.0 / np.sqrt(head_dim)
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        
        # CPU OPERATION 2: Softmax
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
        
        # CPU OPERATION 3: Apply attention
        attn_output = np.matmul(attn_weights, v)
        
        # Reshape back: [batch, seq, num_heads * head_dim]
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        
        # Output projection on GPU
        # [batch*seq, 4096] @ [4096, 5376]^T = [batch*seq, 5376]
        attn_flat = attn_output.reshape(-1, attn_output.shape[-1])
        output = self.vulkan_engine.compute_matrix_multiply_persistent(
            attn_flat, o_buffer_info, o_shape, flags=0)
        
        # Reshape back to original shape
        if batch_size == 1 and hidden_states.ndim == 2:
            output = output.reshape(seq_len, -1)
        else:
            output = output.reshape(batch_size, seq_len, -1)
        
        return output, kv_cache
        
    except Exception as e:
        logger.error(f"GPU attention failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return hidden_states, kv_cache

# This function would replace the compute_attention_layer_gpu method in 
# pure_hardware_pipeline_gpu_fixed.py