#!/usr/bin/env python3
"""
Test pipeline with dimension fixes
"""

import numpy as np
import logging
import time

# Import and patch the pipeline
from pure_hardware_pipeline_gpu_fixed import PureHardwarePipelineGPUFixed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Monkey-patch the attention fix
def compute_attention_layer_gpu_fixed(self, layer_idx: int, hidden_states: np.ndarray, 
                                    kv_cache=None):
    """Fixed attention with correct dimensions"""
    
    # Get buffer keys
    q_key = f'layer_{layer_idx}_language_model.model.layers.{layer_idx}.self_attn.q_proj.weight'
    k_key = f'layer_{layer_idx}_language_model.model.layers.{layer_idx}.self_attn.k_proj.weight'
    v_key = f'layer_{layer_idx}_language_model.model.layers.{layer_idx}.self_attn.v_proj.weight'
    o_key = f'layer_{layer_idx}_language_model.model.layers.{layer_idx}.self_attn.o_proj.weight'
    
    if q_key not in self.gpu_buffers:
        logger.warning(f"Attention weights not in GPU for layer {layer_idx}")
        return hidden_states, kv_cache
    
    try:
        # Get GPU buffer info
        q_buffer_info, q_shape = self._get_gpu_buffer_with_shape(q_key)
        k_buffer_info, k_shape = self._get_gpu_buffer_with_shape(k_key)
        v_buffer_info, v_shape = self._get_gpu_buffer_with_shape(v_key)
        o_buffer_info, o_shape = self._get_gpu_buffer_with_shape(o_key)
        
        # Dimensions
        batch_size = 1
        seq_len = hidden_states.shape[0] if hidden_states.ndim == 2 else hidden_states.shape[1]
        hidden_dim = hidden_states.shape[-1]
        
        # Flatten
        hidden_flat = hidden_states.reshape(-1, hidden_dim).astype(np.float32)
        
        # Project (GPU compute)
        q = self.vulkan_engine.compute_matrix_multiply_persistent(
            hidden_flat, q_buffer_info, q_shape, flags=0)
        k = self.vulkan_engine.compute_matrix_multiply_persistent(
            hidden_flat, k_buffer_info, k_shape, flags=0)
        v = self.vulkan_engine.compute_matrix_multiply_persistent(
            hidden_flat, v_buffer_info, v_shape, flags=0)
        
        # Multi-head attention reshape
        q = q.reshape(batch_size, seq_len, 32, 128).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, 16, 128).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, 16, 128).transpose(0, 2, 1, 3)
        
        # GQA expansion
        k = np.repeat(k, 2, axis=1)
        v = np.repeat(v, 2, axis=1)
        
        # Attention (still CPU for now)
        scale = 1.0 / np.sqrt(128)
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
        attn_output = np.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        attn_flat = attn_output.reshape(-1, 4096)
        
        # Output projection
        output = self.vulkan_engine.compute_matrix_multiply_persistent(
            attn_flat, o_buffer_info, o_shape, flags=0)
        
        output = output.reshape(seq_len, -1)
        
        return output, kv_cache
        
    except Exception as e:
        logger.error(f"Attention failed: {e}")
        return hidden_states, kv_cache

# Apply the patch
PureHardwarePipelineGPUFixed.compute_attention_layer_gpu = compute_attention_layer_gpu_fixed

def main():
    logger.info("🔧 Testing fixed pipeline...")
    
    pipeline = PureHardwarePipelineGPUFixed()
    
    # Initialize
    model_path = "quantized_models/gemma-3-27b-it-layer-by-layer"
    logger.info("Loading model...")
    
    if not pipeline.initialize(model_path):
        logger.error("Failed to initialize")
        return
    
    logger.info("✅ Model loaded!")
    
    # Test single layer
    logger.info("\nTesting single layer forward pass...")
    test_input = np.random.randn(1, 10, 5376).astype(np.float32)
    
    start = time.time()
    output, _ = pipeline.forward_layer(0, test_input)
    elapsed = time.time() - start
    
    logger.info(f"✅ Layer 0 forward pass: {elapsed*1000:.2f}ms")
    logger.info(f"   Input shape: {test_input.shape}")
    logger.info(f"   Output shape: {output.shape}")
    
    # Cleanup
    pipeline.cleanup()

if __name__ == "__main__":
    main()