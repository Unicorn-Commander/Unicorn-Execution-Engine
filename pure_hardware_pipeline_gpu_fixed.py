#!/usr/bin/env python3
"""
Pure Hardware Pipeline - PROPERLY FIXED for GPU Compute
This version actually uses GPU buffers instead of loading back to CPU
"""

import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any

from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed

logger = logging.getLogger(__name__)

class PureHardwarePipelineGPUFixed(PureHardwarePipelineFixed):
    """Fixed pipeline that properly uses GPU compute"""
    
    def __init__(self):
        super().__init__()
        self._gpu_compute_enabled = True
        logger.info("🚀 GPU-Fixed Pipeline: Will use GPU buffers directly")
    
    def compute_attention_layer_gpu(self, layer_idx: int, hidden_states: np.ndarray, 
                                  kv_cache: Optional[Tuple] = None) -> Tuple[np.ndarray, Optional[Tuple]]:
        """Compute attention using GPU buffers directly - NO CPU LOADING"""
        
        # Get buffer keys - note the layer_N_ prefix in stored keys
        q_key = f'layer_{layer_idx}_language_model.model.layers.{layer_idx}.self_attn.q_proj.weight'
        k_key = f'layer_{layer_idx}_language_model.model.layers.{layer_idx}.self_attn.k_proj.weight'
        v_key = f'layer_{layer_idx}_language_model.model.layers.{layer_idx}.self_attn.v_proj.weight'
        o_key = f'layer_{layer_idx}_language_model.model.layers.{layer_idx}.self_attn.o_proj.weight'
        
        # Check if buffers exist
        if q_key not in self.gpu_buffers:
            logger.warning(f"Attention weights not in GPU for layer {layer_idx}")
            return hidden_states, kv_cache
        
        try:
            # Get GPU buffer handles and shapes - DO NOT LOAD TO CPU!
            q_buffer_info, q_shape = self._get_gpu_buffer_with_shape(q_key)
            k_buffer_info, k_shape = self._get_gpu_buffer_with_shape(k_key)
            v_buffer_info, v_shape = self._get_gpu_buffer_with_shape(v_key)
            o_buffer_info, o_shape = self._get_gpu_buffer_with_shape(o_key)
            
            # Reshape for batch processing
            batch_size = hidden_states.shape[0] if hidden_states.ndim == 3 else 1
            seq_len = hidden_states.shape[1] if hidden_states.ndim == 3 else hidden_states.shape[0]
            hidden_dim = hidden_states.shape[-1]
            
            # Flatten for matrix multiply
            if hidden_states.ndim == 3:
                hidden_flat = hidden_states.reshape(-1, hidden_dim)
            else:
                hidden_flat = hidden_states.reshape(-1, hidden_dim)
            
            # Compute Q, K, V on GPU using persistent buffers
            logger.debug(f"GPU Attention: {hidden_flat.shape} x {q_shape}")
            
            # Ensure input is float32 for computation
            if hidden_flat.dtype != np.float32:
                hidden_flat = hidden_flat.astype(np.float32)
            
            # Use persistent GPU buffers for computation
            q = self.vulkan_engine.compute_matrix_multiply_persistent(
                hidden_flat, q_buffer_info, q_shape, flags=0)
            k = self.vulkan_engine.compute_matrix_multiply_persistent(
                hidden_flat, k_buffer_info, k_shape, flags=0)
            v = self.vulkan_engine.compute_matrix_multiply_persistent(
                hidden_flat, v_buffer_info, v_shape, flags=0)
            
            # Reshape back to [batch, seq, num_heads, head_dim]
            num_q_heads = 32  # Gemma uses 32 Q heads
            num_kv_heads = 16  # Gemma uses 16 KV heads (GQA)
            
            # Correctly calculate head dimensions based on the actual projection output shapes.
            # The weight shape is (input_features, output_features), so shape[1] is the output dimension.
            q_head_dim = q_shape[1] // num_q_heads
            kv_head_dim = k_shape[1] // num_kv_heads
            
            logger.debug(f"Reshape - Q shape: {q.shape}, Q heads: {num_q_heads}, Q head_dim: {q_head_dim}")
            logger.debug(f"Reshape - K shape: {k.shape}, KV heads: {num_kv_heads}, KV head_dim: {kv_head_dim}")
            
            q = q.reshape(batch_size, seq_len, num_q_heads, q_head_dim)
            k = k.reshape(batch_size, seq_len, num_kv_heads, kv_head_dim)
            v = v.reshape(batch_size, seq_len, num_kv_heads, kv_head_dim)
            
            # Transpose for attention: [batch, num_heads, seq, head_dim]
            q = q.transpose(0, 2, 1, 3)
            k = k.transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)
            
            # Grouped Query Attention - expand k,v to match q heads
            k = np.repeat(k, num_q_heads // num_kv_heads, axis=1)  # 16 -> 32 heads
            v = np.repeat(v, num_q_heads // num_kv_heads, axis=1)
            
            # Scaled dot-product attention
            scale = 1.0 / np.sqrt(q_head_dim)  # Use Q head dimension for scaling
            scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale
            
            # Softmax
            exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
            
            # Apply attention
            attn_output = np.matmul(attn_weights, v)
            
            # Reshape back: [batch, seq, hidden]
            attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
            
            # Output projection on GPU
            attn_flat = attn_output.reshape(-1, attn_output.shape[-1])
            output = self.vulkan_engine.compute_matrix_multiply_persistent(
                attn_flat, o_buffer_info, o_shape, flags=0)
            
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
    
    def compute_ffn_layer_gpu(self, layer_idx: int, hidden_states: np.ndarray) -> np.ndarray:
        """Compute FFN using GPU buffers directly"""
        
        gate_key = f'layer_{layer_idx}_language_model.model.layers.{layer_idx}.mlp.gate_proj.weight'
        up_key = f'layer_{layer_idx}_language_model.model.layers.{layer_idx}.mlp.up_proj.weight'
        down_key = f'layer_{layer_idx}_language_model.model.layers.{layer_idx}.mlp.down_proj.weight'
        
        if gate_key not in self.gpu_buffers:
            logger.warning(f"FFN weights not in GPU for layer {layer_idx}")
            return hidden_states
        
        try:
            # Get GPU buffer handles and shapes
            gate_buffer_info, gate_shape = self._get_gpu_buffer_with_shape(gate_key)
            up_buffer_info, up_shape = self._get_gpu_buffer_with_shape(up_key)
            down_buffer_info, down_shape = self._get_gpu_buffer_with_shape(down_key)
            
            # Check if fused kernel is available
            if hasattr(self.vulkan_engine, 'compute_fused_ffn_persistent_weights'):
                output = self.vulkan_engine.compute_fused_ffn_persistent_weights(
                    hidden_states,
                    gate_buffer_info, gate_shape,
                    up_buffer_info, up_shape,
                    down_buffer_info, down_shape,
                    flags=0  # FP32 for now
                )
                return output
            else:
                # Fallback to separate operations
                batch_size = hidden_states.shape[0] if hidden_states.ndim == 3 else 1
                seq_len = hidden_states.shape[1] if hidden_states.ndim == 3 else hidden_states.shape[0]
                hidden_dim = hidden_states.shape[-1]
                
                # Flatten for matrix multiply
                hidden_flat = hidden_states.reshape(-1, hidden_dim)
                
                # Compute gate and up projections
                gate_output = self.vulkan_engine.compute_matrix_multiply_persistent(
                    hidden_flat, gate_buffer_info, gate_shape, flags=0)
                up_output = self.vulkan_engine.compute_matrix_multiply_persistent(
                    hidden_flat, up_buffer_info, up_shape, flags=0)
                
                # SiLU activation and element-wise multiply
                gate_activated = gate_output * (1.0 / (1.0 + np.exp(-gate_output)))  # SiLU
                intermediate = gate_activated * up_output
                
                # Down projection
                output = self.vulkan_engine.compute_matrix_multiply_persistent(
                    intermediate, down_buffer_info, down_shape, flags=0)
                
                # Reshape back
                if batch_size == 1 and hidden_states.ndim == 2:
                    output = output.reshape(seq_len, -1)
                else:
                    output = output.reshape(batch_size, seq_len, -1)
                
                return output
            
        except Exception as e:
            logger.error(f"GPU FFN failed: {e}")
            return hidden_states
    
    def forward_layer(self, layer_idx: int, hidden_states: np.ndarray,
                     position_ids: Optional[np.ndarray] = None,
                     kv_cache: Optional[Tuple] = None) -> Tuple[np.ndarray, Tuple]:
        """Forward pass through a layer using GPU compute"""
        
        if layer_idx not in self.layer_weights_gpu:
            logger.warning(f"Layer {layer_idx} not in GPU")
            return hidden_states, kv_cache
        
        # Residual connection
        residual = hidden_states
        
        # Layer norm (simplified - should also be on GPU)
        hidden_states = self._layer_norm(hidden_states)
        
        # Attention on GPU
        attention_output, kv_cache = self.compute_attention_layer_gpu(layer_idx, hidden_states, kv_cache)
        hidden_states = residual + attention_output
        
        # Post-attention residual
        residual = hidden_states
        
        # Post-attention layer norm
        hidden_states = self._layer_norm(hidden_states)
        
        # FFN on GPU
        ffn_output = self.compute_ffn_layer_gpu(layer_idx, hidden_states)
        hidden_states = residual + ffn_output
        
        return hidden_states, kv_cache
    
    def _layer_norm(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Simple layer norm (should be GPU kernel too)"""
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)
    
    def benchmark_layer(self, layer_idx: int = 0, num_iterations: int = 50):
        """Benchmark a single layer"""
        # Use correct hidden dimension for Gemma 27B
        test_input = np.random.randn(1, 1, 5376).astype(np.float32)
        
        # Warm up
        for _ in range(10):
            output, _ = self.forward_layer(layer_idx, test_input)
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            output, _ = self.forward_layer(layer_idx, test_input)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        avg_time = np.mean(times[5:])  # Skip first few
        return avg_time


def test_gpu_pipeline():
    """Test the GPU-fixed pipeline"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    logger.info("🚀 Testing GPU-Fixed Pipeline")
    
    # Initialize
    pipeline = PureHardwarePipelineGPUFixed()
    model_path = "quantized_models/gemma-3-27b-it-layer-by-layer"
    
    logger.info("Loading model...")
    start = time.time()
    
    if not pipeline.initialize(model_path):
        logger.error("Failed to initialize")
        return
    
    load_time = time.time() - start
    logger.info(f"✅ Model loaded in {load_time:.1f}s")
    logger.info(f"   Layers in GPU: {len(pipeline.layer_weights_gpu)}")
    
    # Benchmark
    logger.info("\n📊 Benchmarking GPU compute...")
    avg_time = pipeline.benchmark_layer(0, num_iterations=20)
    
    logger.info(f"   Average layer time: {avg_time*1000:.2f}ms")
    
    # Estimate full model
    full_time = avg_time * 62
    tps = 1.0 / full_time
    
    logger.info(f"\n📊 Performance:")
    logger.info(f"   Per layer: {avg_time*1000:.2f}ms")
    logger.info(f"   Full model: {full_time*1000:.0f}ms")
    logger.info(f"   Single-stream TPS: {tps:.1f}")
    
    if tps >= 81:
        logger.info("   ✅ TARGET ACHIEVED!")
    elif tps >= 10:
        logger.info("   ✅ GPU compute working! Need optimization for 81 TPS")
    else:
        logger.info("   ⚠️ Performance still too low")
    
    # Cleanup
    pipeline.cleanup()
    
    return tps


if __name__ == "__main__":
    test_gpu_pipeline()