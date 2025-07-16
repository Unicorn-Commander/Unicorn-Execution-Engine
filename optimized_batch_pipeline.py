#!/usr/bin/env python3
"""
Optimized Batch Pipeline - Next optimization for 81 TPS target
Implements batch processing to increase GPU utilization
"""

import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any

from pure_hardware_pipeline_gpu_fixed import PureHardwarePipelineGPUFixed

logger = logging.getLogger(__name__)

class OptimizedBatchPipeline(PureHardwarePipelineGPUFixed):
    """Optimized pipeline with batch processing for higher throughput"""
    
    def __init__(self, max_batch_size: int = 8):
        super().__init__()
        self.max_batch_size = max_batch_size
        self._batch_buffer = []
        logger.info(f"🚀 Optimized Batch Pipeline: max_batch_size={max_batch_size}")
    
    def compute_attention_layer_gpu_batch(self, layer_idx: int, hidden_states_batch: List[np.ndarray], 
                                        kv_cache_batch: Optional[List[Tuple]] = None) -> Tuple[List[np.ndarray], List[Optional[Tuple]]]:
        """Compute attention for a batch of inputs using GPU buffers directly"""
        
        # Get buffer keys
        q_key = f'layer_{layer_idx}_language_model.model.layers.{layer_idx}.self_attn.q_proj.weight'
        k_key = f'layer_{layer_idx}_language_model.model.layers.{layer_idx}.self_attn.k_proj.weight'
        v_key = f'layer_{layer_idx}_language_model.model.layers.{layer_idx}.self_attn.v_proj.weight'
        o_key = f'layer_{layer_idx}_language_model.model.layers.{layer_idx}.self_attn.o_proj.weight'
        
        if q_key not in self.gpu_buffers:
            logger.warning(f"Attention weights not in GPU for layer {layer_idx}")
            return hidden_states_batch, kv_cache_batch or [None] * len(hidden_states_batch)
        
        try:
            # Get GPU buffer handles and shapes
            q_buffer_info, q_shape = self._get_gpu_buffer_with_shape(q_key)
            k_buffer_info, k_shape = self._get_gpu_buffer_with_shape(k_key)
            v_buffer_info, v_shape = self._get_gpu_buffer_with_shape(v_key)
            o_buffer_info, o_shape = self._get_gpu_buffer_with_shape(o_key)
            
            # Process batch efficiently
            batch_size = len(hidden_states_batch)
            results = []
            
            # Stack inputs for batch processing
            max_seq_len = max(hs.shape[-2] if hs.ndim == 3 else hs.shape[-1] for hs in hidden_states_batch)
            hidden_dim = hidden_states_batch[0].shape[-1]
            
            # Create padded batch tensor
            batch_tensor = np.zeros((batch_size, max_seq_len, hidden_dim), dtype=np.float32)
            seq_lengths = []
            
            for i, hidden_states in enumerate(hidden_states_batch):
                if hidden_states.ndim == 2:
                    seq_len = hidden_states.shape[0]
                    batch_tensor[i, :seq_len, :] = hidden_states
                else:
                    seq_len = hidden_states.shape[1]
                    batch_tensor[i, :seq_len, :] = hidden_states[0]  # Take first batch element
                seq_lengths.append(seq_len)
            
            # Flatten for matrix multiply - process all at once
            batch_flat = batch_tensor.reshape(-1, hidden_dim)
            
            logger.debug(f"Batch GPU Attention: {batch_flat.shape} x {q_shape}")
            
            # Ensure input is float32
            if batch_flat.dtype != np.float32:
                batch_flat = batch_flat.astype(np.float32)
            
            # Compute Q, K, V for entire batch on GPU
            q_batch = self.vulkan_engine.compute_matrix_multiply_persistent(
                batch_flat, q_buffer_info, q_shape, flags=0)
            k_batch = self.vulkan_engine.compute_matrix_multiply_persistent(
                batch_flat, k_buffer_info, k_shape, flags=0)
            v_batch = self.vulkan_engine.compute_matrix_multiply_persistent(
                batch_flat, v_buffer_info, v_shape, flags=0)
            
            # Reshape and process attention for each sequence
            num_q_heads = 32
            num_kv_heads = 16
            q_head_dim = q_shape[0] // num_q_heads
            kv_head_dim = k_shape[0] // num_kv_heads
            
            q_batch = q_batch.reshape(batch_size, max_seq_len, num_q_heads, q_head_dim)
            k_batch = k_batch.reshape(batch_size, max_seq_len, num_kv_heads, kv_head_dim)
            v_batch = v_batch.reshape(batch_size, max_seq_len, num_kv_heads, kv_head_dim)
            
            # Process each sequence in the batch
            output_batch = []
            for i, seq_len in enumerate(seq_lengths):
                # Extract this sequence
                q = q_batch[i, :seq_len, :, :].transpose(1, 0, 2)  # [heads, seq, head_dim]
                k = k_batch[i, :seq_len, :, :].transpose(1, 0, 2)  # [heads, seq, head_dim]
                v = v_batch[i, :seq_len, :, :].transpose(1, 0, 2)  # [heads, seq, head_dim]
                
                # Expand KV heads for GQA
                k = np.repeat(k, num_q_heads // num_kv_heads, axis=0)
                v = np.repeat(v, num_q_heads // num_kv_heads, axis=0)
                
                # Scaled dot-product attention
                scale = 1.0 / np.sqrt(q_head_dim)
                scores = np.matmul(q, k.transpose(0, 2, 1)) * scale
                
                # Softmax
                exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
                attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
                
                # Apply attention
                attn_output = np.matmul(attn_weights, v)  # [heads, seq, head_dim]
                attn_output = attn_output.transpose(1, 0, 2).reshape(seq_len, -1)  # [seq, hidden]
                
                output_batch.append(attn_output)
            
            # Output projection for batch
            batch_outputs = []
            for i, attn_output in enumerate(output_batch):
                output = self.vulkan_engine.compute_matrix_multiply_persistent(
                    attn_output, o_buffer_info, o_shape, flags=0)
                batch_outputs.append(output)
            
            return batch_outputs, kv_cache_batch or [None] * len(hidden_states_batch)
            
        except Exception as e:
            logger.error(f"Batch GPU attention failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return hidden_states_batch, kv_cache_batch or [None] * len(hidden_states_batch)
    
    def compute_ffn_layer_gpu_batch(self, layer_idx: int, hidden_states_batch: List[np.ndarray]) -> List[np.ndarray]:
        """Compute FFN for a batch using GPU buffers directly"""
        
        gate_key = f'layer_{layer_idx}_language_model.model.layers.{layer_idx}.mlp.gate_proj.weight'
        up_key = f'layer_{layer_idx}_language_model.model.layers.{layer_idx}.mlp.up_proj.weight'
        down_key = f'layer_{layer_idx}_language_model.model.layers.{layer_idx}.mlp.down_proj.weight'
        
        if gate_key not in self.gpu_buffers:
            logger.warning(f"FFN weights not in GPU for layer {layer_idx}")
            return hidden_states_batch
        
        try:
            # Get GPU buffer handles and shapes
            gate_buffer_info, gate_shape = self._get_gpu_buffer_with_shape(gate_key)
            up_buffer_info, up_shape = self._get_gpu_buffer_with_shape(up_key)
            down_buffer_info, down_shape = self._get_gpu_buffer_with_shape(down_key)
            
            # Try fused FFN for each sequence
            batch_outputs = []
            for hidden_states in hidden_states_batch:
                if hasattr(self.vulkan_engine, 'compute_fused_ffn_persistent_weights'):
                    output = self.vulkan_engine.compute_fused_ffn_persistent_weights(
                        hidden_states,
                        gate_buffer_info, gate_shape,
                        up_buffer_info, up_shape,
                        down_buffer_info, down_shape,
                        flags=0
                    )
                    batch_outputs.append(output)
                else:
                    # Fallback to separate operations
                    seq_len, hidden_dim = hidden_states.shape
                    
                    # Gate and up projections
                    gate_output = self.vulkan_engine.compute_matrix_multiply_persistent(
                        hidden_states, gate_buffer_info, gate_shape, flags=0)
                    up_output = self.vulkan_engine.compute_matrix_multiply_persistent(
                        hidden_states, up_buffer_info, up_shape, flags=0)
                    
                    # SiLU activation and element-wise multiply
                    gate_activated = gate_output * (1.0 / (1.0 + np.exp(-gate_output)))
                    intermediate = gate_activated * up_output
                    
                    # Down projection
                    output = self.vulkan_engine.compute_matrix_multiply_persistent(
                        intermediate, down_buffer_info, down_shape, flags=0)
                    
                    batch_outputs.append(output)
            
            return batch_outputs
            
        except Exception as e:
            logger.error(f"Batch GPU FFN failed: {e}")
            return hidden_states_batch
    
    def forward_layer_batch(self, layer_idx: int, hidden_states_batch: List[np.ndarray],
                           position_ids_batch: Optional[List[np.ndarray]] = None,
                           kv_cache_batch: Optional[List[Tuple]] = None) -> Tuple[List[np.ndarray], List[Tuple]]:
        """Forward pass through a layer using GPU compute for batch"""
        
        if layer_idx not in self.layer_weights_gpu:
            logger.warning(f"Layer {layer_idx} not in GPU")
            return hidden_states_batch, kv_cache_batch or [None] * len(hidden_states_batch)
        
        # Residual connections
        residual_batch = [hs.copy() for hs in hidden_states_batch]
        
        # Layer norm
        normalized_batch = [self._layer_norm(hs) for hs in hidden_states_batch]
        
        # Attention on GPU (batch)
        attention_outputs, kv_cache_batch = self.compute_attention_layer_gpu_batch(
            layer_idx, normalized_batch, kv_cache_batch)
        hidden_states_batch = [res + attn for res, attn in zip(residual_batch, attention_outputs)]
        
        # Post-attention residual
        residual_batch = [hs.copy() for hs in hidden_states_batch]
        
        # Post-attention layer norm
        normalized_batch = [self._layer_norm(hs) for hs in hidden_states_batch]
        
        # FFN on GPU (batch)
        ffn_outputs = self.compute_ffn_layer_gpu_batch(layer_idx, normalized_batch)
        hidden_states_batch = [res + ffn for res, ffn in zip(residual_batch, ffn_outputs)]
        
        return hidden_states_batch, kv_cache_batch
    
    def benchmark_batch_performance(self, batch_sizes: List[int] = [1, 2, 4, 8], 
                                   layer_idx: int = 0, num_iterations: int = 20):
        """Benchmark batch performance"""
        results = {}
        
        for batch_size in batch_sizes:
            logger.info(f"\n📊 Benchmarking batch_size={batch_size}...")
            
            # Create batch input
            test_inputs = [np.random.randn(1, 5376).astype(np.float32) for _ in range(batch_size)]
            
            # Warm up
            for _ in range(5):
                outputs, _ = self.forward_layer_batch(layer_idx, test_inputs)
            
            # Benchmark
            times = []
            for _ in range(num_iterations):
                start = time.perf_counter()
                outputs, _ = self.forward_layer_batch(layer_idx, test_inputs)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            
            avg_time = np.mean(times[3:])  # Skip first few
            tokens_per_second = batch_size / avg_time
            
            results[batch_size] = {
                'avg_time': avg_time,
                'tps': tokens_per_second,
                'time_per_token': avg_time / batch_size
            }
            
            logger.info(f"   Batch {batch_size}: {avg_time*1000:.1f}ms total, {tokens_per_second:.1f} TPS")
        
        return results
    
    def estimate_full_model_batch_performance(self, batch_size: int = 4):
        """Estimate full model performance with batch processing"""
        logger.info(f"\n🚀 Estimating full model performance with batch_size={batch_size}")
        
        # Benchmark single layer with batch
        layer_time = self.benchmark_batch_performance([batch_size], num_iterations=10)[batch_size]['avg_time']
        
        # Estimate full model
        full_model_time = layer_time * 62  # 62 layers
        batch_tps = batch_size / full_model_time
        
        logger.info(f"\n📊 Batch Performance Estimate:")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Layer time: {layer_time*1000:.1f}ms")
        logger.info(f"   Full model time: {full_model_time*1000:.1f}ms")
        logger.info(f"   Batch TPS: {batch_tps:.1f}")
        
        # Compare to single token
        single_tps = 1.0 / (layer_time * 62)
        speedup = batch_tps / 8.5  # Compare to current 8.5 TPS
        
        logger.info(f"   Single token TPS: {single_tps:.1f}")
        logger.info(f"   Speedup vs baseline: {speedup:.1f}x")
        
        if batch_tps >= 81:
            logger.info(f"   ✅ TARGET ACHIEVED with batch_size={batch_size}!")
        else:
            needed_batch = int(np.ceil(81 * full_model_time))
            logger.info(f"   📈 Need batch_size={needed_batch} for 81 TPS target")
        
        return batch_tps


def test_batch_optimization():
    """Test batch processing optimization"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    logger.info("🚀 Testing Batch Processing Optimization")
    
    # Initialize
    pipeline = OptimizedBatchPipeline(max_batch_size=8)
    model_path = "quantized_models/gemma-3-27b-it-layer-by-layer"
    
    logger.info("Loading model...")
    start = time.time()
    
    if not pipeline.initialize(model_path):
        logger.error("Failed to initialize")
        return
    
    load_time = time.time() - start
    logger.info(f"✅ Model loaded in {load_time:.1f}s")
    
    # Benchmark different batch sizes
    results = pipeline.benchmark_batch_performance()
    
    # Show results
    logger.info(f"\n📊 Batch Performance Results:")
    for batch_size, metrics in results.items():
        logger.info(f"   Batch {batch_size}: {metrics['tps']:.1f} TPS ({metrics['avg_time']*1000:.1f}ms)")
    
    # Estimate full model with optimal batch size
    best_batch_size = max(results.keys(), key=lambda k: results[k]['tps'])
    pipeline.estimate_full_model_batch_performance(best_batch_size)
    
    # Cleanup
    pipeline.cleanup()
    
    return results


if __name__ == "__main__":
    test_batch_optimization()