#!/usr/bin/env python3
"""
Optimized benchmark with chunked logits computation to avoid GPU crashes
"""

import fix_vulkan_imports

import numpy as np
import time
import logging
from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedPipeline(PureHardwarePipelineFixed):
    """Pipeline with optimized logits computation"""
    
    def generate_tokens_optimized(self, input_ids, max_tokens=10, temperature=1.0, top_p=0.9):
        """Generate tokens with chunked logits computation"""
        if not self.initialized:
            raise RuntimeError("Pipeline not initialized")
            
        generated_ids = []
        hidden_states = None
        kv_cache = [None] * 34  # 34 layers for Gemma3 4B
        
        # Get embedding weights info
        embed_key = 'shared_language_model.model.embed_tokens.weight'
        embed_info = self.gpu_buffers.get(embed_key)
        if not embed_info:
            raise RuntimeError("Embedding weights not found")
            
        # Initial embedding lookup
        hidden_states = self.vulkan_engine.compute_embedding_lookup_gpu(
            input_ids, embed_info['buffer_info']
        )
        
        for i in range(max_tokens):
            try:
                # Forward through all layers
                for layer_idx in range(34):
                    hidden_states, kv_cache[layer_idx] = self.forward_layer(
                        layer_idx, hidden_states, kv_cache=kv_cache[layer_idx]
                    )
                
                # Apply final layer norm
                hidden_states = self._layer_norm(hidden_states, layer_idx=33, is_final=True)
                
                # Compute logits in chunks to avoid GPU memory issues
                batch_size, seq_len, hidden_dim = hidden_states.shape
                hidden_flat = hidden_states.reshape(-1, hidden_dim)
                
                # Read embedding weights back from GPU (temporary)
                embed_buffer, embed_memory, embed_size = embed_info['buffer_info']
                embed_data = self.vulkan_engine._read_buffer(embed_buffer, embed_memory, embed_size)
                embed_weights = np.frombuffer(embed_data, dtype=np.float32).reshape(embed_info['shape'])
                
                # Chunked logits computation
                vocab_size = embed_weights.shape[0]
                chunk_size = 16384  # Process vocabulary in chunks
                logits = np.zeros((batch_size * seq_len, vocab_size), dtype=np.float32)
                
                for start_idx in range(0, vocab_size, chunk_size):
                    end_idx = min(start_idx + chunk_size, vocab_size)
                    embed_chunk = embed_weights[start_idx:end_idx].T  # (hidden_dim, chunk_size)
                    
                    # Compute logits for this chunk
                    logits_chunk = np.dot(hidden_flat, embed_chunk)
                    logits[:, start_idx:end_idx] = logits_chunk
                
                logits = logits.reshape(batch_size, seq_len, -1)
                
                # Get logits for the last token
                last_token_logits = logits[:, -1, :]
                
                # Simple greedy sampling for stability
                next_token_id = np.argmax(last_token_logits, axis=-1)
                generated_ids.append(next_token_id.item())
                
                # Update hidden_states for next iteration
                next_token_embedding = self.vulkan_engine.compute_embedding_lookup_gpu(
                    [next_token_id.item()], embed_info['buffer_info']
                )
                hidden_states = next_token_embedding
                
            except Exception as e:
                logger.error(f"Error at token {i+1}: {e}")
                break
                
        return generated_ids

def run_optimized_benchmark():
    """Run optimized benchmark"""
    logger.info("🚀 Starting Optimized Gemma-3-4B Benchmark...")
    logger.info("=" * 70)
    
    pipeline = OptimizedPipeline()
    model_path = "/home/ucadmin/Development/Unicorn-Execution-Engine/quantized_models/gemma-3-4b-it-quantized"
    
    if not pipeline.initialize(model_path=model_path):
        logger.error("❌ Failed to initialize pipeline")
        return
        
    # Check NPU status
    if pipeline.npu_kernel:
        logger.info(f"✅ NPU Acceleration: {type(pipeline.npu_kernel).__name__}")
    else:
        logger.warning("⚠️ No NPU acceleration available")
    
    # Warmup
    logger.info("🔥 Warming up...")
    _ = pipeline.generate_tokens_optimized([1, 2, 3], max_tokens=1)
    
    # Benchmark
    num_tokens = 50
    logger.info(f"📊 Generating {num_tokens} tokens...")
    
    start_time = time.time()
    generated = pipeline.generate_tokens_optimized([1, 2, 3, 4, 5], max_tokens=num_tokens)
    elapsed = time.time() - start_time
    
    tps = len(generated) / elapsed if elapsed > 0 else 0
    
    # NPU metrics
    if pipeline.npu_total_layers > 0:
        npu_avg = (pipeline.npu_total_time / pipeline.npu_total_layers) * 1000
        npu_contribution = pipeline.npu_total_time / elapsed * 100
    else:
        npu_avg = 0
        npu_contribution = 0
    
    logger.info("=" * 70)
    logger.info("📊 BENCHMARK RESULTS 📊")
    logger.info("=" * 70)
    logger.info(f"✅ Generated {len(generated)} tokens in {elapsed:.2f}s")
    logger.info(f"🚀 Overall TPS: {tps:.2f} tokens/second")
    logger.info(f"🧠 NPU Average: {npu_avg:.2f}ms per attention layer")
    logger.info(f"📈 NPU Contribution: {npu_contribution:.1f}% of compute time")
    logger.info("=" * 70)
    
    # Analysis
    if tps >= 150:
        logger.info("🎉 SUCCESS! Achieved 150+ TPS target!")
    elif tps >= 100:
        logger.info("✅ Good performance! Over 100 TPS")
    elif tps >= 50:
        logger.info("📈 Decent performance. Room for optimization")
    else:
        logger.info("⚠️ Performance below expectations")
        
    # Cleanup
    pipeline.cleanup()

if __name__ == "__main__":
    run_optimized_benchmark()