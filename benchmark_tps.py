#!/usr/bin/env python3
"""
Benchmark TPS (Tokens Per Second) for the INT8 GPU Pipeline
Target: 81 TPS
"""

import time
import logging
import numpy as np
from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def benchmark_single_token(pipeline, num_tokens=100):
    """Benchmark single token generation"""
    logger.info(f"🔄 Benchmarking {num_tokens} tokens...")
    
    # Use a simple input
    input_ids = [1, 2, 3]  # Start tokens
    
    # Get embedding
    embed_weight = pipeline.get_weight_from_gpu('shared_language_model.model.embed_tokens.weight')
    if embed_weight is None:
        logger.error("Failed to get embeddings")
        return 0
    
    # Create initial hidden states
    hidden_states = embed_weight[input_ids]
    if hidden_states.ndim == 2:
        hidden_states = hidden_states[np.newaxis, :]
    
    # Initialize KV cache
    kv_cache = None
    
    # Warm up (first token is slower)
    logger.info("⏱️ Warming up...")
    for _ in range(5):
        output, kv_cache = pipeline.forward_layer(0, hidden_states, kv_cache)
    
    # Benchmark
    logger.info("⏱️ Starting benchmark...")
    token_times = []
    
    for i in range(num_tokens):
        start = time.perf_counter()
        
        # Process through all layers
        current_hidden = hidden_states
        for layer_idx in range(62):  # All 62 layers
            current_hidden, kv_cache = pipeline.forward_layer(layer_idx, current_hidden, kv_cache)
        
        # Time for one complete forward pass
        token_time = time.perf_counter() - start
        token_times.append(token_time)
        
        if i % 10 == 0:
            current_tps = 1.0 / token_time
            logger.info(f"   Token {i}: {token_time*1000:.1f}ms ({current_tps:.1f} TPS)")
    
    # Calculate statistics
    token_times = token_times[5:]  # Skip first few for stability
    avg_time = np.mean(token_times)
    std_time = np.std(token_times)
    min_time = np.min(token_times)
    max_time = np.max(token_times)
    
    avg_tps = 1.0 / avg_time
    max_tps = 1.0 / min_time
    
    return {
        'avg_tps': avg_tps,
        'max_tps': max_tps,
        'avg_time_ms': avg_time * 1000,
        'std_time_ms': std_time * 1000,
        'min_time_ms': min_time * 1000,
        'max_time_ms': max_time * 1000
    }

def benchmark_batch(pipeline, batch_sizes=[1, 4, 8, 16]):
    """Benchmark different batch sizes"""
    results = {}
    
    for batch_size in batch_sizes:
        logger.info(f"\n📊 Testing batch size: {batch_size}")
        
        # Create batch input
        input_ids = [[1, 2, 3] for _ in range(batch_size)]
        
        # Get embeddings
        embed_weight = pipeline.get_weight_from_gpu('shared_language_model.model.embed_tokens.weight')
        if embed_weight is None:
            continue
        
        # Create batch hidden states
        hidden_states = []
        for ids in input_ids:
            h = embed_weight[ids]
            hidden_states.append(h)
        hidden_states = np.stack(hidden_states)  # [batch, seq, hidden]
        
        # Warm up
        kv_cache = None
        for _ in range(3):
            output, kv_cache = pipeline.forward_layer(0, hidden_states, kv_cache)
        
        # Benchmark
        start = time.perf_counter()
        num_tokens = 50
        
        for _ in range(num_tokens):
            current_hidden = hidden_states
            for layer_idx in range(62):
                current_hidden, kv_cache = pipeline.forward_layer(layer_idx, current_hidden, kv_cache)
        
        total_time = time.perf_counter() - start
        tokens_generated = num_tokens * batch_size
        batch_tps = tokens_generated / total_time
        
        results[batch_size] = {
            'tps': batch_tps,
            'time_per_token_ms': (total_time / tokens_generated) * 1000
        }
        
        logger.info(f"   Batch TPS: {batch_tps:.1f}")
        logger.info(f"   Time per token: {results[batch_size]['time_per_token_ms']:.1f}ms")
    
    return results

def main():
    """Run TPS benchmark"""
    logger.info("🚀 TPS Benchmark for INT8 GPU Pipeline")
    logger.info("   Target: 81 TPS")
    logger.info("   Hardware: AMD Radeon 780M + Phoenix NPU")
    
    # Initialize pipeline
    pipeline = PureHardwarePipelineFixed()
    model_path = "quantized_models/gemma-3-27b-it-layer-by-layer"
    
    logger.info("\n🔄 Loading model...")
    start_load = time.time()
    
    if not pipeline.initialize(model_path):
        logger.error("Failed to initialize pipeline")
        return
    
    load_time = time.time() - start_load
    logger.info(f"✅ Model loaded in {load_time:.1f}s")
    
    # Single token benchmark
    logger.info("\n📊 Single Token Benchmark")
    single_results = benchmark_single_token(pipeline, num_tokens=100)
    
    logger.info("\n📊 Results:")
    logger.info(f"   Average TPS: {single_results['avg_tps']:.1f}")
    logger.info(f"   Maximum TPS: {single_results['max_tps']:.1f}")
    logger.info(f"   Avg latency: {single_results['avg_time_ms']:.1f}ms ± {single_results['std_time_ms']:.1f}ms")
    logger.info(f"   Min latency: {single_results['min_time_ms']:.1f}ms")
    logger.info(f"   Max latency: {single_results['max_time_ms']:.1f}ms")
    
    # Compare to target
    if single_results['avg_tps'] >= 81:
        logger.info("✅ TARGET ACHIEVED! 81+ TPS!")
    else:
        shortfall = 81 - single_results['avg_tps']
        logger.info(f"❌ Below target by {shortfall:.1f} TPS")
        logger.info(f"   Need {shortfall/single_results['avg_tps']*100:.0f}% speedup")
    
    # Batch benchmark
    logger.info("\n📊 Batch Processing Benchmark")
    batch_results = benchmark_batch(pipeline, batch_sizes=[1, 4, 8])
    
    logger.info("\n📊 Batch Results Summary:")
    for batch_size, results in batch_results.items():
        logger.info(f"   Batch {batch_size}: {results['tps']:.1f} TPS")
    
    # Find optimal batch size for 81 TPS
    for batch_size, results in batch_results.items():
        if results['tps'] >= 81:
            logger.info(f"\n✅ Batch size {batch_size} achieves {results['tps']:.1f} TPS!")
            break
    
    # Cleanup
    pipeline.cleanup()

if __name__ == "__main__":
    main()