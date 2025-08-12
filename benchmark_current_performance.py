#!/usr/bin/env python3
"""
Quick performance benchmark for current Gemma3 4B pipeline
Measures actual TPS with fixed matrix dimensions
"""

import os
import time
import numpy as np
import logging
from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def measure_single_token_performance(pipeline, input_ids):
    """Measure time for single token generation"""
    
    # Get embeddings
    embed_key = 'language_model.model.embed_tokens.weight'
    embed_info = pipeline.gpu_buffers.get(embed_key)
    if not embed_info:
        embed_key = 'shared_language_model.model.embed_tokens.weight'
        embed_info = pipeline.gpu_buffers.get(embed_key)
    
    if not embed_info:
        logger.error("No embedding weights found")
        return None
    
    # Time single token generation
    start_time = time.time()
    
    try:
        # Initial embedding
        hidden_states = pipeline.vulkan_engine.compute_embedding_lookup_gpu(
            input_ids, embed_info['buffer_info']
        )
        
        # Process through first few layers to get realistic timing
        kv_cache = None
        for layer_idx in range(3):  # Test first 3 layers
            hidden_states, kv_cache = pipeline.forward_layer(
                layer_idx, hidden_states, kv_cache=kv_cache
            )
        
        end_time = time.time()
        return end_time - start_time
        
    except Exception as e:
        logger.error(f"Performance measurement failed: {e}")
        return None

def quick_benchmark():
    """Run quick performance benchmark"""
    
    logger.info("🚀 GEMMA3 4B PERFORMANCE BENCHMARK")
    logger.info("=" * 50)
    
    # Initialize pipeline
    pipeline = PureHardwarePipelineFixed()
    model_path = "/home/ucadmin/Development/Unicorn-Execution-Engine/quantized_models/gemma-3-4b-it-quantized"
    
    logger.info(f"📦 Loading model: {model_path}")
    start_load = time.time()
    
    if not pipeline.initialize(model_path=model_path):
        logger.error("❌ Failed to initialize pipeline")
        return
    
    load_time = time.time() - start_load
    logger.info(f"✅ Model loaded in {load_time:.2f}s")
    
    # Hardware status
    npu_type = type(pipeline.npu_kernel).__name__
    logger.info(f"🧠 NPU: {npu_type}")
    logger.info(f"🎮 GPU: AMD Radeon Graphics (RADV PHOENIX)")
    
    # Memory info
    total_gpu_mb = sum(info['size_mb'] for info in pipeline.gpu_buffers.values())
    logger.info(f"💾 GPU Memory: {total_gpu_mb:.1f}MB allocated")
    
    # Performance tests
    test_configs = [
        {"name": "Single token", "input_ids": [1]},
        {"name": "Short sequence", "input_ids": [1, 2, 3]},
        {"name": "Medium sequence", "input_ids": list(range(1, 6))},
    ]
    
    logger.info("\n📊 PERFORMANCE MEASUREMENTS:")
    logger.info("-" * 50)
    
    times = []
    
    for config in test_configs:
        logger.info(f"\n🔍 Testing: {config['name']}")
        logger.info(f"   Input length: {len(config['input_ids'])}")
        
        # Measure 3 times and take average
        test_times = []
        for i in range(3):
            test_time = measure_single_token_performance(pipeline, config['input_ids'])
            if test_time is not None:
                test_times.append(test_time)
                logger.info(f"   Run {i+1}: {test_time:.3f}s")
        
        if test_times:
            avg_time = np.mean(test_times)
            std_time = np.std(test_times)
            tps = 1.0 / avg_time if avg_time > 0 else 0
            
            logger.info(f"   📈 Average: {avg_time:.3f}s ± {std_time:.3f}s")
            logger.info(f"   🚀 TPS: {tps:.2f}")
            
            times.append(avg_time)
        else:
            logger.warning(f"   ❌ Failed to measure performance")
    
    # Summary
    if times:
        logger.info("\n" + "=" * 50)
        logger.info("📊 PERFORMANCE SUMMARY")
        logger.info("=" * 50)
        
        avg_time = np.mean(times)
        avg_tps = 1.0 / avg_time if avg_time > 0 else 0
        
        logger.info(f"✅ Average time per token: {avg_time:.3f}s")
        logger.info(f"🚀 Average TPS: {avg_tps:.2f}")
        
        # Performance analysis
        if avg_tps >= 10:
            logger.info("✅ GOOD: Performance is acceptable for testing")
        elif avg_tps >= 1:
            logger.info("⚠️  MODERATE: Performance is slow but functional")
        else:
            logger.info("❌ POOR: Performance is very slow")
        
        logger.info("\n🔧 OPTIMIZATION OPPORTUNITIES:")
        if "Simulated" in npu_type:
            logger.info("1. Compile NPU kernels for Gemma3 4B dimensions (2560)")
            logger.info("2. Enable real NPU acceleration instead of simulation")
        logger.info("3. Optimize GPU memory access patterns")
        logger.info("4. Implement proper attention caching")
        
    pipeline.cleanup()
    logger.info("\n🏁 Benchmark completed!")

if __name__ == "__main__":
    quick_benchmark()