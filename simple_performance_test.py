#!/usr/bin/env python3
"""
Simple Performance Test - Focus on Working Vulkan FFN
Measure actual tokens per second with real iGPU acceleration
"""

import torch
import numpy as np
import logging
import time
from vulkan_ffn_compute_engine import VulkanFFNComputeEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_vulkan_ffn_performance():
    """Test Vulkan FFN performance for tokens per second calculation"""
    
    logger.info("🚀 Testing Vulkan FFN Performance for Tokens/Second")
    
    # Initialize Vulkan FFN engine
    ffn_engine = VulkanFFNComputeEngine()
    if not ffn_engine.initialize():
        logger.error("❌ Vulkan FFN engine initialization failed")
        return
    
    # Gemma 3 27B dimensions
    hidden_size = 4096
    intermediate_size = 16384
    
    # Test different sequence lengths and batch sizes
    test_configs = [
        {"batch_size": 1, "seq_len": 1, "name": "Single token"},
        {"batch_size": 1, "seq_len": 32, "name": "32 tokens"},
        {"batch_size": 1, "seq_len": 64, "name": "64 tokens"},
        {"batch_size": 1, "seq_len": 128, "name": "128 tokens"},
        {"batch_size": 1, "seq_len": 256, "name": "256 tokens"},
    ]
    
    results = []
    
    for config in test_configs:
        batch_size = config["batch_size"]
        seq_len = config["seq_len"]
        name = config["name"]
        
        logger.info(f"\n🔬 Testing {name}: {batch_size}x{seq_len}x{hidden_size}")
        
        # Create test tensors
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        gate_proj_weight = torch.randn(intermediate_size, hidden_size)
        up_proj_weight = torch.randn(intermediate_size, hidden_size)
        down_proj_weight = torch.randn(hidden_size, intermediate_size)

        # Pre-load weights to the GPU
        ffn_engine.load_weights(gate_proj_weight, up_proj_weight, down_proj_weight)
        
        # Warmup runs
        logger.info("   🔥 Warmup runs...")
        for _ in range(2):
            _ = ffn_engine.compute_ffn_layer(hidden_states)
        
        # Benchmark runs
        logger.info("   📊 Benchmark runs...")
        times = []
        
        for run in range(5):
            start_time = time.time()
            result = ffn_engine.compute_ffn_layer(hidden_states)
            end_time = time.time()
            
            run_time = end_time - start_time
            times.append(run_time)
            
            logger.info(f"      Run {run+1}: {run_time*1000:.2f}ms")
        
        # Calculate statistics
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        # Calculate tokens per second
        total_tokens = batch_size * seq_len
        tokens_per_second = total_tokens / avg_time
        
        result_data = {
            "config": name,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "total_tokens": total_tokens,
            "avg_time_ms": avg_time * 1000,
            "min_time_ms": min_time * 1000,
            "max_time_ms": max_time * 1000,
            "tokens_per_second": tokens_per_second
        }
        
        results.append(result_data)
        
        logger.info(f"   ✅ {name}: {tokens_per_second:.2f} tokens/sec")
        logger.info(f"      Average time: {avg_time*1000:.2f}ms")
        logger.info(f"      Range: {min_time*1000:.2f}ms - {max_time*1000:.2f}ms")
    
    # Summary
    logger.info("\n🎉 VULKAN FFN PERFORMANCE SUMMARY:")
    logger.info("=" * 60)
    
    for result in results:
        logger.info(f"{result['config']:15}: {result['tokens_per_second']:6.2f} tokens/sec ({result['avg_time_ms']:6.1f}ms)")
    
    # Calculate performance metrics for single token generation
    single_token_result = results[0]  # Single token test
    logger.info(f"\n🎯 SINGLE TOKEN GENERATION:")
    logger.info(f"   FFN Processing Time: {single_token_result['avg_time_ms']:.2f}ms")
    logger.info(f"   Effective Rate: {single_token_result['tokens_per_second']:.2f} tokens/sec")
    
    # Estimate full model performance
    # Assuming: FFN takes ~70% of total layer time, 62 layers total
    ffn_time_per_token = single_token_result['avg_time_ms']
    estimated_layer_time = ffn_time_per_token / 0.7  # FFN is ~70% of layer time
    estimated_full_model_time = estimated_layer_time * 62  # 62 layers
    estimated_tokens_per_sec = 1000 / estimated_full_model_time  # Convert ms to seconds
    
    logger.info(f"\n📊 ESTIMATED FULL MODEL PERFORMANCE:")
    logger.info(f"   FFN time per token: {ffn_time_per_token:.1f}ms")
    logger.info(f"   Estimated layer time: {estimated_layer_time:.1f}ms")
    logger.info(f"   Estimated full model time: {estimated_full_model_time:.1f}ms")
    logger.info(f"   Estimated tokens/sec: {estimated_tokens_per_sec:.2f}")
    
    # Get Vulkan performance stats
    stats = ffn_engine.get_performance_stats()
    logger.info(f"\n🎮 VULKAN HARDWARE PERFORMANCE:")
    logger.info(f"   Total FFN operations: {stats['total_ffn_operations']}")
    logger.info(f"   Average FFN time: {stats['avg_ffn_time_ms']:.1f}ms")
    logger.info(f"   Min FFN time: {stats['min_ffn_time_ms']:.1f}ms")
    logger.info(f"   Max FFN time: {stats['max_ffn_time_ms']:.1f}ms")

def estimate_layer_breakdown(vulkan_ffn_time_ms):
    """Estimate performance breakdown for a complete transformer layer"""
    
    logger.info("\n🔍 TRANSFORMER LAYER PERFORMANCE BREAKDOWN:")
    logger.info("=" * 50)
    
    # Based on typical transformer architectures
    # FFN: ~60-70% of computation time
    # Attention: ~25-35% of computation time  
    # LayerNorm/Residuals: ~5-10% of computation time
    
    # Our measured Vulkan FFN time (from previous tests)
    #vulkan_ffn_time_ms = 2600  # Average from our benchmark
    
    # Estimate other components
    attention_time_ms = vulkan_ffn_time_ms * 0.4  # Attention is ~40% of FFN time
    layernorm_time_ms = vulkan_ffn_time_ms * 0.05  # LayerNorm is ~5% of FFN time

    
    total_layer_time_ms = vulkan_ffn_time_ms + attention_time_ms + layernorm_time_ms
    
    logger.info(f"   FFN (Vulkan iGPU):  {vulkan_ffn_time_ms:6.0f}ms ({vulkan_ffn_time_ms/total_layer_time_ms*100:.1f}%)")
    logger.info(f"   Attention (NPU):    {attention_time_ms:6.0f}ms ({attention_time_ms/total_layer_time_ms*100:.1f}%)")
    logger.info(f"   LayerNorm (CPU):    {layernorm_time_ms:6.0f}ms ({layernorm_time_ms/total_layer_time_ms*100:.1f}%)")
    logger.info(f"   Total per layer:    {total_layer_time_ms:6.0f}ms")
    
    # Full model estimation (62 layers)
    total_model_time_ms = total_layer_time_ms * 62
    tokens_per_second = 1000 / total_model_time_ms
    
    logger.info(f"\n📊 FULL GEMMA 3 27B ESTIMATION:")
    logger.info(f"   Layers: 62")
    logger.info(f"   Time per layer: {total_layer_time_ms/1000:.2f}s")
    logger.info(f"   Total model time: {total_model_time_ms/1000:.1f}s")
    logger.info(f"   🎯 ESTIMATED TOKENS/SEC: {tokens_per_second:.3f}")
    
    # Convert to more readable format
    if tokens_per_second < 1:
        logger.info(f"   🎯 ESTIMATED PERFORMANCE: {1/tokens_per_second:.1f} seconds per token")
    else:
        logger.info(f"   🎯 ESTIMATED PERFORMANCE: {tokens_per_second:.2f} tokens per second")

if __name__ == "__main__":
    results = test_vulkan_ffn_performance()
    if results:
        single_token_ffn_time = results[0]['avg_time_ms']
        estimate_layer_breakdown(single_token_ffn_time)