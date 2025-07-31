#!/usr/bin/env python3
"""
Test REAL NPU+iGPU Performance with Gemma3n
No simulation - actual hardware only
"""

import os
import sys
import time
import numpy as np
import logging

# Add the necessary imports
from gemma3n_e4b_npu_acceleration import NPUPhoenixAccelerator
from gemma3n_e4b_vulkan_acceleration import VulkanRadeonAccelerator as VulkanAccelerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_real_hardware_performance():
    """Test actual NPU+iGPU performance with real data"""
    
    logger.info("🚀 REAL HARDWARE PERFORMANCE TEST - NO SIMULATION")
    logger.info("=" * 60)
    
    # Initialize NPU
    logger.info("⚡ Initializing NPU Phoenix...")
    npu = NPUPhoenixAccelerator()
    
    if not npu.npu_available:
        logger.error("❌ NPU not available - cannot proceed with real test")
        return
    
    # Initialize iGPU (Vulkan)
    logger.info("🎮 Initializing AMD Radeon iGPU...")
    gpu = VulkanAccelerator()
    
    if not gpu.vulkan_available:
        logger.error("❌ iGPU not available - cannot proceed with real test")
        return
    
    logger.info("✅ Both NPU and iGPU initialized successfully!")
    
    # Test parameters (Gemma3n E4B dimensions)
    batch_size = 1
    seq_length = 512
    hidden_size = 3072
    num_heads = 24
    head_dim = 128
    
    logger.info(f"\n📊 Test Configuration:")
    logger.info(f"  Model: Gemma3n E4B")
    logger.info(f"  Batch Size: {batch_size}")
    logger.info(f"  Sequence Length: {seq_length}")
    logger.info(f"  Hidden Size: {hidden_size}")
    logger.info(f"  Attention Heads: {num_heads}")
    
    # Create real test data
    logger.info("\n🔄 Creating test data...")
    input_data = np.random.randn(batch_size, seq_length, hidden_size).astype(np.float32)
    
    # Warm-up run
    logger.info("\n🔥 Warming up hardware...")
    try:
        import torch
        input_tensor = torch.from_numpy(input_data)
        _ = npu.run_attention(input_tensor)
    except:
        logger.warning("⚠️ Warm-up failed, continuing anyway")
    
    # Benchmark NPU attention
    logger.info("\n⚡ Testing NPU Attention Performance...")
    num_iterations = 10
    npu_times = []
    
    for i in range(num_iterations):
        start_time = time.time()
        try:
            # Run real NPU attention
            output = npu.run_attention(torch.from_numpy(input_data))
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.time() - start_time
            npu_times.append(elapsed)
            logger.info(f"  Iteration {i+1}: {elapsed*1000:.2f}ms")
        except Exception as e:
            logger.error(f"  NPU execution failed: {e}")
            break
    
    if npu_times:
        avg_npu_time = np.mean(npu_times) * 1000  # Convert to ms
        logger.info(f"\n✅ NPU Attention Average: {avg_npu_time:.2f}ms")
        
        # Calculate tokens per second for attention
        # For autoregressive generation, each token requires one attention computation
        attention_tps = 1000 / avg_npu_time
        logger.info(f"🚀 NPU Attention TPS: {attention_tps:.2f} tokens/second")
    
    # Test iGPU FFN performance
    logger.info("\n🎮 Testing iGPU FFN Performance...")
    intermediate_size = 8192  # Gemma3n E4B intermediate size
    
    # Create FFN weight matrices
    gate_weight = np.random.randn(hidden_size, intermediate_size).astype(np.float32)
    up_weight = np.random.randn(hidden_size, intermediate_size).astype(np.float32)
    down_weight = np.random.randn(intermediate_size, hidden_size).astype(np.float32)
    
    gpu_times = []
    for i in range(num_iterations):
        start_time = time.time()
        try:
            # Run real GPU FFN
            output = gpu.compute_ffn(input_data[0], gate_weight, up_weight, down_weight)
            elapsed = time.time() - start_time
            gpu_times.append(elapsed)
            logger.info(f"  Iteration {i+1}: {elapsed*1000:.2f}ms")
        except Exception as e:
            logger.error(f"  GPU execution failed: {e}")
            break
    
    if gpu_times:
        avg_gpu_time = np.mean(gpu_times) * 1000  # Convert to ms
        logger.info(f"\n✅ iGPU FFN Average: {avg_gpu_time:.2f}ms")
        
        # Calculate tokens per second for FFN
        ffn_tps = 1000 / avg_gpu_time
        logger.info(f"🚀 iGPU FFN TPS: {ffn_tps:.2f} tokens/second")
    
    # Calculate combined performance
    if npu_times and gpu_times:
        # In a transformer, we have attention + FFN per layer
        # Gemma3n E4B has 18 layers
        num_layers = 18
        total_time_per_token = (avg_npu_time + avg_gpu_time) * num_layers
        combined_tps = 1000 / total_time_per_token
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 REAL HARDWARE PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"✅ NPU (Attention): {avg_npu_time:.2f}ms per layer")
        logger.info(f"✅ iGPU (FFN): {avg_gpu_time:.2f}ms per layer")
        logger.info(f"✅ Total per token: {total_time_per_token:.2f}ms ({num_layers} layers)")
        logger.info(f"🚀 REAL TPS: {combined_tps:.2f} tokens/second")
        logger.info("=" * 60)
        
        # Performance analysis
        if combined_tps >= 150:
            logger.info("🎉 EXCELLENT! Achieved target of 150+ TPS!")
        elif combined_tps >= 100:
            logger.info("✅ Good performance! Over 100 TPS")
        elif combined_tps >= 50:
            logger.info("📈 Decent performance. Room for optimization")
        else:
            logger.info("⚠️ Performance below expectations")
    
    # Show hardware utilization
    logger.info("\n📊 Hardware Status:")
    npu_status = npu.get_status()
    logger.info(f"  NPU: {npu_status}")
    
    return combined_tps if (npu_times and gpu_times) else 0

if __name__ == "__main__":
    try:
        tps = test_real_hardware_performance()
        logger.info(f"\n✅ Test completed successfully! Real TPS: {tps:.2f}")
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()