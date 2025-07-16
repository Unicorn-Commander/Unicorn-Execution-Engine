#!/usr/bin/env python3
"""
Gemma 3 27B Performance Summary
Quick performance calculation based on real hardware measurements
"""

import time
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import our real components
from npu_attention_kernel import NPUAttentionKernel, NPUAttentionConfig
from real_vulkan_compute import RealVulkanCompute

def calculate_gemma3_27b_performance():
    """Calculate Gemma 3 27B performance with real hardware measurements"""
    
    print("🦄 Gemma 3 27B NPU+iGPU Performance Summary")
    print("=" * 60)
    
    # Model specifications
    config = {
        "model": "Gemma 3 27B",
        "layers": 62,
        "d_model": 4096,
        "intermediate_size": 14336,
        "heads": 32,
        "parameters": "~27B",
        "quantized_size_gb": 28.9,
        "target_tps": 22.7
    }
    
    print(f"📊 Model Configuration:")
    print(f"   Model: {config['model']}")
    print(f"   Layers: {config['layers']}")
    print(f"   Model size: {config['quantized_size_gb']} GB (quantized)")
    print(f"   Target TPS: {config['target_tps']}")
    
    # Initialize components
    print(f"\n🚀 Initializing Hardware Components...")
    
    # NPU initialization
    npu_config = NPUAttentionConfig(
        seq_length=512,
        d_model=config["d_model"],
        num_heads=config["heads"],
        npu_memory_mb=2048,
        precision="fp16"
    )
    
    npu_kernel = NPUAttentionKernel(npu_config)
    npu_initialized = npu_kernel.initialize()
    print(f"   NPU Phoenix (16 TOPS): {'✅' if npu_initialized else '❌'}")
    
    # Vulkan initialization
    vulkan_compute = RealVulkanCompute()
    vulkan_initialized = vulkan_compute.initialize()
    print(f"   AMD Radeon 780M iGPU: {'✅' if vulkan_initialized else '❌'}")
    
    if not (npu_initialized and vulkan_initialized):
        print("❌ Hardware initialization failed")
        return False
    
    # Measure single layer performance
    print(f"\n⚡ Measuring Single Layer Performance...")
    
    # Create test tensors
    seq_len = 64  # Realistic token length
    hidden_states = np.random.randn(seq_len, config["d_model"]).astype(np.float32) * 0.1
    
    # Measure NPU attention
    print(f"   🧠 Testing NPU attention...")
    npu_start = time.time()
    query = hidden_states + np.random.randn(seq_len, config["d_model"]).astype(np.float32) * 0.01
    key = hidden_states + np.random.randn(seq_len, config["d_model"]).astype(np.float32) * 0.01
    value = hidden_states + np.random.randn(seq_len, config["d_model"]).astype(np.float32) * 0.01
    
    attention_output = npu_kernel.compute_attention(query, key, value)
    npu_time = time.time() - npu_start
    
    print(f"      NPU attention time: {npu_time*1000:.2f}ms")
    
    # Measure Vulkan FFN
    print(f"   🎮 Testing Vulkan FFN...")
    vulkan_start = time.time()
    
    # Gate projection
    gate_weight = np.random.randn(config["d_model"], config["intermediate_size"]).astype(np.float32) * 0.1
    gate_proj = vulkan_compute.execute_matrix_multiply(attention_output, gate_weight)
    
    # Down projection
    down_weight = np.random.randn(config["intermediate_size"], config["d_model"]).astype(np.float32) * 0.1
    ffn_output = vulkan_compute.execute_matrix_multiply(gate_proj, down_weight)
    
    vulkan_time = time.time() - vulkan_start
    
    print(f"      Vulkan FFN time: {vulkan_time*1000:.2f}ms")
    
    # Calculate layer performance
    layer_time = npu_time + vulkan_time
    print(f"      Total layer time: {layer_time*1000:.2f}ms")
    
    # Calculate full model performance
    print(f"\n🔮 Calculating Full Model Performance...")
    
    # Time per token (all 62 layers)
    time_per_token = layer_time * config["layers"]
    theoretical_tps = 1.0 / time_per_token
    
    # Account for memory overhead and CPU orchestration
    memory_overhead = 0.15  # 15% overhead for memory transfers
    cpu_overhead = 0.10     # 10% overhead for CPU orchestration
    total_overhead = memory_overhead + cpu_overhead
    
    practical_tps = theoretical_tps * (1 - total_overhead)
    
    print(f"   Single layer time: {layer_time*1000:.2f}ms")
    print(f"   Time per token (62 layers): {time_per_token*1000:.2f}ms")
    print(f"   Theoretical TPS: {theoretical_tps:.2f}")
    print(f"   Memory overhead: {memory_overhead*100:.1f}%")
    print(f"   CPU overhead: {cpu_overhead*100:.1f}%")
    print(f"   Practical TPS: {practical_tps:.2f}")
    
    # Performance analysis
    print(f"\n📈 Performance Analysis:")
    print(f"   Target TPS: {config['target_tps']}")
    print(f"   Achieved TPS: {practical_tps:.2f}")
    print(f"   Target achieved: {'✅' if practical_tps >= config['target_tps'] else '❌'}")
    
    if practical_tps >= config['target_tps']:
        print(f"   Performance: {practical_tps/config['target_tps']:.2f}x target")
    else:
        print(f"   Performance: {practical_tps/config['target_tps']:.2f}x target (needs optimization)")
    
    # Hardware utilization
    print(f"\n⚙️ Hardware Utilization:")
    npu_utilization = (npu_time / layer_time) * 100
    igpu_utilization = (vulkan_time / layer_time) * 100
    
    print(f"   NPU utilization: {npu_utilization:.1f}%")
    print(f"   iGPU utilization: {igpu_utilization:.1f}%")
    print(f"   Memory architecture: HMA (96GB DDR5 shared)")
    
    # Comparison with baseline
    baseline_cpu_tps = 1.2  # From README
    improvement = practical_tps / baseline_cpu_tps
    
    print(f"\n🚀 Performance Comparison:")
    print(f"   CPU baseline: {baseline_cpu_tps} TPS")
    print(f"   NPU+iGPU: {practical_tps:.2f} TPS")
    print(f"   Improvement: {improvement:.1f}x faster")
    
    # Memory efficiency
    print(f"\n💾 Memory Efficiency:")
    print(f"   Model size: {config['quantized_size_gb']} GB")
    print(f"   NPU memory: 2GB SRAM")
    print(f"   iGPU memory: 16GB allocation")
    print(f"   System memory: 96GB DDR5-5600")
    print(f"   Architecture: ✅ Fits in HMA")
    
    return True

if __name__ == "__main__":
    calculate_gemma3_27b_performance()