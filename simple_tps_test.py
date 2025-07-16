#!/usr/bin/env python3
"""
Simple TPS Test - Measure real tokens per second with simulated model
"""

import time
import logging
import numpy as np
from real_vulkan_matrix_compute import VulkanMatrixCompute
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_gpu_memory():
    """Check GPU memory usage"""
    try:
        result = subprocess.run(['radeontop', '-d', '-', '-l', '1'], 
                              capture_output=True, text=True, timeout=1)
        if result.stdout:
            import re
            vram_match = re.search(r'vram\s+(\d+\.\d+)%\s+(\d+\.\d+)mb', result.stdout)
            gtt_match = re.search(r'gtt\s+(\d+\.\d+)%\s+(\d+\.\d+)mb', result.stdout)
            gpu_match = re.search(r'gpu\s+(\d+\.\d+)%', result.stdout)
            
            if vram_match and gtt_match and gpu_match:
                return {
                    'gpu_pct': float(gpu_match.group(1)),
                    'vram_mb': float(vram_match.group(2)),
                    'gtt_mb': float(gtt_match.group(2))
                }
    except:
        pass
    return {'gpu_pct': 0, 'vram_mb': 0, 'gtt_mb': 0}

def simulate_transformer_layer(vulkan, hidden_states, layer_weights):
    """Simulate one transformer layer computation"""
    batch_size, seq_len, hidden_dim = hidden_states.shape
    
    # 1. Multi-head attention
    # Q, K, V projections
    qkv = vulkan.matrix_multiply(
        hidden_states.reshape(-1, hidden_dim),
        layer_weights['qkv_weight']
    )
    
    # 2. FFN Gate and Up projections (fused)
    ffn_hidden = vulkan.matrix_multiply(
        hidden_states.reshape(-1, hidden_dim),
        layer_weights['gate_weight']
    )
    
    # 3. FFN Down projection
    output = vulkan.matrix_multiply(
        ffn_hidden,
        layer_weights['down_weight']
    )
    
    return output.reshape(batch_size, seq_len, hidden_dim)

def main():
    logger.info("🚀 SIMPLE TPS TEST - Measuring real performance")
    
    # Initialize Vulkan
    logger.info("Initializing Vulkan compute...")
    vulkan = VulkanMatrixCompute()
    vulkan.initialize()
    
    # Model parameters (Gemma 27B)
    hidden_dim = 5376
    intermediate_dim = hidden_dim * 4  # 21504
    num_layers = 62
    
    # Check initial memory
    mem_start = check_gpu_memory()
    logger.info(f"Initial GPU memory: VRAM={mem_start['vram_mb']:.0f}MB, GTT={mem_start['gtt_mb']:.0f}MB")
    
    # Create and load model weights to GPU
    logger.info("Creating model weights...")
    
    # Simulate loading 10 layers worth of weights
    layer_weights_list = []
    for i in range(10):  # Just 10 layers for testing
        layer_weights = {
            'qkv_weight': np.random.randn(hidden_dim, hidden_dim * 3).astype(np.float32) * 0.02,
            'gate_weight': np.random.randn(hidden_dim, intermediate_dim).astype(np.float32) * 0.02,
            'down_weight': np.random.randn(intermediate_dim, hidden_dim).astype(np.float32) * 0.02
        }
        
        # Allocate to GPU
        logger.info(f"Allocating layer {i} to GPU...")
        gpu_weights = {}
        for name, weight in layer_weights.items():
            gpu_buffer = vulkan._allocate_gpu_memory(weight)
            gpu_weights[name] = weight  # Keep CPU copy for now
        
        layer_weights_list.append(gpu_weights)
    
    # Check memory after loading
    mem_after = check_gpu_memory()
    logger.info(f"After loading weights: VRAM={mem_after['vram_mb']:.0f}MB (+{mem_after['vram_mb']-mem_start['vram_mb']:.0f}MB)")
    
    # Test token generation
    logger.info("\n📊 Testing token generation speed...")
    
    batch_size = 1
    seq_len = 256
    num_tokens = 50
    
    # Initialize hidden states
    hidden_states = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
    
    # Warm up
    logger.info("Warming up...")
    for i in range(5):
        for layer_idx in range(len(layer_weights_list)):
            hidden_states = simulate_transformer_layer(vulkan, hidden_states, layer_weights_list[layer_idx])
    
    # Actual benchmark
    logger.info("Running benchmark...")
    token_times = []
    
    for token_idx in range(num_tokens):
        token_start = time.time()
        
        # Process through all layers
        current_hidden = hidden_states
        for layer_idx in range(len(layer_weights_list)):
            current_hidden = simulate_transformer_layer(vulkan, current_hidden, layer_weights_list[layer_idx])
        
        token_time = time.time() - token_start
        token_times.append(token_time)
        
        if token_idx % 10 == 0:
            current_tps = 1.0 / (sum(token_times) / len(token_times))
            mem_current = check_gpu_memory()
            logger.info(f"Token {token_idx}: {current_tps:.1f} TPS, GPU={mem_current['gpu_pct']:.1f}%")
    
    # Final results
    avg_token_time = sum(token_times) / len(token_times)
    tps = 1.0 / avg_token_time
    
    mem_final = check_gpu_memory()
    
    logger.info("\n🎯 RESULTS:")
    logger.info(f"Average token time: {avg_token_time*1000:.1f}ms")
    logger.info(f"Tokens per second: {tps:.1f} TPS")
    logger.info(f"GPU usage: {mem_final['gpu_pct']:.1f}%")
    logger.info(f"VRAM usage: {mem_final['vram_mb']:.0f}MB")
    logger.info(f"GTT usage: {mem_final['gtt_mb']:.0f}MB")
    
    # Performance analysis
    logger.info("\n📊 PERFORMANCE ANALYSIS:")
    flops_per_token = (
        # QKV projection
        batch_size * seq_len * hidden_dim * hidden_dim * 3 * 2 +
        # FFN gate
        batch_size * seq_len * hidden_dim * intermediate_dim * 2 +
        # FFN down
        batch_size * seq_len * intermediate_dim * hidden_dim * 2
    ) * len(layer_weights_list)
    
    gflops = flops_per_token / (avg_token_time * 1e9)
    theoretical_max = 8900  # 8.9 TFLOPS for Radeon 780M
    efficiency = (gflops / theoretical_max) * 100
    
    logger.info(f"FLOPS per token: {flops_per_token/1e9:.1f} GFLOPS")
    logger.info(f"Achieved: {gflops:.1f} GFLOPS")
    logger.info(f"Theoretical max: {theoretical_max} GFLOPS")
    logger.info(f"Efficiency: {efficiency:.1f}%")
    
    # Cleanup
    vulkan.cleanup()

if __name__ == "__main__":
    main()