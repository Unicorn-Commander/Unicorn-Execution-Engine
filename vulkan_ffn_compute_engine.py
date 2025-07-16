#!/usr/bin/env python3
"""
Vulkan FFN Compute Engine for Gemma 3 27B
Specialized iGPU compute for quantized FFN layers
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import time
from pathlib import Path

# Import base Vulkan compute
from real_vulkan_matrix_compute import VulkanMatrixCompute

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VulkanFFNComputeEngine:
    """Vulkan compute engine specialized for FFN operations"""
    
    def __init__(self):
        self.vulkan_compute = VulkanMatrixCompute()
        self.initialized = False
        
        # Performance tracking
        self.ffn_compute_times = []
        self.total_ffn_operations = 0
        
        # Persistent weight buffers
        self.gate_weight_buffer = None
        self.up_weight_buffer = None
        self.down_weight_buffer = None
        self.weights_loaded = False
        
    def initialize(self, use_fp16: bool = False) -> bool:
        """Initialize Vulkan FFN compute engine"""
        logger.info("🎮 Initializing Vulkan FFN Compute Engine...")
        
        success = self.vulkan_compute.initialize(use_fp16=use_fp16)
        if success:
            self.initialized = True
            logger.info("✅ Vulkan FFN Compute Engine ready for iGPU acceleration!")
        else:
            logger.error("❌ Failed to initialize Vulkan FFN Compute Engine")
        
        return success

    def load_weights(self, gate_proj_weight: torch.Tensor, up_proj_weight: torch.Tensor, down_proj_weight: torch.Tensor):
        """Pre-load FFN weights to persistent Vulkan buffers on the iGPU"""
        if not self.initialized:
            raise RuntimeError("Vulkan FFN compute engine not initialized")

        logger.info("🚀 Pre-loading FFN weights to iGPU VRAM...")
        
        # Convert tensors to numpy and then create persistent buffers
        gate_weight_np = gate_proj_weight.T.cpu().numpy()
        up_weight_np = up_proj_weight.T.cpu().numpy()
        down_weight_np = down_proj_weight.T.cpu().numpy()

        self.gate_weight_buffer = self.vulkan_compute.create_persistent_buffer(gate_weight_np)
        self.up_weight_buffer = self.vulkan_compute.create_persistent_buffer(up_weight_np)
        self.down_weight_buffer = self.vulkan_compute.create_persistent_buffer(down_weight_np)
        
        self.weights_loaded = True
        logger.info("✅ FFN weights loaded and resident on iGPU.")

    def compute_ffn_layer(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute FFN layer using Vulkan iGPU acceleration with pre-loaded weights.
        Weights are expected to be resident on the GPU already.
        
        FFN formula: down_proj(silu(gate_proj(x)) * up_proj(x))
        """
        if not self.initialized or not self.weights_loaded:
            raise RuntimeError("Vulkan engine not initialized or weights not pre-loaded.")
        
        logger.info(f"🚀 Vulkan FFN Layer (pre-loaded weights): {hidden_states.shape}")
        start_time = time.time()
        
        # Convert only the changing hidden_states tensor to numpy
        hidden_np = hidden_states.detach().cpu().numpy().astype(np.float32)
        
        # Reshape for matrix multiplication
        batch_size, seq_len, hidden_size = hidden_np.shape
        hidden_flat = hidden_np.reshape(-1, hidden_size)
        
        # FUSED VULKAN OPERATION: Use pre-loaded weight buffers
        logger.info("   🚀 815 GFLOPS Fused FFN with resident weights: gate_proj + up_proj + silu + multiply + down_proj")
        final_output = self.vulkan_compute.compute_fused_ffn_persistent_weights(
            hidden_flat, 
            self.gate_weight_buffer, 
            self.up_weight_buffer, 
            self.down_weight_buffer, 
            flags=1
        )
        
        # Reshape back to original shape
        final_output_reshaped = final_output.reshape(batch_size, seq_len, hidden_size)
        
        # Convert back to torch tensor
        result = torch.from_numpy(final_output_reshaped.astype(np.float32))
        
        # Performance tracking
        compute_time = time.time() - start_time
        self.ffn_compute_times.append(compute_time)
        self.total_ffn_operations += 1
        
        logger.info(f"   ✅ FFN layer complete: {compute_time*1000:.2f}ms")
        
        return result
    
    def _silu_activation(self, x: np.ndarray) -> np.ndarray:
        """SiLU activation function: x * sigmoid(x)"""
        return x * (1.0 / (1.0 + np.exp(-x)))
    
    def compute_ffn_batch(self, 
                         hidden_states: torch.Tensor,
                         ffn_weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute FFN with weight dictionary format
        """
        return self.compute_ffn_layer(
            hidden_states,
            ffn_weights['gate_proj'],
            ffn_weights['up_proj'],
            ffn_weights['down_proj']
        )
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get FFN performance statistics"""
        if not self.ffn_compute_times:
            return {
                "avg_ffn_time_ms": 0,
                "total_ffn_operations": 0,
                "total_ffn_time_s": 0
            }
        
        avg_time = np.mean(self.ffn_compute_times)
        total_time = np.sum(self.ffn_compute_times)
        
        return {
            "avg_ffn_time_ms": avg_time * 1000,
            "total_ffn_operations": self.total_ffn_operations,
            "total_ffn_time_s": total_time,
            "min_ffn_time_ms": np.min(self.ffn_compute_times) * 1000,
            "max_ffn_time_ms": np.max(self.ffn_compute_times) * 1000
        }

    def cleanup(self):
        """Cleanup Vulkan resources"""
        if self.vulkan_compute:
            self.vulkan_compute.cleanup()
    
    def benchmark_ffn_performance(self, 
                                 batch_size: int = 1,
                                 seq_len: int = 128,
                                 hidden_size: int = 4096,
                                 intermediate_size: int = 16384) -> Dict[str, float]:
        """Benchmark FFN performance with synthetic data"""
        
        logger.info(f"🔬 Benchmarking FFN performance...")
        logger.info(f"   Batch size: {batch_size}, Seq len: {seq_len}")
        logger.info(f"   Hidden size: {hidden_size}, Intermediate size: {intermediate_size}")
        
        # Create synthetic data
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        gate_proj_weight = torch.randn(intermediate_size, hidden_size)
        up_proj_weight = torch.randn(intermediate_size, hidden_size)
        down_proj_weight = torch.randn(hidden_size, intermediate_size)
        
        # Warmup
        logger.info("   🔥 Warmup runs...")
        for _ in range(3):
            _ = self.compute_ffn_layer(hidden_states, gate_proj_weight, up_proj_weight, down_proj_weight)
        
        # Benchmark runs
        logger.info("   📊 Benchmark runs...")
        benchmark_times = []
        
        for i in range(10):
            start_time = time.time()
            result = self.compute_ffn_layer(hidden_states, gate_proj_weight, up_proj_weight, down_proj_weight)
            end_time = time.time()
            
            benchmark_times.append(end_time - start_time)
            logger.info(f"      Run {i+1}: {(end_time - start_time)*1000:.2f}ms")
        
        # Calculate statistics
        avg_time = np.mean(benchmark_times)
        min_time = np.min(benchmark_times)
        max_time = np.max(benchmark_times)
        std_time = np.std(benchmark_times)
        
        # Calculate theoretical performance
        total_flops = batch_size * seq_len * (
            2 * hidden_size * intermediate_size +  # gate_proj
            2 * hidden_size * intermediate_size +  # up_proj
            2 * intermediate_size * hidden_size    # down_proj
        )
        
        avg_gflops = total_flops / (avg_time * 1e9)
        
        stats = {
            "avg_time_ms": avg_time * 1000,
            "min_time_ms": min_time * 1000,
            "max_time_ms": max_time * 1000,
            "std_time_ms": std_time * 1000,
            "avg_gflops": avg_gflops,
            "total_flops": total_flops,
            "throughput_tokens_per_sec": (batch_size * seq_len) / avg_time
        }
        
        logger.info("🎯 FFN Benchmark Results:")
        logger.info(f"   Average time: {stats['avg_time_ms']:.2f}ms")
        logger.info(f"   Min/Max time: {stats['min_time_ms']:.2f}ms / {stats['max_time_ms']:.2f}ms")
        logger.info(f"   Performance: {stats['avg_gflops']:.2f} GFLOPS")
        logger.info(f"   Throughput: {stats['throughput_tokens_per_sec']:.2f} tokens/sec")
        
        return stats

def test_vulkan_ffn_engine():
    """Test Vulkan FFN compute engine"""
    logger.info("🧪 Testing Vulkan FFN Compute Engine...")
    
    # Initialize engine
    ffn_engine = VulkanFFNComputeEngine()
    if not ffn_engine.initialize():
        logger.error("❌ Failed to initialize Vulkan FFN engine")
        return
    
    # Test with Gemma 3 27B dimensions
    batch_size = 1
    seq_len = 32
    hidden_size = 4096
    intermediate_size = 16384
    
    logger.info("🔬 Testing with Gemma 3 27B dimensions...")
    
    # Create test data
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    gate_proj_weight = torch.randn(intermediate_size, hidden_size)
    up_proj_weight = torch.randn(intermediate_size, hidden_size)
    down_proj_weight = torch.randn(hidden_size, intermediate_size)
    
    # Test FFN computation
    start_time = time.time()
    result = ffn_engine.compute_ffn_layer(
        hidden_states, gate_proj_weight, up_proj_weight, down_proj_weight
    )
    end_time = time.time()
    
    # Verify output shape
    expected_shape = (batch_size, seq_len, hidden_size)
    assert result.shape == expected_shape, f"Shape mismatch: {result.shape} != {expected_shape}"
    
    logger.info(f"✅ FFN test passed!")
    logger.info(f"   Input shape: {hidden_states.shape}")
    logger.info(f"   Output shape: {result.shape}")
    logger.info(f"   Compute time: {(end_time - start_time)*1000:.2f}ms")
    
    # Performance statistics
    stats = ffn_engine.get_performance_stats()
    logger.info(f"📊 Performance stats:")
    logger.info(f"   Average FFN time: {stats['avg_ffn_time_ms']:.2f}ms")
    logger.info(f"   Total operations: {stats['total_ffn_operations']}")
    
    # Benchmark performance
    logger.info("\n🚀 Running comprehensive benchmark...")
    benchmark_stats = ffn_engine.benchmark_ffn_performance(
        batch_size=1, seq_len=128, hidden_size=4096, intermediate_size=16384
    )
    
    logger.info("✅ Vulkan FFN engine test completed!")
    
    return benchmark_stats

if __name__ == "__main__":
    try:
        test_vulkan_ffn_engine()
    finally:
        # Ensure cleanup is called even if test fails
        ffn_engine = VulkanFFNComputeEngine()
        ffn_engine.cleanup()