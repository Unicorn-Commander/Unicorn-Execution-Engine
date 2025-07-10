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
        
    def initialize(self) -> bool:
        """Initialize Vulkan FFN compute engine"""
        logger.info("🎮 Initializing Vulkan FFN Compute Engine...")
        
        success = self.vulkan_compute.initialize()
        if success:
            self.initialized = True
            logger.info("✅ Vulkan FFN Compute Engine ready for iGPU acceleration!")
        else:
            logger.error("❌ Failed to initialize Vulkan FFN Compute Engine")
        
        return success
    
    def compute_ffn_layer(self, 
                         hidden_states: torch.Tensor,
                         gate_proj_weight: torch.Tensor,
                         up_proj_weight: torch.Tensor,
                         down_proj_weight: torch.Tensor) -> torch.Tensor:
        """
        Compute FFN layer using Vulkan iGPU acceleration
        
        FFN formula: down_proj(silu(gate_proj(x)) * up_proj(x))
        """
        if not self.initialized:
            raise RuntimeError("Vulkan FFN compute engine not initialized")
        
        logger.info(f"🚀 Vulkan FFN Layer: {hidden_states.shape}")
        start_time = time.time()
        
        # Convert tensors to numpy float16 for optimized Vulkan compute
        hidden_np = hidden_states.detach().cpu().numpy().astype(np.float16)
        gate_weight_np = gate_proj_weight.detach().cpu().numpy().astype(np.float16)
        up_weight_np = up_proj_weight.detach().cpu().numpy().astype(np.float16)
        down_weight_np = down_proj_weight.detach().cpu().numpy().astype(np.float16)
        
        # Reshape for matrix multiplication
        batch_size, seq_len, hidden_size = hidden_np.shape
        hidden_flat = hidden_np.reshape(-1, hidden_size)  # (batch*seq, hidden)
        
        # FUSED VULKAN OPERATION: All FFN computation on GPU
        logger.info("   🚀 Fused FFN: gate_proj + up_proj + silu + multiply + down_proj (Vulkan)")
        final_output = self.vulkan_compute.compute_fused_ffn(
            hidden_flat, gate_weight_np.T, up_weight_np.T, down_weight_np.T
        )
        
        # Reshape back to original shape
        final_output_reshaped = final_output.reshape(batch_size, seq_len, hidden_size)
        
        # Convert back to torch tensor
        result = torch.from_numpy(final_output_reshaped).to(hidden_states.device)
        
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
    test_vulkan_ffn_engine()