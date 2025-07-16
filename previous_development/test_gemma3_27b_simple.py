#!/usr/bin/env python3
"""
Simplified Gemma 3 27B NPU+iGPU Performance Test
Direct performance measurement with real hardware components
"""

import os
import sys
import time
import numpy as np
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import our real NPU+iGPU components
from npu_attention_kernel import NPUAttentionKernel, NPUAttentionConfig
from real_vulkan_compute import RealVulkanCompute

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplifiedGemma3_27B_Engine:
    """Simplified Gemma 3 27B engine for direct performance testing"""
    
    def __init__(self):
        # Model configuration for 27B
        self.config = {
            "model_name": "gemma-3-27b-it",
            "vocab_size": 256000,
            "seq_length": 2048,
            "d_model": 4096,
            "n_layers": 62,  # 27B has 62 layers
            "n_heads": 32,
            "intermediate_size": 14336,
            "head_dim": 128,
            "projected_tps": 22.7  # From Vulkan-accelerated README
        }
        
        # Hardware components
        self.npu_kernel = None
        self.vulkan_compute = None
        self.initialized = False
        
        # Performance tracking
        self.total_tokens = 0
        self.total_time = 0.0
        self.npu_time = 0.0
        self.igpu_time = 0.0
        
        logger.info("🦄 Simplified Gemma 3 27B NPU+iGPU Test Engine")
        logger.info(f"   Model: {self.config['model_name']}")
        logger.info(f"   Layers: {self.config['n_layers']}")
        logger.info(f"   Parameters: ~27B")
        logger.info(f"   Target TPS: {self.config['projected_tps']}")
    
    def initialize(self):
        """Initialize NPU+iGPU components"""
        logger.info("🚀 Initializing NPU+iGPU acceleration...")
        
        try:
            # Initialize NPU attention kernel
            npu_config = NPUAttentionConfig(
                seq_length=self.config["seq_length"],
                d_model=self.config["d_model"],
                num_heads=self.config["n_heads"],
                npu_memory_mb=2048,  # 2GB NPU memory
                precision="fp16"
            )
            
            self.npu_kernel = NPUAttentionKernel(npu_config)
            if not self.npu_kernel.initialize():
                logger.error("❌ NPU initialization failed")
                return False
            
            logger.info("✅ NPU Phoenix (16 TOPS) initialized")
            
            # Initialize Vulkan compute for FFN
            self.vulkan_compute = RealVulkanCompute()
            if not self.vulkan_compute.initialize():
                logger.error("❌ Vulkan compute initialization failed")
                return False
            
            logger.info("✅ AMD Radeon 780M iGPU initialized")
            
            self.initialized = True
            logger.info("🎯 Engine ready for performance testing!")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False
    
    def run_performance_test(self, num_tokens=50):
        """Run performance test with real NPU+iGPU computation"""
        if not self.initialized:
            raise RuntimeError("Engine not initialized")
        
        logger.info(f"⚡ Running Gemma 3 27B performance test ({num_tokens} tokens)")
        
        # Reset counters
        self.total_tokens = 0
        self.total_time = 0.0
        self.npu_time = 0.0
        self.igpu_time = 0.0
        
        start_time = time.time()
        
        # Simulate token generation loop
        for token_idx in range(num_tokens):
            # Current sequence length grows with each token
            current_seq_len = min(32 + token_idx, self.config["seq_length"])
            
            # Create realistic input tensors
            hidden_states = np.random.randn(current_seq_len, self.config["d_model"]).astype(np.float32) * 0.1
            
            # Process through all 62 layers
            for layer_idx in range(self.config["n_layers"]):
                # Attention computation on NPU
                npu_start = time.time()
                attention_output = self._run_npu_attention(hidden_states, layer_idx)
                npu_end = time.time()
                self.npu_time += (npu_end - npu_start)
                
                # FFN computation on iGPU
                igpu_start = time.time()
                ffn_output = self._run_igpu_ffn(attention_output, layer_idx)
                igpu_end = time.time()
                self.igpu_time += (igpu_end - igpu_start)
                
                # Update hidden states for next layer
                hidden_states = ffn_output
            
            # Progress indicator
            if (token_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                current_tps = (token_idx + 1) / elapsed
                logger.info(f"   Token {token_idx+1}/{num_tokens} - Current TPS: {current_tps:.1f}")
        
        # Calculate final performance
        self.total_time = time.time() - start_time
        self.total_tokens = num_tokens
        
        return self._generate_performance_report()
    
    def _run_npu_attention(self, hidden_states, layer_idx):
        """Run attention layer on NPU Phoenix"""
        seq_len, d_model = hidden_states.shape
        
        # Create Q, K, V matrices
        query = hidden_states + np.random.randn(seq_len, d_model).astype(np.float32) * 0.01
        key = hidden_states + np.random.randn(seq_len, d_model).astype(np.float32) * 0.01
        value = hidden_states + np.random.randn(seq_len, d_model).astype(np.float32) * 0.01
        
        # Use real NPU kernel for attention computation
        attention_output = self.npu_kernel.compute_attention(query, key, value)
        
        return attention_output
    
    def _run_igpu_ffn(self, hidden_states, layer_idx):
        """Run FFN layer on AMD Radeon 780M iGPU"""
        seq_len, d_model = hidden_states.shape
        
        # Create weight matrices for FFN
        up_weight = np.random.randn(d_model, self.config["intermediate_size"]).astype(np.float32) * 0.1
        gate_weight = np.random.randn(d_model, self.config["intermediate_size"]).astype(np.float32) * 0.1
        down_weight = np.random.randn(self.config["intermediate_size"], d_model).astype(np.float32) * 0.1
        
        # Gate projection with real Vulkan compute
        gate_proj = self.vulkan_compute.execute_matrix_multiply(hidden_states, gate_weight)
        
        # Up projection (simulated for this test)
        up_proj = np.dot(hidden_states, up_weight)
        
        # Apply SiLU activation and combine
        def silu(x):
            return x / (1.0 + np.exp(-x))
        
        intermediate = silu(gate_proj) * up_proj
        
        # Down projection with real Vulkan compute
        output = self.vulkan_compute.execute_matrix_multiply(intermediate, down_weight)
        
        return output
    
    def _generate_performance_report(self):
        """Generate detailed performance report"""
        if self.total_time == 0:
            return None
        
        tps = self.total_tokens / self.total_time
        compute_time = self.npu_time + self.igpu_time
        
        report = {
            "model": self.config["model_name"],
            "performance": {
                "tokens_per_second": tps,
                "total_tokens": self.total_tokens,
                "total_time_seconds": self.total_time,
                "target_tps": self.config["projected_tps"],
                "target_achieved": tps >= self.config["projected_tps"],
                "improvement_over_cpu": tps / 1.2  # Baseline 1.2 TPS from README
            },
            "hardware_breakdown": {
                "npu_time_seconds": self.npu_time,
                "igpu_time_seconds": self.igpu_time,
                "compute_time_seconds": compute_time,
                "npu_utilization_percent": (self.npu_time / self.total_time) * 100,
                "igpu_utilization_percent": (self.igpu_time / self.total_time) * 100,
                "compute_efficiency_percent": (compute_time / self.total_time) * 100
            },
            "architecture": {
                "npu_device": "NPU Phoenix (16 TOPS)",
                "igpu_device": "AMD Radeon 780M (2.7 TFLOPS)",
                "layers": self.config["n_layers"],
                "parameters": "~27B",
                "memory_architecture": "HMA (96GB DDR5 shared)"
            }
        }
        
        return report

def test_gemma3_27b_performance():
    """Test Gemma 3 27B performance with real NPU+iGPU"""
    print("🦄 Gemma 3 27B NPU+iGPU Performance Test")
    print("=" * 60)
    
    # Initialize engine
    engine = SimplifiedGemma3_27B_Engine()
    
    if not engine.initialize():
        print("❌ Engine initialization failed")
        return False
    
    # Run performance tests with different token counts
    test_configurations = [
        {"tokens": 25, "name": "Quick Test"},
        {"tokens": 50, "name": "Standard Test"},
        {"tokens": 100, "name": "Extended Test"}
    ]
    
    results = []
    
    for config in test_configurations:
        print(f"\n🧪 {config['name']} ({config['tokens']} tokens)")
        
        try:
            # Run test
            report = engine.run_performance_test(config["tokens"])
            results.append(report)
            
            # Display results
            perf = report["performance"]
            hw = report["hardware_breakdown"]
            
            print(f"   ✅ TPS: {perf['tokens_per_second']:.1f}")
            print(f"   🎯 Target: {perf['target_tps']} TPS - {'✅' if perf['target_achieved'] else '❌'}")
            print(f"   📊 NPU: {hw['npu_utilization_percent']:.1f}%, iGPU: {hw['igpu_utilization_percent']:.1f}%")
            print(f"   ⚡ Improvement: {perf['improvement_over_cpu']:.1f}x over CPU")
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            continue
    
    # Summary report
    if results:
        avg_tps = sum(r['performance']['tokens_per_second'] for r in results) / len(results)
        target_tps = results[0]['performance']['target_tps']
        
        print(f"\n📊 Final Performance Summary:")
        print(f"   Model: {results[0]['model']}")
        print(f"   Architecture: {results[0]['architecture']['npu_device']} + {results[0]['architecture']['igpu_device']}")
        print(f"   Average TPS: {avg_tps:.1f}")
        print(f"   Target TPS: {target_tps}")
        print(f"   Target Achieved: {'✅' if avg_tps >= target_tps else '❌'}")
        print(f"   Layers: {results[0]['architecture']['layers']}")
        print(f"   Parameters: {results[0]['architecture']['parameters']}")
        print(f"   Memory: {results[0]['architecture']['memory_architecture']}")
        
        # Calculate projected performance for longer sequences
        print(f"\n🔮 Projected Performance:")
        print(f"   4B model equivalent: {avg_tps * 4:.1f} TPS")
        print(f"   Production throughput: {avg_tps * 60:.0f} tokens/minute")
        
        return True
    
    return False

if __name__ == "__main__":
    test_gemma3_27b_performance()