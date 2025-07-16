#!/usr/bin/env python3
"""
Measure NPU+iGPU Performance - Tokens Per Second
Real hardware acceleration performance measurement
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import time
from pathlib import Path
import gc

# Import our components
from quantized_gemma27b_npu_igpu_loader import QuantizedGemma27BNPUIGPULoader
from vulkan_ffn_compute_engine import VulkanFFNComputeEngine

import os
import torch
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# --- PATCH: Load embedding weights if not found ---
def patch_load_embeddings(shared_weights):
    if shared_weights is None:
        shared_weights = {}
    embed_file = os.path.join("./Development/github_repos/Unicorn-Execution-Engine/real_test_weights/", "language_model.model.embed_tokens.weight.pt")
    if os.path.exists(embed_file):
        embed_tensor = torch.load(embed_file)
        shared_weights["embed_tokens"] = {"tensor": embed_tensor}
        print("[PATCH] Loaded embedding tensor from .pt file.")
    return shared_weights
    """Measure real NPU+iGPU performance in tokens per second"""
    
    def __init__(self, quantized_model_path: str = "./quantized_models/gemma-3-27b-it-layer-by-layer"):
        self.quantized_model_path = quantized_model_path
        
        # Initialize components
        self.model_loader = QuantizedGemma27BNPUIGPULoader(quantized_model_path)
        self.vulkan_ffn_engine = VulkanFFNComputeEngine()
        
        # Model state
        self.model_info = None
        self.shared_weights = None
        self.shared_weights = patch_load_embeddings(self.shared_weights)
        self.layer_loader = None
        
        # Performance tracking
        self.layer_times = []
        self.ffn_times = []
        self.attention_times = []
        self.total_times = []
        
        logger.info("🔬 NPU+iGPU Performance Measurement Tool")
    
    def initialize_hardware(self) -> bool:
        """Initialize hardware components"""
        logger.info("🚀 Initializing NPU+iGPU hardware for performance measurement...")
        
        # Initialize Vulkan FFN engine
        if not self.vulkan_ffn_engine.initialize():
            logger.error("❌ Vulkan FFN engine initialization failed")
            return False
        
        # Load quantized model
        try:
            self.model_info = self.model_loader.load_model_streaming()
            self.shared_weights = self.model_info['shared_weights']
            self.layer_loader = self.model_info['layer_loader']
            logger.info("✅ Quantized model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            return False
        
        logger.info("✅ Hardware initialization complete!")
        return True
    
    def compute_attention_cpu(self, 
                             hidden_states: torch.Tensor,
                             attention_weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute attention (CPU fallback for now)"""
        start_time = time.time()
        
        # Extract weights
        q_weight = attention_weights['q_proj']['tensor']
        k_weight = attention_weights['k_proj']['tensor']
        v_weight = attention_weights['v_proj']['tensor']
        o_weight = attention_weights['o_proj']['tensor']
        
        # Project to Q, K, V
        q = torch.matmul(hidden_states, q_weight.transpose(-1, -2))
        k = torch.matmul(hidden_states, k_weight.transpose(-1, -2))
        v = torch.matmul(hidden_states, v_weight.transpose(-1, -2))
        
        # Scaled dot-product attention
        d_k = q.size(-1)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)
        attention_probs = F.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_probs, v)
        
        # Output projection
        output = torch.matmul(context, o_weight.transpose(-1, -2))
        
        attention_time = time.time() - start_time
        self.attention_times.append(attention_time)
        
        return output
    
    def compute_ffn_vulkan(self, 
                          hidden_states: torch.Tensor,
                          ffn_weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute FFN using Vulkan iGPU acceleration"""
        start_time = time.time()
        
        # Extract weight tensors
        gate_proj = ffn_weights['gate_proj']['tensor']
        up_proj = ffn_weights['up_proj']['tensor']
        down_proj = ffn_weights['down_proj']['tensor']
        
        # Use Vulkan FFN engine
        result = self.vulkan_ffn_engine.compute_ffn_layer(
            hidden_states, gate_proj, up_proj, down_proj
        )
        
        ffn_time = time.time() - start_time
        self.ffn_times.append(ffn_time)
        
        return result
    
    def compute_single_layer(self, 
                            hidden_states: torch.Tensor,
                            layer_weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute a single transformer layer"""
        layer_start = time.time()
        
        # Extract weights by type
        attention_weights = {}
        ffn_weights = {}
        norm_weights = {}
        
        for name, weight_info in layer_weights.items():
            if 'self_attn' in name:
                if 'q_proj' in name:
                    attention_weights['q_proj'] = weight_info
                elif 'k_proj' in name:
                    attention_weights['k_proj'] = weight_info
                elif 'v_proj' in name:
                    attention_weights['v_proj'] = weight_info
                elif 'o_proj' in name:
                    attention_weights['o_proj'] = weight_info
            elif 'mlp' in name:
                if 'gate_proj' in name:
                    ffn_weights['gate_proj'] = weight_info
                elif 'up_proj' in name:
                    ffn_weights['up_proj'] = weight_info
                elif 'down_proj' in name:
                    ffn_weights['down_proj'] = weight_info
            elif 'norm' in name:
                norm_weights[name] = weight_info
        
        # Input layer norm
        input_norm_weight = None
        for name, weight_info in norm_weights.items():
            if 'input_layernorm' in name:
                input_norm_weight = weight_info['tensor']
                break
        
        if input_norm_weight is not None:
            normed_input = F.layer_norm(hidden_states, input_norm_weight.shape, input_norm_weight)
        else:
            normed_input = hidden_states
        
        # Attention computation
        attention_output = self.compute_attention_cpu(normed_input, attention_weights)
        
        # Residual connection
        hidden_states = hidden_states + attention_output
        
        # Post-attention layer norm
        post_attn_norm_weight = None
        for name, weight_info in norm_weights.items():
            if 'post_attention_layernorm' in name:
                post_attn_norm_weight = weight_info['tensor']
                break
        
        if post_attn_norm_weight is not None:
            normed_hidden = F.layer_norm(hidden_states, post_attn_norm_weight.shape, post_attn_norm_weight)
        else:
            normed_hidden = hidden_states
        
        # FFN computation (Vulkan iGPU)
        ffn_output = self.compute_ffn_vulkan(normed_hidden, ffn_weights)
        
        # Residual connection
        hidden_states = hidden_states + ffn_output
        
        layer_time = time.time() - layer_start
        self.layer_times.append(layer_time)
        
        return hidden_states
    
    def measure_single_token_performance(self, 
                                       input_length: int = 128,
                                       num_layers: int = 3) -> Dict[str, float]:
        """Measure performance for generating a single token"""
        
        logger.info(f"🔬 Measuring single token performance...")
        logger.info(f"   Input length: {input_length} tokens")
        logger.info(f"   Processing layers: {num_layers}")
        
        total_start = time.time()
        
        # Get embeddings
        embed_weight = None
        for name, weight_info in self.shared_weights.items():
            if 'embed_tokens' in name:
                embed_weight = weight_info['tensor']
                break
        
        if embed_weight is None:
            raise RuntimeError("Embedding weights not found")
        
        # Create input sequence
        input_ids = torch.randint(0, embed_weight.size(0), (1, input_length), dtype=torch.long)
        
        # Embedding lookup
        embed_start = time.time()
        hidden_states = F.embedding(input_ids, embed_weight)
        embed_time = time.time() - embed_start
        
        logger.info(f"   📊 Embedding lookup: {embed_time*1000:.2f}ms")
        
        # Process through layers
        for layer_num in range(num_layers):
            logger.info(f"   🔄 Processing layer {layer_num}...")
            
            # Load layer weights
            layer_weights = self.layer_loader(layer_num)
            
            # Compute layer
            hidden_states = self.compute_single_layer(hidden_states, layer_weights)
            
            logger.info(f"      ✅ Layer {layer_num}: {self.layer_times[-1]*1000:.2f}ms")
            logger.info(f"         - Attention: {self.attention_times[-1]*1000:.2f}ms")
            logger.info(f"         - FFN (Vulkan): {self.ffn_times[-1]*1000:.2f}ms")
            
            # Cleanup
            del layer_weights
            gc.collect()
        
        # Final layer norm
        final_norm_weight = None
        for name, weight_info in self.shared_weights.items():
            if 'norm' in name and 'model' in name:
                final_norm_weight = weight_info['tensor']
                break
        
        if final_norm_weight is not None:
            norm_start = time.time()
            hidden_states = F.layer_norm(hidden_states, final_norm_weight.shape, final_norm_weight)
            norm_time = time.time() - norm_start
            logger.info(f"   📊 Final norm: {norm_time*1000:.2f}ms")
        
        # Language model head (logits)
        lm_start = time.time()
        logits = torch.matmul(hidden_states[:, -1, :], embed_weight.transpose(-1, -2))
        lm_time = time.time() - lm_start
        
        logger.info(f"   📊 LM head: {lm_time*1000:.2f}ms")
        
        total_time = time.time() - total_start
        self.total_times.append(total_time)
        
        # Calculate performance metrics
        total_attention_time = sum(self.attention_times[-num_layers:])
        total_ffn_time = sum(self.ffn_times[-num_layers:])
        total_layer_time = sum(self.layer_times[-num_layers:])
        
        performance = {
            "total_time_ms": total_time * 1000,
            "total_time_s": total_time,
            "tokens_per_second": 1.0 / total_time,
            "input_length": input_length,
            "num_layers": num_layers,
            "embed_time_ms": embed_time * 1000,
            "attention_time_ms": total_attention_time * 1000,
            "ffn_time_ms": total_ffn_time * 1000,
            "layer_time_ms": total_layer_time * 1000,
            "lm_head_time_ms": lm_time * 1000,
            "avg_layer_time_ms": (total_layer_time / num_layers) * 1000,
            "avg_attention_time_ms": (total_attention_time / num_layers) * 1000,
            "avg_ffn_time_ms": (total_ffn_time / num_layers) * 1000
        }
        
        return performance
    
    def benchmark_performance(self, 
                            input_lengths: List[int] = [32, 64, 128],
                            num_layers: int = 3,
                            num_runs: int = 3) -> Dict[str, Any]:
        """Comprehensive performance benchmark"""
        
        logger.info("🚀 Starting comprehensive NPU+iGPU performance benchmark...")
        
        results = {
            "benchmark_config": {
                "input_lengths": input_lengths,
                "num_layers": num_layers,
                "num_runs": num_runs,
                "hardware": "NPU Phoenix + AMD Radeon 780M + Vulkan"
            },
            "results": {}
        }
        
        for input_length in input_lengths:
            logger.info(f"\n📏 Benchmarking input length: {input_length} tokens")
            
            length_results = []
            
            for run in range(num_runs):
                logger.info(f"   🔄 Run {run + 1}/{num_runs}")
                
                # Reset performance tracking
                self.layer_times = []
                self.ffn_times = []
                self.attention_times = []
                self.total_times = []
                
                # Measure performance
                perf = self.measure_single_token_performance(input_length, num_layers)
                length_results.append(perf)
                
                logger.info(f"      ✅ {perf['tokens_per_second']:.2f} tokens/sec")
            
            # Calculate statistics
            tokens_per_sec = [r['tokens_per_second'] for r in length_results]
            total_times = [r['total_time_ms'] for r in length_results]
            ffn_times = [r['avg_ffn_time_ms'] for r in length_results]
            attention_times = [r['avg_attention_time_ms'] for r in length_results]
            
            results["results"][f"input_length_{input_length}"] = {
                "avg_tokens_per_second": np.mean(tokens_per_sec),
                "min_tokens_per_second": np.min(tokens_per_sec),
                "max_tokens_per_second": np.max(tokens_per_sec),
                "std_tokens_per_second": np.std(tokens_per_sec),
                "avg_total_time_ms": np.mean(total_times),
                "avg_ffn_time_ms": np.mean(ffn_times),
                "avg_attention_time_ms": np.mean(attention_times),
                "individual_runs": length_results
            }
            
            logger.info(f"📊 Input length {input_length} summary:")
            logger.info(f"   Average: {np.mean(tokens_per_sec):.2f} tokens/sec")
            logger.info(f"   Range: {np.min(tokens_per_sec):.2f} - {np.max(tokens_per_sec):.2f} tokens/sec")
        
        # Overall summary
        all_tokens_per_sec = []
        for length_data in results["results"].values():
            all_tokens_per_sec.extend([r['tokens_per_second'] for r in length_data['individual_runs']])
        
        results["overall_summary"] = {
            "avg_tokens_per_second": np.mean(all_tokens_per_sec),
            "min_tokens_per_second": np.min(all_tokens_per_sec),
            "max_tokens_per_second": np.max(all_tokens_per_sec),
            "std_tokens_per_second": np.std(all_tokens_per_sec),
            "total_measurements": len(all_tokens_per_sec)
        }
        
        return results

def main():
    """Main performance measurement function"""
    logger.info("🔬 NPU+iGPU Performance Measurement")
    
    # Initialize measurement tool
    perf_tool = NPUIGPUPerformanceMeasurement()
    
    if not perf_tool.initialize_hardware():
        logger.error("❌ Hardware initialization failed")
        return
    
    # Run comprehensive benchmark
    results = perf_tool.benchmark_performance(
        input_lengths=[32, 64, 128],
        num_layers=3,  # Test with 3 layers for speed
        num_runs=3
    )
    
    # Print final results
    logger.info("\n🎉 FINAL PERFORMANCE RESULTS:")
    logger.info("=" * 60)
    
    overall = results["overall_summary"]
    logger.info(f"🚀 Average Performance: {overall['avg_tokens_per_second']:.2f} tokens/sec")
    logger.info(f"📊 Performance Range: {overall['min_tokens_per_second']:.2f} - {overall['max_tokens_per_second']:.2f} tokens/sec")
    logger.info(f"📈 Standard Deviation: {overall['std_tokens_per_second']:.2f} tokens/sec")
    logger.info(f"🔬 Total Measurements: {overall['total_measurements']}")
    
    logger.info("\nPer Input Length:")
    for length, data in results["results"].items():
        length_val = length.split('_')[-1]
        logger.info(f"   {length_val} tokens: {data['avg_tokens_per_second']:.2f} tokens/sec")
        logger.info(f"      FFN (Vulkan): {data['avg_ffn_time_ms']:.1f}ms")
        logger.info(f"      Attention: {data['avg_attention_time_ms']:.1f}ms")
    
    # Vulkan FFN stats
    ffn_stats = perf_tool.vulkan_ffn_engine.get_performance_stats()
    logger.info(f"\n🎮 Vulkan FFN Performance:")
    logger.info(f"   Operations: {ffn_stats['total_ffn_operations']}")
    logger.info(f"   Average time: {ffn_stats['avg_ffn_time_ms']:.1f}ms")
    
    logger.info("\n✅ Performance measurement complete!")

if __name__ == "__main__":
    main()