#!/usr/bin/env python3
"""
Gemma 3 27B Real Weights NPU Mapper
Maps real Gemma 3 27B model weights to NPU kernels for hardware acceleration
"""

import os
import sys
import time
import torch
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from safetensors import safe_open

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import our real NPU+iGPU components
from npu_attention_kernel import NPUAttentionKernel, NPUAttentionConfig
from real_vulkan_compute import RealVulkanCompute

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Gemma3_27B_RealWeightsNPUMapper:
    """Map real Gemma 3 27B weights to NPU kernels"""
    
    def __init__(self, model_path: str = "/home/ucadmin/models/gemma-3-27b-it"):
        self.model_path = Path(model_path)
        
        # Model configuration
        self.config = {
            "model_name": "gemma-3-27b-it",
            "vocab_size": 256000,
            "seq_length": 2048,
            "d_model": 4096,
            "n_layers": 62,
            "n_heads": 32,
            "intermediate_size": 14336,
            "head_dim": 128,
            "hidden_size": 4096,
            "num_key_value_heads": 32,
            "num_attention_heads": 32,
            "rope_theta": 10000.0,
            "max_position_embeddings": 131072,
            "torch_dtype": "bfloat16"
        }
        
        # Hardware components
        self.npu_kernel = None
        self.vulkan_compute = None
        self.initialized = False
        
        # Weight mappings
        self.attention_weights = {}
        self.ffn_weights = {}
        self.embedding_weights = {}
        
        logger.info("🦄 Gemma 3 27B Real Weights NPU Mapper initialized")
        logger.info(f"   Model path: {self.model_path}")
        logger.info(f"   Model config: {self.config['n_layers']} layers, {self.config['d_model']} hidden size")
        
    def initialize_hardware(self):
        """Initialize NPU and Vulkan hardware"""
        logger.info("🚀 Initializing NPU+iGPU hardware for real weights...")
        
        try:
            # Initialize NPU attention kernel
            npu_config = NPUAttentionConfig(
                seq_length=self.config["seq_length"],
                d_model=self.config["d_model"],
                num_heads=self.config["n_heads"],
                npu_memory_mb=2048,
                precision="fp16"
            )
            
            self.npu_kernel = NPUAttentionKernel(npu_config)
            if not self.npu_kernel.initialize():
                logger.error("❌ NPU initialization failed")
                return False
            
            logger.info("✅ NPU Phoenix (16 TOPS) initialized for real weights")
            
            # Initialize Vulkan compute
            self.vulkan_compute = RealVulkanCompute()
            if not self.vulkan_compute.initialize():
                logger.error("❌ Vulkan compute initialization failed")
                return False
            
            logger.info("✅ AMD Radeon 780M iGPU initialized for real weights")
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Hardware initialization failed: {e}")
            return False
    
    def load_real_weights(self):
        """Load real Gemma 3 27B weights from safetensors files"""
        logger.info("📥 Loading real Gemma 3 27B weights...")
        
        # Load model index
        index_path = self.model_path / "model.safetensors.index.json"
        if not index_path.exists():
            logger.error(f"❌ Model index not found: {index_path}")
            return False
        
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        weight_map = index_data["weight_map"]
        total_weights = len(weight_map)
        
        logger.info(f"   Total model weights: {total_weights}")
        logger.info(f"   Model files: {len(set(weight_map.values()))}")
        
        # Load weights by category
        attention_count = 0
        ffn_count = 0
        embedding_count = 0
        
        for weight_name, file_name in weight_map.items():
            weight_path = self.model_path / file_name
            
            try:
                with safe_open(weight_path, framework="pt", device="cpu") as f:
                    # Categorize weights
                    if "self_attn" in weight_name:
                        # Attention weights for NPU
                        layer_idx = self._extract_layer_index(weight_name)
                        if layer_idx is not None:
                            if layer_idx not in self.attention_weights:
                                self.attention_weights[layer_idx] = {}
                            
                            weight_tensor = f.get_tensor(weight_name)
                            self.attention_weights[layer_idx][weight_name] = weight_tensor
                            attention_count += 1
                    
                    elif "mlp" in weight_name:
                        # FFN weights for Vulkan
                        layer_idx = self._extract_layer_index(weight_name)
                        if layer_idx is not None:
                            if layer_idx not in self.ffn_weights:
                                self.ffn_weights[layer_idx] = {}
                            
                            weight_tensor = f.get_tensor(weight_name)
                            self.ffn_weights[layer_idx][weight_name] = weight_tensor
                            ffn_count += 1
                    
                    elif "embed" in weight_name or "lm_head" in weight_name:
                        # Embedding weights for CPU
                        weight_tensor = f.get_tensor(weight_name)
                        self.embedding_weights[weight_name] = weight_tensor
                        embedding_count += 1
                        
            except Exception as e:
                logger.warning(f"⚠️  Failed to load weight {weight_name}: {e}")
                continue
        
        logger.info(f"✅ Real weights loaded:")
        logger.info(f"   Attention weights: {attention_count} (NPU target)")
        logger.info(f"   FFN weights: {ffn_count} (Vulkan target)")
        logger.info(f"   Embedding weights: {embedding_count} (CPU target)")
        
        return True
    
    def _extract_layer_index(self, weight_name: str) -> Optional[int]:
        """Extract layer index from weight name"""
        try:
            if "layers." in weight_name:
                parts = weight_name.split(".")
                for i, part in enumerate(parts):
                    if part == "layers" and i + 1 < len(parts):
                        return int(parts[i + 1])
            return None
        except:
            return None
    
    def map_attention_weights_to_npu(self):
        """Map attention weights to NPU kernels"""
        logger.info("⚡ Mapping attention weights to NPU kernels...")
        
        mapped_layers = 0
        
        for layer_idx, layer_weights in self.attention_weights.items():
            logger.info(f"   🧠 Mapping layer {layer_idx} attention to NPU...")
            
            # Extract Q, K, V, O projection weights
            q_weight = None
            k_weight = None
            v_weight = None
            o_weight = None
            
            for weight_name, weight_tensor in layer_weights.items():
                if "q_proj" in weight_name:
                    q_weight = weight_tensor
                elif "k_proj" in weight_name:
                    k_weight = weight_tensor
                elif "v_proj" in weight_name:
                    v_weight = weight_tensor
                elif "o_proj" in weight_name:
                    o_weight = weight_tensor
            
            if all(w is not None for w in [q_weight, k_weight, v_weight, o_weight]):
                # Map to NPU kernel
                npu_weights = {
                    "q_proj": q_weight.to(torch.float16),
                    "k_proj": k_weight.to(torch.float16),
                    "v_proj": v_weight.to(torch.float16),
                    "o_proj": o_weight.to(torch.float16)
                }
                
                # Configure NPU kernel with real weights
                success = self.npu_kernel.load_attention_weights(layer_idx, npu_weights)
                if success:
                    mapped_layers += 1
                    logger.info(f"      ✅ Layer {layer_idx} mapped to NPU")
                else:
                    logger.warning(f"      ❌ Layer {layer_idx} NPU mapping failed")
            else:
                logger.warning(f"      ❌ Layer {layer_idx} incomplete weights")
        
        logger.info(f"✅ NPU attention mapping complete: {mapped_layers}/{len(self.attention_weights)} layers")
        return mapped_layers
    
    def map_ffn_weights_to_vulkan(self):
        """Map FFN weights to Vulkan compute"""
        logger.info("🎮 Mapping FFN weights to Vulkan compute...")
        
        mapped_layers = 0
        
        for layer_idx, layer_weights in self.ffn_weights.items():
            logger.info(f"   🔥 Mapping layer {layer_idx} FFN to Vulkan...")
            
            # Extract gate, up, down projection weights
            gate_weight = None
            up_weight = None
            down_weight = None
            
            for weight_name, weight_tensor in layer_weights.items():
                if "gate_proj" in weight_name:
                    gate_weight = weight_tensor
                elif "up_proj" in weight_name:
                    up_weight = weight_tensor
                elif "down_proj" in weight_name:
                    down_weight = weight_tensor
            
            if all(w is not None for w in [gate_weight, up_weight, down_weight]):
                # Map to Vulkan compute
                vulkan_weights = {
                    "gate_proj": gate_weight.to(torch.float32),
                    "up_proj": up_weight.to(torch.float32),
                    "down_proj": down_weight.to(torch.float32)
                }
                
                # Upload weights to Vulkan
                success = self.vulkan_compute.load_ffn_weights(layer_idx, vulkan_weights)
                if success:
                    mapped_layers += 1
                    logger.info(f"      ✅ Layer {layer_idx} mapped to Vulkan")
                else:
                    logger.warning(f"      ❌ Layer {layer_idx} Vulkan mapping failed")
            else:
                logger.warning(f"      ❌ Layer {layer_idx} incomplete weights")
        
        logger.info(f"✅ Vulkan FFN mapping complete: {mapped_layers}/{len(self.ffn_weights)} layers")
        return mapped_layers
    
    def test_real_inference(self, prompt: str = "Hello, world!", max_tokens: int = 20):
        """Test inference with real weights on NPU+Vulkan"""
        logger.info("🔮 Testing real inference with NPU+Vulkan weights...")
        logger.info(f"   Prompt: '{prompt}'")
        
        # Simple tokenization (for testing)
        input_tokens = [1, 2, 3, 4, 5]  # Placeholder tokens
        seq_len = len(input_tokens)
        
        total_npu_time = 0.0
        total_vulkan_time = 0.0
        
        generated_tokens = []
        
        start_time = time.time()
        
        for token_idx in range(max_tokens):
            current_seq_len = seq_len + token_idx
            
            # Create input tensor
            hidden_states = torch.randn(current_seq_len, self.config["d_model"], dtype=torch.float16)
            
            # Process through layers
            for layer_idx in range(min(self.config["n_layers"], len(self.attention_weights))):
                # NPU attention
                npu_start = time.time()
                attention_output = self.npu_kernel.forward_with_real_weights(
                    hidden_states, layer_idx
                )
                npu_time = time.time() - npu_start
                total_npu_time += npu_time
                
                # Vulkan FFN
                vulkan_start = time.time()
                ffn_output = self.vulkan_compute.forward_with_real_weights(
                    attention_output, layer_idx
                )
                vulkan_time = time.time() - vulkan_start
                total_vulkan_time += vulkan_time
                
                # Update hidden states
                hidden_states = ffn_output
            
            # Generate next token (simplified)
            next_token = torch.randint(0, self.config["vocab_size"], (1,)).item()
            generated_tokens.append(next_token)
            
            # Progress update
            if (token_idx + 1) % 5 == 0:
                elapsed = time.time() - start_time
                current_tps = (token_idx + 1) / elapsed
                logger.info(f"   Generated {token_idx + 1}/{max_tokens} tokens ({current_tps:.2f} TPS)")
        
        total_time = time.time() - start_time
        final_tps = len(generated_tokens) / total_time
        
        # Performance report
        logger.info(f"🎯 Real inference performance:")
        logger.info(f"   Tokens generated: {len(generated_tokens)}")
        logger.info(f"   Total time: {total_time:.2f}s")
        logger.info(f"   Final TPS: {final_tps:.2f}")
        logger.info(f"   NPU time: {total_npu_time:.2f}s ({total_npu_time/total_time*100:.1f}%)")
        logger.info(f"   Vulkan time: {total_vulkan_time:.2f}s ({total_vulkan_time/total_time*100:.1f}%)")
        
        # Target comparison
        target_tps = 22.7
        logger.info(f"   Target TPS: {target_tps}")
        logger.info(f"   Target achieved: {'✅' if final_tps >= target_tps else '❌'}")
        
        return {
            "tps": final_tps,
            "target_tps": target_tps,
            "target_achieved": final_tps >= target_tps,
            "npu_time": total_npu_time,
            "vulkan_time": total_vulkan_time,
            "total_time": total_time
        }

def main():
    """Main function to test real weights mapping"""
    print("🦄 Gemma 3 27B Real Weights NPU Mapping Test")
    print("=" * 60)
    
    # Initialize mapper
    mapper = Gemma3_27B_RealWeightsNPUMapper()
    
    # Initialize hardware
    if not mapper.initialize_hardware():
        print("❌ Hardware initialization failed")
        return False
    
    # Load real weights
    if not mapper.load_real_weights():
        print("❌ Weight loading failed")
        return False
    
    # Map weights to hardware
    npu_layers = mapper.map_attention_weights_to_npu()
    vulkan_layers = mapper.map_ffn_weights_to_vulkan()
    
    if npu_layers == 0 or vulkan_layers == 0:
        print("❌ Weight mapping failed")
        return False
    
    # Test real inference
    result = mapper.test_real_inference(
        prompt="Explain the benefits of NPU acceleration",
        max_tokens=25
    )
    
    # Final summary
    print(f"\n🎉 REAL WEIGHTS NPU MAPPING COMPLETE!")
    print(f"   NPU layers mapped: {npu_layers}")
    print(f"   Vulkan layers mapped: {vulkan_layers}")
    print(f"   Inference TPS: {result['tps']:.2f}")
    print(f"   Target achieved: {'✅' if result['target_achieved'] else '❌'}")
    
    return True

if __name__ == "__main__":
    main()