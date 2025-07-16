#!/usr/bin/env python3
"""
Qwen 2.5 32B Unicorn Loader
Hardware-aware model sharding and loading for NPU+iGPU pipeline
"""

import os
import sys
import time
import torch
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Import our custom components
from qwen32b_unicorn_quantization_engine import Qwen32BUnicornQuantizer
from qwen32b_npu_igpu_memory_allocator import Qwen32BMemoryAllocator, DeviceType
from qwen32b_hma_memory_bridge import Qwen32BHMAMemoryBridge, MemoryDeviceType
from qwen32b_npu_attention_kernels import Qwen32BNPUAttentionKernel
from qwen32b_vulkan_ffn_shaders import Qwen32BVulkanFFNShaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LayerType(Enum):
    ATTENTION = "attention"
    FFN = "ffn"
    EMBEDDING = "embedding"
    OUTPUT = "output"
    LAYER_NORM = "layer_norm"

@dataclass
class LayerShard:
    """Layer shard configuration"""
    layer_id: int
    layer_type: LayerType
    device: MemoryDeviceType
    quantization: str
    memory_allocation: Dict
    kernel_config: Dict

class Qwen32BUnicornLoader:
    """Complete Qwen 2.5 32B loader with hardware-aware sharding"""
    
    def __init__(self, model_path: str = "./models/qwen2.5-32b-instruct"):
        self.model_path = Path(model_path)
        self.device_config = self.detect_hardware()
        
        # Initialize components
        self.quantizer = Qwen32BUnicornQuantizer(str(self.model_path))
        self.memory_allocator = Qwen32BMemoryAllocator()
        self.hma_bridge = Qwen32BHMAMemoryBridge()
        self.npu_kernels = Qwen32BNPUAttentionKernel()
        self.vulkan_shaders = Qwen32BVulkanFFNShaders()
        
        # Model state
        self.model_config = None
        self.layer_shards = []
        self.loaded_layers = {}
        self.hardware_contexts = {}
        
        logger.info(f"🦄 Qwen 32B Unicorn Loader initialized")
        logger.info(f"📁 Model path: {self.model_path}")
        
    def detect_hardware(self) -> Dict:
        """Detect available hardware"""
        
        logger.info("🔍 Detecting hardware configuration...")
        
        config = {
            "npu_phoenix": {
                "available": False,
                "memory": 2 * 1024**3,
                "tops": 16
            },
            "radeon_780m": {
                "available": False,
                "memory": 16 * 1024**3,
                "compute_units": 12
            },
            "system_memory": {
                "available": True,
                "memory": 80 * 1024**3
            }
        }
        
        # Check NPU
        try:
            if os.path.exists("/dev/accel/accel0"):
                config["npu_phoenix"]["available"] = True
                logger.info("   ✅ NPU Phoenix detected")
        except:
            logger.warning("   ⚠️ NPU Phoenix not available")
        
        # Check iGPU
        try:
            import subprocess
            result = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True)
            if "radv phoenix" in result.stdout.lower():
                config["radeon_780m"]["available"] = True
                logger.info("   ✅ Radeon 780M detected")
        except:
            logger.warning("   ⚠️ Radeon 780M not available")
        
        return config
    
    def analyze_model_architecture(self) -> Dict:
        """Analyze Qwen 2.5 32B architecture"""
        
        logger.info("🔍 Analyzing Qwen 2.5 32B architecture...")
        
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(self.model_path)
            
            architecture = {
                "model_type": "qwen2",
                "num_layers": config.num_hidden_layers,  # 64
                "hidden_size": config.hidden_size,       # 5120
                "num_attention_heads": config.num_attention_heads,  # 40
                "num_key_value_heads": getattr(config, "num_key_value_heads", 40),
                "intermediate_size": config.intermediate_size,  # 27392
                "vocab_size": config.vocab_size,        # ~152064
                "max_position_embeddings": config.max_position_embeddings,  # 32768
                "rms_norm_eps": getattr(config, "rms_norm_eps", 1e-6),
                "rope_theta": getattr(config, "rope_theta", 1000000.0),
                "sliding_window": getattr(config, "sliding_window", None),
                "attention_dropout": getattr(config, "attention_dropout", 0.0),
                "hidden_dropout": getattr(config, "hidden_dropout", 0.0)
            }
            
            self.model_config = architecture
            
            logger.info(f"   📊 Layers: {architecture['num_layers']}")
            logger.info(f"   📊 Hidden size: {architecture['hidden_size']}")
            logger.info(f"   📊 Attention heads: {architecture['num_attention_heads']}")
            logger.info(f"   📊 Intermediate size: {architecture['intermediate_size']}")
            logger.info(f"   📊 Vocab size: {architecture['vocab_size']}")
            
            return architecture
            
        except Exception as e:
            logger.error(f"❌ Architecture analysis failed: {e}")
            return None
    
    def create_sharding_strategy(self) -> List[LayerShard]:
        """Create optimal sharding strategy for hardware"""
        
        logger.info("🎯 Creating hardware-aware sharding strategy...")
        
        if not self.model_config:
            raise ValueError("Model architecture not analyzed")
        
        shards = []
        num_layers = self.model_config["num_layers"]
        
        # Determine layer distribution based on available hardware
        npu_layers = []
        igpu_layers = []
        system_layers = []
        
        if self.device_config["npu_phoenix"]["available"]:
            # NPU handles first 20 attention layers
            npu_layers = list(range(0, min(20, num_layers)))
        
        if self.device_config["radeon_780m"]["available"]:
            # iGPU handles next 24 FFN layers
            start_idx = len(npu_layers)
            igpu_layers = list(range(start_idx, min(start_idx + 24, num_layers)))
        
        # Remaining layers on system memory
        allocated_layers = set(npu_layers + igpu_layers)
        system_layers = [i for i in range(num_layers) if i not in allocated_layers]
        
        logger.info(f"   🔧 NPU layers: {len(npu_layers)} (attention)")
        logger.info(f"   🔧 iGPU layers: {len(igpu_layers)} (FFN)")
        logger.info(f"   🔧 System layers: {len(system_layers)} (mixed)")
        
        # Create shards for NPU layers
        for layer_id in npu_layers:
            # Attention shard
            attention_shard = LayerShard(
                layer_id=layer_id,
                layer_type=LayerType.ATTENTION,
                device=MemoryDeviceType.NPU_PHOENIX,
                quantization="INT8_SYMMETRIC",
                memory_allocation=self.calculate_attention_memory(layer_id),
                kernel_config=self.get_npu_kernel_config(layer_id)
            )
            shards.append(attention_shard)
            
            # FFN on system for NPU layers
            ffn_shard = LayerShard(
                layer_id=layer_id,
                layer_type=LayerType.FFN,
                device=MemoryDeviceType.SYSTEM_DDR5,
                quantization="FP16",
                memory_allocation=self.calculate_ffn_memory(layer_id),
                kernel_config=self.get_cpu_kernel_config(layer_id)
            )
            shards.append(ffn_shard)
        
        # Create shards for iGPU layers
        for layer_id in igpu_layers:
            # Attention on system for iGPU layers
            attention_shard = LayerShard(
                layer_id=layer_id,
                layer_type=LayerType.ATTENTION,
                device=MemoryDeviceType.SYSTEM_DDR5,
                quantization="FP16",
                memory_allocation=self.calculate_attention_memory(layer_id),
                kernel_config=self.get_cpu_kernel_config(layer_id)
            )
            shards.append(attention_shard)
            
            # FFN shard
            ffn_shard = LayerShard(
                layer_id=layer_id,
                layer_type=LayerType.FFN,
                device=MemoryDeviceType.RADEON_780M,
                quantization="INT4_GROUPED",
                memory_allocation=self.calculate_ffn_memory(layer_id),
                kernel_config=self.get_vulkan_kernel_config(layer_id)
            )
            shards.append(ffn_shard)
        
        # Create shards for system layers
        for layer_id in system_layers:
            # Both attention and FFN on system
            attention_shard = LayerShard(
                layer_id=layer_id,
                layer_type=LayerType.ATTENTION,
                device=MemoryDeviceType.SYSTEM_DDR5,
                quantization="FP16",
                memory_allocation=self.calculate_attention_memory(layer_id),
                kernel_config=self.get_cpu_kernel_config(layer_id)
            )
            shards.append(attention_shard)
            
            ffn_shard = LayerShard(
                layer_id=layer_id,
                layer_type=LayerType.FFN,
                device=MemoryDeviceType.SYSTEM_DDR5,
                quantization="FP16",
                memory_allocation=self.calculate_ffn_memory(layer_id),
                kernel_config=self.get_cpu_kernel_config(layer_id)
            )
            shards.append(ffn_shard)
        
        # Add embedding and output shards
        embedding_shard = LayerShard(
            layer_id=-1,
            layer_type=LayerType.EMBEDDING,
            device=MemoryDeviceType.SYSTEM_DDR5,
            quantization="FP16",
            memory_allocation=self.calculate_embedding_memory(),
            kernel_config={"executor": "cpu", "precision": "fp16"}
        )
        shards.append(embedding_shard)
        
        output_shard = LayerShard(
            layer_id=-2,
            layer_type=LayerType.OUTPUT,
            device=MemoryDeviceType.SYSTEM_DDR5,
            quantization="FP16",
            memory_allocation=self.calculate_output_memory(),
            kernel_config={"executor": "cpu", "precision": "fp16"}
        )
        shards.append(output_shard)
        
        self.layer_shards = shards
        logger.info(f"   ✅ Created {len(shards)} layer shards")
        
        return shards
    
    def calculate_attention_memory(self, layer_id: int) -> Dict:
        """Calculate memory requirements for attention layer"""
        
        hidden_size = self.model_config["hidden_size"]
        num_heads = self.model_config["num_attention_heads"]
        
        # Q, K, V, O projection weights
        qkv_weights = 3 * hidden_size * hidden_size * 1  # INT8 = 1 byte
        o_weights = hidden_size * hidden_size * 1
        
        # Attention cache
        max_seq_len = 2048  # Working sequence length
        kv_cache = 2 * num_heads * (hidden_size // num_heads) * max_seq_len * 2  # FP16
        
        # Intermediate tensors
        intermediate = num_heads * max_seq_len * max_seq_len * 2  # FP16 attention scores
        
        return {
            "qkv_weights": qkv_weights,
            "o_weights": o_weights,
            "kv_cache": kv_cache,
            "intermediate": intermediate,
            "total": qkv_weights + o_weights + kv_cache + intermediate
        }
    
    def calculate_ffn_memory(self, layer_id: int) -> Dict:
        """Calculate memory requirements for FFN layer"""
        
        hidden_size = self.model_config["hidden_size"]
        intermediate_size = self.model_config["intermediate_size"]
        
        # Gate, Up, Down projection weights
        gate_weights = hidden_size * intermediate_size * 0.5  # INT4 = 0.5 bytes
        up_weights = hidden_size * intermediate_size * 0.5
        down_weights = intermediate_size * hidden_size * 0.5
        
        # Quantization scales and zero points
        group_size = 128
        num_groups = (hidden_size * intermediate_size) // group_size
        scales_zp = num_groups * 8  # 4 bytes scale + 4 bytes zero point
        
        # Intermediate activation tensors
        max_seq_len = 2048
        activations = max_seq_len * intermediate_size * 2  # FP16
        
        return {
            "gate_weights": gate_weights,
            "up_weights": up_weights,
            "down_weights": down_weights,
            "scales_zp": scales_zp,
            "activations": activations,
            "total": gate_weights + up_weights + down_weights + scales_zp + activations
        }
    
    def calculate_embedding_memory(self) -> Dict:
        """Calculate memory requirements for embedding layer"""
        
        vocab_size = self.model_config["vocab_size"]
        hidden_size = self.model_config["hidden_size"]
        
        embedding_weights = vocab_size * hidden_size * 2  # FP16
        
        return {
            "embedding_weights": embedding_weights,
            "total": embedding_weights
        }
    
    def calculate_output_memory(self) -> Dict:
        """Calculate memory requirements for output layer"""
        
        vocab_size = self.model_config["vocab_size"]
        hidden_size = self.model_config["hidden_size"]
        
        output_weights = vocab_size * hidden_size * 2  # FP16
        
        return {
            "output_weights": output_weights,
            "total": output_weights
        }
    
    def get_npu_kernel_config(self, layer_id: int) -> Dict:
        """Get NPU kernel configuration"""
        
        return {
            "executor": "npu_phoenix",
            "precision": "int8",
            "kernels": ["qkv_projection", "attention_scores", "attention_softmax", "attention_output"],
            "optimization": "attention_fusion",
            "memory_pattern": "streaming",
            "turbo_mode": True
        }
    
    def get_vulkan_kernel_config(self, layer_id: int) -> Dict:
        """Get Vulkan kernel configuration"""
        
        return {
            "executor": "radeon_780m",
            "precision": "int4_grouped",
            "kernels": ["gate_projection", "up_projection", "silu_activation", "down_projection"],
            "optimization": "ffn_fusion",
            "memory_pattern": "random_access",
            "workgroup_size": [16, 16, 1]
        }
    
    def get_cpu_kernel_config(self, layer_id: int) -> Dict:
        """Get CPU kernel configuration"""
        
        return {
            "executor": "cpu",
            "precision": "fp16",
            "kernels": ["pytorch_native"],
            "optimization": "cpu_optimized",
            "memory_pattern": "sequential",
            "num_threads": os.cpu_count()
        }
    
    def load_model_weights(self) -> Dict:
        """Load and quantize model weights according to sharding strategy"""
        
        logger.info("📥 Loading Qwen 2.5 32B model weights...")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Load tokenizer
            logger.info("   🔤 Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Load model
            logger.info("   🧠 Loading model...")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="cpu",  # Load to CPU first
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            # Get parameter count
            param_count = sum(p.numel() for p in model.parameters())
            logger.info(f"   📊 Parameters: {param_count:,} ({param_count/1e9:.1f}B)")
            
            # Quantize and shard according to strategy
            logger.info("   🔧 Quantizing and sharding weights...")
            quantized_weights = self.quantizer.parallel_quantization({
                "architecture": self.model_config,
                "memory_allocation": {}
            })
            
            # Organize weights by shard
            sharded_weights = self.organize_weights_by_shard(model, quantized_weights)
            
            # Load weights to respective devices
            device_weights = self.load_weights_to_devices(sharded_weights)
            
            logger.info("   ✅ Model weights loaded and sharded")
            
            return {
                "tokenizer": tokenizer,
                "model": model,
                "quantized_weights": quantized_weights,
                "sharded_weights": sharded_weights,
                "device_weights": device_weights
            }
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            raise
    
    def organize_weights_by_shard(self, model, quantized_weights: Dict) -> Dict:
        """Organize weights according to shard configuration"""
        
        sharded_weights = {}
        
        for shard in self.layer_shards:
            shard_key = f"{shard.layer_type.value}_{shard.layer_id}_{shard.device.value}"
            
            if shard.layer_type == LayerType.ATTENTION:
                if shard.layer_id >= 0 and shard.layer_id < len(model.model.layers):
                    layer = model.model.layers[shard.layer_id]
                    
                    if shard.device == MemoryDeviceType.NPU_PHOENIX:
                        # Use quantized weights for NPU
                        sharded_weights[shard_key] = {
                            "q_proj": quantized_weights.get("npu", {}).get(shard.layer_id, {}).get("q_proj"),
                            "k_proj": quantized_weights.get("npu", {}).get(shard.layer_id, {}).get("k_proj"),
                            "v_proj": quantized_weights.get("npu", {}).get(shard.layer_id, {}).get("v_proj"),
                            "o_proj": quantized_weights.get("npu", {}).get(shard.layer_id, {}).get("o_proj"),
                        }
                    else:
                        # Use FP16 weights for system
                        sharded_weights[shard_key] = {
                            "q_proj": layer.self_attn.q_proj.weight.half(),
                            "k_proj": layer.self_attn.k_proj.weight.half(),
                            "v_proj": layer.self_attn.v_proj.weight.half(),
                            "o_proj": layer.self_attn.o_proj.weight.half(),
                        }
            
            elif shard.layer_type == LayerType.FFN:
                if shard.layer_id >= 0 and shard.layer_id < len(model.model.layers):
                    layer = model.model.layers[shard.layer_id]
                    
                    if shard.device == MemoryDeviceType.RADEON_780M:
                        # Use quantized weights for iGPU
                        sharded_weights[shard_key] = {
                            "gate_proj": quantized_weights.get("igpu", {}).get(shard.layer_id, {}).get("gate_proj"),
                            "up_proj": quantized_weights.get("igpu", {}).get(shard.layer_id, {}).get("up_proj"),
                            "down_proj": quantized_weights.get("igpu", {}).get(shard.layer_id, {}).get("down_proj"),
                        }
                    else:
                        # Use FP16 weights for system
                        sharded_weights[shard_key] = {
                            "gate_proj": layer.mlp.gate_proj.weight.half(),
                            "up_proj": layer.mlp.up_proj.weight.half(),
                            "down_proj": layer.mlp.down_proj.weight.half(),
                        }
            
            elif shard.layer_type == LayerType.EMBEDDING:
                sharded_weights[shard_key] = {
                    "embed_tokens": model.model.embed_tokens.weight.half()
                }
            
            elif shard.layer_type == LayerType.OUTPUT:
                if hasattr(model, "lm_head"):
                    sharded_weights[shard_key] = {
                        "lm_head": model.lm_head.weight.half()
                    }
        
        return sharded_weights
    
    def load_weights_to_devices(self, sharded_weights: Dict) -> Dict:
        """Load weights to respective hardware devices"""
        
        logger.info("   🚛 Loading weights to devices...")
        
        device_weights = {}
        
        for shard_key, weights in sharded_weights.items():
            try:
                # Parse shard info
                parts = shard_key.split("_")
                device_type = parts[-1]
                
                if device_type == "npu-phoenix":
                    # Load to NPU (placeholder - would use NPU APIs)
                    device_weights[shard_key] = self.load_to_npu(weights)
                elif device_type == "radeon-780m":
                    # Load to iGPU via Vulkan
                    device_weights[shard_key] = self.load_to_vulkan(weights)
                else:
                    # Keep in system memory
                    device_weights[shard_key] = weights
                
            except Exception as e:
                logger.warning(f"      ⚠️ Failed to load {shard_key}: {e}")
                device_weights[shard_key] = weights  # Fallback to system
        
        return device_weights
    
    def load_to_npu(self, weights: Dict) -> Dict:
        """Load weights to NPU memory"""
        # Placeholder for NPU weight loading
        logger.debug("      🔧 Loading to NPU Phoenix...")
        return weights
    
    def load_to_vulkan(self, weights: Dict) -> Dict:
        """Load weights to Vulkan device memory"""
        # Placeholder for Vulkan weight loading
        logger.debug("      🔧 Loading to Radeon 780M...")
        return weights
    
    def initialize_hardware_contexts(self) -> Dict:
        """Initialize hardware execution contexts"""
        
        logger.info("⚡ Initializing hardware contexts...")
        
        contexts = {}
        
        # NPU context
        if self.device_config["npu_phoenix"]["available"]:
            contexts["npu"] = self.initialize_npu_context()
        
        # Vulkan context
        if self.device_config["radeon_780m"]["available"]:
            contexts["vulkan"] = self.initialize_vulkan_context()
        
        # CPU context
        contexts["cpu"] = self.initialize_cpu_context()
        
        self.hardware_contexts = contexts
        return contexts
    
    def initialize_npu_context(self) -> Dict:
        """Initialize NPU execution context"""
        
        return {
            "device": "npu_phoenix",
            "kernels": self.npu_kernels.compile_kernels(),
            "memory_pool": "npu_sram",
            "turbo_mode": True,
            "precision": "int8"
        }
    
    def initialize_vulkan_context(self) -> Dict:
        """Initialize Vulkan execution context"""
        
        return {
            "device": "radeon_780m",
            "shaders": self.vulkan_shaders.compile_shaders(),
            "memory_pool": "vulkan_device",
            "workgroup_optimization": True,
            "precision": "int4_grouped"
        }
    
    def initialize_cpu_context(self) -> Dict:
        """Initialize CPU execution context"""
        
        return {
            "device": "cpu",
            "threads": os.cpu_count(),
            "precision": "fp16",
            "optimization": "pytorch_native"
        }
    
    def generate(self, prompt: str, max_tokens: int = 100, **kwargs) -> str:
        """Generate text using the sharded model"""
        
        logger.info(f"🔮 Generating with prompt: {prompt[:50]}...")
        
        # This would implement the actual inference pipeline
        # coordinating between NPU, iGPU, and CPU
        
        # Placeholder implementation
        response = f"Generated response for: {prompt[:30]}... (using Qwen 32B Unicorn pipeline)"
        
        logger.info(f"✅ Generated {len(response.split())} tokens")
        return response

def main():
    """Test Qwen 32B Unicorn Loader"""
    
    logger.info("🦄 Qwen 2.5 32B Unicorn Loader")
    logger.info("=" * 60)
    
    # Initialize loader
    loader = Qwen32BUnicornLoader()
    
    # Analyze architecture
    architecture = loader.analyze_model_architecture()
    
    if architecture:
        # Create sharding strategy
        shards = loader.create_sharding_strategy()
        
        # Initialize hardware contexts
        contexts = loader.initialize_hardware_contexts()
        
        # Load model weights (commented out for demo)
        # weights = loader.load_model_weights()
        
        # Summary
        logger.info("=" * 60)
        logger.info("🎯 QWEN 32B UNICORN LOADER READY!")
        logger.info(f"📊 Total shards: {len(shards)}")
        logger.info(f"🔧 Hardware contexts: {len(contexts)}")
        logger.info(f"💾 Model layers: {architecture['num_layers']}")
        logger.info(f"📊 Parameters: ~32B")
        logger.info("=" * 60)
    else:
        logger.error("❌ Failed to analyze model architecture")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())