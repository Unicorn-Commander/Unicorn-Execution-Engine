#!/usr/bin/env python3.13
"""
Magic Unicorn Model Loader
Complete implementation for loading and managing LLM models with safetensors
"""

import torch
import json
import numpy as np
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
import time
from typing import Dict, Optional, List, Tuple, Any
import logging
import hashlib
import shutil
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantizationType(Enum):
    """Supported quantization types"""
    NONE = "none"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"


@dataclass
class ModelConfig:
    """Model configuration dataclass"""
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    max_position_embeddings: int
    rope_theta: float = 10000.0
    layer_norm_epsilon: float = 1e-6
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = False
    quantization: QuantizationType = QuantizationType.NONE
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create ModelConfig from dictionary"""
        # Handle nested text_config for multimodal models
        if 'text_config' in config_dict:
            text_config = config_dict['text_config']
        else:
            text_config = config_dict
            
        return cls(
            model_type=config_dict.get('model_type', 'unknown'),
            hidden_size=text_config['hidden_size'],
            intermediate_size=text_config['intermediate_size'],
            num_hidden_layers=text_config['num_hidden_layers'],
            num_attention_heads=text_config['num_attention_heads'],
            num_key_value_heads=text_config.get('num_key_value_heads', 
                                               text_config['num_attention_heads']),
            vocab_size=text_config['vocab_size'],
            max_position_embeddings=text_config.get('max_position_embeddings', 8192),
            rope_theta=text_config.get('rope_theta', 10000.0),
            layer_norm_epsilon=text_config.get('layer_norm_epsilon', 1e-6),
            use_cache=text_config.get('use_cache', True),
            pad_token_id=text_config.get('pad_token_id', 0),
            bos_token_id=text_config.get('bos_token_id', 1),
            eos_token_id=text_config.get('eos_token_id', 2),
            tie_word_embeddings=text_config.get('tie_word_embeddings', False),
        )


class MagicUnicornModelLoader:
    """Complete model loader with quantization support"""
    
    def __init__(self, cache_dir: str = "~/.cache/unicorn"):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_models = {}
        
        logger.info(f"🦄 Magic Unicorn Model Loader initialized")
        logger.info(f"   Cache directory: {self.cache_dir}")
        
    def load_model(self, 
                   model_path: str,
                   quantization: QuantizationType = QuantizationType.NONE,
                   device: str = "cpu") -> Tuple[ModelConfig, Dict[str, torch.Tensor]]:
        """Load model with optional quantization"""
        model_path = Path(model_path)
        
        # Generate cache key
        cache_key = self._get_cache_key(model_path, quantization)
        
        # Check if already loaded
        if cache_key in self.loaded_models:
            logger.info(f"Model already loaded from cache: {cache_key}")
            return self.loaded_models[cache_key]
            
        logger.info(f"Loading model: {model_path}")
        logger.info(f"Quantization: {quantization.value}")
        
        # Load configuration
        config = self._load_config(model_path)
        config.quantization = quantization
        
        # Check for quantized version in cache
        quantized_path = self.cache_dir / f"{cache_key}.safetensors"
        if quantized_path.exists() and quantization != QuantizationType.NONE:
            logger.info(f"Loading pre-quantized model from cache")
            weights = self._load_safetensors(quantized_path, device)
        else:
            # Load original weights
            weights = self._load_weights(model_path, config, device)
            
            # Apply quantization if requested
            if quantization != QuantizationType.NONE:
                logger.info(f"Applying {quantization.value} quantization...")
                weights = self._quantize_weights(weights, quantization)
                
                # Save quantized version to cache
                if device == "cpu":  # Only cache CPU tensors
                    logger.info(f"Saving quantized model to cache")
                    save_file(weights, quantized_path)
                    
        # Store in memory cache
        self.loaded_models[cache_key] = (config, weights)
        
        # Print model info
        self._print_model_info(config, weights)
        
        return config, weights
        
    def _load_config(self, model_path: Path) -> ModelConfig:
        """Load model configuration"""
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
            
        return ModelConfig.from_dict(config_dict)
        
    def _load_weights(self, model_path: Path, config: ModelConfig, device: str) -> Dict[str, torch.Tensor]:
        """Load model weights from safetensors files"""
        # Check for index file (sharded model)
        index_path = model_path / "model.safetensors.index.json"
        
        if index_path.exists():
            return self._load_sharded_weights(model_path, index_path, device)
        else:
            # Single file model
            model_file = model_path / "model.safetensors"
            if not model_file.exists():
                # Try pytorch_model.bin as fallback
                model_file = model_path / "pytorch_model.bin"
                if model_file.exists():
                    logger.warning("Loading from pytorch_model.bin (safetensors preferred)")
                    return torch.load(model_file, map_location=device)
                else:
                    raise FileNotFoundError(f"Model weights not found in {model_path}")
                    
            return self._load_safetensors(model_file, device)
            
    def _load_sharded_weights(self, model_path: Path, index_path: Path, device: str) -> Dict[str, torch.Tensor]:
        """Load sharded model weights"""
        with open(index_path, 'r') as f:
            index = json.load(f)
            
        weight_map = index['weight_map']
        weights = {}
        loaded_files = set()
        
        # Group weights by file
        file_to_weights = {}
        for weight_name, file_name in weight_map.items():
            if file_name not in file_to_weights:
                file_to_weights[file_name] = []
            file_to_weights[file_name].append(weight_name)
            
        # Load each file
        for file_name, weight_names in file_to_weights.items():
            if file_name in loaded_files:
                continue
                
            file_path = model_path / file_name
            logger.info(f"Loading shard: {file_name}")
            
            with safe_open(file_path, framework="pt", device=device) as f:
                for weight_name in weight_names:
                    weights[weight_name] = f.get_tensor(weight_name)
                    
            loaded_files.add(file_name)
            
        return weights
        
    def _load_safetensors(self, file_path: Path, device: str) -> Dict[str, torch.Tensor]:
        """Load weights from a single safetensors file"""
        weights = {}
        with safe_open(file_path, framework="pt", device=device) as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
        return weights
        
    def _quantize_weights(self, weights: Dict[str, torch.Tensor], 
                         quantization: QuantizationType) -> Dict[str, torch.Tensor]:
        """Quantize model weights"""
        quantized = {}
        
        for name, tensor in weights.items():
            # Skip non-weight tensors
            if 'weight' not in name or tensor.ndim < 2:
                quantized[name] = tensor
                continue
                
            if quantization == QuantizationType.FP16:
                quantized[name] = tensor.half()
                
            elif quantization == QuantizationType.INT8:
                # Symmetric INT8 quantization
                scale = tensor.abs().max() / 127.0
                quantized_tensor = torch.round(tensor / scale).clamp(-128, 127).to(torch.int8)
                
                # Store scale for dequantization
                quantized[name] = quantized_tensor
                quantized[f"{name}_scale"] = scale
                
            elif quantization == QuantizationType.INT4:
                # Symmetric INT4 quantization with packing
                scale = tensor.abs().max() / 7.0
                quantized_tensor = torch.round(tensor / scale).clamp(-8, 7).to(torch.int8)
                
                # Pack INT4 values (2 per byte)
                if tensor.shape[-1] % 2 == 0:
                    # Reshape to ensure even number of elements
                    reshaped = quantized_tensor.reshape(-1, 2)
                    # Pack: first value in lower 4 bits, second in upper 4 bits
                    packed = ((reshaped[:, 0] & 0xF) | ((reshaped[:, 1] & 0xF) << 4)).to(torch.uint8)
                    packed = packed.reshape(*tensor.shape[:-1], -1)
                    quantized[name] = packed
                else:
                    # Fallback to INT8 for odd dimensions
                    logger.warning(f"Cannot pack {name} to INT4 (odd dimension), using INT8")
                    quantized[name] = quantized_tensor
                    
                quantized[f"{name}_scale"] = scale
                quantized[f"{name}_packed"] = torch.tensor(True)  # Mark as packed
                
        return quantized
        
    def dequantize_weight(self, weight_name: str, weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Dequantize a weight tensor for use"""
        tensor = weights[weight_name]
        
        # Check if it's quantized
        scale_name = f"{weight_name}_scale"
        if scale_name not in weights:
            return tensor  # Not quantized
            
        scale = weights[scale_name]
        
        # Check if INT4 packed
        if f"{weight_name}_packed" in weights and weights[f"{weight_name}_packed"]:
            # Unpack INT4
            unpacked_shape = list(tensor.shape)
            unpacked_shape[-1] *= 2
            
            unpacked = torch.zeros(unpacked_shape, dtype=torch.int8, device=tensor.device)
            flat_packed = tensor.flatten()
            flat_unpacked = unpacked.flatten()
            
            # Extract lower and upper 4 bits
            flat_unpacked[0::2] = (flat_packed & 0xF) - 8  # Sign extend
            flat_unpacked[1::2] = ((flat_packed >> 4) & 0xF) - 8  # Sign extend
            
            unpacked = flat_unpacked.reshape(unpacked_shape)
            return unpacked.float() * scale
        else:
            # INT8 dequantization
            return tensor.float() * scale
            
    def prepare_for_inference(self, config: ModelConfig, 
                            weights: Dict[str, torch.Tensor],
                            layer_idx: int) -> Dict[str, torch.Tensor]:
        """Prepare layer weights for inference"""
        layer_weights = {}
        prefix = f"language_model.layers.{layer_idx}." if "language_model.layers.0" in weights else f"layers.{layer_idx}."
        
        # Standard transformer weight names
        weight_map = {
            'q_proj': f'{prefix}self_attn.q_proj.weight',
            'k_proj': f'{prefix}self_attn.k_proj.weight',
            'v_proj': f'{prefix}self_attn.v_proj.weight',
            'o_proj': f'{prefix}self_attn.o_proj.weight',
            'gate_proj': f'{prefix}mlp.gate_proj.weight',
            'up_proj': f'{prefix}mlp.up_proj.weight',
            'down_proj': f'{prefix}mlp.down_proj.weight',
        }
        
        for key, weight_name in weight_map.items():
            if weight_name in weights:
                # Dequantize if needed
                if config.quantization != QuantizationType.NONE:
                    layer_weights[key] = self.dequantize_weight(weight_name, weights)
                else:
                    layer_weights[key] = weights[weight_name]
            else:
                logger.warning(f"Weight {weight_name} not found")
                
        return layer_weights
        
    def _get_cache_key(self, model_path: Path, quantization: QuantizationType) -> str:
        """Generate cache key for model"""
        # Use model path and quantization type
        key_str = f"{model_path.name}_{quantization.value}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
        
    def _print_model_info(self, config: ModelConfig, weights: Dict[str, torch.Tensor]):
        """Print model information"""
        # Calculate parameter count
        total_params = 0
        quantized_params = 0
        
        for name, tensor in weights.items():
            if 'weight' in name and '_scale' not in name and '_packed' not in name:
                params = tensor.numel()
                if f"{name}_scale" in weights:
                    # Quantized weight
                    if config.quantization == QuantizationType.INT4:
                        params = params * 4 / 8  # 4 bits per param
                    elif config.quantization == QuantizationType.INT8:
                        params = params * 8 / 32  # 8 bits per param
                    quantized_params += params
                else:
                    total_params += params
                    
        total_params += quantized_params
        
        print(f"\n📊 MODEL INFORMATION:")
        print(f"   Type: {config.model_type}")
        print(f"   Hidden size: {config.hidden_size}")
        print(f"   Layers: {config.num_hidden_layers}")
        print(f"   Parameters: {total_params/1e9:.2f}B")
        
        if config.quantization != QuantizationType.NONE:
            print(f"   Quantization: {config.quantization.value}")
            
            # Calculate size reduction
            if config.quantization == QuantizationType.FP16:
                reduction = 2.0
            elif config.quantization == QuantizationType.INT8:
                reduction = 4.0
            elif config.quantization == QuantizationType.INT4:
                reduction = 8.0
            else:
                reduction = 1.0
                
            print(f"   Size reduction: {reduction:.1f}x")
            
    def clear_cache(self):
        """Clear the model cache"""
        logger.info("Clearing model cache...")
        
        # Clear memory cache
        self.loaded_models.clear()
        
        # Clear disk cache
        for file in self.cache_dir.glob("*.safetensors"):
            file.unlink()
            
        logger.info("Cache cleared")


def test_model_loader():
    """Test the model loader"""
    print("🦄 Testing Magic Unicorn Model Loader")
    print("=" * 60)
    
    loader = MagicUnicornModelLoader()
    
    # Test with dummy model
    test_model_path = Path("models/test_model")
    
    if not test_model_path.exists():
        print("Creating test model...")
        test_model_path.mkdir(parents=True, exist_ok=True)
        
        # Create dummy config
        config = {
            "model_type": "gemma",
            "hidden_size": 2048,
            "intermediate_size": 16384,
            "num_hidden_layers": 18,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "vocab_size": 256000,
            "max_position_embeddings": 8192
        }
        
        with open(test_model_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
            
        # Create dummy weights
        weights = {}
        for i in range(18):
            prefix = f"layers.{i}."
            weights[f"{prefix}self_attn.q_proj.weight"] = torch.randn(2048, 2048)
            weights[f"{prefix}self_attn.k_proj.weight"] = torch.randn(512, 2048)
            weights[f"{prefix}self_attn.v_proj.weight"] = torch.randn(512, 2048)
            weights[f"{prefix}self_attn.o_proj.weight"] = torch.randn(2048, 2048)
            weights[f"{prefix}mlp.gate_proj.weight"] = torch.randn(16384, 2048)
            weights[f"{prefix}mlp.up_proj.weight"] = torch.randn(16384, 2048)
            weights[f"{prefix}mlp.down_proj.weight"] = torch.randn(2048, 16384)
            
        save_file(weights, test_model_path / "model.safetensors")
        print("Test model created")
        
    # Test different quantization types
    for quant in [QuantizationType.NONE, QuantizationType.FP16, 
                  QuantizationType.INT8, QuantizationType.INT4]:
        print(f"\n{'='*60}")
        print(f"Testing {quant.value} quantization")
        print(f"{'='*60}")
        
        start_time = time.time()
        config, weights = loader.load_model(test_model_path, quantization=quant)
        load_time = time.time() - start_time
        
        print(f"   Load time: {load_time:.2f}s")
        
        # Test inference preparation
        layer_weights = loader.prepare_for_inference(config, weights, layer_idx=0)
        print(f"   Layer 0 weights prepared: {list(layer_weights.keys())}")
        
        # Test dequantization speed
        if quant != QuantizationType.NONE:
            start_time = time.time()
            for _ in range(10):
                _ = loader.prepare_for_inference(config, weights, layer_idx=0)
            dequant_time = (time.time() - start_time) / 10
            print(f"   Dequantization time: {dequant_time*1000:.1f}ms per layer")
            
    print("\n✅ Model loader test complete!")


if __name__ == "__main__":
    test_model_loader()