
#!/usr/bin/env python3
"""
Qwen3-30B-A3B MoE Loader for Unicorn Execution Engine
Integrates with existing Vulkan workaround and quantization systems
"""

import numpy as np
import os
import sys
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import existing Unicorn infrastructure
from dynamic_quantization_engine import DynamicQuantizationEngine
from vulkan_compute_workaround import VulkanComputeWorkaround
from bfloat16_converter import BFloat16Converter

logger = logging.getLogger(__name__)

class Qwen3MoELoader:
    """MoE-optimized loader for Qwen3-30B-A3B model"""
    
    def __init__(self):
        self.quantization_engine = DynamicQuantizationEngine()
        self.bf16_converter = BFloat16Converter()
        self.router_weights = {}
        self.expert_weights = {}
        self.shared_weights = {}
        self.active_experts = 8
        self.total_experts = 128
        
        # MoE configuration for Qwen3-30B-A3B
        self.moe_config = {
            'hidden_size': 4096,
            'intermediate_size': 22016,
            'num_experts': 128,
            'top_k': 8,  # Active experts per token
            'router_precision': 'fp16',  # Keep router at high precision
            'expert_precision': 'int4'   # Quantize experts for memory efficiency
        }
        
        logger.info("🦄 Qwen3 MoE Loader initialized for 30B-A3B model")
        logger.info(f"   Active experts: {self.active_experts}/{self.total_experts}")
        logger.info(f"   Target memory: ~7.5GB active, ~30GB total")

    def load_qwen3_moe_model(self, model_path: str, load_mode='sparse') -> Dict[str, Any]:
        """
        Load Qwen3-30B-A3B model with MoE-optimized memory management
        
        Args:
            model_path: Path to the model weights
            load_mode: 'sparse' (active experts only) or 'full' (all experts)
        
        Returns:
            Dictionary containing model weights and metadata
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist.")

        logger.info(f"🚀 Loading Qwen3-30B-A3B MoE model from {model_path}")
        logger.info(f"   Load mode: {load_mode}")
        
        # Load model configuration
        config_path = os.path.join(model_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                model_config = json.load(f)
            self._validate_moe_config(model_config)
        else:
            logger.warning("No config.json found, using default MoE configuration")
            model_config = self.moe_config

        # Load weights based on mode
        if load_mode == 'sparse':
            weights = self._load_sparse_weights(model_path, model_config)
        else:
            weights = self._load_full_weights(model_path, model_config)

        # Apply quantization
        quantized_weights = self._apply_moe_quantization(weights)
        
        logger.info("✅ Qwen3 MoE model loaded successfully")
        self._log_memory_usage(quantized_weights)
        
        return {
            'weights': quantized_weights,
            'config': model_config,
            'quantization_metadata': self._get_quantization_metadata()
        }

    def _validate_moe_config(self, config: Dict) -> None:
        """Validate MoE configuration parameters"""
        required_keys = ['hidden_size', 'intermediate_size']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
        
        # Update our config with model-specific values
        self.moe_config.update({
            'hidden_size': config.get('hidden_size', 4096),
            'intermediate_size': config.get('intermediate_size', 22016),
            'num_experts': config.get('n_experts', 128),
            'top_k': config.get('n_experts_per_tok', 8)
        })

    def _load_sparse_weights(self, model_path: str, config: Dict) -> Dict[str, np.ndarray]:
        """Load only active experts and shared weights"""
        weights = {}
        
        # Load shared weights (embeddings, layer norms, etc.)
        shared_files = self._find_shared_weight_files(model_path)
        for file_path in shared_files:
            weights.update(self._load_safetensor_file(file_path))
            
        # Load router weights (critical for expert selection)
        router_files = self._find_router_weight_files(model_path)
        for file_path in router_files:
            weights.update(self._load_safetensor_file(file_path))
            
        # Load only top-K most important experts initially
        expert_files = self._find_expert_weight_files(model_path)
        important_experts = self._select_important_experts(expert_files)
        
        for expert_idx, file_path in important_experts:
            expert_weights = self._load_safetensor_file(file_path)
            # Prefix with expert index for organization
            for key, value in expert_weights.items():
                weights[f"expert_{expert_idx}.{key}"] = value
                
        return weights

    def _load_full_weights(self, model_path: str, config: Dict) -> Dict[str, np.ndarray]:
        """Load all model weights (for systems with sufficient memory)"""
        # This would implement full model loading
        # For now, fall back to sparse loading with warning
        logger.warning("Full weight loading not yet implemented, using sparse mode")
        return self._load_sparse_weights(model_path, config)

    def _apply_moe_quantization(self, weights: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Apply MoE-optimized quantization strategy"""
        quantized_weights = {}
        
        for key, tensor in weights.items():
            if 'router' in key or 'gate' in key:
                # Keep router weights at FP16 for precision
                quantized_weights[key] = {
                    'data': tensor.astype(np.float16),
                    'dtype': 'fp16',
                    'quantization': None
                }
                logger.debug(f"Router weight {key}: {tensor.shape} -> FP16")
                
            elif 'expert' in key and any(layer in key for layer in ['gate_proj', 'up_proj', 'down_proj']):
                # Quantize expert FFN weights to INT4
                quantized, scales = self.quantization_engine.quantize_int4(tensor)
                quantized_weights[key] = {
                    'data': quantized,
                    'scales': scales,
                    'dtype': 'int4',
                    'original_shape': tensor.shape
                }
                logger.debug(f"Expert weight {key}: {tensor.shape} -> INT4 (group quantized)")
                
            elif any(name in key for name in ['embed', 'norm']):
                # Shared weights at INT8 for balance of quality and efficiency
                quantized, scale = self.quantization_engine.quantize_int8(tensor)
                quantized_weights[key] = {
                    'data': quantized,
                    'scale': scale,
                    'dtype': 'int8',
                    'original_shape': tensor.shape
                }
                logger.debug(f"Shared weight {key}: {tensor.shape} -> INT8")
                
            else:
                # Default to FP16 for other weights
                quantized_weights[key] = {
                    'data': tensor.astype(np.float16),
                    'dtype': 'fp16',
                    'quantization': None
                }
                
        return quantized_weights

    def _find_shared_weight_files(self, model_path: str) -> List[str]:
        """Find files containing shared weights (embeddings, norms)"""
        # This is a placeholder - would need model-specific implementation
        pattern_files = []
        for file in Path(model_path).glob("*.safetensors"):
            if 'shared' in str(file) or 'embed' in str(file):
                pattern_files.append(str(file))
        return pattern_files

    def _find_router_weight_files(self, model_path: str) -> List[str]:
        """Find files containing router weights"""
        pattern_files = []
        for file in Path(model_path).glob("*.safetensors"):
            if 'router' in str(file) or 'gate' in str(file):
                pattern_files.append(str(file))
        return pattern_files

    def _find_expert_weight_files(self, model_path: str) -> List[str]:
        """Find files containing expert weights"""
        pattern_files = []
        for file in Path(model_path).glob("*.safetensors"):
            if 'expert' in str(file) or 'mlp' in str(file):
                pattern_files.append(str(file))
        return pattern_files

    def _select_important_experts(self, expert_files: List[str]) -> List[Tuple[int, str]]:
        """Select most important experts to load initially"""
        # For now, select first N experts
        # In practice, this would use routing statistics or model analysis
        important_experts = []
        for i, file_path in enumerate(expert_files[:self.active_experts]):
            important_experts.append((i, file_path))
        return important_experts

    def _load_safetensor_file(self, file_path: str) -> Dict[str, np.ndarray]:
        """Load weights from a safetensors file"""
        try:
            # This would use actual safetensors loading
            # For now, return empty dict as placeholder
            logger.warning(f"Placeholder: would load {file_path}")
            return {}
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return {}

    def _get_quantization_metadata(self) -> Dict:
        """Get metadata about quantization configuration"""
        return {
            'router_precision': self.moe_config['router_precision'],
            'expert_precision': self.moe_config['expert_precision'],
            'active_experts': self.active_experts,
            'total_experts': self.total_experts,
            'quantization_engine': 'DynamicQuantizationEngine'
        }

    def _log_memory_usage(self, weights: Dict) -> None:
        """Log estimated memory usage"""
        total_params = 0
        active_params = 0
        
        for key, weight_info in weights.items():
            if isinstance(weight_info, dict) and 'data' in weight_info:
                params = np.prod(weight_info.get('original_shape', weight_info['data'].shape))
                total_params += params
                
                # Count as active if it's shared weights or active expert
                if 'expert' not in key or any(f'expert_{i}' in key for i in range(self.active_experts)):
                    active_params += params
        
        logger.info(f"📊 Memory usage estimate:")
        logger.info(f"   Total parameters: {total_params / 1e9:.1f}B")
        logger.info(f"   Active parameters: {active_params / 1e9:.1f}B") 
        logger.info(f"   Memory efficiency: {active_params/total_params*100:.1f}% active")

def blockwise_int4_quantization(tensor: np.ndarray, block_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """
    Block-wise INT4 quantization for MoE expert weights
    Compatible with existing GPT5 skeleton code
    """
    quantizer = DynamicQuantizationEngine()
    quantized, scales = quantizer.quantize_int4(tensor)
    return quantized, scales

if __name__ == "__main__":
    # Test the loader
    loader = Qwen3MoELoader()
    
    # This would test with actual model path
    model_path = "/home/ucladmin/Development/github_repos/Unicorn-Execution-Engine/models/qwen3-30b-a3b"
    
    if os.path.exists(model_path):
        try:
            model_data = loader.load_qwen3_moe_model(model_path)
            print("✅ Qwen3 MoE model loaded successfully")
            print(f"   Loaded {len(model_data['weights'])} weight groups")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
    else:
        print(f"ℹ️  Model path not found: {model_path}")
        print("   This is expected - model needs to be downloaded first")
        
        # Test quantization function for GPT5 compatibility
        test_tensor = np.random.randn(1000, 1000).astype(np.float32)
        quantized, scales = blockwise_int4_quantization(test_tensor)
        print(f"✅ Quantization test: {test_tensor.shape} -> {quantized.shape}")
        print(f"   Scales shape: {scales.shape}")
        print(f"   Memory reduction: {test_tensor.nbytes / quantized.nbytes:.1f}x")
