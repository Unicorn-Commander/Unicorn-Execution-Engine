#!/usr/bin/env python3
"""
Direct Safetensors Loader for Gemma3n E2B
Bypass transformers library limitations by loading safetensors directly
Extract real model weights for NPU acceleration
"""

import os
import sys
import json
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DirectSafetensorsLoader:
    """
    Direct loader for Gemma3n E2B model from safetensors files
    Bypasses transformers architecture limitations
    """
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.config = None
        self.tokenizer_config = None
        self.model_weights = {}
        self.safetensors_files = []
        
        # Check if model path exists
        if not self.model_path.exists():
            raise ValueError(f"Model path not found: {model_path}")
        
        logger.info(f"Initializing direct loader for: {self.model_path}")
        
    def load_config(self) -> Dict[str, Any]:
        """Load model configuration"""
        try:
            config_path = self.model_path / "config.json"
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            
            logger.info(f"✅ Model config loaded: {self.config.get('model_type', 'unknown')}")
            
            # Extract text config (where the important parameters are)
            if 'text_config' in self.config:
                text_config = self.config['text_config']
                logger.info(f"📊 Model Architecture:")
                logger.info(f"  Hidden size: {text_config.get('hidden_size', '?')}")
                logger.info(f"  Layers: {text_config.get('num_hidden_layers', '?')}")
                logger.info(f"  Attention heads: {text_config.get('num_attention_heads', '?')}")
                logger.info(f"  KV heads: {text_config.get('num_key_value_heads', '?')}")
                logger.info(f"  Vocab size: {text_config.get('vocab_size', '?')}")
                
                # Show sparsity pattern
                sparsity = text_config.get('activation_sparsity_pattern', [])
                if sparsity:
                    sparse_layers = [i for i, s in enumerate(sparsity) if s > 0]
                    dense_layers = [i for i, s in enumerate(sparsity) if s == 0]
                    logger.info(f"  Sparse layers (95%): {sparse_layers}")
                    logger.info(f"  Dense layers (0%): {dense_layers}")
            
            return self.config
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def load_tokenizer_info(self) -> Dict[str, Any]:
        """Load tokenizer configuration"""
        try:
            tokenizer_path = self.model_path / "tokenizer_config.json"
            if tokenizer_path.exists():
                with open(tokenizer_path, 'r') as f:
                    self.tokenizer_config = json.load(f)
                logger.info(f"✅ Tokenizer config loaded")
            else:
                logger.warning("No tokenizer config found")
                
            return self.tokenizer_config
            
        except Exception as e:
            logger.warning(f"Failed to load tokenizer config: {e}")
            return {}
    
    def find_safetensors_files(self) -> List[Path]:
        """Find all safetensors files in the model directory"""
        safetensors_files = list(self.model_path.glob("*.safetensors"))
        safetensors_files.sort()  # Ensure consistent ordering
        
        logger.info(f"📁 Found {len(safetensors_files)} safetensors files:")
        for i, file in enumerate(safetensors_files):
            logger.info(f"  {i+1}. {file.name}")
        
        self.safetensors_files = safetensors_files
        return safetensors_files
    
    def load_safetensors_file(self, file_path: Path) -> Dict[str, np.ndarray]:
        """Load a single safetensors file and return weights as numpy arrays"""
        try:
            # Try to import safetensors
            try:
                from safetensors import safe_open
            except ImportError:
                logger.error("safetensors library not available. Install with: pip install safetensors")
                raise
            
            weights = {}
            
            with safe_open(file_path, framework="pt") as f:  # Use PyTorch framework for bfloat16 support
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    
                    # Convert to numpy, handling bfloat16
                    if hasattr(tensor, 'numpy'):
                        # PyTorch tensor
                        import torch
                        if tensor.dtype == torch.bfloat16:
                            # Convert bfloat16 to float32 for numpy compatibility
                            tensor_np = tensor.float().numpy()
                        else:
                            tensor_np = tensor.numpy()
                    else:
                        # Already numpy
                        tensor_np = tensor
                    
                    weights[key] = tensor_np
                    logger.debug(f"Loaded {key}: {tensor_np.shape} {tensor_np.dtype}")
            
            logger.info(f"✅ Loaded {len(weights)} tensors from {file_path.name}")
            return weights
            
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            raise
    
    def load_all_weights(self) -> Dict[str, np.ndarray]:
        """Load all model weights from safetensors files"""
        logger.info("📦 Loading all model weights...")
        
        if not self.safetensors_files:
            self.find_safetensors_files()
        
        all_weights = {}
        total_params = 0
        total_size_mb = 0
        
        for file_path in self.safetensors_files:
            file_weights = self.load_safetensors_file(file_path)
            all_weights.update(file_weights)
        
        # Calculate statistics
        for name, tensor in all_weights.items():
            total_params += tensor.size
            total_size_mb += tensor.nbytes / 1024 / 1024
        
        logger.info(f"✅ Total model loaded: {total_params:,} parameters, {total_size_mb:.1f} MB")
        
        self.model_weights = all_weights
        return all_weights
    
    def organize_weights_by_layer(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Organize weights by transformer layer for easier processing"""
        if not self.model_weights:
            self.load_all_weights()
        
        organized = {
            'embeddings': {},
            'layers': {},
            'output': {},
            'other': {}
        }
        
        # Debug: Show first few weight names to understand structure
        weight_names = list(self.model_weights.keys())[:10]
        logger.info(f"Sample weight names: {weight_names}")
        
        # Find language model weights specifically
        language_model_weights = [name for name in self.model_weights.keys() if 'language_model' in name]
        logger.info(f"Language model weights found: {len(language_model_weights)}")
        if language_model_weights:
            logger.info(f"Language model sample: {language_model_weights[:5]}")
            
        # Find transformer layers specifically
        layer_weights = [name for name in self.model_weights.keys() if 'language_model.layers.' in name]
        logger.info(f"Transformer layer weights found: {len(layer_weights)}")
        if layer_weights:
            logger.info(f"Transformer layer sample: {layer_weights[:5]}")
        
        for name, tensor in self.model_weights.items():
            # Focus on language model components
            if 'language_model' not in name:
                organized['other'][name] = tensor
                continue
                
            if 'embed_tokens' in name:
                organized['embeddings'][name] = tensor
            elif 'lm_head' in name:
                organized['output'][name] = tensor
            elif 'language_model.layers.' in name:
                # Extract layer number from model.language_model.layers.N.xxx
                parts = name.split('.')
                if len(parts) >= 5 and parts[3].isdigit():
                    layer_idx = int(parts[3])
                    if layer_idx not in organized['layers']:
                        organized['layers'][layer_idx] = {}
                    organized['layers'][layer_idx][name] = tensor
                else:
                    organized['other'][name] = tensor
            else:
                organized['other'][name] = tensor
        
        logger.info(f"📊 Organized weights:")
        logger.info(f"  Embeddings: {len(organized['embeddings'])} tensors")
        logger.info(f"  Layers: {len(organized['layers'])} layers")
        logger.info(f"  Output: {len(organized['output'])} tensors")
        logger.info(f"  Other: {len(organized['other'])} tensors")
        
        return organized
    
    def extract_layer_weights(self, layer_idx: int) -> Dict[str, np.ndarray]:
        """Extract weights for a specific transformer layer"""
        organized = self.organize_weights_by_layer()
        
        if layer_idx not in organized['layers']:
            raise ValueError(f"Layer {layer_idx} not found in model")
        
        layer_weights = organized['layers'][layer_idx]
        
        # Organize by component type
        extracted = {
            'attention': {},
            'mlp': {},
            'norms': {}
        }
        
        for name, tensor in layer_weights.items():
            if 'self_attn' in name:
                extracted['attention'][name] = tensor
            elif 'mlp' in name:
                extracted['mlp'][name] = tensor
            elif 'norm' in name:
                extracted['norms'][name] = tensor
            else:
                extracted.setdefault('other', {})[name] = tensor
        
        return extracted
    
    def get_sparsity_info(self) -> Dict[int, float]:
        """Get sparsity information for each layer"""
        if not self.config:
            self.load_config()
        
        sparsity_info = {}
        
        if 'text_config' in self.config:
            sparsity_pattern = self.config['text_config'].get('activation_sparsity_pattern', [])
            for i, sparsity in enumerate(sparsity_pattern):
                sparsity_info[i] = sparsity
        
        return sparsity_info
    
    def create_npu_compatible_weights(self) -> Dict[str, Any]:
        """Create NPU-compatible weight format for our acceleration framework"""
        logger.info("🔄 Creating NPU-compatible weight format...")
        
        organized = self.organize_weights_by_layer()
        sparsity_info = self.get_sparsity_info()
        
        npu_weights = {
            'embedding': None,
            'output_projection': None,
            'layers': {}
        }
        
        # Extract embeddings (use standard embedding, not per-layer)
        for name, tensor in organized['embeddings'].items():
            if 'embed_tokens' in name and 'per_layer' not in name:
                npu_weights['embedding'] = tensor.astype(np.float32)
                logger.info(f"✅ Embedding: {tensor.shape}")
                break  # Use first standard embedding found
        
        # Extract output projection (look for lm_head or use embedding weights transposed)
        output_found = False
        for name, tensor in organized['output'].items():
            if 'lm_head' in name:
                # Transpose weight for correct matrix multiplication: (input_dim, output_dim)
                npu_weights['output_projection'] = tensor.astype(np.float32).T
                logger.info(f"✅ Output projection: {tensor.shape} -> {tensor.T.shape}")
                output_found = True
                break
        
        # If no lm_head found, use embedding weights transposed (tied weights)
        if not output_found and npu_weights['embedding'] is not None:
            npu_weights['output_projection'] = npu_weights['embedding'].T
            logger.info(f"✅ Output projection (tied weights): {npu_weights['embedding'].T.shape}")
        
        # Extract layer weights
        for layer_idx, layer_weights in organized['layers'].items():
            layer_info = {
                'sparsity': sparsity_info.get(layer_idx, 0.0),
                'attention': {},
                'mlp': {},
                'norms': {}
            }
            
            for name, tensor in layer_weights.items():
                # Convert to float32 for compatibility
                tensor_f32 = tensor.astype(np.float32)
                
                if 'self_attn.q_proj' in name:
                    # Transpose weight for correct matrix multiplication: (input_dim, output_dim)
                    layer_info['attention']['q_proj'] = tensor_f32.T
                elif 'self_attn.k_proj' in name:
                    # Transpose weight for correct matrix multiplication: (input_dim, output_dim)
                    layer_info['attention']['k_proj'] = tensor_f32.T
                elif 'self_attn.v_proj' in name:
                    # Transpose weight for correct matrix multiplication: (input_dim, output_dim)
                    layer_info['attention']['v_proj'] = tensor_f32.T
                elif 'self_attn.o_proj' in name:
                    # Transpose weight for correct matrix multiplication: (input_dim, output_dim)
                    layer_info['attention']['o_proj'] = tensor_f32.T
                elif 'mlp.gate_proj' in name:
                    # Transpose weight for correct matrix multiplication: (input_dim, output_dim)
                    layer_info['mlp']['gate_proj'] = tensor_f32.T
                elif 'mlp.up_proj' in name:
                    # Transpose weight for correct matrix multiplication: (input_dim, output_dim)
                    layer_info['mlp']['up_proj'] = tensor_f32.T
                elif 'mlp.down_proj' in name:
                    # Transpose weight for correct matrix multiplication: (input_dim, output_dim)
                    layer_info['mlp']['down_proj'] = tensor_f32.T
                elif 'input_layernorm' in name:
                    layer_info['norms']['attn_norm'] = tensor_f32
                elif 'post_attention_layernorm' in name:
                    layer_info['norms']['ffn_norm'] = tensor_f32
            
            npu_weights['layers'][layer_idx] = layer_info
            
            # Log layer info
            sparsity = layer_info['sparsity']
            device = 'NPU' if sparsity > 0 else 'iGPU'
            logger.info(f"Layer {layer_idx:2d}: {sparsity*100:3.0f}% sparse → {device}")
        
        logger.info(f"✅ NPU-compatible weights created for {len(npu_weights['layers'])} layers")
        return npu_weights

def test_direct_safetensors_loader():
    """Test the direct safetensors loader"""
    logger.info("🧪 Testing Direct Safetensors Loader")
    logger.info("=" * 60)
    
    model_path = "/home/ucadmin/Development/AI-Models/gemma-3n-E2B-it"
    
    try:
        # Initialize loader
        loader = DirectSafetensorsLoader(model_path)
        
        # Load configuration
        config = loader.load_config()
        tokenizer_info = loader.load_tokenizer_info()
        
        # Find safetensors files
        files = loader.find_safetensors_files()
        
        # Load all weights
        weights = loader.load_all_weights()
        
        # Test layer extraction
        logger.info("\n🔍 Testing layer weight extraction:")
        for layer_idx in [0, 9, 10, 29]:  # Test sparse and dense layers
            try:
                layer_weights = loader.extract_layer_weights(layer_idx)
                attention_count = len(layer_weights['attention'])
                mlp_count = len(layer_weights['mlp'])
                norm_count = len(layer_weights['norms'])
                logger.info(f"  Layer {layer_idx}: {attention_count} attn + {mlp_count} mlp + {norm_count} norm weights")
            except Exception as e:
                logger.warning(f"  Layer {layer_idx}: {e}")
        
        # Create NPU-compatible format
        npu_weights = loader.create_npu_compatible_weights()
        
        # Show statistics
        logger.info(f"\n📊 NPU Weight Statistics:")
        if npu_weights['embedding'] is not None:
            logger.info(f"  Embedding: {npu_weights['embedding'].shape}")
        if npu_weights['output_projection'] is not None:
            logger.info(f"  Output: {npu_weights['output_projection'].shape}")
        
        sparse_layers = [i for i, info in npu_weights['layers'].items() if info['sparsity'] > 0]
        dense_layers = [i for i, info in npu_weights['layers'].items() if info['sparsity'] == 0]
        
        logger.info(f"  Sparse layers (NPU): {len(sparse_layers)} layers")
        logger.info(f"  Dense layers (iGPU): {len(dense_layers)} layers")
        
        logger.info("\n✅ Direct safetensors loading successful!")
        logger.info("🚀 Ready for integration with NPU acceleration")
        
        return loader, npu_weights
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    test_direct_safetensors_loader()