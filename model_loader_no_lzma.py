#!/usr/bin/env python3
"""
Model Loader Without LZMA Dependency
Alternative approach to load models without requiring LZMA compression
"""

import os
import sys
import torch
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import safetensors
from safetensors.torch import load_file as load_safetensors

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NoLZMAModelLoader:
    """
    Model loader that avoids LZMA dependencies by using only safetensors
    and basic JSON configuration files
    """
    
    def __init__(self, model_path: str):
        """
        Initialize model loader
        
        Args:
            model_path: Path to model directory
        """
        self.model_path = Path(model_path)
        self.config = None
        self.model_weights = None
        
    def load_config(self) -> Dict[str, Any]:
        """Load model configuration from config.json"""
        
        config_path = self.model_path / "config.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            
            logger.info(f"✅ Loaded config from {config_path}")
            return self.config
            
        except Exception as e:
            logger.error(f"❌ Failed to load config: {e}")
            raise
    
    def list_weight_files(self) -> list:
        """List available weight files (prioritize safetensors)"""
        
        # Look for safetensors files first (no compression issues)
        safetensors_files = list(self.model_path.glob("*.safetensors"))
        
        if safetensors_files:
            logger.info(f"✅ Found {len(safetensors_files)} safetensors files")
            return sorted(safetensors_files)
        
        # Look for pytorch files as fallback
        pytorch_files = list(self.model_path.glob("pytorch_model*.bin"))
        
        if pytorch_files:
            logger.info(f"✅ Found {len(pytorch_files)} PyTorch files")
            return sorted(pytorch_files)
        
        raise FileNotFoundError("No weight files found (.safetensors or .bin)")
    
    def load_weights_safetensors(self) -> Dict[str, torch.Tensor]:
        """Load model weights from safetensors files"""
        
        weight_files = self.list_weight_files()
        all_weights = {}
        
        for weight_file in weight_files:
            if weight_file.suffix == '.safetensors':
                try:
                    logger.info(f"📦 Loading weights from {weight_file}")
                    weights = load_safetensors(str(weight_file))
                    all_weights.update(weights)
                    
                    logger.info(f"✅ Loaded {len(weights)} tensors from {weight_file.name}")
                    
                except Exception as e:
                    logger.warning(f"⚠️  Failed to load {weight_file}: {e}")
                    continue
            
            elif weight_file.suffix == '.bin':
                try:
                    logger.info(f"📦 Loading PyTorch weights from {weight_file}")
                    weights = torch.load(str(weight_file), map_location='cpu')
                    all_weights.update(weights)
                    
                    logger.info(f"✅ Loaded {len(weights)} tensors from {weight_file.name}")
                    
                except Exception as e:
                    logger.warning(f"⚠️  Failed to load {weight_file}: {e}")
                    continue
        
        if not all_weights:
            raise RuntimeError("Failed to load any model weights")
        
        self.model_weights = all_weights
        logger.info(f"✅ Total loaded weights: {len(all_weights)} tensors")
        
        return all_weights
    
    def create_simple_tokenizer(self) -> 'SimpleTokenizer':
        """Create a simple tokenizer without dependencies"""
        
        # Look for tokenizer files
        tokenizer_files = [
            "tokenizer.json",
            "vocab.txt", 
            "tokenizer_config.json"
        ]
        
        found_files = []
        for tf in tokenizer_files:
            if (self.model_path / tf).exists():
                found_files.append(tf)
        
        if found_files:
            logger.info(f"✅ Found tokenizer files: {found_files}")
            return SimpleTokenizer(self.model_path, found_files)
        else:
            logger.warning("⚠️  No tokenizer files found, creating basic tokenizer")
            return SimpleTokenizer(self.model_path, [])
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information summary"""
        
        if not self.config:
            self.load_config()
        
        if not self.model_weights:
            self.load_weights_safetensors()
        
        # Calculate model size
        total_params = 0
        total_size = 0
        
        for name, tensor in self.model_weights.items():
            total_params += tensor.numel()
            total_size += tensor.numel() * tensor.element_size()
        
        info = {
            'model_path': str(self.model_path),
            'config': self.config,
            'total_parameters': total_params,
            'total_size_gb': total_size / (1024**3),
            'num_weight_tensors': len(self.model_weights),
            'architecture': self.config.get('architectures', ['unknown'])[0] if self.config else 'unknown',
            'hidden_size': self.config.get('hidden_size', 0) if self.config else 0,
            'num_attention_heads': self.config.get('num_attention_heads', 0) if self.config else 0,
            'num_layers': self.config.get('num_hidden_layers', 0) if self.config else 0,
            'vocab_size': self.config.get('vocab_size', 0) if self.config else 0
        }
        
        return info

class SimpleTokenizer:
    """Simple tokenizer implementation without heavy dependencies"""
    
    def __init__(self, model_path: Path, tokenizer_files: list):
        """Initialize simple tokenizer"""
        
        self.model_path = model_path
        self.tokenizer_files = tokenizer_files
        self.vocab = None
        self.pad_token = None
        self.eos_token = None
        
        self._load_basic_vocab()
    
    def _load_basic_vocab(self):
        """Load basic vocabulary"""
        
        # Try to load tokenizer config
        config_path = self.model_path / "tokenizer_config.json"
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                self.pad_token = config.get('pad_token', '</s>')
                self.eos_token = config.get('eos_token', '</s>')
                
                logger.info(f"✅ Loaded tokenizer config: pad='{self.pad_token}', eos='{self.eos_token}'")
                
            except Exception as e:
                logger.warning(f"⚠️  Failed to load tokenizer config: {e}")
        
        # Set defaults if not found
        if not self.pad_token:
            self.pad_token = '</s>'
        if not self.eos_token:
            self.eos_token = '</s>'
        
        # Create basic vocabulary (placeholder)
        self.vocab = {
            '<pad>': 0,
            '<unk>': 1,
            '<s>': 2,
            '</s>': 3
        }
        
        logger.info("✅ Created basic tokenizer")
    
    def encode(self, text: str, **kwargs) -> torch.Tensor:
        """Encode text to token IDs (basic implementation)"""
        
        # Very basic encoding - just return some placeholder tokens
        # In a real implementation, this would use the actual tokenizer
        tokens = [2, 3, 4, 5, 6]  # Placeholder token sequence
        
        if kwargs.get('return_tensors') == 'pt':
            return torch.tensor([tokens])
        
        return tokens
    
    def decode(self, token_ids, **kwargs) -> str:
        """Decode token IDs to text (basic implementation)"""
        
        # Basic decoding - return placeholder
        return f"Generated text from tokens: {token_ids}"

def test_no_lzma_loader():
    """Test the LZMA-free model loader"""
    
    logger.info("🧪 Testing LZMA-free model loader...")
    
    model_path = "/home/ucadmin/Development/Unicorn-Execution-Engine/quantized_models/gemma-3-4b-it-quantized"
    
    # Test if model path exists
    if not Path(model_path).exists():
        logger.error(f"❌ Model path not found: {model_path}")
        return False
    
    try:
        # Initialize loader
        loader = NoLZMAModelLoader(model_path)
        
        # Load configuration
        config = loader.load_config()
        logger.info(f"✅ Model config: {config.get('architectures', 'unknown')}")
        
        # Load weights
        weights = loader.load_weights_safetensors()
        logger.info(f"✅ Loaded {len(weights)} weight tensors")
        
        # Create simple tokenizer
        tokenizer = loader.create_simple_tokenizer()
        logger.info("✅ Simple tokenizer created")
        
        # Get model info
        info = loader.get_model_info()
        logger.info("📊 Model Information:")
        logger.info(f"   Architecture: {info['architecture']}")
        logger.info(f"   Parameters: {info['total_parameters']:,}")
        logger.info(f"   Size: {info['total_size_gb']:.2f}GB")
        logger.info(f"   Hidden size: {info['hidden_size']}")
        logger.info(f"   Attention heads: {info['num_attention_heads']}")
        logger.info(f"   Layers: {info['num_layers']}")
        
        logger.info("✅ LZMA-free model loading successful!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return False

if __name__ == "__main__":
    test_no_lzma_loader()