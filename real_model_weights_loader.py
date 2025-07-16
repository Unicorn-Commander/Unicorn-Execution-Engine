#!/usr/bin/env python3
"""
Real Model Weights Loader - No Simulation
Load actual Gemma 3 model weights and run real inference
"""

import os
import sys
import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealModelWeightsLoader:
    """Load and manage real Gemma 3 model weights"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_config = None
        self.weights_loaded = False
        
        logger.info(f"🦄 Real Model Weights Loader initialized")
        logger.info(f"   Model path: {model_path}")
        logger.info(f"   Device: {self.device}")
    
    def load_model_weights(self) -> bool:
        """Load real model weights from disk"""
        logger.info("📥 Loading real model weights...")
        
        try:
            # Check if model path exists
            if not self.model_path.exists():
                logger.error(f"❌ Model path does not exist: {self.model_path}")
                return False
            
            # Load tokenizer
            logger.info("🔤 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
                use_fast=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"   ✅ Tokenizer loaded: {len(self.tokenizer)} tokens")
            
            # Load model configuration
            config_path = self.model_path / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.model_config = json.load(f)
                logger.info(f"   ✅ Model config loaded: {self.model_config.get('model_type', 'unknown')}")
            
            # Load model weights
            logger.info("🧠 Loading model weights...")
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Get model info
            param_count = sum(p.numel() for p in self.model.parameters())
            logger.info(f"   ✅ Model loaded: {param_count:,} parameters")
            
            # Print model architecture info
            if hasattr(self.model, 'config'):
                config = self.model.config
                logger.info(f"   📊 Hidden size: {getattr(config, 'hidden_size', 'N/A')}")
                logger.info(f"   📊 Num layers: {getattr(config, 'num_hidden_layers', 'N/A')}")
                logger.info(f"   📊 Attention heads: {getattr(config, 'num_attention_heads', 'N/A')}")
                logger.info(f"   📊 Vocab size: {getattr(config, 'vocab_size', 'N/A')}")
            
            self.weights_loaded = True
            logger.info("✅ Real model weights loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load real model weights: {e}")
            return False
    
    def get_model_weights(self) -> Dict[str, torch.Tensor]:
        """Extract model weights for hardware deployment"""
        if not self.weights_loaded:
            raise RuntimeError("Model weights not loaded")
        
        logger.info("🔍 Extracting model weights...")
        
        weights = {}
        for name, param in self.model.named_parameters():
            # Convert to numpy for hardware deployment
            weights[name] = param.detach().cpu().numpy()
        
        logger.info(f"   ✅ Extracted {len(weights)} weight tensors")
        return weights
    
    def get_attention_weights(self, layer_idx: int) -> Dict[str, np.ndarray]:
        """Get attention weights for specific layer"""
        if not self.weights_loaded:
            raise RuntimeError("Model weights not loaded")
        
        layer_weights = {}
        prefix = f"model.layers.{layer_idx}.self_attn"
        
        for name, param in self.model.named_parameters():
            if name.startswith(prefix):
                # Remove prefix and convert to numpy
                key = name[len(prefix)+1:]  # Remove prefix and dot
                layer_weights[key] = param.detach().cpu().numpy()
        
        return layer_weights
    
    def get_ffn_weights(self, layer_idx: int) -> Dict[str, np.ndarray]:
        """Get FFN weights for specific layer"""
        if not self.weights_loaded:
            raise RuntimeError("Model weights not loaded")
        
        layer_weights = {}
        prefix = f"model.layers.{layer_idx}.mlp"
        
        for name, param in self.model.named_parameters():
            if name.startswith(prefix):
                # Remove prefix and convert to numpy
                key = name[len(prefix)+1:]  # Remove prefix and dot
                layer_weights[key] = param.detach().cpu().numpy()
        
        return layer_weights
    
    def tokenize_input(self, text: str) -> torch.Tensor:
        """Tokenize input text"""
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not loaded")
        
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        return tokens.input_ids
    
    def detokenize_output(self, token_ids: torch.Tensor) -> str:
        """Convert token IDs back to text"""
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not loaded")
        
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def run_baseline_inference(self, text: str, max_tokens: int = 50) -> str:
        """Run baseline inference using the loaded model"""
        if not self.weights_loaded:
            raise RuntimeError("Model weights not loaded")
        
        logger.info(f"🔮 Running baseline inference...")
        logger.info(f"   Input: {text}")
        
        start_time = time.time()
        
        try:
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            inference_time = time.time() - start_time
            tps = max_tokens / inference_time
            
            logger.info(f"   ✅ Baseline inference completed in {inference_time:.2f}s")
            logger.info(f"   📊 Baseline TPS: {tps:.1f}")
            logger.info(f"   Output: {generated_text}")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"❌ Baseline inference failed: {e}")
            return f"ERROR: {e}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        if not self.weights_loaded:
            return {"error": "Model not loaded"}
        
        info = {
            "model_path": str(self.model_path),
            "device": self.device,
            "weights_loaded": self.weights_loaded,
            "tokenizer_vocab_size": len(self.tokenizer) if self.tokenizer else 0,
            "model_config": self.model_config
        }
        
        if self.model:
            info["parameter_count"] = sum(p.numel() for p in self.model.parameters())
            info["model_dtype"] = str(next(self.model.parameters()).dtype)
            info["model_device"] = str(next(self.model.parameters()).device)
            
            if hasattr(self.model, 'config'):
                config = self.model.config
                info["hidden_size"] = getattr(config, 'hidden_size', None)
                info["num_layers"] = getattr(config, 'num_hidden_layers', None)
                info["num_attention_heads"] = getattr(config, 'num_attention_heads', None)
                info["vocab_size"] = getattr(config, 'vocab_size', None)
        
        return info

def main():
    """Test real model weights loader"""
    logger.info("🦄 Testing Real Model Weights Loader")
    
    # Test with available models
    test_models = [
        "./models/gemma-3-4b-it",
        "./models/gemma-3-27b-it",
        "./quantized_models/gemma-3-4b-it-quantized"
    ]
    
    for model_path in test_models:
        if os.path.exists(model_path):
            logger.info(f"📂 Testing model: {model_path}")
            
            # Load model
            loader = RealModelWeightsLoader(model_path)
            
            if loader.load_model_weights():
                # Get model info
                info = loader.get_model_info()
                logger.info(f"   📊 Parameters: {info.get('parameter_count', 0):,}")
                
                # Run test inference
                test_text = "Explain quantum computing in simple terms."
                result = loader.run_baseline_inference(test_text, max_tokens=30)
                
                logger.info(f"   ✅ Test completed for {model_path}")
                break
            else:
                logger.warning(f"   ⚠️ Failed to load {model_path}")
        else:
            logger.info(f"   ❌ Model not found: {model_path}")
    
    logger.info("✅ Real model weights loader test completed!")

if __name__ == "__main__":
    main()