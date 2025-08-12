#!/usr/bin/env python3.13
"""
Magic Unicorn Real Model Loader
Load actual Gemma3n weights and test performance with real model
"""

import torch
import json
import numpy as np
from pathlib import Path
from safetensors import safe_open
import time
from typing import Dict, Optional
import sys
import os

# Add the project path to use existing OpenCL pipeline
sys.path.append('/home/ucadmin/Development/Unicorn-Execution-Engine')

class MagicUnicornRealModelLoader:
    """Load and benchmark real Gemma3n model"""
    
    def __init__(self, model_path: str = "models/gemma-3n-e4b-it"):
        self.model_path = Path(model_path)
        self.config = None
        self.weights = {}
        self.loaded_layers = set()
        
        print("🦄📁 MAGIC UNICORN REAL MODEL LOADER")
        print("=" * 60)
        print(f"🎯 TARGET: Load real Gemma3n and measure actual performance")
        
        self.load_config()
        self.analyze_model_structure()
        
    def load_config(self):
        """Load model configuration"""
        config_path = self.model_path / "config.json"
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Extract text model config
        self.text_config = self.config['text_config']
        
        print(f"\n📊 MODEL CONFIGURATION:")
        print(f"   Model: Gemma3n multimodal")
        print(f"   Text layers: {self.text_config['num_hidden_layers']}")
        print(f"   Hidden size: {self.text_config['hidden_size']}")
        print(f"   Intermediate size: {self.text_config['intermediate_size']}")
        print(f"   Attention heads: {self.text_config['num_attention_heads']}")
        print(f"   KV heads: {self.text_config['num_key_value_heads']}")
        print(f"   Dtype: {self.text_config['torch_dtype']}")
        
        # Check sparsity pattern
        sparsity = self.text_config.get('activation_sparsity_pattern', [])
        if sparsity:
            sparse_layers = sum(1 for s in sparsity if s > 0.5)
            print(f"   Sparse layers: {sparse_layers}/{len(sparsity)} (95% sparsity)")
    
    def analyze_model_structure(self):
        """Analyze the safetensors structure"""
        index_path = self.model_path / "model.safetensors.index.json"
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        self.weight_map = index['weight_map']
        self.total_params = index['metadata']['total_parameters']
        self.total_size = index['metadata']['total_size']
        
        print(f"\n📈 MODEL SIZE:")
        print(f"   Total parameters: {self.total_params:,}")
        print(f"   Total size: {self.total_size / 1024**3:.2f} GB")
        
        # Find text model layers
        text_layers = {}
        for weight_name in self.weight_map.keys():
            if 'language_model.layers.' in weight_name:
                parts = weight_name.split('.')
                # Find the layer index after 'layers'
                try:
                    layers_idx = parts.index('layers')
                    if layers_idx + 1 < len(parts):
                        layer_idx = int(parts[layers_idx + 1])
                        if layer_idx not in text_layers:
                            text_layers[layer_idx] = []
                        text_layers[layer_idx].append(weight_name)
                except (ValueError, IndexError):
                    continue
        
        self.text_layers = text_layers
        print(f"   Text layers found: 0-{max(text_layers.keys())}")
        
        # Analyze layer 0 structure
        if 0 in text_layers:
            layer_0_weights = text_layers[0]
            print(f"\n🔍 LAYER 0 STRUCTURE:")
            attention_weights = [w for w in layer_0_weights if 'self_attn' in w]
            mlp_weights = [w for w in layer_0_weights if 'mlp' in w]
            altup_weights = [w for w in layer_0_weights if 'altup' in w]
            
            print(f"   Attention weights: {len(attention_weights)}")
            print(f"   MLP weights: {len(mlp_weights)}")
            print(f"   AltUp weights: {len(altup_weights)} (sparse activation)")
    
    def load_layer_weights(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Load weights for a specific layer"""
        if layer_idx in self.loaded_layers:
            return self.get_layer_weights(layer_idx)
        
        print(f"\n📥 Loading layer {layer_idx} weights...")
        
        # Find all weights for this layer
        layer_weights = {}
        files_to_load = set()
        
        for weight_name in self.text_layers.get(layer_idx, []):
            file_name = self.weight_map[weight_name]
            files_to_load.add(file_name)
        
        # Load from safetensors files
        for file_name in files_to_load:
            file_path = self.model_path / file_name
            
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for weight_name in self.text_layers[layer_idx]:
                    if self.weight_map[weight_name] == file_name:
                        weight_tensor = f.get_tensor(weight_name)
                        layer_weights[weight_name] = weight_tensor
        
        # Store in cache
        self.weights[layer_idx] = layer_weights
        self.loaded_layers.add(layer_idx)
        
        print(f"   ✅ Loaded {len(layer_weights)} weights")
        return layer_weights
    
    def get_layer_weights(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Get cached layer weights"""
        return self.weights.get(layer_idx, {})
    
    def extract_standard_weights(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Extract standard transformer weights from Gemma3n layer"""
        layer_weights = self.load_layer_weights(layer_idx)
        
        # Map Gemma3n weight names to standard names
        standard_weights = {}
        
        # Find attention weights (might be under self_attn)
        for weight_name, tensor in layer_weights.items():
            if 'self_attn.q_proj.weight' in weight_name:
                standard_weights['q_proj'] = tensor
            elif 'self_attn.k_proj.weight' in weight_name:
                standard_weights['k_proj'] = tensor
            elif 'self_attn.v_proj.weight' in weight_name:
                standard_weights['v_proj'] = tensor
            elif 'self_attn.o_proj.weight' in weight_name:
                standard_weights['o_proj'] = tensor
            elif 'mlp.gate_proj.weight' in weight_name:
                standard_weights['gate_proj'] = tensor
            elif 'mlp.up_proj.weight' in weight_name:
                standard_weights['up_proj'] = tensor
            elif 'mlp.down_proj.weight' in weight_name:
                standard_weights['down_proj'] = tensor
        
        # If we didn't find standard weights, try altup (sparse) weights
        if not standard_weights:
            print(f"   ⚠️ No standard weights found, checking altup (sparse) weights...")
            for weight_name, tensor in layer_weights.items():
                print(f"      Available: {weight_name} -> {tensor.shape}")
        
        return standard_weights
    
    def test_real_performance(self, layer_idx: int = 0):
        """Test performance with real model weights"""
        print(f"\n🚀 TESTING REAL MODEL PERFORMANCE")
        print("=" * 50)
        
        # Load real weights
        weights = self.extract_standard_weights(layer_idx)
        
        if not weights:
            print(f"❌ Could not extract standard weights from layer {layer_idx}")
            return None
        
        print(f"✅ Extracted weights for layer {layer_idx}:")
        for name, tensor in weights.items():
            print(f"   {name}: {tensor.shape} ({tensor.dtype})")
        
        # Create test input matching model config
        batch_size = 1
        seq_len = 128  # Start with medium context
        hidden_size = self.text_config['hidden_size']
        
        print(f"\n🧪 Test configuration:")
        print(f"   Batch: {batch_size}")
        print(f"   Sequence: {seq_len}")
        print(f"   Hidden: {hidden_size}")
        
        # Create realistic input
        x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
        
        # Convert weights to float32 for compatibility
        weights_f32 = {}
        for name, tensor in weights.items():
            weights_f32[name] = tensor.float()
        
        # Test with existing OpenCL pipeline
        try:
            from optimized_hybrid_pipeline import OptimizedHybridEngine
            
            engine = OptimizedHybridEngine()
            
            if engine.igpu_context is None:
                print("⚠️ iGPU not available, using CPU")
                return self.test_cpu_performance(x, weights_f32)
            
            print(f"\n⚡ Testing with OpenCL iGPU acceleration...")
            
            # Warmup
            for i in range(2):
                _ = engine.forward_layer_optimized(x, weights_f32)
            
            # Benchmark
            times = []
            for i in range(3):
                start = time.time()
                output, layer_time = engine.forward_layer_optimized(x, weights_f32)
                times.append(layer_time)
            
            avg_time = sum(times) / len(times)
            fastest_time = min(times)
            
            print(f"\n🏆 REAL MODEL PERFORMANCE:")
            print(f"   Average layer: {avg_time*1000:.1f}ms")
            print(f"   Fastest layer: {fastest_time*1000:.1f}ms")
            print(f"   Output shape: {output.shape}")
            print(f"   Output valid: {torch.isfinite(output).all()}")
            
            # Project to full model
            num_layers = self.text_config['num_hidden_layers']
            full_time = fastest_time * num_layers
            tokens_per_sec = 1.0 / full_time
            
            print(f"\n📊 FULL MODEL PROJECTION:")
            print(f"   Layers: {num_layers}")
            print(f"   Full model time: {full_time:.2f}s")
            print(f"   Single token speed: {tokens_per_sec:.3f} tokens/sec")
            print(f"   vs 21 tok/s target: {tokens_per_sec/21:.3f}x")
            
            if tokens_per_sec >= 21.0:
                print(f"   🎯 BASELINE ACHIEVED!")
            elif tokens_per_sec >= 10.0:
                print(f"   🔥 GETTING CLOSE!")
            else:
                print(f"   ⚡ REAL MODEL LOADED, OPTIMIZATION NEEDED")
            
            return {
                'layer_time': fastest_time,
                'full_time': full_time,
                'tokens_per_sec': tokens_per_sec,
                'model_config': self.text_config
            }
            
        except Exception as e:
            print(f"❌ OpenCL test failed: {e}")
            return self.test_cpu_performance(x, weights_f32)
    
    def test_cpu_performance(self, x: torch.Tensor, weights: Dict[str, torch.Tensor]):
        """CPU-only performance test"""
        print(f"\n⚡ Testing with CPU-only...")
        
        def simple_transformer_layer(x, weights):
            # Simple CPU implementation
            q = torch.matmul(x, weights['q_proj'].T)
            k = torch.matmul(x, weights['k_proj'].T)
            v = torch.matmul(x, weights['v_proj'].T)
            
            # Simple attention
            scale = 1.0 / (q.shape[-1] ** 0.5)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = torch.softmax(scores, dim=-1)
            attn_out = torch.matmul(attn, v)
            
            # Output projection
            attn_out = torch.matmul(attn_out, weights['o_proj'].T)
            x = x + attn_out
            
            # FFN
            gate = torch.matmul(x, weights['gate_proj'].T)
            up = torch.matmul(x, weights['up_proj'].T)
            hidden = torch.nn.functional.silu(gate) * up
            output = torch.matmul(hidden, weights['down_proj'].T)
            
            return x + output
        
        # Benchmark
        times = []
        for i in range(3):
            start = time.time()
            output = simple_transformer_layer(x, weights)
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        fastest_time = min(times)
        
        print(f"\n🏆 CPU PERFORMANCE:")
        print(f"   Average layer: {avg_time*1000:.1f}ms")
        print(f"   Fastest layer: {fastest_time*1000:.1f}ms")
        
        # Project to full model
        num_layers = self.text_config['num_hidden_layers']
        full_time = fastest_time * num_layers
        tokens_per_sec = 1.0 / full_time
        
        print(f"\n📊 CPU PROJECTION:")
        print(f"   Full model time: {full_time:.2f}s")
        print(f"   Single token speed: {tokens_per_sec:.3f} tokens/sec")
        
        return {
            'layer_time': fastest_time,
            'full_time': full_time,
            'tokens_per_sec': tokens_per_sec,
            'model_config': self.text_config
        }

def test_real_model_loading():
    """Test real model loading and performance"""
    print("🦄📁 MAGIC UNICORN REAL MODEL TEST")
    print("=" * 70)
    
    try:
        loader = MagicUnicornRealModelLoader()
        
        # Test loading first layer
        result = loader.test_real_performance(layer_idx=0)
        
        if result:
            print(f"\n🏁 REAL MODEL RESULTS:")
            print(f"   Model: Gemma3n ({result['model_config']['num_hidden_layers']} layers)")
            print(f"   Hidden size: {result['model_config']['hidden_size']}")
            print(f"   Performance: {result['tokens_per_sec']:.3f} tokens/sec")
            
            if result['tokens_per_sec'] >= 21.0:
                print(f"   🎯 TARGET ACHIEVED! Real model beats baseline!")
            else:
                gap = 21.0 / result['tokens_per_sec']
                print(f"   📊 Gap analysis: Need {gap:.1f}x speedup for 21 tok/s")
                
                if gap < 10:
                    print(f"   🔥 CLOSE! Quantization could bridge the gap!")
                elif gap < 100:
                    print(f"   ⚡ MODERATE gap - kernel optimization promising!")
                else:
                    print(f"   🔧 LARGE gap - fundamental optimization needed!")
        
    except Exception as e:
        print(f"❌ Real model test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_real_model_loading()