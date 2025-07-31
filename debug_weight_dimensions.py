#!/usr/bin/env python3.13
"""
Debug weight dimensions to fix the projection issues
"""

import json
import mmap
import struct
import numpy as np
from pathlib import Path

def debug_weights():
    """Debug the actual weight dimensions"""
    print("🔍 Debugging Weight Dimensions")
    print("=" * 50)
    
    model_path = Path("/home/ucadmin/Development/Unicorn-Execution-Engine/quantized_models/gemma-3-4b-it-quantized")
    weight_files = list(model_path.glob("*.safetensors"))
    
    # Load config
    with open(model_path / "config.json") as f:
        config = json.load(f)
    
    print(f"Config dimensions:")
    print(f"  Hidden size: {config.get('hidden_size', 2560)}")
    print(f"  Num heads: {config.get('num_attention_heads', 20)}")
    print(f"  Head dim: {config.get('hidden_size', 2560) // config.get('num_attention_heads', 20)}")
    
    # Load first weight file
    weight_file = weight_files[0]
    print(f"\nAnalyzing {weight_file.name}...")
    
    with open(weight_file, 'rb') as f:
        # Read header
        header_len = struct.unpack('<Q', f.read(8))[0]
        header_data = f.read(header_len)
        header = json.loads(header_data.decode('utf-8'))
        
        # Check layer 0 attention weights
        layer_0_weights = {}
        for name, info in header.items():
            if name.startswith("language_model.model.layers.0.self_attn") and 'shape' in info:
                layer_0_weights[name] = info['shape']
        
        print(f"\nLayer 0 attention weights:")
        for name, shape in layer_0_weights.items():
            print(f"  {name}: {shape}")
            
        # Check embedding weights
        for name, info in header.items():
            if "embed_tokens.weight" in name and 'shape' in info:
                print(f"\nEmbedding weights:")
                print(f"  {name}: {info['shape']}")
                break

if __name__ == "__main__":
    debug_weights()