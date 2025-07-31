#!/usr/bin/env python3.13
"""
Load real safetensors weights test
"""
import os
import sys
import struct
import json
from pathlib import Path

print("🦄 Loading Real Model Weights")
print("=" * 50)

model_path = Path("/home/ucadmin/Development/Unicorn-Execution-Engine/quantized_models/gemma-3-4b-it-quantized")
weight_files = list(model_path.glob("*.safetensors"))

print(f"Found {len(weight_files)} weight files:")
for f in weight_files:
    size_gb = f.stat().st_size / (1024**3)
    print(f"  - {f.name}: {size_gb:.2f} GB")

# Read safetensors header from first file
weight_file = weight_files[0]
print(f"\n📦 Reading {weight_file.name}...")

try:
    with open(weight_file, 'rb') as f:
        # Read header length (first 8 bytes)
        header_len = struct.unpack('<Q', f.read(8))[0]
        print(f"   Header length: {header_len} bytes")
        
        # Read header
        header_data = f.read(header_len)
        header = json.loads(header_data.decode('utf-8'))
        
        print(f"   Found {len(header)} tensors:")
        for name, info in list(header.items())[:5]:  # First 5
            if isinstance(info, dict):
                shape = info.get('shape', [])
                dtype = info.get('dtype', 'unknown')
                print(f"     - {name}: {shape} ({dtype})")

    print("✅ Safetensors file readable!")
    print("🎉 Ready for real weight loading!")
    
except Exception as e:
    print(f"❌ Error reading safetensors: {e}")