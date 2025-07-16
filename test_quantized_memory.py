#!/usr/bin/env python3
"""Test quantized model memory usage"""

import os
import json
import struct
import numpy as np

def check_model_size(model_path):
    """Check actual size of quantized model"""
    print(f"Checking model at: {model_path}")
    
    # Check all safetensor files
    total_size = 0
    total_params = 0
    file_count = 0
    
    for root, dirs, files in os.walk(model_path):
        for file in files:
            if file.endswith('.safetensors'):
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                total_size += file_size
                file_count += 1
                
                # Read header to get tensor count
                with open(file_path, 'rb') as f:
                    # First 8 bytes is header size
                    header_size = struct.unpack('<Q', f.read(8))[0]
                    header_data = f.read(header_size)
                    header = json.loads(header_data)
                    
                    # Count tensors
                    tensor_count = sum(1 for k, v in header.items() if isinstance(v, dict) and 'dtype' in v)
                    print(f"  {file}: {file_size/(1024*1024):.1f}MB, {tensor_count} tensors")
    
    print(f"\nTotal model size: {total_size/(1024*1024*1024):.2f}GB across {file_count} files")
    print(f"Expected for 27B int8 model: ~27GB")
    
    # Check specific tensor to understand dtype
    sample_file = os.path.join(model_path, "model-00001-of-00012_layer_0.safetensors")
    if os.path.exists(sample_file):
        with open(sample_file, 'rb') as f:
            header_size = struct.unpack('<Q', f.read(8))[0]
            header_data = f.read(header_size)
            header = json.loads(header_data)
            
            print("\nSample tensor dtypes:")
            for name, info in list(header.items())[:5]:
                if isinstance(info, dict) and 'dtype' in info:
                    print(f"  {name}: dtype={info['dtype']}, shape={info.get('shape', [])}")

if __name__ == "__main__":
    model_path = "quantized_models/gemma-3-27b-it-layer-by-layer"
    check_model_size(model_path)