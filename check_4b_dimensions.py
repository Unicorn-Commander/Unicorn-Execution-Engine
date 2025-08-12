#!/usr/bin/env python3
"""
Check the actual dimensions of the 4B model
"""

import safetensors.torch
from pathlib import Path

def check_4b_dimensions():
    model_path = Path("/home/ucadmin/Development/Unicorn-Execution-Engine/quantized_models/gemma-3-4b-it-quantized")
    
    # Check embedding dimensions
    for file in model_path.glob("*.safetensors"):
        print(f"\nFile: {file.name}")
        with safetensors.torch.safe_open(file, framework="pt") as f:
            for key in f.keys():
                if 'embed_tokens' in key:
                    tensor = f.get_tensor(key)
                    print(f"  {key}: {tensor.shape}")
                elif 'layers.0.' in key and any(x in key for x in ['q_proj', 'k_proj', 'v_proj']):
                    tensor = f.get_tensor(key)
                    print(f"  {key}: {tensor.shape}")
        break  # Just check first file

if __name__ == "__main__":
    check_4b_dimensions()