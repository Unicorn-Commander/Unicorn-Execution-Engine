#!/usr/bin/env python3
"""
Download the quantized Gemma-3 27B model from HuggingFace
"""
import os
import sys
from huggingface_hub import snapshot_download
from pathlib import Path

def download_quantized_model():
    """Download the quantized Gemma-3 27B model"""
    
    # Model info
    model_id = "magicunicorn/gemma-3-27b-npu-quantized"
    local_dir = Path("quantized_models/gemma-3-27b-it-layer-by-layer")
    
    print(f"🦄 Downloading quantized model from HuggingFace...")
    print(f"Model ID: {model_id}")
    print(f"Target directory: {local_dir}")
    
    # Create directory
    local_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download model
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"✅ Model downloaded successfully to {local_dir}")
        
        # List downloaded files
        print("\nDownloaded files:")
        for file in sorted(local_dir.rglob("*")):
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  - {file.relative_to(local_dir)}: {size_mb:.1f} MB")
                
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        print("\nTo download manually:")
        print(f"1. Visit https://huggingface.co/{model_id}")
        print(f"2. Download all .safetensors files")
        print(f"3. Place them in {local_dir}")
        sys.exit(1)

if __name__ == "__main__":
    download_quantized_model()