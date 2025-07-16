#!/usr/bin/env python3
"""
Download the quantized Gemma-3 27B model with resume capability
"""
import os
import sys
from huggingface_hub import hf_hub_download, list_repo_files
from pathlib import Path
from tqdm import tqdm

def download_quantized_model():
    """Download the quantized Gemma-3 27B model with progress tracking"""
    
    # Model info
    model_id = "magicunicorn/gemma-3-27b-npu-quantized"
    local_dir = Path("quantized_models/gemma-3-27b-it-layer-by-layer")
    
    print(f"🦄 Downloading quantized model from HuggingFace...")
    print(f"Model ID: {model_id}")
    print(f"Target directory: {local_dir}")
    print(f"Note: This is a ~27GB download and may take a while\n")
    
    # Create directory
    local_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Get list of files in the repo
        print("Fetching file list...")
        files = list_repo_files(model_id)
        
        # Filter for model files
        model_files = [f for f in files if f.endswith(('.safetensors', '.json', '.txt', '.model'))]
        print(f"Found {len(model_files)} files to download\n")
        
        # Download each file with progress
        for file in tqdm(model_files, desc="Downloading files"):
            local_path = local_dir / file
            
            # Skip if already downloaded
            if local_path.exists():
                size_mb = local_path.stat().st_size / (1024 * 1024)
                print(f"✓ {file} already exists ({size_mb:.1f} MB)")
                continue
            
            # Create subdirectories if needed
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download file
            print(f"\nDownloading {file}...")
            hf_hub_download(
                repo_id=model_id,
                filename=file,
                local_dir=str(local_dir),
                force_download=False
            )
            
        print(f"\n✅ Model downloaded successfully to {local_dir}")
        
        # List downloaded files with sizes
        print("\nDownloaded files:")
        total_size = 0
        for file in sorted(local_dir.rglob("*")):
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                total_size += size_mb
                if size_mb > 100:  # Only show large files
                    print(f"  - {file.name}: {size_mb:.1f} MB")
        
        print(f"\nTotal size: {total_size/1024:.1f} GB")
                
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted. Run this script again to resume.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("\nTo download manually:")
        print(f"1. Visit https://huggingface.co/{model_id}")
        print(f"2. Download all .safetensors files")
        print(f"3. Place them in {local_dir}")
        sys.exit(1)

if __name__ == "__main__":
    download_quantized_model()