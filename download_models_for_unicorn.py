#!/usr/bin/env python3
"""
Download models for Unicorn Execution Engine testing
Uses our custom quantization, not pre-quantized formats
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
import argparse

def download_phi4_mini(base_path: str):
    """Download Microsoft Phi-4-mini-instruct (3.8B)"""
    model_id = "microsoft/Phi-4-mini-instruct"
    target_dir = os.path.join(base_path, "phi-4-mini-instruct")
    
    print(f"🚀 Downloading {model_id}...")
    print(f"   Target: {target_dir}")
    print(f"   Size: ~7.6GB (3.8B parameters)")
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=target_dir,
            ignore_patterns=["*.bin", "*.gguf", "*.onnx"],  # Only safetensors
            resume_download=True,
            max_workers=4
        )
        print(f"✅ Successfully downloaded {model_id}")
        return True
    except Exception as e:
        print(f"❌ Error downloading {model_id}: {e}")
        return False

def download_granite_3b(base_path: str):
    """Download IBM Granite-3.3-8B-instruct"""
    model_id = "ibm-granite/granite-3.3-8b-instruct"
    target_dir = os.path.join(base_path, "granite-3.3-8b-instruct")
    
    print(f"🚀 Downloading {model_id}...")
    print(f"   Target: {target_dir}")
    print(f"   Size: ~16GB (8B parameters)")
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=target_dir,
            ignore_patterns=["*.bin", "*.gguf", "*.onnx"],
            resume_download=True,
            max_workers=4
        )
        print(f"✅ Successfully downloaded {model_id}")
        return True
    except Exception as e:
        print(f"❌ Error downloading {model_id}: {e}")
        return False

def download_qwen3_moe(base_path: str, version: str = "fp8"):
    """Download Qwen3-30B-A3B MoE model"""
    model_map = {
        "fp8": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        "full": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "thinking": "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "thinking-fp8": "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8"
    }
    
    if version not in model_map:
        print(f"❌ Unknown version: {version}")
        return False
        
    model_id = model_map[version]
    target_dir = os.path.join(base_path, f"qwen3-30b-a3b-{version}")
    
    print(f"🚀 Downloading {model_id}...")
    print(f"   Target: {target_dir}")
    print(f"   Size: ~{'30GB' if 'fp8' in version else '60GB'}")
    print(f"   Type: MoE with 3.3B active params")
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=target_dir,
            ignore_patterns=["*.bin", "*.gguf", "*.onnx"],
            resume_download=True,
            max_workers=4
        )
        print(f"✅ Successfully downloaded {model_id}")
        return True
    except Exception as e:
        print(f"❌ Error downloading {model_id}: {e}")
        return False

def check_existing_models(base_path: str):
    """Check what models are already downloaded"""
    print("\n📁 Checking existing models...")
    
    models = {
        "phi-4-mini-instruct": "Phi-4-mini (14B)",
        "granite-3.3-8b-instruct": "Granite-3.3 (8B)",
        "qwen3-30b-a3b-fp8": "Qwen3 MoE FP8",
        "qwen3-30b-a3b-full": "Qwen3 MoE Full",
        "qwen3-30b-a3b-gguf": "Qwen3 GGUF (wrong format)"
    }
    
    for dir_name, desc in models.items():
        path = os.path.join(base_path, dir_name)
        if os.path.exists(path):
            files = list(Path(path).glob("*.safetensors"))
            if files:
                print(f"✅ {desc}: {len(files)} safetensor files")
            else:
                other_files = len(list(Path(path).iterdir()))
                print(f"⚠️  {desc}: exists but no safetensors ({other_files} other files)")
        else:
            print(f"❌ {desc}: not downloaded")

def main():
    parser = argparse.ArgumentParser(description="Download models for Unicorn Engine")
    parser.add_argument("--model", choices=["phi4", "granite", "qwen3", "all"], 
                       help="Which model to download")
    parser.add_argument("--qwen-version", choices=["fp8", "full", "thinking", "thinking-fp8"],
                       default="fp8", help="Qwen3 version to download")
    args = parser.parse_args()
    
    base_path = "/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/models"
    os.makedirs(base_path, exist_ok=True)
    
    print("🦄 Unicorn Engine Model Downloader")
    print("=" * 50)
    
    # Check existing models
    check_existing_models(base_path)
    
    if not args.model:
        print("\n📋 Download Options:")
        print("1. Phi-4-mini-instruct (3.8B) - Perfect for iGPU testing, smallest model")
        print("2. Granite-3.3-8B-instruct - Smallest, fastest")
        print("3. Qwen3-30B-A3B MoE - Target model")
        print("4. Download all in order")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            download_phi4_mini(base_path)
        elif choice == "2":
            download_granite_3b(base_path)
        elif choice == "3":
            version = input("Version (fp8/full/thinking/thinking-fp8) [fp8]: ").strip() or "fp8"
            download_qwen3_moe(base_path, version)
        elif choice == "4":
            print("\n🚀 Downloading all models in recommended order...")
            if download_phi4_mini(base_path):
                print("\n✅ Phi-4 ready for testing!")
            if download_granite_3b(base_path):
                print("\n✅ Granite ready for testing!")
            if download_qwen3_moe(base_path, "fp8"):
                print("\n✅ Qwen3 MoE ready for testing!")
        else:
            print("Exiting...")
            return
    else:
        if args.model == "phi4":
            download_phi4_mini(base_path)
        elif args.model == "granite":
            download_granite_3b(base_path)
        elif args.model == "qwen3":
            download_qwen3_moe(base_path, args.qwen_version)
        elif args.model == "all":
            download_phi4_mini(base_path)
            download_granite_3b(base_path)
            download_qwen3_moe(base_path, args.qwen_version)
    
    print("\n📝 Next Steps:")
    print("1. Start with Phi-4-mini for iGPU-only testing")
    print("2. Apply our custom INT4/INT8 quantization")
    print("3. Test on our Unicorn execution engine")
    print("4. Measure TPS and optimize")

if __name__ == "__main__":
    main()