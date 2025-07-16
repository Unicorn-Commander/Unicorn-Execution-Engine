#!/usr/bin/env python3
"""
Check the status of the quantized model download
"""
import os
from pathlib import Path
import json

def check_model_status():
    model_dir = Path("quantized_models/gemma-3-27b-it-layer-by-layer")
    
    print("🔍 Checking model status...")
    print(f"📁 Model directory: {model_dir}")
    
    if not model_dir.exists():
        print("❌ Model directory does not exist!")
        return
    
    # Count files by type
    safetensor_files = list(model_dir.glob("*.safetensors"))
    json_files = list(model_dir.glob("*.json"))
    
    print(f"\n📊 File counts:")
    print(f"  - Safetensor files: {len(safetensor_files)}")
    print(f"  - JSON files: {len(json_files)}")
    
    # Check for critical files
    print("\n🔍 Checking for critical files:")
    critical_files = [
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model-00001-of-00012_shared_weights.safetensors"
    ]
    
    for file in critical_files:
        if (model_dir / file).exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - MISSING")
    
    # Check layer coverage
    print("\n📊 Layer coverage:")
    layer_nums = []
    for f in safetensor_files:
        if "_layer_" in f.name:
            try:
                layer_num = int(f.name.split("_layer_")[1].split(".")[0])
                layer_nums.append(layer_num)
            except:
                pass
    
    if layer_nums:
        layer_nums.sort()
        print(f"  - Layers found: {len(layer_nums)}")
        print(f"  - Range: {min(layer_nums)} to {max(layer_nums)}")
        
        # Check for missing layers
        expected_layers = set(range(62))  # Gemma 27B has 62 layers
        found_layers = set(layer_nums)
        missing = expected_layers - found_layers
        if missing:
            print(f"  ❌ Missing layers: {sorted(missing)}")
        else:
            print(f"  ✅ All 62 layers present!")
    
    # Check total size
    total_size = sum(f.stat().st_size for f in safetensor_files)
    print(f"\n💾 Total model size: {total_size / (1024**3):.1f} GB")
    
    # Check for shared weights
    shared_weights = [f for f in safetensor_files if "shared" in f.name.lower()]
    if shared_weights:
        print(f"\n✅ Found shared weights: {[f.name for f in shared_weights]}")
    else:
        print("\n❌ No shared weights file found - this contains embeddings!")
        print("   The model needs a shared_weights.safetensors file")
    
    print("\n🎯 Recommendation:")
    if len(json_files) == 0:
        print("  - Model configuration files are missing")
    if not shared_weights:
        print("  - Shared weights file is missing (embeddings, layer norms)")
        print("  - This is critical for inference!")
    if len(safetensor_files) < 63:  # 62 layers + 1 shared
        print(f"  - Continue downloading ({len(safetensor_files)}/63 files)")

if __name__ == "__main__":
    check_model_status()