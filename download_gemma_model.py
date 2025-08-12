#!/usr/bin/env python3
"""
Download and convert Gemma model for NPU testing
"""

import os
import subprocess
import sys
import requests
import json
from pathlib import Path

def check_dependencies():
    """Check if required tools are installed"""
    print("🔍 Checking dependencies...")
    
    # Check for huggingface-cli
    hf_cli = subprocess.run(["which", "huggingface-cli"], capture_output=True).returncode == 0
    if not hf_cli:
        print("📦 Installing huggingface-hub...")
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface-hub", "--quiet"])
    
    # Check for convert script
    convert_script = "llama.cpp/convert_hf_to_gguf.py"
    if not os.path.exists(convert_script):
        print("❌ Convert script not found at", convert_script)
        return False
    
    return True

def download_gemma_2b():
    """Download Gemma 2B model from Hugging Face"""
    print("\n📥 Downloading Gemma 2B model...")
    
    # Create models directory
    os.makedirs("models/gemma-2b-it", exist_ok=True)
    
    # First check if we can find a pre-converted GGUF
    gguf_repos = [
        "bartowski/gemma-2-2b-it-GGUF",
        "QuantFactory/gemma-2-2b-it-GGUF", 
        "lmstudio-community/gemma-2-2b-it-GGUF"
    ]
    
    for repo in gguf_repos:
        print(f"\n🔍 Checking {repo} for GGUF files...")
        
        # Try to download Q4_K_M version
        for filename in ["gemma-2-2b-it-Q4_K_M.gguf", "gemma-2-2b-it.Q4_K_M.gguf", "gemma-2-2b-it-q4_k_m.gguf"]:
            url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
            
            print(f"   Trying: {filename}")
            try:
                # Check if file exists with HEAD request
                response = requests.head(url, allow_redirects=True, timeout=10)
                if response.status_code == 200:
                    # Download the file
                    print(f"   ✅ Found! Downloading {filename}...")
                    
                    cmd = [
                        "wget", 
                        "--progress=bar:force",
                        "-O", "gemma-2b-it-q4_k_m.gguf",
                        url
                    ]
                    
                    result = subprocess.run(cmd)
                    if result.returncode == 0 and os.path.exists("gemma-2b-it-q4_k_m.gguf"):
                        file_size = os.path.getsize("gemma-2b-it-q4_k_m.gguf") / (1024**3)
                        print(f"\n✅ Successfully downloaded Gemma 2B GGUF ({file_size:.2f} GB)")
                        return True
            except Exception as e:
                continue
    
    print("\n📥 No pre-converted GGUF found. Downloading original model...")
    
    # Download original model using huggingface-cli
    cmd = [
        sys.executable, "-m", "huggingface_hub", "download",
        "google/gemma-2b-it",
        "--local-dir", "models/gemma-2b-it",
        "--local-dir-use-symlinks", "False"
    ]
    
    print("This may take a while...")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("❌ Failed to download model")
        return False
    
    return True

def convert_to_gguf():
    """Convert Gemma model to GGUF format"""
    if os.path.exists("gemma-2b-it-q4_k_m.gguf"):
        print("✅ GGUF model already exists")
        return True
        
    print("\n🔄 Converting to GGUF format...")
    
    # Check if we have the original model
    if not os.path.exists("models/gemma-2b-it/config.json"):
        print("❌ Original model not found")
        return False
    
    # Convert to GGUF
    cmd = [
        sys.executable,
        "llama.cpp/convert_hf_to_gguf.py",
        "models/gemma-2b-it",
        "--outfile", "gemma-2b-it-f16.gguf",
        "--outtype", "f16"
    ]
    
    print("Converting to F16 GGUF...")
    result = subprocess.run(cmd)
    
    if result.returncode != 0 or not os.path.exists("gemma-2b-it-f16.gguf"):
        print("❌ Conversion failed")
        return False
    
    # Quantize to Q4_K_M
    print("\n📦 Quantizing to Q4_K_M...")
    cmd = [
        "./llama.cpp/build/bin/llama-quantize",
        "gemma-2b-it-f16.gguf",
        "gemma-2b-it-q4_k_m.gguf",
        "Q4_K_M"
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0 and os.path.exists("gemma-2b-it-q4_k_m.gguf"):
        print("✅ Successfully created Q4_K_M quantized model")
        # Clean up F16 version to save space
        if os.path.exists("gemma-2b-it-f16.gguf"):
            os.remove("gemma-2b-it-f16.gguf")
        return True
    
    return False

def test_inference():
    """Test inference with the Gemma model"""
    if not os.path.exists("gemma-2b-it-q4_k_m.gguf"):
        print("❌ Model not found")
        return
        
    print("\n🚀 Testing Gemma 2B inference...")
    print("=" * 60)
    
    # Test 1: GPU-only baseline
    print("\n📊 Test 1: GPU-only (Vulkan)")
    cmd = [
        "./llama.cpp/build/bin/llama-cli",
        "-m", "gemma-2b-it-q4_k_m.gguf",
        "-p", "The future of AI acceleration is",
        "-n", "50",
        "--gpu-layers", "999",
        "-c", "2048"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-500:] if result.stdout else "No output")
    
    # Test 2: NPU+GPU acceleration
    print("\n📊 Test 2: NPU+GPU acceleration")
    cmd = [
        "./llama.cpp/build/bin/llama-cli",
        "-m", "gemma-2b-it-q4_k_m.gguf",
        "-p", "The future of AI acceleration is",
        "-n", "50",
        "--npu-attention",
        "--gpu-layers", "999",
        "-c", "2048"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-500:] if result.stdout else "No output")
    
    # Also show any NPU messages from stderr
    if "NPU" in result.stderr:
        print("\n🔍 NPU Status:")
        for line in result.stderr.split('\n'):
            if "NPU" in line or "npu" in line:
                print(f"   {line}")

def main():
    print("""
    🦄 Gemma Model Download & Convert
    ==================================
    This will download and prepare a Gemma model for NPU testing
    """)
    
    if not check_dependencies():
        print("❌ Missing dependencies")
        return
    
    # Try to download GGUF directly first
    if download_gemma_2b():
        if os.path.exists("gemma-2b-it-q4_k_m.gguf"):
            test_inference()
        else:
            # Need to convert
            if convert_to_gguf():
                test_inference()
            else:
                print("❌ Conversion failed")
    else:
        print("❌ Download failed")
        print("\n💡 You can manually download from:")
        print("   https://huggingface.co/bartowski/gemma-2-2b-it-GGUF")
        print("   Look for Q4_K_M quantized version")

if __name__ == "__main__":
    main()