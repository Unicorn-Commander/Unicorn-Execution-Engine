#!/usr/bin/env python3
"""
Download a small Gemma model for NPU testing
"""

import os
import subprocess
import sys

def download_gemma_2b():
    """Download Gemma 2B from Hugging Face"""
    print("🔍 Checking for Gemma 2B model...")
    
    # Check if already exists
    if os.path.exists("gemma-2b-it-q4_k_m.gguf"):
        print("✅ Gemma 2B model already exists")
        return True
        
    print("📥 Downloading Gemma 2B from Hugging Face...")
    
    # Try to download using wget
    urls = [
        # Quantized versions that might be available
        "https://huggingface.co/google/gemma-2b-it-GGUF/resolve/main/gemma-2b-it-q4_k_m.gguf",
        "https://huggingface.co/TheBloke/gemma-2b-it-GGUF/resolve/main/gemma-2b-it.Q4_K_M.gguf",
        "https://huggingface.co/ggml-org/gemma-2b-it-Q4_K_M-GGUF/resolve/main/gemma-2b-it-q4_k_m.gguf"
    ]
    
    for url in urls:
        print(f"Trying: {url}")
        result = subprocess.run(
            ["wget", "-q", "--show-progress", "-O", "gemma-2b-it-q4_k_m.gguf", url],
            capture_output=False
        )
        if result.returncode == 0 and os.path.exists("gemma-2b-it-q4_k_m.gguf"):
            print("✅ Successfully downloaded Gemma 2B")
            return True
            
    print("❌ Could not download Gemma 2B automatically")
    print("\nManual download instructions:")
    print("1. Visit: https://huggingface.co/google/gemma-2b-it")
    print("2. Download the GGUF quantized version")
    print("3. Place it in this directory as 'gemma-2b-it-q4_k_m.gguf'")
    return False

def run_inference_test():
    """Run a quick inference test"""
    if not os.path.exists("gemma-2b-it-q4_k_m.gguf"):
        print("❌ Model not found")
        return
        
    print("\n🚀 Running NPU+GPU inference test...")
    
    # First try GPU-only to get baseline
    print("\n📊 GPU-only baseline:")
    subprocess.run([
        "./llama.cpp/build/bin/llama-cli",
        "-m", "gemma-2b-it-q4_k_m.gguf",
        "-p", "The future of AI acceleration is",
        "-n", "30",
        "--gpu-layers", "999",
        "--log-disable"
    ])
    
    print("\n📊 NPU+GPU acceleration:")
    subprocess.run([
        "./llama.cpp/build/bin/llama-cli",
        "-m", "gemma-2b-it-q4_k_m.gguf",
        "-p", "The future of AI acceleration is",
        "-n", "30",
        "--npu-attention",
        "--gpu-layers", "999",
        "--log-disable"
    ])

if __name__ == "__main__":
    if download_gemma_2b():
        run_inference_test()
    else:
        print("\n💡 For now, let's test with a compatible model size")
        print("The NPU kernels are optimized for Gemma architecture models")