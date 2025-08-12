#!/usr/bin/env python3.13
"""
Test complete NPU inference pipeline
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Test NPU inference"""
    
    print("🦄 Testing NPU Inference Pipeline")
    print("=" * 60)
    
    # Use virtual environment
    venv_python = "npu_kernel_env/bin/python3.13"
    
    # Step 1: Test direct runtime
    print("\n📊 Step 1: Testing NPU Direct Runtime...")
    print("-" * 40)
    
    result = subprocess.run([
        venv_python,
        "npu_direct_runtime.py"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
        
    # Step 2: Run inference executor
    print("\n📊 Step 2: Running NPU Inference...")
    print("-" * 40)
    
    result = subprocess.run([
        venv_python,
        "npu_inference_executor.py"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
        
    # Step 3: Show generated kernels
    print("\n📊 Step 3: Generated inference kernels...")
    print("-" * 40)
    
    kernels_dir = Path("npu_kernels_inference")
    if kernels_dir.exists():
        for model_dir in kernels_dir.iterdir():
            if model_dir.is_dir():
                print(f"\n{model_dir.name}:")
                for kernel in model_dir.glob("*.npu"):
                    size = kernel.stat().st_size
                    print(f"  {kernel.name}: {size:,} bytes")
    
    print("\n✅ NPU inference test complete!")
    
    return 0


if __name__ == "__main__":
    exit(main())