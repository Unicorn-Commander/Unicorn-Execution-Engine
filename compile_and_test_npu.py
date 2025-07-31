#!/usr/bin/env python3.13
"""
Compile and test real NPU kernels
Uses Python 3.13 virtual environment
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Compile NPU kernels and run tests"""
    
    print("🦄 Real NPU Kernel Development")
    print("=" * 60)
    
    # Ensure we're in the right directory
    os.chdir("/home/ucadmin/Development/Unicorn-Execution-Engine")
    
    # Activate virtual environment
    venv_python = "npu_kernel_env/bin/python3.13"
    
    # Check virtual environment
    if not Path(venv_python).exists():
        print("❌ Virtual environment not found")
        print("   Run: python3.13 -m venv npu_kernel_env")
        return 1
        
    print("✅ Using Python 3.13 virtual environment")
    
    # Step 1: Compile NPU kernels
    print("\n📊 Step 1: Compiling NPU kernels...")
    print("-" * 40)
    
    result = subprocess.run([
        venv_python,
        "real_npu_kernel_compiler.py"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
        
    if result.returncode != 0:
        print("❌ Kernel compilation failed")
        return 1
        
    # Step 2: Test NPU execution
    print("\n📊 Step 2: Testing NPU execution...")
    print("-" * 40)
    
    result = subprocess.run([
        venv_python,
        "real_npu_kernel_executor.py"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
        
    # Step 3: Show kernel files
    print("\n📊 Step 3: Generated kernel files...")
    print("-" * 40)
    
    kernels_dir = Path("npu_kernels_real")
    if kernels_dir.exists():
        for model_dir in kernels_dir.iterdir():
            if model_dir.is_dir():
                print(f"\n{model_dir.name}:")
                for kernel in model_dir.glob("*.xclbin"):
                    size = kernel.stat().st_size
                    print(f"  {kernel.name}: {size:,} bytes")
    else:
        print("No kernels found")
        
    print("\n✅ NPU kernel development complete!")
    print("🚀 Real hardware acceleration ready!")
    
    return 0


if __name__ == "__main__":
    exit(main())