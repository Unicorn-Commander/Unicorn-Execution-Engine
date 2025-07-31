#!/usr/bin/env python3.13
"""
🔍 Check what acceleration is actually being used
Verify if we're using CPU or GPU
"""

import os
import numpy as np
import time
import psutil
import subprocess

def check_numpy_config():
    """Check NumPy configuration"""
    print("🔍 NumPy Configuration:")
    print(f"   Version: {np.__version__}")
    print(f"   BLAS: {np.show_config()}")
    
def check_gpu_available():
    """Check if GPU compute is available"""
    print("\n🖥️  GPU Check:")
    
    # Check for AMD GPU
    try:
        result = subprocess.run(['lspci'], capture_output=True, text=True)
        if 'AMD' in result.stdout and ('Radeon' in result.stdout or 'Rembrandt' in result.stdout):
            print("   ✅ AMD GPU detected")
        else:
            print("   ❌ No AMD GPU found")
    except:
        print("   ⚠️  Cannot check GPU")
    
    # Check for ROCm
    if os.path.exists('/opt/rocm'):
        print("   ✅ ROCm installed")
    else:
        print("   ❌ ROCm not found")
    
    # Check for compute libraries
    print("\n📚 Compute Libraries:")
    
    # Check CuPy (ROCm version)
    try:
        import cupy
        print(f"   ✅ CuPy available: {cupy.__version__}")
    except:
        print("   ❌ CuPy not available")
    
    # Check JAX
    try:
        import jax
        print(f"   ✅ JAX available: {jax.__version__}")
        print(f"      Default backend: {jax.default_backend()}")
    except:
        print("   ❌ JAX not available")
    
    # Check PyTorch
    try:
        import torch
        print(f"   ✅ PyTorch available: {torch.__version__}")
        print(f"      CUDA available: {torch.cuda.is_available()}")
        if hasattr(torch, 'hip'):
            print(f"      HIP/ROCm available: {torch.hip.is_available()}")
    except:
        print("   ❌ PyTorch not available")

def test_computation_location():
    """Test where computation actually happens"""
    print("\n🧪 Testing Computation Location:")
    
    # Create large matrices
    size = 4096
    print(f"\n   Creating {size}x{size} matrices...")
    a = np.random.randn(size, size).astype(np.float32)
    b = np.random.randn(size, size).astype(np.float32)
    
    # Monitor CPU before
    cpu_before = psutil.cpu_percent(interval=0.1)
    
    print("   Running matrix multiplication...")
    start = time.time()
    
    # This will use CPU with standard NumPy
    c = np.matmul(a, b)
    
    elapsed = time.time() - start
    cpu_after = psutil.cpu_percent(interval=0.1)
    
    gflops = (2 * size**3) / (elapsed * 1e9)
    
    print(f"\n   Results:")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Performance: {gflops:.1f} GFLOPS")
    print(f"   CPU usage: {cpu_before:.1f}% → {cpu_after:.1f}%")
    
    if cpu_after > cpu_before + 20:
        print("   📍 Computation on CPU (CPU usage increased)")
    else:
        print("   📍 Computation might be on GPU")

def show_actual_situation():
    """Show what's really happening"""
    print("\n" + "="*60)
    print("💡 ACTUAL SITUATION:")
    print("="*60)
    
    print("\n1. NumPy Status:")
    print("   ❌ NumPy uses CPU by default")
    print("   ❌ No automatic GPU acceleration")
    print("   ✅ Good for CPU computation")
    
    print("\n2. Our 'NPU' Claims:")
    print("   ⚠️  pyxrt is loaded but NPU kernels aren't being executed")
    print("   ⚠️  The XCLBIN files exist but aren't being used in inference")
    print("   ❌ No actual NPU acceleration happening")
    
    print("\n3. GPU Acceleration:")
    print("   ❌ Not using ROCm/HIP for compute")
    print("   ❌ Not using OpenCL or Vulkan compute")
    print("   ❌ Standard NumPy = CPU only")
    
    print("\n4. Real Performance:")
    print("   🔹 4B model: ~1-5 TPS on CPU (realistic)")
    print("   🔹 27B model: ~0.2-0.5 TPS on CPU (realistic)")
    print("   🔹 Our claims of 42/287 TPS: Incorrect measurements")

def suggest_real_acceleration():
    """Suggest how to get real acceleration"""
    print("\n" + "="*60)
    print("🚀 TO GET REAL GPU ACCELERATION:")
    print("="*60)
    
    print("\n1. For AMD GPU (ROCm):")
    print("   pip install torch --index-url https://download.pytorch.org/whl/rocm5.7")
    print("   pip install cupy-rocm-5-0")
    
    print("\n2. Replace NumPy operations:")
    print("   # Instead of:")
    print("   np.matmul(a, b)")
    print("   # Use:")
    print("   torch.matmul(a_tensor.cuda(), b_tensor.cuda())")
    
    print("\n3. For NPU:")
    print("   # Actually execute XCLBIN kernels:")
    print("   kernel = pyxrt.kernel(device, xclbin, 'attention_kernel')")
    print("   kernel(buffer_q, buffer_k, buffer_v, buffer_out)")

if __name__ == "__main__":
    print("🔍 ACCELERATION CHECK")
    print("=" * 60)
    
    check_numpy_config()
    check_gpu_available()
    test_computation_location()
    show_actual_situation()
    suggest_real_acceleration()
    
    print("\n\n🎯 CONCLUSION:")
    print("We're running on CPU, not GPU/NPU!")
    print("The performance numbers need to be revised.")
    print("Real acceleration requires proper GPU libraries.")