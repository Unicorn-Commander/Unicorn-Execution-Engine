#!/usr/bin/env python3.13
"""
Simple hardware test with correct APIs
"""

import os
import sys
import numpy as np

# Set XRT environment
os.environ['XILINX_XRT'] = '/opt/xilinx/xrt'
os.environ['PYTHONPATH'] = '/opt/xilinx/xrt/python:' + os.environ.get('PYTHONPATH', '')

print("🦄 Simple Hardware Test")
print("=" * 50)

# Test 1: NPU via pyxrt
print("\n1️⃣ Testing NPU...")
try:
    import pyxrt
    print("✅ pyxrt imported")
    
    # Try to create device directly
    try:
        device = pyxrt.device(0)
        print("✅ NPU device created")
        print(f"   Device: {device}")
        npu_ok = True
    except Exception as e:
        print(f"❌ NPU device error: {e}")
        npu_ok = False
        
except ImportError as e:
    print(f"❌ pyxrt import failed: {e}")
    npu_ok = False

# Test 2: Basic numpy computation
print("\n2️⃣ Testing computation...")
try:
    # Small matrix multiply
    size = 128
    a = np.random.randn(size, size).astype(np.float32)
    b = np.random.randn(size, size).astype(np.float32)
    c = np.matmul(a, b)
    print(f"✅ Matrix multiply {size}x{size} successful")
    compute_ok = True
except Exception as e:
    print(f"❌ Compute failed: {e}")
    compute_ok = False

# Test 3: Check model files
print("\n3️⃣ Checking model files...")
model_path = "/home/ucadmin/Development/Unicorn-Execution-Engine/quantized_models/gemma-3-4b-it-quantized"
if os.path.exists(model_path):
    files = os.listdir(model_path)
    print(f"✅ Found {len(files)} model files")
    for f in sorted(files)[:5]:  # Show first 5
        print(f"   - {f}")
    model_ok = True
else:
    print("❌ Model path not found")
    model_ok = False

# Summary
print("\n" + "=" * 50)
print("📊 SUMMARY")
print("=" * 50)
print(f"NPU Available: {'✅' if npu_ok else '❌'}")
print(f"Compute Ready: {'✅' if compute_ok else '❌'}")  
print(f"Model Files: {'✅' if model_ok else '❌'}")

if npu_ok or compute_ok:
    print("\n✅ Basic hardware test passed!")
    print("🚀 Ready to attempt inference")
else:
    print("\n❌ Hardware not ready")

print("\n💡 Next: Let's create a minimal inference test")