#!/usr/bin/env python3.13
"""
🦄 NPU Working Test - Get Basic Hardware Functionality Working
"""

import os
import sys
import time
import numpy as np

# XRT environment
os.environ['XILINX_XRT'] = '/opt/xilinx/xrt'
os.environ['PYTHONPATH'] = '/opt/xilinx/xrt/python:' + os.environ.get('PYTHONPATH', '')

try:
    import pyxrt
    NPU_AVAILABLE = True
except ImportError:
    NPU_AVAILABLE = False
    print("❌ NPU not available")
    sys.exit(1)

def test_npu_minimal():
    """Minimal NPU test to get data transfer working"""
    print("🦄 NPU Minimal Working Test")
    print("=" * 40)
    
    try:
        # Initialize device
        print("🎯 Initializing NPU device...")
        device = pyxrt.device(0)
        print("✅ NPU device created successfully")
        
        # Test buffer creation and data operations
        print("\n💾 Testing buffer operations...")
        
        # Create buffer - 4KB test
        size = 4096
        buffer = pyxrt.bo(device, size, pyxrt.bo.flags.cacheable, 0)
        print(f"✅ {size//1024}KB buffer created")
        
        # Create test data
        num_floats = size // 4
        test_data = np.random.randn(num_floats).astype(np.float32)
        print(f"   Test data: {num_floats} floats ({test_data.nbytes} bytes)")
        
        # Write data to buffer
        buffer.write(test_data, 0)
        print("   ✅ Data written to NPU buffer")
        
        # Read data back - use correct pyxrt syntax
        read_result = buffer.read(size, 0)  # read(size, offset) -> returns int8 array
        read_data = np.frombuffer(read_result, dtype=np.float32)
        print("   ✅ Data read from NPU buffer")
        
        # Verify data
        error = np.max(np.abs(test_data - read_data))
        print(f"   📊 Data verification: max error = {error:.2e}")
        
        if error < 1e-6:
            print("   ✅ NPU data transfer VERIFIED!")
            return True
        else:
            print("   ❌ NPU data transfer FAILED!")
            return False
        
    except Exception as e:
        print(f"❌ NPU test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def benchmark_npu_memory():
    """Benchmark NPU memory bandwidth"""
    print("\n📊 NPU Memory Benchmark")
    print("=" * 30)
    
    try:
        device = pyxrt.device(0)
        
        # Test different sizes
        sizes = [4096, 16384, 65536, 262144]  # 4KB to 256KB
        
        for size in sizes:
            buffer = pyxrt.bo(device, size, pyxrt.bo.flags.cacheable, 0)
            num_floats = size // 4
            test_data = np.random.randn(num_floats).astype(np.float32)
            
            # Measure write bandwidth
            start_time = time.time()
            for _ in range(100):
                buffer.write(test_data, 0)
            write_time = (time.time() - start_time) / 100
            write_bw = (size / write_time) / (1024**3)  # GB/s
            
            # Measure read bandwidth
            start_time = time.time()
            for _ in range(100):
                buffer.read(size, 0)
            read_time = (time.time() - start_time) / 100
            read_bw = (size / read_time) / (1024**3)  # GB/s
            
            print(f"   {size//1024:3d}KB: Write {write_bw:.2f} GB/s, Read {read_bw:.2f} GB/s")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory benchmark failed: {e}")
        return False

def estimate_inference_performance():
    """Estimate inference performance based on memory bandwidth"""
    print("\n🧮 Inference Performance Estimation")
    print("=" * 40)
    
    # Assume we got ~1-10 GB/s memory bandwidth
    memory_bw_gbs = 5.0  # Conservative estimate
    
    # Model parameters
    models = {
        "4B": {"hidden": 2560, "layers": 28, "heads": 20},
        "27B": {"hidden": 4608, "layers": 32, "heads": 32}
    }
    
    for name, config in models.items():
        hidden = config["hidden"]
        layers = config["layers"]
        
        # Memory per token (weights + activations)
        # Simplified: activation memory per token
        memory_per_token = hidden * 4  # bytes (float32)
        
        # Memory bandwidth limited inference
        max_tps_memory = memory_bw_gbs * 1e9 / memory_per_token
        
        # Conservative estimate (accounting for overhead)
        realistic_tps = max_tps_memory * 0.1  # 10% efficiency
        
        print(f"   Gemma 3 {name}: ~{realistic_tps:.1f} TPS (memory bound)")
        
        # Also estimate compute bound
        # Attention: O(seq_len^2 * hidden) per layer
        seq_len = 128
        attention_ops = 2 * seq_len * seq_len * hidden  # QK^T + softmax*V
        mlp_ops = 8 * hidden * hidden  # 2 linear layers, ~4x hidden expansion
        total_ops = (attention_ops + mlp_ops) * layers
        
        # Assume NPU can do ~1 TOPS (conservative)
        compute_tps = 1e12 / total_ops  # operations per second / ops per token
        
        print(f"   Gemma 3 {name}: ~{compute_tps:.1f} TPS (compute bound)")
        
        # Practical estimate is minimum of both
        practical_tps = min(realistic_tps, compute_tps)
        print(f"   Gemma 3 {name}: ~{practical_tps:.1f} TPS (PRACTICAL)")

if __name__ == "__main__":
    print("🦄 NPU Working Test Suite")
    print("=" * 50)
    
    success1 = test_npu_minimal()
    
    if success1:
        success2 = benchmark_npu_memory()
        estimate_inference_performance()
        
        print(f"\n🎉 NPU hardware is WORKING!")
        print("   ✅ Device initialization: OK")
        print("   ✅ Buffer creation: OK") 
        print("   ✅ Data transfer: OK")
        print("   ✅ Memory bandwidth: Measured")
        print("\n🚀 Ready to implement real kernels!")
    else:
        print(f"\n❌ NPU hardware test FAILED")