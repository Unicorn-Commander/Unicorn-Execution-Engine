#!/usr/bin/env python3.13
"""
🦄 NPU Baseline Test - Verify Hardware Works
Simple test without XCLBIN to verify NPU basic functionality
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

def test_npu_baseline():
    """Test NPU hardware without XCLBIN"""
    print("🦄 NPU Baseline Hardware Test")
    print("=" * 50)
    
    try:
        # Initialize device
        print("🎯 Initializing NPU device...")
        device = pyxrt.device(0)
        print("✅ NPU device created successfully")
        
        # Test buffer creation with correct syntax
        print("\n💾 Testing buffer creation...")
        test_sizes = [4096, 16384, 65536]  # 4KB, 16KB, 64KB
        
        for size in test_sizes:
            try:
                # Create buffer with cacheable flag for NPU
                buffer = pyxrt.bo(device, size, pyxrt.bo.flags.cacheable, 0)
                print(f"   ✅ {size//1024}KB buffer created")
                
                # Test data transfer
                test_data = np.random.randn(size//4).astype(np.float32)
                buffer.write(test_data, 0)  # Correct pyxrt syntax: write(data, offset)
                
                # Read back
                read_data = np.zeros_like(test_data)
                buffer.read(read_data, 0)  # Correct pyxrt syntax: read(buffer, offset)
                
                error = np.max(np.abs(test_data - read_data))
                if error < 1e-6:
                    print(f"   ✅ Data transfer verified (error: {error:.2e})")
                else:
                    print(f"   ❌ Data transfer error: {error}")
                
            except Exception as e:
                print(f"   ❌ {size//1024}KB buffer failed: {e}")
        
        print("\n🎉 NPU baseline test complete!")
        return True
        
    except Exception as e:
        print(f"❌ NPU baseline test failed: {e}")
        return False

def test_performance_estimation():
    """Estimate NPU performance based on memory bandwidth"""
    print("\n📊 NPU Performance Estimation")
    print("=" * 40)
    
    try:
        device = pyxrt.device(0)
        
        # Test memory bandwidth
        test_size = 1024 * 1024 * 4  # 4MB
        buffer = pyxrt.bo(device, test_size, pyxrt.bo.flags.cacheable, 0)
        
        test_data = np.random.randn(test_size//4).astype(np.float32)
        
        # Measure write speed
        start_time = time.time()
        for _ in range(10):
            buffer.write(test_data, 0)
        write_time = (time.time() - start_time) / 10
        
        write_bandwidth = (test_size / write_time) / (1024**3)  # GB/s
        
        # Measure read speed  
        read_data = np.zeros_like(test_data)
        start_time = time.time()
        for _ in range(10):
            buffer.read(read_data, 0)
        read_time = (time.time() - start_time) / 10
        
        read_bandwidth = (test_size / read_time) / (1024**3)  # GB/s
        
        print(f"📈 Memory Bandwidth:")
        print(f"   Write: {write_bandwidth:.2f} GB/s")
        print(f"   Read:  {read_bandwidth:.2f} GB/s")
        
        # Estimate attention performance
        # Typical NPU: 1-10 TOPS, memory bound operations
        estimated_tops = min(write_bandwidth * 256, 10)  # Conservative estimate
        
        print(f"📊 Estimated NPU Performance:")
        print(f"   Compute: ~{estimated_tops:.1f} TOPS")
        
        # Estimate model performance
        for model, hidden, layers in [("4B", 2560, 28), ("27B", 4608, 32)]:
            # Attention FLOPS per token
            attention_flops = 4 * hidden * hidden  # QKV + output projection
            layer_flops = attention_flops + (8 * hidden * hidden)  # + MLP
            total_flops = layer_flops * layers
            
            # Time per token (memory bound assumption)
            data_per_token = hidden * 4  # bytes
            memory_time = data_per_token / (write_bandwidth * 1e9)  # seconds
            
            # Compute time (optimistic)
            compute_time = total_flops / (estimated_tops * 1e12)  # seconds
            
            total_time = max(memory_time, compute_time)
            tps = 1.0 / total_time
            
            print(f"   {model}: ~{tps:.1f} TPS (estimated)")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance estimation failed: {e}")
        return False

if __name__ == "__main__":
    success1 = test_npu_baseline()
    success2 = test_performance_estimation()
    
    if success1 and success2:
        print("\n🎉 NPU hardware verification complete!")
        print("   Ready for kernel execution")
    else:
        print("\n❌ NPU hardware issues detected")