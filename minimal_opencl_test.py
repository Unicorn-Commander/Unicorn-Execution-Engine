#!/usr/bin/env python3.13
"""
Minimal OpenCL test to diagnose environment issues
"""

import pyopencl as cl
import numpy as np
import sys

def test_basic_opencl():
    """Test if basic OpenCL functionality works"""
    print("🔧 Basic OpenCL Environment Test")
    print("=" * 50)
    
    try:
        # List platforms
        platforms = cl.get_platforms()
        print(f"✓ Found {len(platforms)} OpenCL platforms")
        
        for i, platform in enumerate(platforms):
            print(f"  Platform {i}: {platform.name}")
            
            devices = platform.get_devices()
            for j, device in enumerate(devices):
                print(f"    Device {j}: {device.name}")
                print(f"      Type: {cl.device_type.to_string(device.type)}")
                print(f"      Memory: {device.global_mem_size / 1024**3:.1f} GB")
        
        return True
        
    except Exception as e:
        print(f"✗ Platform enumeration failed: {e}")
        return False

def test_context_creation():
    """Test OpenCL context creation"""
    print("\n🔧 Context Creation Test")
    print("=" * 50)
    
    try:
        platforms = cl.get_platforms()
        gpu_devices = []
        
        for platform in platforms:
            devices = platform.get_devices(device_type=cl.device_type.GPU)
            gpu_devices.extend(devices)
        
        if not gpu_devices:
            print("✗ No GPU devices found")
            return False
        
        device = gpu_devices[0]
        print(f"Testing with: {device.name}")
        
        # Create context
        ctx = cl.Context([device])
        queue = cl.CommandQueue(ctx)
        
        print("✓ Context and queue created successfully")
        return True
        
    except Exception as e:
        print(f"✗ Context creation failed: {e}")
        return False

def test_minimal_kernel():
    """Test ultra-minimal kernel execution"""
    print("\n🔧 Minimal Kernel Test")
    print("=" * 50)
    
    try:
        # Setup
        platforms = cl.get_platforms()
        gpu_devices = []
        for platform in platforms:
            devices = platform.get_devices(device_type=cl.device_type.GPU)
            gpu_devices.extend(devices)
        
        if not gpu_devices:
            return False
        
        ctx = cl.Context([gpu_devices[0]])
        queue = cl.CommandQueue(ctx)
        
        # Ultra-simple kernel - just copy a value
        kernel_source = """
        __kernel void copy_value(__global float* input, __global float* output) {
            int i = get_global_id(0);
            if (i == 0) {
                output[0] = input[0];
            }
        }
        """
        
        print("Building kernel...")
        program = cl.Program(ctx, kernel_source).build()
        
        # Single element test
        input_data = np.array([42.0], dtype=np.float32)
        output_data = np.zeros(1, dtype=np.float32)
        
        # Create buffers
        mf = cl.mem_flags
        input_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_data)
        output_buf = cl.Buffer(ctx, mf.WRITE_ONLY, output_data.nbytes)
        
        print("Executing kernel...")
        
        # Execute with single work item
        program.copy_value(queue, (1,), None, input_buf, output_buf)
        
        print("Reading result...")
        cl.enqueue_copy(queue, output_data, output_buf)
        queue.finish()
        
        print(f"Input: {input_data[0]}, Output: {output_data[0]}")
        
        if abs(output_data[0] - input_data[0]) < 1e-6:
            print("✓ Minimal kernel test passed")
            return True
        else:
            print("✗ Kernel output incorrect")
            return False
        
    except Exception as e:
        print(f"✗ Minimal kernel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_add_kernel():
    """Test simple addition kernel"""
    print("\n🔧 Addition Kernel Test")
    print("=" * 50)
    
    try:
        # Setup
        platforms = cl.get_platforms()
        gpu_devices = []
        for platform in platforms:
            devices = platform.get_devices(device_type=cl.device_type.GPU)
            gpu_devices.extend(devices)
        
        ctx = cl.Context([gpu_devices[0]])
        queue = cl.CommandQueue(ctx)
        
        # Simple addition kernel
        kernel_source = """
        __kernel void add_two(__global float* a, __global float* b, __global float* c) {
            int i = get_global_id(0);
            if (i == 0) {
                c[0] = a[0] + b[0];
            }
        }
        """
        
        program = cl.Program(ctx, kernel_source).build()
        
        # Test data
        a = np.array([1.5], dtype=np.float32)
        b = np.array([2.5], dtype=np.float32)
        c = np.zeros(1, dtype=np.float32)
        
        # Buffers
        mf = cl.mem_flags
        a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
        c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)
        
        # Execute
        program.add_two(queue, (1,), None, a_buf, b_buf, c_buf)
        cl.enqueue_copy(queue, c, c_buf)
        queue.finish()
        
        expected = a[0] + b[0]
        print(f"a={a[0]}, b={b[0]}, c={c[0]}, expected={expected}")
        
        if abs(c[0] - expected) < 1e-6:
            print("✓ Addition kernel test passed")
            return True
        else:
            print("✗ Addition kernel output incorrect")
            return False
        
    except Exception as e:
        print(f"✗ Addition kernel test failed: {e}")
        return False

def check_system_info():
    """Check system information related to GPU"""
    print("\n🔧 System Information")
    print("=" * 50)
    
    import subprocess
    import os
    
    try:
        # Check if ROCm is available
        rocm_smi = subprocess.run(['rocm-smi'], capture_output=True, text=True, timeout=5)
        if rocm_smi.returncode == 0:
            print("✓ ROCm is available")
        else:
            print("⚠️  ROCm not found or not working")
    except:
        print("⚠️  ROCm tools not available")
    
    try:
        # Check GPU driver
        lspci = subprocess.run(['lspci', '|', 'grep', 'VGA'], 
                              shell=True, capture_output=True, text=True)
        print(f"GPU: {lspci.stdout.strip()}")
    except:
        print("Could not get GPU info")
    
    # Check environment variables
    print(f"PYOPENCL_CTX: {os.environ.get('PYOPENCL_CTX', 'Not set')}")
    print(f"GPU_MAX_ALLOC_PERCENT: {os.environ.get('GPU_MAX_ALLOC_PERCENT', 'Not set')}")

def main():
    """Run comprehensive OpenCL diagnostics"""
    print("🚨 OpenCL Environment Diagnostics")
    print("=" * 60)
    print("Testing to diagnose GPU hang issues...")
    
    # System info first
    check_system_info()
    
    # Test sequence
    tests = [
        ("Platform Detection", test_basic_opencl),
        ("Context Creation", test_context_creation),
        ("Minimal Kernel", test_minimal_kernel),
        ("Addition Kernel", test_add_kernel),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
            
            if not success:
                print(f"\n❌ {test_name} failed - stopping here")
                break
                
        except Exception as e:
            print(f"\n💥 {test_name} crashed: {e}")
            results.append((test_name, False))
            break
    
    # Summary
    print("\n" + "=" * 60)
    print("🏁 Diagnostic Summary:")
    
    for test_name, success in results:
        status = "✓" if success else "✗"
        print(f"   {status} {test_name}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n✅ Basic OpenCL functionality works!")
        print("   The issue might be specific to complex kernels or larger data")
    else:
        print("\n❌ Basic OpenCL functionality is broken!")
        print("   This confirms a driver/environment issue")
        print("\n🔧 Recommended actions:")
        print("   1. Update AMD GPU drivers")
        print("   2. Check system logs: dmesg | grep amdgpu")
        print("   3. Try: sudo modprobe -r amdgpu && sudo modprobe amdgpu")
        print("   4. Consider using CPU-only Phase 1 implementation")

if __name__ == "__main__":
    main()