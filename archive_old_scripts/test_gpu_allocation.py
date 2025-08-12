#!/usr/bin/env python3
"""Test GPU memory allocation directly"""

import numpy as np
import time
from real_vulkan_matrix_compute import VulkanMatrixCompute

def test_gpu_allocation():
    print("Testing GPU memory allocation...")
    
    # Initialize Vulkan
    vulkan = VulkanMatrixCompute()
    if not vulkan.initialize():
        print("Failed to initialize Vulkan")
        return
    
    print("✅ Vulkan initialized")
    
    # Test allocating different sizes
    test_sizes_mb = [100, 500, 1000, 2000]
    
    allocated_buffers = []
    
    for size_mb in test_sizes_mb:
        size_bytes = size_mb * 1024 * 1024
        elements = size_bytes // 4  # float32
        
        # Create test tensor
        tensor = np.random.randn(elements).astype(np.float32)
        
        print(f"\nAllocating {size_mb}MB to GPU...")
        
        try:
            # Test VRAM allocation
            gpu_buffer = vulkan._allocate_gpu_memory(tensor)
            allocated_buffers.append(gpu_buffer)
            print(f"✅ Allocated {size_mb}MB to VRAM")
            
            # Test GTT allocation
            gtt_buffer = vulkan._allocate_gtt_memory(tensor)
            allocated_buffers.append(gtt_buffer)
            print(f"✅ Allocated {size_mb}MB to GTT")
            
        except Exception as e:
            print(f"❌ Allocation failed: {e}")
    
    # Check memory usage
    print("\nChecking GPU memory usage...")
    import subprocess
    result = subprocess.run(['radeontop', '-d', '-', '-l', '1'], 
                          capture_output=True, text=True)
    
    # Parse VRAM and GTT from output
    output = result.stdout
    if 'vram' in output:
        lines = output.strip().split('\n')
        for line in lines:
            if 'vram' in line and 'gtt' in line:
                print(f"GPU Memory Status: {line}")
                break
    
    print("\nTest complete. Buffers allocated:", len(allocated_buffers))
    
    # Keep buffers alive for a moment to see memory usage
    time.sleep(2)

if __name__ == "__main__":
    test_gpu_allocation()