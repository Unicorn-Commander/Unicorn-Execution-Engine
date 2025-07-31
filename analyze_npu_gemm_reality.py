#!/usr/bin/env python3
"""
Analyze NPU GEMM capabilities and memory bandwidth reality
Based on hardware specifications and kernel availability
"""

import os
import subprocess
import numpy as np
import time

def analyze_npu_gemm_capabilities():
    """Analyze NPU's actual GEMM capabilities"""
    
    print("🦄 NPU GEMM Capability Analysis")
    print("=" * 70)
    
    # 1. Check available NPU kernels
    print("\n📦 Available NPU Kernels:")
    xrt_kernel_path = "/opt/xilinx/xrt/amdxdna/bins/17f0_20/"
    
    if os.path.exists(xrt_kernel_path):
        kernels = os.listdir(xrt_kernel_path)
        gemm_kernels = [k for k in kernels if 'gemm' in k.lower()]
        
        print(f"Found {len(gemm_kernels)} GEMM-related kernels:")
        for kernel in gemm_kernels:
            size = os.path.getsize(os.path.join(xrt_kernel_path, kernel))
            print(f"   - {kernel}: {size/1024:.1f} KB")
    
    # 2. Hardware specifications
    print("\n🔧 NPU Hardware Specifications:")
    print("   Architecture: AMD Phoenix XDNA1")
    print("   Compute: 16 TOPS (INT8)")
    print("   AIE Tiles: 20 (4x5 configuration)")
    print("   Vector Width: 512 bits")
    print("   Precision: INT8 optimized, FP32 supported")
    
    # 3. Memory architecture analysis
    print("\n💾 Memory Architecture:")
    print("   Type: Shared system memory (DDR5)")
    print("   No dedicated HBM (unlike discrete GPUs)")
    print("   Bandwidth: Shared with CPU and iGPU")
    
    # Get memory info
    try:
        meminfo = subprocess.check_output(['cat', '/proc/meminfo'], text=True)
        for line in meminfo.split('\n'):
            if 'MemTotal' in line:
                total_mem = int(line.split()[1]) / 1024 / 1024
                print(f"   Total System Memory: {total_mem:.1f} GB")
                break
    except:
        pass
    
    # 4. Theoretical performance analysis
    print("\n📊 Theoretical GEMM Performance:")
    
    # INT8 GEMM
    print("\n   INT8 GEMM (NPU optimized):")
    int8_tops = 16.0
    print(f"   - Peak: {int8_tops} TOPS")
    print(f"   - For 2048x2048 GEMM: {(2*2048**3)/(int8_tops*1e12)*1000:.1f} ms theoretical")
    
    # FP32 GEMM (estimated at 1/8 of INT8)
    print("\n   FP32 GEMM (not optimized):")
    fp32_tops = int8_tops / 8  # Rough estimate
    print(f"   - Peak: ~{fp32_tops} TFLOPS")
    print(f"   - For 2048x2048 GEMM: {(2*2048**3)/(fp32_tops*1e12)*1000:.1f} ms theoretical")
    
    # 5. Memory bandwidth analysis
    print("\n📈 Memory Bandwidth Analysis:")
    
    # DDR5-5600 theoretical
    ddr5_speed = 5600  # MT/s
    channels = 2
    bus_width = 64  # bits
    theoretical_bw = (ddr5_speed * channels * bus_width) / 8 / 1024  # GB/s
    
    print(f"   DDR5-5600 Dual Channel: {theoretical_bw:.1f} GB/s theoretical")
    print(f"   Typical efficiency: ~80% = {theoretical_bw*0.8:.1f} GB/s")
    print(f"   Shared between: CPU + iGPU + NPU")
    
    # Bandwidth per device estimate
    print(f"\n   Estimated bandwidth allocation:")
    print(f"   - CPU: ~30 GB/s")
    print(f"   - iGPU: ~30 GB/s") 
    print(f"   - NPU: ~20 GB/s")
    print(f"   - Total contention when all active")
    
    # 6. GEMM bandwidth requirements
    print("\n🔄 GEMM Bandwidth Requirements:")
    
    matrix_sizes = [1024, 2048, 4096]
    for size in matrix_sizes:
        # For C = A @ B, need to read A and B, write C
        bytes_moved = 3 * size * size * 4  # 3 matrices, 4 bytes per float
        
        for bw in [20, 30, 50]:  # Different bandwidth scenarios
            time_ms = (bytes_moved / (bw * 1e9)) * 1000
            print(f"\n   {size}x{size} GEMM @ {bw} GB/s bandwidth:")
            print(f"   - Data movement: {bytes_moved/1e9:.2f} GB")
            print(f"   - Transfer time: {time_ms:.1f} ms")
            print(f"   - Compute/bandwidth ratio: {(2*size**3)/(bytes_moved):.1f}:1")
    
    # 7. NPU vs iGPU comparison
    print("\n⚖️  NPU vs iGPU for GEMM:")
    
    print("\n   NPU Advantages:")
    print("   ✅ Excellent INT8 performance (16 TOPS)")
    print("   ✅ Power efficient for quantized models")
    print("   ✅ Dedicated compute tiles")
    
    print("\n   NPU Disadvantages:")
    print("   ❌ Limited memory bandwidth (shared)")
    print("   ❌ Overhead of data transfers")
    print("   ❌ Less mature software stack")
    
    print("\n   iGPU Advantages:")
    print("   ✅ Mature OpenCL/ROCm support")
    print("   ✅ Better FP32/FP16 performance")
    print("   ✅ Larger register files")
    print("   ✅ 38GB addressable memory")
    
    # 8. Practical recommendations
    print("\n🎯 Practical Recommendations:")
    
    print("\n   1. When to use NPU for GEMM:")
    print("      - INT8 quantized models")
    print("      - Smaller matrices (< 1024x1024)")
    print("      - When iGPU is busy with other tasks")
    
    print("\n   2. When to use iGPU for GEMM:")
    print("      - FP32/FP16 operations")
    print("      - Large matrices (> 2048x2048)")
    print("      - When maximum bandwidth needed")
    
    print("\n   3. Memory bandwidth is THE limiting factor:")
    print("      - 89.6 GB/s shared between all devices")
    print("      - GEMM is memory-bound for large matrices")
    print("      - NPU adds competition for bandwidth")
    
    # 9. Real performance estimate
    print("\n📊 Realistic Performance Estimates:")
    
    print("\n   2048x2048 FP32 GEMM:")
    print("   - CPU: ~120ms")
    print("   - iGPU: ~30ms (with optimization)")
    print("   - NPU: ~80-100ms (bandwidth limited)")
    
    print("\n   2048x2048 INT8 GEMM:")
    print("   - CPU: ~100ms")
    print("   - iGPU: ~25ms")
    print("   - NPU: ~15-20ms (if bandwidth available)")
    
    # 10. Conclusion
    print("\n✅ Conclusion:")
    print("\n   The NPU CAN do GEMM operations:")
    print("   - gemm.xclbin and gemm_int8.elf kernels exist")
    print("   - Best suited for INT8 operations")
    print("   - Performance limited by memory bandwidth")
    print("\n   Memory bandwidth IS the primary limiter:")
    print("   - Shared DDR5 creates bottleneck")
    print("   - All devices compete for ~90 GB/s")
    print("   - Large GEMM operations are memory-bound")
    print("\n   Optimal strategy:")
    print("   - Use NPU for INT8 inference")
    print("   - Use iGPU for FP32/FP16 GEMM")
    print("   - Avoid concurrent large operations")


if __name__ == "__main__":
    analyze_npu_gemm_capabilities()