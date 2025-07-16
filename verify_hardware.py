#!/usr/bin/env python3
"""
Verify hardware setup for Unicorn Execution Engine
"""
import os
import subprocess
import sys

def check_npu():
    """Check NPU availability"""
    print("🔍 Checking NPU...")
    
    # Check device
    if os.path.exists("/dev/accel/accel0"):
        print("  ✅ NPU device found: /dev/accel/accel0")
    else:
        print("  ❌ NPU device not found")
        return False
    
    # Check XRT
    try:
        result = subprocess.run(["xrt-smi", "examine"], capture_output=True, text=True)
        if "NPU Phoenix" in result.stdout:
            print("  ✅ NPU Phoenix detected via XRT")
        else:
            print("  ⚠️  XRT installed but NPU not detected")
    except FileNotFoundError:
        print("  ❌ XRT not installed (xrt-smi not found)")
    
    # Check driver
    driver_loaded = subprocess.run(["lsmod"], capture_output=True, text=True)
    if "amdxdna" in driver_loaded.stdout:
        print("  ✅ AMDXDNA driver loaded")
    else:
        print("  ⚠️  AMDXDNA driver not loaded")
    
    return True

def check_gpu():
    """Check GPU availability"""
    print("\n🔍 Checking GPU...")
    
    # Check Vulkan
    try:
        result = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True)
        if "AMD Radeon" in result.stdout:
            print("  ✅ AMD Radeon GPU detected via Vulkan")
            # Extract GPU name
            for line in result.stdout.split('\n'):
                if "deviceName" in line and "AMD" in line:
                    print(f"  ✅ {line.strip()}")
                    break
        else:
            print("  ❌ No AMD GPU detected via Vulkan")
            return False
    except FileNotFoundError:
        print("  ❌ Vulkan not installed (vulkaninfo not found)")
        return False
    
    # Check ROCm
    if os.path.exists("/opt/rocm/bin/rocminfo"):
        print("  ✅ ROCm installed")
    else:
        print("  ⚠️  ROCm not found (not required for pure hardware)")
    
    return True

def check_memory():
    """Check system memory"""
    print("\n🔍 Checking Memory...")
    
    # Get memory info
    with open('/proc/meminfo', 'r') as f:
        for line in f:
            if line.startswith('MemTotal'):
                mem_kb = int(line.split()[1])
                mem_gb = mem_kb / (1024 * 1024)
                print(f"  ✅ Total RAM: {mem_gb:.1f} GB")
                if mem_gb < 32:
                    print("  ⚠️  Less than 32GB RAM may cause issues")
                break
    
    # Check GPU memory
    try:
        vram_files = [
            "/sys/class/drm/card0/device/mem_info_vram_total",
            "/sys/class/drm/card1/device/mem_info_vram_total"
        ]
        for vram_file in vram_files:
            if os.path.exists(vram_file):
                with open(vram_file, 'r') as f:
                    vram_bytes = int(f.read().strip())
                    vram_gb = vram_bytes / (1024**3)
                    print(f"  ✅ GPU VRAM: {vram_gb:.1f} GB")
                break
    except:
        print("  ⚠️  Could not read GPU VRAM info")

def check_vulkan_shaders():
    """Check if Vulkan shaders are compiled"""
    print("\n🔍 Checking Vulkan Shaders...")
    
    shader_files = [
        "matrix_multiply.spv",
        "gate_up_silu_mul.spv",
        "down_proj.spv",
        "rdna3_optimized.spv",
        "rdna3_attention.spv",
        "rdna3_int4.spv"
    ]
    
    missing = []
    for shader in shader_files:
        if os.path.exists(shader):
            print(f"  ✅ {shader}")
        else:
            missing.append(shader)
    
    if missing:
        print(f"  ⚠️  Missing shaders: {', '.join(missing)}")
        print("  ℹ️  Run compile_rdna3_shaders.sh to compile")
    
    return len(missing) == 0

def check_npu_kernels():
    """Check if NPU kernels are present"""
    print("\n🔍 Checking NPU Kernels...")
    
    kernel_dir = "npu_kernels"
    if os.path.exists(kernel_dir):
        kernels = [f for f in os.listdir(kernel_dir) if f.endswith('.bin')]
        if kernels:
            print(f"  ✅ Found {len(kernels)} NPU kernels:")
            for kernel in sorted(kernels):
                size_kb = os.path.getsize(os.path.join(kernel_dir, kernel)) / 1024
                print(f"    - {kernel}: {size_kb:.1f} KB")
        else:
            print("  ⚠️  No NPU kernel binaries found")
    else:
        print("  ❌ NPU kernels directory not found")

def main():
    """Run all hardware checks"""
    print("🦄 Unicorn Execution Engine - Hardware Verification\n")
    
    # Check each component
    npu_ok = check_npu()
    gpu_ok = check_gpu()
    check_memory()
    check_vulkan_shaders()
    check_npu_kernels()
    
    # Summary
    print("\n📊 Summary:")
    if npu_ok and gpu_ok:
        print("  ✅ Hardware ready for Unicorn Execution Engine!")
        print("  ℹ️  NPU may have SMU errors - GPU-only mode works fine")
    else:
        print("  ❌ Some hardware components missing")
        print("  ℹ️  GPU-only mode may still work")
    
    return 0 if (npu_ok and gpu_ok) else 1

if __name__ == "__main__":
    sys.exit(main())