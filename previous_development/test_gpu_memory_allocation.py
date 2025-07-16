#!/usr/bin/env python3
"""
Test GPU Memory Allocation
Verify that model loads to VRAM/GTT instead of RAM
"""

import os
import sys
import subprocess
import psutil
import time

def get_gpu_memory_info():
    """Get GPU memory usage from amdgpu_top or rocm-smi"""
    
    try:
        # Try amdgpu_top first
        result = subprocess.run(['amdgpu_top', '-d'], capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            output = result.stdout
            # Parse VRAM and GTT usage
            for line in output.split('\n'):
                if 'VRAM' in line and 'GTT' in line:
                    print(f"📊 {line.strip()}")
            return True
    except:
        pass
    
    try:
        # Try rocm-smi
        result = subprocess.run(['rocm-smi', '--showmeminfo', 'vram'], capture_output=True, text=True)
        if result.returncode == 0:
            print("📊 VRAM Info:")
            print(result.stdout)
        
        result = subprocess.run(['rocm-smi', '--showmeminfo', 'gtt'], capture_output=True, text=True)
        if result.returncode == 0:
            print("📊 GTT Info:")
            print(result.stdout)
        return True
    except:
        pass
    
    # Fallback to sysfs
    try:
        from pathlib import Path
        gpu_paths = list(Path('/sys/class/drm').glob('card*/device'))
        
        for gpu_path in gpu_paths:
            vendor_path = gpu_path / 'vendor'
            if vendor_path.exists() and vendor_path.read_text().strip() == '0x1002':
                # AMD GPU found
                vram_total = gpu_path / 'mem_info_vram_total'
                vram_used = gpu_path / 'mem_info_vram_used'
                gtt_total = gpu_path / 'mem_info_gtt_total'
                gtt_used = gpu_path / 'mem_info_gtt_used'
                
                if vram_total.exists():
                    vram_total_gb = int(vram_total.read_text()) / (1024**3)
                    vram_used_gb = int(vram_used.read_text()) / (1024**3) if vram_used.exists() else 0
                    print(f"📊 VRAM: {vram_used_gb:.1f}GB / {vram_total_gb:.1f}GB")
                
                if gtt_total.exists():
                    gtt_total_gb = int(gtt_total.read_text()) / (1024**3)
                    gtt_used_gb = int(gtt_used.read_text()) / (1024**3) if gtt_used.exists() else 0
                    print(f"📊 GTT: {gtt_used_gb:.1f}GB / {gtt_total_gb:.1f}GB")
                
                return True
    except Exception as e:
        print(f"⚠️ Could not read GPU memory info: {e}")
    
    return False

def get_system_memory_info():
    """Get system RAM usage"""
    
    mem = psutil.virtual_memory()
    print(f"📊 System RAM: {mem.used / (1024**3):.1f}GB / {mem.total / (1024**3):.1f}GB ({mem.percent:.1f}%)")
    
    # Get process memory
    process = psutil.Process()
    rss = process.memory_info().rss / (1024**3)
    print(f"📊 Process RSS: {rss:.1f}GB")

def main():
    print("🧪 Testing GPU Memory Allocation")
    print("=" * 60)
    
    # Show initial memory state
    print("\n📊 Initial Memory State:")
    get_system_memory_info()
    get_gpu_memory_info()
    
    # Test different allocation methods
    print("\n🔧 Testing memory allocation methods...")
    
    # Method 1: Test with HMA allocator
    print("\n1️⃣ Testing HMA GPU Memory Allocator:")
    try:
        from hma_gpu_memory_allocator import HMAGPUMemoryAllocator
        
        allocator = HMAGPUMemoryAllocator()
        stats = allocator.get_memory_stats()
        
        print(f"   VRAM available: {stats['vram_free_gb']:.1f}GB")
        print(f"   GTT available: {stats['gtt_free_gb']:.1f}GB")
        
        # Test allocation
        print("\n   Allocating 2GB to GTT...")
        buffer = allocator.allocate_gpu_memory(2 * 1024**3, 'gtt')
        
        time.sleep(1)
        print("\n   After allocation:")
        get_gpu_memory_info()
        
    except Exception as e:
        print(f"   ❌ HMA allocator test failed: {e}")
    
    # Method 2: Test with PyTorch ROCm
    print("\n2️⃣ Testing PyTorch ROCm allocation:")
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"   ✅ ROCm available: {torch.cuda.get_device_name()}")
            
            # Enable GTT allocation
            os.environ['HSA_FORCE_FINE_GRAIN_PCIE'] = '1'
            os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'expandable_segments:True'
            
            # Allocate tensor
            print("\n   Allocating 4GB tensor to GPU...")
            tensor = torch.zeros((1024, 1024, 1024), dtype=torch.float32, device='cuda')
            
            time.sleep(1)
            print("\n   After allocation:")
            get_gpu_memory_info()
            
            # Check memory info
            print(f"\n   PyTorch GPU memory: {torch.cuda.memory_allocated() / (1024**3):.1f}GB allocated")
            print(f"   PyTorch GPU memory: {torch.cuda.memory_reserved() / (1024**3):.1f}GB reserved")
            
        else:
            print("   ❌ PyTorch ROCm not available")
            
    except ImportError:
        print("   ❌ PyTorch not installed")
    except Exception as e:
        print(f"   ❌ PyTorch test failed: {e}")
    
    # Method 3: Check environment variables
    print("\n3️⃣ Environment variables for HMA:")
    important_vars = [
        'HSA_ENABLE_UNIFIED_MEMORY',
        'HSA_FORCE_FINE_GRAIN_PCIE',
        'PYTORCH_HIP_ALLOC_CONF',
        'HIP_VISIBLE_DEVICES',
        'HSA_OVERRIDE_GFX_VERSION'
    ]
    
    for var in important_vars:
        value = os.environ.get(var, 'Not set')
        print(f"   {var}: {value}")
    
    # Final memory state
    print("\n📊 Final Memory State:")
    get_system_memory_info()
    get_gpu_memory_info()
    
    print("\n✅ GPU memory allocation test complete!")


if __name__ == "__main__":
    main()