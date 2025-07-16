#!/usr/bin/env python3
"""
Demonstrates the difference between fake and real GPU memory allocation
"""

import os
import subprocess
import time
import numpy as np

def get_memory_usage():
    """Get current memory usage"""
    # System RAM
    result = subprocess.run(['free', '-b'], capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if line.startswith('Mem:'):
            parts = line.split()
            ram_used_gb = int(parts[2]) / (1024**3)
            break
    
    # GPU VRAM
    vram_used_gb = 0
    result = subprocess.run(['rocm-smi', '--showmeminfo', 'vram'], capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if 'Used Memory' in line and 'GPU[0]' in line:
            vram_used_bytes = int(line.split(':')[-1].strip())
            vram_used_gb = vram_used_bytes / (1024**3)
            break
    
    # GPU GTT  
    gtt_used_gb = 0
    result = subprocess.run(['rocm-smi', '--showmeminfo', 'gtt'], capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if 'Used Memory' in line and 'GPU[0]' in line:
            gtt_used_bytes = int(line.split(':')[-1].strip())
            gtt_used_gb = gtt_used_bytes / (1024**3)
            break
    
    return ram_used_gb, vram_used_gb, gtt_used_gb

def test_fake_gpu_allocation():
    """Current approach - doesn't actually use GPU memory"""
    print("\n❌ FAKE GPU Allocation (Current Approach)")
    print("=" * 50)
    
    # Get initial memory
    ram_before, vram_before, gtt_before = get_memory_usage()
    print(f"Before: RAM={ram_before:.1f}GB, VRAM={vram_before:.1f}GB, GTT={gtt_before:.1f}GB")
    
    # Allocate 4GB with numpy (goes to RAM)
    print("Allocating 4GB with numpy...")
    data = np.zeros((1024, 1024, 1024), dtype=np.float32)  # 4GB
    
    # Check memory after
    ram_after, vram_after, gtt_after = get_memory_usage()
    print(f"After:  RAM={ram_after:.1f}GB, VRAM={vram_after:.1f}GB, GTT={gtt_after:.1f}GB")
    
    print(f"\nResult:")
    print(f"  RAM increased: {ram_after - ram_before:.1f}GB ❌")
    print(f"  VRAM increased: {vram_after - vram_before:.1f}GB") 
    print(f"  GTT increased: {gtt_after - gtt_before:.1f}GB")
    print("  ❌ Data is in system RAM, not GPU memory!")

def test_real_gpu_allocation():
    """Correct approach - actually uses GPU memory"""
    print("\n✅ REAL GPU Allocation (With PyTorch+ROCm)")
    print("=" * 50)
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("❌ PyTorch+ROCm not available")
            return
            
        # Get initial memory
        ram_before, vram_before, gtt_before = get_memory_usage()
        print(f"Before: RAM={ram_before:.1f}GB, VRAM={vram_before:.1f}GB, GTT={gtt_before:.1f}GB")
        
        # Allocate 4GB on GPU
        print("Allocating 4GB to GPU with PyTorch...")
        device = torch.device('cuda:0')
        tensor = torch.zeros((1024, 1024, 1024), dtype=torch.float32, device=device)
        
        # Force synchronization
        torch.cuda.synchronize()
        time.sleep(1)
        
        # Check memory after
        ram_after, vram_after, gtt_after = get_memory_usage()
        print(f"After:  RAM={ram_after:.1f}GB, VRAM={vram_after:.1f}GB, GTT={gtt_after:.1f}GB")
        
        print(f"\nResult:")
        print(f"  RAM increased: {ram_after - ram_before:.1f}GB")
        print(f"  VRAM increased: {vram_after - vram_before:.1f}GB ✅") 
        print(f"  GTT increased: {gtt_after - gtt_before:.1f}GB")
        print("  ✅ Data is in GPU memory (VRAM or GTT)!")
        
    except ImportError:
        print("❌ PyTorch not installed")
        print("To fix: pip install torch --index-url https://download.pytorch.org/whl/rocm6.2")

def main():
    print("🔍 GPU Memory Allocation Demo")
    print("This shows why the model isn't loading to VRAM/GTT")
    
    # Enable AMD APU optimizations
    os.environ['HSA_ENABLE_UNIFIED_MEMORY'] = '1'
    os.environ['HSA_FORCE_FINE_GRAIN_PCIE'] = '1'
    
    test_fake_gpu_allocation()
    test_real_gpu_allocation()
    
    print("\n📝 Summary:")
    print("- numpy/mmap ALWAYS allocates in system RAM")
    print("- To use VRAM/GTT, you MUST use PyTorch+ROCm or HIP APIs")
    print("- Environment variables alone don't move data to GPU")
    print("- The current implementation needs PyTorch+ROCm to work")

if __name__ == "__main__":
    main()