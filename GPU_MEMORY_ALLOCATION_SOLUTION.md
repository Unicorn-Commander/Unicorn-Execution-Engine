# GPU Memory Allocation Issue & Solution

## The Problem

The model is loading entirely into system RAM instead of using VRAM/GTT because:

1. **No PyTorch+ROCm installed**: The current environment lacks PyTorch with ROCm support
2. **numpy/mmap = RAM only**: These methods ALWAYS allocate in system RAM
3. **Environment variables insufficient**: Setting HMA flags alone doesn't move data to GPU

## Current Situation

- System has 16GB VRAM + 38GB GTT available
- Model loads to RAM because we're using numpy arrays
- Without PyTorch+ROCm or HIP APIs, we CANNOT allocate GPU memory

## The Solution

### Step 1: Install PyTorch with ROCm Support

```bash
# For ROCm 6.2 (latest)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Alternative for ROCm 6.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
```

### Step 2: Use Real GPU Memory Allocator

Replace the current numpy-based approach with PyTorch tensors:

```python
import torch

# Enable AMD APU optimizations
os.environ['HSA_ENABLE_UNIFIED_MEMORY'] = '1'
os.environ['HSA_FORCE_FINE_GRAIN_PCIE'] = '1'
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'expandable_segments:True'

# Allocate directly to GPU
device = torch.device('cuda:0')  # ROCm uses CUDA interface

# Load tensor to VRAM
tensor = torch.zeros(shape, device=device)

# Or load from numpy and transfer to GPU
gpu_tensor = torch.from_numpy(numpy_array).to(device)
```

### Step 3: Load Model Directly to GPU

Use the `real_gpu_memory_allocator.py` implementation:

```python
from real_gpu_memory_allocator import RealGPUMemoryAllocator

allocator = RealGPUMemoryAllocator()

# This actually loads to VRAM/GTT
gpu_tensors = allocator.load_model_to_gpu(model_path)
```

## Why Current Approach Fails

| Method | Memory Location | GPU Accessible |
|--------|----------------|----------------|
| numpy.array() | System RAM | No |
| mmap() | System RAM | No |
| "Pinned memory" (our implementation) | System RAM | Theoretically |
| torch.tensor(device='cuda') | VRAM/GTT | Yes ✅ |

## Memory Architecture on AMD APU

```
┌─────────────────────────────────────┐
│         96GB DDR5 Total             │
├─────────────────────────────────────┤
│  16GB VRAM (GPU-dedicated)          │ ← Need PyTorch+ROCm to access
│  38GB GTT (GPU-accessible)          │ ← Need PyTorch+ROCm to access  
│  42GB System RAM                    │ ← numpy/mmap always goes here
└─────────────────────────────────────┘
```

## Verification

After implementing the solution:

```bash
# Check GPU memory usage
rocm-smi --showmeminfo vram
rocm-smi --showmeminfo gtt

# You should see:
# VRAM Used: Several GB (not just 1GB)
# GTT Used: Several GB (not just 0.1GB)
```

## Alternative: Use Existing PyTorch Models

If installing PyTorch+ROCm is not feasible, consider using the existing PyTorch-based implementations in the codebase that already have GPU support:

- `igpu_optimization_engine.py`
- `gpu_memory_pool.py`
- `torch_gemma27b_loader.py`

These implementations already handle GPU memory allocation correctly when PyTorch+ROCm is available.