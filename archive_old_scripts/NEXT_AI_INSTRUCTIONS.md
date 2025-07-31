# 🦄 Instructions for Next AI - NPU+iGPU Fast Loading & Inference

## Current Status
- **Model**: Fully downloaded at `/home/ucadmin/Development/Unicorn-Execution-Engine/quantized_models/gemma-3-27b-it-layer-by-layer/` (25.9GB)
- **Hardware**: NPU detected, GPU working, all drivers loaded
- **Problem 1**: Model loading takes 2+ minutes (single CPU core + transpose operations)
- **Problem 2**: Inference uses CPU instead of strict NPU+iGPU

## Critical Files to Fix

### 1. Main Pipeline File (NEEDS FIXING)
**Path**: `/home/ucadmin/Development/Unicorn-Execution-Engine/pure_hardware_pipeline_fixed.py`

**Current Issues**:
- Line 203: `actual_tensor = self.loader.get_tensor(weight_info)` - loads to CPU RAM
- Line 209: `actual_tensor = actual_tensor.T` - CPU transpose operation
- Uses slow single-threaded loader

**Fix Required**:
- Replace loader with `LightningFastLoader` or `LightningLoaderNPUOptimized`
- Remove transpose operations (or do on GPU)
- Implement direct GPU memory mapping

### 2. Fast Loader (ALREADY EXISTS)
**Path**: `/home/ucadmin/Development/Unicorn-Execution-Engine/lightning_fast_loader.py`
- Uses all 16 CPU cores
- Memory maps files
- Just needs integration

### 3. NPU+iGPU Strict Pipeline (CREATED BUT NEEDS INTEGRATION)
**Path**: `/home/ucadmin/Development/Unicorn-Execution-Engine/pure_hardware_pipeline_npu_strict.py`
- Enforces NPU for attention, GPU for FFN
- No CPU fallback allowed
- Needs proper integration with model loading

## Key Requirements

### Memory Distribution
```
NPU SRAM (2GB): Attention weights for first 2-4 layers
GPU VRAM (16GB): Embeddings + critical layers  
GPU GTT (40GB): Bulk FFN weights
```

### Strict Execution Rules
1. **NPU MUST handle ALL attention** - no fallback to GPU/CPU
2. **GPU MUST handle ALL FFN** - no fallback to CPU
3. **Zero CPU compute during inference**

## Test Commands

### Check Current Performance
```bash
cd /home/ucadmin/Development/Unicorn-Execution-Engine
source /home/ucadmin/ai-env-py311/bin/activate

# This currently takes 2+ minutes to load
python test_npu_gpu_strict.py
```

### Monitor Hardware Usage
```bash
# In separate terminal - watch GPU usage
watch -n 0.5 'radeontop -d - -l 1'

# Check CPU usage - should be near 0% during inference
htop
```

## Expected Results After Fix
- **Loading time**: 10-15 seconds (like Ollama)
- **Inference**: 81+ TPS
- **CPU usage**: ~0% during inference
- **GPU usage**: High during FFN computation
- **NPU usage**: Active during attention

## Critical Integration Points

### 1. Fix Model Loading
```python
# In pure_hardware_pipeline_fixed.py, replace:
self.loader = PureMemoryMappedLoader(model_path)

# With:
from lightning_fast_loader import LightningFastLoader
self.loader = LightningFastLoader(model_path)
```

### 2. Remove CPU Operations
```python
# Remove all transpose operations on CPU
# Instead, either:
# 1. Pre-transpose during quantization
# 2. Transpose on GPU after loading
# 3. Use transposed operations in compute
```

### 3. Enforce NPU Usage
```python
# In forward_layer, enforce NPU for attention:
if not self.npu_kernel.available:
    raise RuntimeError("NPU required - no fallback!")
attention_output = self.npu_kernel.compute_attention(...)
```

## Environment Setup
```bash
cd /home/ucadmin/Development/Unicorn-Execution-Engine
source /home/ucadmin/ai-env-py311/bin/activate
```

## Key People/Context
- This is for "Magic Unicorn Unconventional Technology & Stuff" - an AI company
- The goal is unconventional direct hardware execution
- NO PyTorch/frameworks during inference
- Must achieve Ollama-like loading speed
- Must use NPU+iGPU with zero CPU compute

## Files Already Created
- `/home/ucadmin/Development/Unicorn-Execution-Engine/lightning_loader_npu_optimized.py` - Optimized parallel loader
- `/home/ucadmin/Development/Unicorn-Execution-Engine/pure_hardware_pipeline_npu_strict.py` - Strict NPU+GPU pipeline
- `/home/ucadmin/Development/Unicorn-Execution-Engine/NPU_GPU_OPTIMIZATION_PLAN.md` - Detailed plan

The infrastructure exists - it just needs proper integration!