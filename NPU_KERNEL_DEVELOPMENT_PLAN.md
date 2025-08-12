# NPU Kernel Development Plan for Gemma Models

## Overview

We need to build **real NPU kernels** specifically for each Gemma model variant. The transcription project showed us HOW the Phoenix NPU works - now we need to build kernels for LLM inference.

## Current Status

### ✅ What We Have:
1. **NPU Architecture Knowledge** (from transcription project):
   - AMD Phoenix NPU (XDNA1, 16 TOPS, AIE v1.1)
   - Memory banks: 131071 (DMA), 65536/65537 (compute)
   - IOCTL interface for direct hardware access
   - 16 AIE tiles in 4x4 configuration

2. **Infrastructure Ready**:
   - llama.cpp with NPU integration (`--npu-attention` flag)
   - Direct NPU Runtime integrated from transcription project
   - NPU device accessible (`/dev/accel/accel0`)

3. **MLIR Kernel Sources**:
   - `npu_kernels/gemma-3-4b-attention/attention_kernel.mlir`
   - Templates for attention operations

### ❌ What We Need:
1. **Compiled NPU Kernels** for:
   - Gemma3n (1.5B parameters)
   - Gemma3 4B 
   - Gemma3 27B

2. **Kernel Variants** for different sequence lengths:
   - 128, 256, 512, 1024, 2048 tokens

## Development Approach

### Option A: MLIR-AIE Compilation (Recommended)
```bash
# 1. Install MLIR-AIE toolchain
python3 install_mlir_aie2_toolchain.py

# 2. Build kernels for all models
python3 build_gemma_npu_kernels.py

# 3. Test with llama.cpp
./llama.cpp/build/bin/llama-cli \
    -m gemma-3-4b.gguf \
    --npu-attention \
    --gpu-layers 999
```

### Option B: Direct Binary Kernel Creation
If MLIR-AIE is unavailable, we can create kernels directly:
- Use binary format from transcription project
- Implement attention operations in NPU assembly
- Package as XCLBIN files

## Kernel Specifications

### Gemma3n (1.5B)
```
Hidden Size: 1536
Heads: 12 (no GQA)
Head Dim: 128
Intermediate: 6144
```

### Gemma3 4B
```
Hidden Size: 2560
Heads: 32, KV Heads: 16 (GQA 2:1)
Head Dim: 80
Intermediate: 10240
```

### Gemma3 27B
```
Hidden Size: 4608
Heads: 48, KV Heads: 8 (GQA 6:1)
Head Dim: 96
Intermediate: 18432
```

## NPU Kernel Architecture

### Attention Operation Flow:
1. **QKV Projections** (INT8 GEMM)
   - Distributed across 16 AIE tiles
   - Use DMA bank (131071) for weight streaming

2. **Attention Scores** (FP16)
   - Q @ K^T computation
   - Softmax in FP16 precision
   - Use compute banks (65536/65537)

3. **Value Application** (FP16 → INT8)
   - Attention weights @ V
   - Quantize back to INT8

### Memory Layout:
```
Bank 131071 (2MB): Weight streaming, DMA operations
Bank 65536 (1MB): Activations, compute buffer 0
Bank 65537 (1MB): Activations, compute buffer 1
```

## Expected Performance

Based on transcription project (2,985x RT = 220x CPU speedup):

### Current (GPU-only):
- TinyLlama: 58 tok/s
- Gemma 2B: 40 tok/s

### Target (NPU+GPU):
- Gemma3n: ~8,000 tok/s
- Gemma3 4B: ~5,000 tok/s
- Gemma3 27B: ~1,000 tok/s

## Implementation Steps

### 1. Install Toolchain
```bash
# Install MLIR-AIE
python3 install_mlir_aie2_toolchain.py
```

### 2. Build Kernels
```bash
# Build all Gemma kernels
python3 build_gemma_npu_kernels.py
```

### 3. Update NPU Integration
Update `npu_stub.cpp` to load correct kernels based on model architecture:
- Detect model dimensions
- Select appropriate kernel
- Handle GQA expansion

### 4. Test & Benchmark
```bash
# Test each model
python3 test_gemma_npu_integration.py

# Benchmark performance
python3 benchmark_npu_igpu_gemma.py
```

## Alternative: Custom Execution Engine

If llama.cpp integration proves difficult, we can use the custom Unicorn Execution Engine:
- Direct control over NPU scheduling
- Custom memory management
- Optimized for Phoenix NPU

```python
# Use custom engine
python3 unicorn_execution_engine.py \
    --model gemma-3-4b \
    --npu-kernels npu_kernels_compiled/gemma3_4b/
```

## Next Actions

1. **Immediate**: Run `build_gemma_npu_kernels.py` to create kernels
2. **Test**: Verify kernel loading with simple inference
3. **Optimize**: Tune kernel parameters for maximum performance
4. **Deploy**: Integrate with llama.cpp or custom engine

## Success Criteria

- [ ] MLIR kernels compile to XCLBIN format
- [ ] NPU executes attention operations
- [ ] 100x+ speedup vs CPU demonstrated
- [ ] All three Gemma models supported
- [ ] Stable inference at high token rates

---

The path is clear: we need to compile model-specific NPU kernels using the architecture knowledge from the transcription project. This will unlock the full 16 TOPS of the Phoenix NPU for LLM inference! 🦄✨