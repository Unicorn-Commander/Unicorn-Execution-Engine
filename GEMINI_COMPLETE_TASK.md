# GEMINI-CLI COMPREHENSIVE TASK: Fix GPU Loading & Quantize Gemma-4B

## 🚀 PROJECT CONTEXT
**Working Directory**: `/home/ucadmin/Development/Unicorn-Execution-Engine/`
**Environment**: `source /home/ucadmin/activate-uc1-ai-py311.sh`
**Goal**: Achieve 81+ TPS using NPU+iGPU only (no CPU compute)

## 🔴 CRITICAL ISSUE
The GPU loading is broken. Model weights are NOT loading to VRAM/GTT, preventing all performance testing. All optimizations are implemented (30,600x theoretical speedup) but we can't test until GPU loading works.

## 📋 YOUR TASKS

### TASK 1: Fix GPU Loading in LightningFastLoader

**Problem**: The `LightningFastLoader` returns "pre-loaded weights" (references) instead of actual tensor data that can be transferred to GPU.

**Files to examine**:
```bash
/home/ucadmin/Development/Unicorn-Execution-Engine/lightning_fast_loader.py
/home/ucadmin/Development/Unicorn-Execution-Engine/pure_hardware_pipeline_fixed.py
/home/ucadmin/Development/Unicorn-Execution-Engine/real_vulkan_matrix_compute.py
```

**What's happening**:
1. `LightningFastLoader` loads model to CPU memory
2. Returns dictionary with "pre-loaded weights" message
3. `_load_tensor_to_gpu` expects raw tensor data but gets references
4. GPU memory stays at baseline (~1GB instead of ~16GB expected)

**Fix needed**:
- Modify `LightningFastLoader` to support direct GPU loading OR
- Return actual tensor data that can be transferred to GPU OR
- Create a new GPU-specific loader

**Test the fix**:
```bash
# Monitor GPU memory
watch -n 0.5 'radeontop -d - -l 1 2>/dev/null | grep -E "(vram|gtt)"'

# Run test
python3 benchmark_final_performance.py
```

**Success = VRAM increases to ~16GB**

---

### TASK 2: Quantize Gemma-3-4B-IT Model

**Why**: Testing with 4B model (4GB) is easier than 27B model (26GB) - avoids memory issues.

**Source**: `/home/ucadmin/Development/AI-Models/gemma-3-4b-it/`
**Target**: `/home/ucadmin/Development/Unicorn-Execution-Engine/quantized_models/gemma-3-4b-it.uce`

**Reference format**: Check how the 27B model is structured:
```bash
ls -la /home/ucadmin/Development/Unicorn-Execution-Engine/quantized_models/gemma-3-27b-it-layer-by-layer/
```

**Quantization requirements**:
1. INT8 asymmetric for weights <1MB
2. INT4 grouped for weights ≥1MB
3. Preserve exact tensor names (e.g., `language_model.model.layers.0.self_attn.q_proj.weight`)
4. Include quantization metadata for dequantization

**Custom UCE format suggestion**:
```python
# Single file with custom extension: .uce (Unicorn Compressed Engine)
# Structure:
{
    "format_version": "1.0",
    "model_info": {
        "name": "gemma-3-4b-it",
        "layers": 28,  # or whatever 4B has
        "hidden_size": 3072,  # or actual size
        "quantization": "mixed_int8_int4"
    },
    "layer_data": {
        0: {
            "self_attn.q_proj.weight": {"data": <bytes>, "dtype": "int4", "shape": [...], "metadata": {...}},
            "self_attn.k_proj.weight": {...},
            ...
        },
        1: {...},
        ...
    }
}
```

**OR keep layer-by-layer files** if that's what the loader expects.

---

### TASK 3: Update Pipeline for 4B Model

Once quantized, update paths to test with 4B model:
```python
# In benchmark_final_performance.py or test script
model_path = "/home/ucadmin/Development/Unicorn-Execution-Engine/quantized_models/gemma-3-4b-it.uce"
# OR
model_path = "/home/ucadmin/Development/Unicorn-Execution-Engine/quantized_models/gemma-3-4b-it-layer-by-layer/"
```

---

## 🔧 KEY TECHNICAL DETAILS

### GPU Memory Allocation Methods
```python
# In real_vulkan_matrix_compute.py
self._allocate_gpu_memory(buffer)  # Allocates to VRAM
self._allocate_gtt_memory(buffer)  # Allocates to GTT
# Both return: (buffer, memory, size_bytes)
```

### Expected Memory Distribution
- **4B Model**: ~2GB VRAM + ~2GB GTT = 4GB total
- **27B Model**: ~16GB VRAM + ~10GB GTT = 26GB total

### Vision Components
Currently skipped with:
```python
if 'vision_tower' in weight_name or 'vision' in weight_name:
    logger.info(f"Skipping vision component: {weight_name}")
    continue
```

---

## 📊 TESTING CHECKLIST

1. **Quantize 4B model** ✓
2. **Fix GPU loading** ✓
3. **Verify with radeontop**:
   - VRAM should increase by ~2GB for 4B model
   - GPU utilization should be >0% during inference
4. **Run performance test**:
   ```bash
   python3 benchmark_final_performance.py
   ```
5. **If successful with 4B**, apply same fix to 27B model

---

## 🎯 SUCCESS CRITERIA

1. **GPU Loading Fixed**:
   - VRAM usage increases appropriately (2GB for 4B, 16GB for 27B)
   - No more "pre-loaded weights" issue
   - Tensors actually in GPU memory

2. **4B Model Quantized**:
   - Size ~4GB (50% reduction)
   - Loads with fixed pipeline
   - Same format as 27B model

3. **Performance**:
   - 4B model should achieve proportionally high TPS
   - Proves the optimization stack works
   - Ready to scale to 27B for final 81+ TPS target

---

## 💡 HELPFUL HINTS

1. The issue is in how `LightningFastLoader` handles tensor data
2. Check `self.layer_loader` - it returns pre-loaded data
3. GPU allocation works fine - just needs actual tensor data
4. Consider creating a `.uce` format for cleaner model distribution
5. All optimizations are ready - just need GPU loading to work!

Good luck! Once GPU loading works, we should see incredible performance!