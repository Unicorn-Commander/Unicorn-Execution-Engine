# Prompt for Gemini - Phi-4-mini Implementation

Please implement Phi-4-mini-instruct (3.8B parameters) on our custom Unicorn Execution Engine. This is a dense model perfect for iGPU-only execution and will serve as our baseline for testing our custom quantization methods.

## 🎯 Your Mission
Get Phi-4-mini running on our custom inference engine with our own INT4/INT8 quantization, achieving 50+ TPS (target: 80-100 TPS).

## 📁 Critical File Paths

### Project Root
```
/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/
```

### Implementation Plan & Tracking Documents
- **Detailed Plan**: `/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/PHI4_MINI_IMPLEMENTATION_PLAN.md`
- **Performance Tracker**: `/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/MODEL_PERFORMANCE_TRACKER.md`
- **Quantization Checklist**: `/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/QUANTIZATION_CHECKLIST.md`

### Model Download Script
```bash
cd /home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/
python download_models_for_unicorn.py
# Choose option 1 for Phi-4-mini
```

### Model Location (after download)
```
/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/models/phi-4-mini-instruct/
```

### Existing Infrastructure You'll Use

#### Core Components
- **Quantization Engine**: `dynamic_quantization_engine.py` - Handles INT4/INT8 quantization
- **Vulkan Compute**: `vulkan_compute_workaround.py` - GPU acceleration with NumPy fallback
- **BF16 Converter**: `bfloat16_converter.py` - For tensor format conversion

#### Reference Implementations
- **Model Loader Example**: `gemma_27b_loader_v2.py` - Shows how to load safetensors
- **Pipeline Example**: `gemma_27b_working_pipeline.py` - Shows working inference pipeline
- **Benchmark Example**: `qwen3_30b_moe/benchmark_moe.py` - Comprehensive benchmarking

#### Compiled Vulkan Shaders (ready to use)
- `matrix_multiply_int8.spv` - INT8 matrix multiplication
- `rdna3_int4.spv` - RDNA3-optimized INT4 operations
- `transformer_optimized.spv` - Optimized transformer ops

### Files You'll Create
1. `phi4_mini_loader.py` - Adapt from gemma_27b_loader_v2.py
2. `phi4_mini_pipeline.py` - Inference pipeline for Phi-4
3. `phi4_benchmark_results.json` - Store your benchmark results

## 🔧 Technical Requirements

### Quantization Implementation (Not Just Testing!)
1. **Actually implement our custom quantization** using:
   - Our Vulkan compute shaders (INT4/INT8)
   - RDNA3-specific optimizations 
   - Custom memory layout for our hardware
2. **Use our compiled shaders**:
   - `matrix_multiply_int8.spv` for INT8 operations
   - `rdna3_int4.spv` for INT4 operations  
   - `transformer_optimized.spv` for fused operations
3. **Progressive optimization**:
   - INT8 baseline implementation
   - INT4 for memory-bound layers (FFN)
   - Mixed precision based on layer importance

### Hardware Configuration
- **Primary**: iGPU (AMD Radeon 780M) with our custom shaders
- **Memory**: Optimize VRAM (16GB) + GTT distribution
- **NPU**: Not needed for Phi-4 (dense model), but prepare modular design

### Success Metrics
- **Minimum**: 50 TPS
- **Target**: 80-100 TPS  
- **Quality**: Perplexity within 5% of FP16 baseline

## 📋 Implementation Steps (Not Just Testing!)

1. **Download the model** using the script above

2. **Create phi4_mini_loader.py**
   - Load safetensors files
   - Apply our custom quantization
   - Store quantized weights properly

3. **Implement custom quantization pipeline**
   - Use `dynamic_quantization_engine.py` to quantize weights
   - Create Vulkan buffers for quantized weights
   - Load our compiled shaders (INT8/INT4)

4. **Create phi4_mini_pipeline.py** with:
   ```python
   # Use our actual Vulkan compute
   from vulkan_compute_workaround import VulkanComputeWorkaround
   
   # Initialize with our shaders
   self.vulkan = VulkanComputeWorkaround()
   self.vulkan.load_shader('matrix_multiply_int8.spv')
   self.vulkan.load_shader('rdna3_int4.spv')
   ```

5. **Implement layer execution**
   - Attention: INT8 with `matrix_multiply_int8.spv`
   - FFN: INT4 with `rdna3_int4.spv` 
   - Use `transformer_optimized.spv` for fused operations

6. **Optimize memory layout**
   - Frequently accessed → VRAM
   - Large weights → GTT
   - Stream from disk if needed

7. **Benchmark with our framework**
   - Use existing benchmark_moe.py as template
   - Measure real TPS, not simulated

8. **Document everything** in the trackers

## 🚨 Important Notes

- We're NOT using pre-quantized models (no GGUF/ONNX)
- We're using our CUSTOM quantization engine
- The model should run on iGPU only (no NPU for dense models)
- Start with batch size 1, then test 2, 4, 8
- Document EVERYTHING in the performance tracker

## 💡 Tips

- Look at `vulkan_compute_workaround.py` - it handles Vulkan errors gracefully
- The INT4 quantization in `dynamic_quantization_engine.py` uses group size 16
- Phi-4 uses standard transformer architecture, similar to Gemma
- With only 3.8B params, you should see very high TPS

## ✅ Definition of Done

- [ ] Model downloads successfully
- [ ] Baseline FP16 inference works
- [ ] INT8 quantization applied and tested
- [ ] INT4 quantization applied and tested
- [ ] Achieved >50 TPS (ideally 80-100)
- [ ] Results documented in MODEL_PERFORMANCE_TRACKER.md
- [ ] Optimal configuration identified and saved

Good luck! This should be a great starting point for our custom inference engine. Remember - we're building something that runs on our custom stack, not using any pre-made solutions!