# GEMINI-CLI TASK: Quantize Gemma-3-4B-IT for Unicorn Execution Engine

## 📍 Context
We need to test the GPU loading fix with a smaller model (4B instead of 27B) to avoid memory issues. The Unicorn Execution Engine requires models in a specific quantized format.

## 🎯 Your Mission
Quantize the Gemma-3-4B-IT model to match the format expected by our custom pipeline.

## 📂 Locations
- **Source Model**: `/home/ucadmin/Development/AI-Models/gemma-3-4b-it/`
- **Target Directory**: `/home/ucadmin/Development/Unicorn-Execution-Engine/quantized_models/gemma-3-4b-it-layer-by-layer/`
- **Reference Format**: Look at `/home/ucadmin/Development/Unicorn-Execution-Engine/quantized_models/gemma-3-27b-it-layer-by-layer/` for the expected structure

## 🔧 Requirements

### 1. **Layer-by-Layer Organization**
The 27B model is split into files like:
```
model-00001-of-00012_layer_0.safetensors (414MB)
model-00001-of-00012_layer_1.safetensors (190MB)
model-00001-of-00012_layer_10.safetensors (15MB)
...
```

### 2. **Quantization Format**
- Use INT8 asymmetric quantization for most weights
- Use INT4 grouped quantization for large weights (>1MB)
- Preserve the exact tensor naming convention
- Keep metadata for dequantization

### 3. **Expected Structure**
Each layer file should contain tensors like:
```
language_model.model.layers.0.self_attn.q_proj.weight
language_model.model.layers.0.self_attn.k_proj.weight
language_model.model.layers.0.self_attn.v_proj.weight
language_model.model.layers.0.mlp.gate_proj.weight
...
```

### 4. **Integration with Pipeline**
The quantized model must work with:
- `LightningFastLoader` 
- `pure_hardware_pipeline_fixed.py`
- The existing INT8/INT4 Vulkan compute shaders

## 📝 Steps

1. **Analyze the existing 4B model structure**:
   ```bash
   ls -la /home/ucadmin/Development/AI-Models/gemma-3-4b-it/
   ```

2. **Check the quantization code** used for the 27B model:
   - Look for any quantization scripts in the project
   - Match the exact format and metadata

3. **Quantize the 4B model**:
   - Split into layer-by-layer files
   - Apply INT8/INT4 quantization
   - Preserve tensor names and structure

4. **Verify the output**:
   - Total size should be ~4GB (from ~8-9GB unquantized)
   - Files should follow the same naming pattern
   - Test loading with the pipeline

## 🎯 Success Criteria
- Quantized model in `quantized_models/gemma-3-4b-it-layer-by-layer/`
- Same file structure as the 27B model
- Loads successfully with `LightningFastLoader`
- ~50% size reduction from quantization

## 💡 Hints
- The project might have existing quantization scripts
- Check how `lightning_fast_loader.py` expects the data
- INT4 quantization is used for weights >1MB to save memory
- The layer-by-layer split helps with parallel loading

Good luck! This smaller model will help us debug the GPU loading issue much faster.