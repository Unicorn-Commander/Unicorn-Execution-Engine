# URGENT: Custom Quantize Gemma-3-4B-IT for NPU+iGPU

## 📍 Current Situation
- GPU loading is WORKING ✅
- We loaded ~8GB which is likely the UNQUANTIZED Gemma-3-4B model
- We need to QUANTIZE it first for optimal NPU+iGPU performance

## 🎯 Task: Quantize Gemma-3-4B-IT

### Source Model
```bash
/home/ucadmin/Development/AI-Models/gemma-3-4b-it/
```

### Target Output
```bash
/home/ucadmin/Development/Unicorn-Execution-Engine/quantized_models/gemma-3-4b-it-quantized/
```

### Quantization Requirements

1. **INT8/INT4 Mixed Precision**:
   - **Attention weights** (Q/K/V/O): INT8 for NPU
   - **FFN weights** (gate/up/down): INT4 for iGPU  
   - **Small weights** (norms, biases): Keep FP16/32

2. **Hardware Allocation**:
   ```python
   # NPU gets attention (optimized for matrix ops)
   if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'o_proj' in name:
       device = 'npu'  # Will use GTT memory
   
   # iGPU gets FFN (massive parallel compute)
   elif 'gate_proj' in name or 'up_proj' in name or 'down_proj' in name:
       device = 'igpu'  # Will use VRAM
   
   # Small weights stay accessible
   else:
       device = 'cpu'  # Quick access for norms/biases
   ```

3. **Expected Size**:
   - Original 4B model: ~8GB
   - Quantized target: ~2-3GB total
   - Should fit entirely in VRAM (16GB available)

## 🔧 Implementation Steps

1. **Check existing quantization scripts**:
   ```bash
   ls -la *.py | grep quant
   ```

2. **Create quantization script** if needed:
   ```python
   # Key components:
   - Load original model
   - Apply INT8 to attention weights
   - Apply INT4 to FFN weights
   - Save in layer-by-layer format
   - Include metadata for dequantization
   ```

3. **Use the same format as 27B model**:
   - Layer-by-layer safetensor files
   - Proper naming convention
   - Compatible with LightningFastLoader

## ✅ Success Criteria

1. **Size Reduction**: 8GB → 2-3GB
2. **Format**: Compatible with existing pipeline
3. **Loading**: Fits entirely in VRAM
4. **Performance**: Ready for NPU+iGPU execution

## 🚀 Quick Test After Quantization

```bash
# Update model path in benchmark
sed -i 's/gemma-3-27b-it-layer-by-layer/gemma-3-4b-it-quantized/g' benchmark_final_performance.py

# Run test
python3 benchmark_final_performance.py
```

## 💡 Benefits of 4B Model

1. **Fits in VRAM**: No GTT needed, faster access
2. **Quick testing**: Iterate optimizations faster
3. **NPU friendly**: Smaller attention matrices
4. **Memory safe**: No OOM issues

## ⚡ STRICT REQUIREMENT

**NPU+iGPU ONLY** - No CPU compute allowed!

Once quantized, we can properly test the full optimization stack with the 4B model before scaling to 27B.