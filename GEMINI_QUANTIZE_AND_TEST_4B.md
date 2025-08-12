# Task: Quantize Gemma-3-4B and Test Performance

## 🎯 Objective
Quantize the Gemma-3-4B model for optimal NPU+iGPU performance and test it with the fixed GPU loading pipeline.

## 📋 Steps

### 1. Quantize Gemma-3-4B Model

```bash
# Navigate to project directory
cd /home/ucadmin/Development/Unicorn-Execution-Engine/

# Activate environment
source /home/ucadmin/activate-uc1-ai-py311.sh

# Run the universal quantizer for 4B model
python3 universal_gemma3_quantizer.py --variant 4b
```

Expected output:
- Input: `/home/ucadmin/Development/AI-Models/gemma-3-4b-it/` (~8GB)
- Output: `./quantized_models/gemma-3-4b-it-quantized/` (~2.5GB)
- INT4 for FFN weights (iGPU)
- INT8 for attention weights (NPU)
- FP16 for small weights

### 2. Update Benchmark to Use 4B Model

```bash
# Create a 4B-specific benchmark
cp benchmark_final_performance.py benchmark_4b_performance.py

# Update the model path
sed -i 's|gemma-3-27b-it-layer-by-layer|gemma-3-4b-it-quantized|g' benchmark_4b_performance.py
```

### 3. Clear Memory and Test

```bash
# Clear file cache
sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"

# Monitor GPU in another terminal
watch -n 0.5 'radeontop -d - -l 1 2>/dev/null | grep -E "(vram|gtt)"'

# Run the benchmark
python3 benchmark_4b_performance.py
```

## 📊 Expected Results

### Memory Usage
- **Quantized 4B model**: ~2.5GB total
- Should fit **entirely in VRAM** (16GB available)
- No GTT needed = faster access

### Performance Targets
- With all optimizations: 200-500 TPS expected
- NPU handles attention efficiently
- iGPU processes FFN layers with INT4

## 🔍 What to Monitor

1. **Successful Quantization**:
   - Size reduction: 8GB → 2.5GB
   - Proper INT4/INT8 distribution
   - All files created

2. **GPU Loading**:
   - VRAM should increase by ~2.5GB
   - No GTT usage needed
   - All layers loaded

3. **Performance**:
   - Actual TPS measurement
   - GPU utilization >80%
   - No CPU compute

## 🚀 Benefits of Testing with 4B

1. **Faster iteration**: Quick load/test cycles
2. **Memory safe**: No OOM issues
3. **Full VRAM**: Everything fits in fast memory
4. **Proves concept**: Same optimizations scale to 27B

## ⚡ Strict Requirements

- **NPU+iGPU ONLY** - No CPU compute!
- All weights must load to GPU
- Must use quantized version

## 📝 Report Back

After testing, please report:
1. Quantization success/stats
2. GPU memory usage (VRAM/GTT)
3. Actual TPS achieved
4. Any issues encountered

This will validate our optimization stack before scaling to the 27B model!