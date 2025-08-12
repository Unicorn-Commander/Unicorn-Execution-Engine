# 🎯 Gemma Model Performance Summary

## Models Tested

### 1. **Gemma 2B Q4_K_M** (1.6GB)
- **CPU**: 28.5 tokens/second
- **GPU**: 39.4 tokens/second 
- **GPU+NPU**: 29.4 tokens/second
- **Best for**: Fast responses, interactive chat, lower quality acceptable

### 2. **Gemma 3n Q8_0** (6.8GB) 
- **CPU**: 10.4 tokens/second
- **GPU**: 13.6 tokens/second
- **GPU+NPU**: 12.4 tokens/second  
- **Best for**: Higher quality outputs, better reasoning

## Key Findings

### 🚀 Performance Insights
1. **Q4 vs Q8 Quantization**: 
   - Q4 models are **2.7x faster** than Q8
   - Minor quality tradeoff for major speed gain
   
2. **GPU Acceleration**:
   - Provides **30-40% speedup** over CPU
   - GPU usage reaches 80-88% during inference
   
3. **NPU Impact**:
   - Currently shows mixed results (sometimes slower)
   - NPU kernels are loading but may need optimization

### 📊 Model Size vs Performance Tradeoff

| Model Size | Quantization | Speed (GPU) | Quality |
|------------|--------------|-------------|---------|
| 1.6GB | Q4_K_M | 39.4 tok/s | Good |
| 6.8GB | Q8_0 | 13.6 tok/s | Excellent |
| ~5.5GB | Q4_K_M (9B) | ~20-25 tok/s* | Very Good |
| ~15GB | Q4_K_M (27B) | ~5-8 tok/s* | Best |

*Estimated based on model size

## Recommendations

### For Speed Priority:
- Use **Gemma 2B Q4_K_M** 
- Enable GPU offloading: `--n-gpu-layers 999`
- Get 35-40 tokens/second

### For Quality Priority:
- Use **Gemma 3n Q8_0** or download **Gemma 9B Q4_K_M**
- Enable GPU offloading
- Get 13-20 tokens/second with better outputs

### For Maximum Quality:
- Download **Gemma 27B Q4_K_M** (requires 15GB+ download)
- Expect 5-8 tokens/second
- Best reasoning and output quality

## Download Commands

```bash
# Gemma 2 9B Q4 (balanced speed/quality)
wget -c https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q4_K_M.gguf

# Gemma 27B Q4 (maximum quality)  
wget -c https://huggingface.co/mradermacher/Gemma-2-Ataraxy-v4d-27B-GGUF/resolve/main/Gemma-2-Ataraxy-v4d-27B.Q4_K_M.gguf
```

## Running Models

```bash
# Fast inference (Gemma 2B Q4)
./llama.cpp/build/bin/llama-cli -m gemma-2b-it-q4_k_m.gguf -p "Your prompt" -n 100 --n-gpu-layers 999

# Quality inference (Gemma 3n Q8)
./llama.cpp/build/bin/llama-cli -m gemma-3n-E4B-it-Q8_0.gguf -p "Your prompt" -n 100 --n-gpu-layers 35 --npu-attention
```