# PROMPT FOR NEXT AI SESSION

## IMMEDIATE CONTEXT
Please read the CLAUDE.md file first for full project status:
```
/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/CLAUDE.md
```

## CURRENT STATUS
- ✅ GPU memory allocation FIXED (16GB VRAM + 2.5GB GTT)
- ✅ Inference pipeline REFACTORED to use GPU buffers
- ✅ Ready for performance testing with real model

## YOUR TASK
Test the refactored GPU inference pipeline and achieve 50-180 TPS performance:

1. **First, activate the environment:**
   ```bash
   source /home/ucadmin/activate-pure-hardware-env.sh
   cd /home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/
   ```

2. **Test token generation:**
   ```python
   from pure_hardware_pipeline import PureHardwarePipeline
   pipeline = PureHardwarePipeline()
   pipeline.initialize('quantized_models/gemma-3-27b-it-layer-by-layer')
   
   # Test generation
   result = pipeline.generate_tokens([1, 2, 3, 4, 5], max_tokens=10)
   print(f"Generated: {result}")
   ```

3. **Monitor performance:**
   - Run `radeontop` in another terminal to watch GPU usage
   - GPU usage should be >50% during inference
   - VRAM should stay at ~16GB, GTT at ~2.5GB+

4. **Measure TPS:**
   ```python
   import time
   start = time.time()
   result = pipeline.generate_tokens([1, 2, 3], max_tokens=100)
   elapsed = time.time() - start
   tps = 100 / elapsed
   print(f"Tokens per second: {tps}")
   ```

## WHAT'S BEEN DONE
- GPU buffers are properly allocated and stored in `self.gpu_buffers`
- `compute_attention_layer_gpu()` uses Vulkan matrix operations
- `compute_ffn_layer_gpu()` uses Vulkan's `compute_fused_ffn_persistent_weights()`
- Forward pass calls GPU methods first, then falls back to CPU if needed

## POTENTIAL ISSUES
1. If you see high memory usage (40GB+), check for stalled Python processes
2. The attention computation still has some CPU parts - could be optimized further
3. Only ~2.5GB GTT is used - target is 10GB for better performance

## SUCCESS CRITERIA
- Model loads without errors
- Token generation works
- GPU usage >50% during inference
- Achieve 50+ TPS (target: 50-180 TPS)

## IMPORTANT
- NO SIMULATIONS - Use real model weights only
- NO CPU FALLBACK - Should use NPU+iGPU only
- Check CLAUDE.md for full architecture details