# GEMINI-CLI TASK: Fix GPU Memory Loading Issue

## 🚨 CRITICAL BLOCKER
The GPU is NOT loading model weights to VRAM/GTT, preventing all performance testing. All optimizations are implemented but we can't measure performance until this is fixed.

## 📍 Working Directory
```bash
cd /home/ucadmin/Development/Unicorn-Execution-Engine/
source /home/ucadmin/activate-uc1-ai-py311.sh
```

## 🔴 Current Problem
When running `benchmark_final_performance.py`:
- **Expected**: VRAM should increase from 802MB → ~16GB
- **Actual**: VRAM stays at 802MB baseline
- **Evidence**: 
  - "Layers loaded: 0"
  - "VRAM: 0.0GB / 16.0GB"
  - "GTT: 0.0GB / 10.0GB"
  - GPU utilization: 0%

## 🎯 Your Mission

### 1. Debug GPU Loading Issue
Find why `_load_tensor_to_gpu` is not being called or failing silently:

```bash
# Key files to investigate:
- pure_hardware_pipeline_gpu_fixed.py (the working pipeline)
- benchmark_final_performance.py (shows the problem)
- LightningFastLoader class (handles tensor loading)
```

Add debug logging to trace the loading flow:
```python
# Add logging to see what's happening:
logger.info(f"Loading tensor {weight_name} to GPU...")
logger.info(f"Tensor size: {tensor_size} bytes")
logger.info(f"Before GPU transfer - VRAM: {current_vram}GB")
# ... actual transfer ...
logger.info(f"After GPU transfer - VRAM: {new_vram}GB")
```

### 2. Fix Vision Tower Errors
The model has vision components causing load failures. Modify the loader to skip them:

```python
# In the tensor loading loop, add:
if 'vision_tower' in weight_name or 'vision' in weight_name:
    logger.info(f"Skipping vision component: {weight_name}")
    continue
```

### 3. Verify GPU Buffer Retention
Ensure GPU buffers aren't being deallocated after creation:
- Check if buffers are stored properly in `self.gpu_buffers`
- Verify the buffer keys match when retrieving
- Ensure no premature cleanup is happening

### 4. Testing Your Fix

#### Monitor GPU in real-time:
```bash
# Terminal 1: Watch GPU memory
watch -n 0.5 'radeontop -d - -l 1 2>/dev/null | grep -E "(gpu|vram|gtt)"'

# Terminal 2: Run the benchmark
python3 benchmark_final_performance.py
```

#### Success Criteria:
- ✅ VRAM increases from 802MB → ~16GB
- ✅ GTT shows ~10GB usage
- ✅ "Layers loaded: N" where N > 0
- ✅ GPU utilization > 0% during inference

## 📊 Context
All optimizations are already implemented:
- Persistent buffers (16.5x speedup)
- Setup overhead fixed (430x improvement) 
- RDNA3 shaders (2.4x speedup)
- INT4 quantization (1.8x speedup)
- **Combined theoretical: ~30,600x improvement**

We just need the GPU to actually load the model to achieve 1000+ TPS!

## 🔧 Helpful Commands

```bash
# Clear file cache before testing
sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"

# Check current GPU memory
radeontop -d - -l 1 2>/dev/null | grep -E "(vram|gtt)"

# Test the fixed pipeline
python3 benchmark_final_performance.py
```

## 📝 Please Report Back
1. What was preventing GPU loading?
2. What changes fixed it?
3. Final VRAM/GTT usage after fix
4. Any remaining issues

Good luck! The entire system is ready to achieve 1000+ TPS once you fix this GPU loading issue.