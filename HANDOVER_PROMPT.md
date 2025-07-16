# HANDOVER PROMPT FOR NEXT AI

I need help fixing a GPU memory loading issue in the Unicorn Execution Engine project. Please read the CLAUDE.md file first to understand the current status:

```
/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/CLAUDE.md
```

## Current Situation:
- We have fixed Vulkan memory mapping errors and GPU compute is working
- The 27B quantized model at `quantized_models/gemma-3-27b-it-layer-by-layer/` is NOT loading to GPU memory
- Only seeing ~1.2GB VRAM usage instead of target 16GB VRAM + 10GB GTT

## The Critical Bug:
In `/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/pure_hardware_pipeline.py` around lines 200-210, the code calls `self.vulkan_engine._allocate_gpu_memory(tensor)` but doesn't store the returned GPU buffer handles. This causes the allocated memory to be freed immediately.

## Your Task:
1. First, read these files to understand the issue:
   - `/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/CLAUDE.md` (project overview)
   - `/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/pure_hardware_pipeline.py` (needs fixing)
   - `/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/real_vulkan_matrix_compute.py` (GPU allocation implementation)

2. Fix the GPU buffer storage issue in `pure_hardware_pipeline.py`:
   - Add `self.gpu_buffers = {}` to store GPU buffer handles
   - Modify the code around line 207 to store the returned buffer info
   - Ensure buffers aren't freed prematurely

3. Test the fix by running:
   ```bash
   python /home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/pure_hardware_pipeline.py
   ```

4. Monitor GPU memory usage with:
   ```bash
   radeontop -d - -l 1
   ```

## Success Criteria:
- VRAM usage should reach ~16GB (from baseline ~1GB)
- GTT usage should reach ~10GB (from baseline ~100MB)
- The model should actually load to GPU memory and stay there
- Then we can measure real TPS (tokens per second)

## Environment:
- Working directory: `/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/`
- Python environment: Already activated
- Model path: `quantized_models/gemma-3-27b-it-layer-by-layer/`
- Server port: 8010 (if you need to run the server)

## Important Notes:
- The model is a quantized Gemma-3 27B model split into 62 layers
- Target performance is 50-180 TPS using NPU+iGPU only (no CPU fallback)
- We're using pure hardware acceleration with Vulkan for iGPU and MLIR-AIE2 for NPU
- NO PyTorch/ROCm dependencies - pure numpy and Vulkan

Please focus on fixing the GPU memory allocation issue first. The model loading is already implemented but the GPU buffers are being lost.