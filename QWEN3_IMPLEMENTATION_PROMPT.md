# Prompt for Implementing Qwen3-30B-A3B MoE Model

Please implement the Qwen3-30B-A3B MoE (Mixture of Experts) model on our custom Unicorn Execution Engine. This model has 30B total parameters but only activates 3B at a time, making it perfect for our hardware.

## Project Location
- **Working Directory**: `/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/`
- **Create New Subdirectory**: `/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/qwen3_30b_moe/`

## Key Existing Files You'll Need

### Core Engine Files:
- `/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/vulkan_compute_workaround.py` - Vulkan compute engine with fallback
- `/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/gemma_27b_loader_v2.py` - Model loader to adapt for MoE
- `/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/bfloat16_converter.py` - BF16 conversion utilities
- `/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/npu_attention_kernel_real.py` - NPU integration for routing

### Compiled Shaders (Ready to Use):
- `/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/rdna3_int4.spv` - INT4 compute shader
- `/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/matrix_multiply_int8.spv` - INT8 matrix multiplication
- `/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/transformer_optimized.spv` - Optimized transformer ops
- `/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/gate_up_silu_mul_int8.spv` - FFN operations

### Reference Implementation:
- `/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/gemma_27b_working_pipeline.py` - Working pipeline example
- `/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/QWEN3_30B_MOE_IMPLEMENTATION_PLAN.md` - Detailed implementation plan

## Environment Setup
```bash
cd /home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/
source /home/ucadmin/activate-pure-hardware-env.sh
```

## Your Tasks

### 1. Download and Prepare Model
Download Qwen3-30B-A3B from HuggingFace and place it in:
```bash
mkdir -p /home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/qwen3_30b_moe/base_model
# Download model to above directory
```

### 2. Create Custom Quantizer
Create `/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/qwen3_30b_moe/qwen3_moe_quantizer.py` that:
- Implements INT4 quantization for expert weights
- Keeps router at FP16 precision
- Uses K-means clustering (similar to Q4_K_M)
- Outputs to custom format optimized for our engine

### 3. Adapt Model Loader
Create `/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/qwen3_30b_moe/qwen3_moe_loader.py` based on `gemma_27b_loader_v2.py` that:
- Handles MoE weight structure (router + 128 experts)
- Implements sparse loading (only loads active experts)
- Supports our custom INT4 format

### 4. Implement MoE Pipeline
Create `/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/qwen3_30b_moe/qwen3_moe_pipeline.py` that:
- Uses the vulkan_compute_workaround.py engine
- Implements MoE routing (NPU if available, else GPU)
- Only computes active 3B parameters per token
- Achieves 40-50 TPS target

### 5. Add Tool Support
Implement tool calling by:
- Adding special tokens: `<function_call>`, `</function_call>`, `<tool_response>`, `</tool_response>`
- Modifying generation loop to detect and execute tools
- Creating `/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/qwen3_30b_moe/tools/` directory

### 6. Create Benchmark Script
Create `/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/qwen3_30b_moe/benchmark_moe.py` that:
- Measures tokens per second
- Reports memory usage (should be ~7.5GB active)
- Validates we achieve 40-50 TPS

## Technical Requirements

### Quantization Strategy:
- Router: FP16 (critical for expert selection)
- Active Experts: INT4 with block size 32
- Inactive Experts: INT4 or INT3
- Embeddings: INT8
- Use existing INT4 shader at `/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/rdna3_int4.spv`

### Memory Layout:
- NPU (if available): Router weights (~100MB)
- GPU VRAM: Active experts (~3GB) + cache
- GPU GTT: Inactive experts
- Target: ~7.5GB total active memory

### Performance Target:
- 40-50 TPS (compared to 17 TPS for dense models)
- Leverage MoE sparsity to reduce memory bandwidth by 10x

## Important Notes

1. The Vulkan initialization has issues, so use `vulkan_compute_workaround.py` which falls back to optimized NumPy when needed

2. The model's MoE structure (8 active out of 128 experts) is key to performance - only compute what's needed

3. We already have compiled Vulkan shaders for INT4/INT8 operations - reuse them

4. The NPU can be used for routing to save memory bandwidth, but it's optional

5. Refer to `/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/CLAUDE.md` for system architecture details

## Success Criteria
- [ ] Model loads and runs inference
- [ ] Achieves 40-50 TPS 
- [ ] Uses <8GB active memory
- [ ] Tool calling works
- [ ] Quantization maintains quality (test with simple prompts)

Please start by examining the existing codebase and implementation plan, then proceed with the implementation. Let me know if you need any clarification!