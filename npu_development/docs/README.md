# NPU Development for Multi-Model Optimization

## Overview
This directory contains NPU Phoenix kernel development for optimizing multiple LLM architectures.

## Supported Models
- **Gemma 3 27B-IT**: Ultra-aggressive quantization + NPU acceleration
- **Qwen2.5 7B/14B/32B**: RoPE attention optimization
- **Gemma 2 2B/9B**: Ultra-fast inference
- **Qwen2.5-VL**: Vision-language fusion

## Directory Structure
```
npu_development/
├── kernels/
│   ├── universal/          # Cross-model kernels
│   ├── gemma/             # Gemma-specific optimizations
│   └── qwen/              # Qwen-specific optimizations
├── tools/                 # Development tools
├── tests/                 # Kernel testing
├── examples/              # Usage examples
└── docs/                  # Documentation
```

## Quick Start
1. Run setup: `python setup_npu_development.py`
2. Test kernels: `python tests/test_npu_kernels.py`
3. Compile kernels: `./tools/compile_kernels.sh`

## Performance Targets
- Gemma 3 27B: 150-200 TPS
- Qwen2.5-7B: 200-300 TPS  
- Gemma 2 2B: 400-600 TPS
