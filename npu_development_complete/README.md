# NPU Development Environment

## MLIR-AIE2 Toolchain Installation Complete

### Available Tools
- `aie-opt`: MLIR optimization for NPU kernels
- `aie-translate`: Binary generation for AMD Phoenix NPU
- `build_kernels.sh`: Automated build script

### Usage
```bash
# Build NPU kernels
./build_kernels.sh

# Manual compilation
aie-opt --help
aie-translate --help
```

### Development Workflow
1. Edit `npu_attention_kernel.mlir`
2. Run `./build_kernels.sh`
3. Test with `custom_execution_engine.py`
4. Deploy to NPU hardware

### Performance Targets
- NPU Phoenix: 16 TOPS INT8
- Target: 50+ TPS per attention layer
- Memory: 2GB NPU local memory
