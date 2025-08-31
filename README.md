# Unicorn Execution Engine

## Multi-Platform AI Execution Framework

A hardware-optimized execution framework for AI models across Intel iGPU, AMD NPU, NVIDIA GPU, and more. Currently featuring **Kokoro TTS v0.19** with Intel iGPU acceleration.

### 🚀 Quick Start

#### Intel iGPU (Kokoro TTS)
```bash
# Install from wheel
pip install wheels/unicorn_execution_engine-1.0.0-linux_x86_64_intel_igpu.whl

# Or use Docker
docker pull magicunicorn/unicorn-execution-engine:kokoro-intel-igpu
```

## 📦 Available Modules

### TTS (Text-to-Speech)

#### Kokoro v0.19 - Intel iGPU Optimized
- **Performance**: 3-5x faster than CPU
- **Power**: 15W TDP (laptop-friendly)
- **Voices**: 50+ professional voices
- **API**: OpenAI-compatible

```python
from tts.kokoro_intel_igpu import KokoroIntelTTS

tts = KokoroIntelTTS(device="igpu")
audio = tts.synthesize("Hello world!", voice="af_bella")
```

## 🏗️ Architecture

```
Unicorn Execution Engine
├── TTS Module
│   ├── Kokoro (Intel iGPU) ✅
│   ├── Whisper (Coming Soon)
│   └── Bark (Planned)
├── LLM Module
│   ├── Llama (AMD NPU) 🚧
│   └── Mistral (NVIDIA) 📋
└── Vision Module
    ├── CLIP (Apple ANE) 📋
    └── SAM (Qualcomm) 📋
```

## Platform Support

### Intel Integrated GPUs
- **Intel Iris Xe** (96 EU) - Tiger Lake, Alder Lake, Raptor Lake
- **Intel Arc iGPU** (128 EU) - Meteor Lake and newer
- **Intel UHD Graphics** (32 EU) - Budget/older systems

## Key Features

### 1. Automatic Hardware Detection
```python
executor = IntelIGPUExecutor()
# Automatically detects Intel GPU capabilities
```

### 2. OpenVINO Optimization
- **FP16 Precision**: Automatic mixed precision for 2x speedup
- **Graph Optimization**: Fuses operations for iGPU
- **Memory Patterns**: Optimized for shared system memory
- **Dynamic Shapes**: Supports variable input sizes

### 3. Power Efficiency
- **15W TDP**: Runs within laptop thermal limits
- **Shared Memory**: No dedicated VRAM needed
- **Balanced Mode**: Optimizes performance/power ratio

## Installation

### Prerequisites
```bash
# Install OpenVINO runtime
pip install openvino==2024.0.0
pip install onnxruntime-openvino==1.17.0

# Intel GPU drivers (Ubuntu/Debian)
sudo apt-get install intel-opencl-icd intel-level-zero-gpu level-zero
```

### Docker Support
```dockerfile
FROM openvino/ubuntu22_runtime:2024.0.0
# Includes all Intel GPU drivers and OpenVINO
```

## Usage Example

### Basic Inference
```python
from intel_igpu_module import IntelIGPUExecutor

# Initialize executor
executor = IntelIGPUExecutor()

# Create optimized session
session = executor.create_session("model.onnx")

# Run inference
inputs = {"input": numpy_array}
outputs = executor.run_inference(session, inputs)
```

### Kokoro TTS v0.19 Integration
```python
from tts.kokoro_intel_igpu import KokoroIntelTTS

# Load model with iGPU optimization
tts = KokoroIntelTTS(
    model_path="models/kokoro-v0_19.onnx",
    voices_path="models/voices-v1.0.bin",
    device="igpu"
)

# Synthesize speech with 50+ voices
audio = tts.synthesize("Hello world!", voice="af_bella", speed=1.0)

# Save output
tts.save_audio(audio, "output.wav")
```

## Performance Benchmarks

### Intel Iris Xe (96 EU) - Laptop
| Model | CPU Time | iGPU Time | Speedup |
|-------|----------|-----------|---------|
| Kokoro TTS | 450ms | 150ms | 3.0x |
| Whisper Base | 800ms | 250ms | 3.2x |
| BERT Base | 120ms | 40ms | 3.0x |

### Power Consumption
- **CPU Only**: 35W average
- **iGPU**: 15W average
- **Battery Life**: 2.3x longer on iGPU

## Architecture Details

### Memory Architecture
```
System RAM (Shared)
    ↓
Intel iGPU ← Zero-Copy → CPU
    ↓
OpenVINO Runtime
    ↓
Optimized Kernels
```

### Optimization Pipeline
1. **Model Loading**: ONNX → OpenVINO IR (cached)
2. **Graph Optimization**: Operation fusion, constant folding
3. **Precision**: FP32 → FP16 automatic conversion
4. **Execution**: Parallel EU (Execution Unit) dispatch

## Comparison with Other Platforms

| Platform | Hardware | Power | Speed | Cost |
|----------|----------|-------|-------|------|
| Intel iGPU | Integrated | 15W | Fast | Free* |
| NVIDIA GPU | Discrete | 75W+ | Fastest | $300+ |
| AMD NPU | Integrated | 10W | Fast | Free* |
| CPU | Any | 35W+ | Slow | Free |

*Included with CPU purchase

## Multi-Platform Strategy

This Intel iGPU module is part of the broader Unicorn Execution Engine supporting:

1. **Intel iGPU** (this module) - Laptops, NUCs
2. **AMD NPU** (coming soon) - Ryzen AI laptops
3. **Apple Neural Engine** (planned) - M-series Macs
4. **Qualcomm Hexagon** (planned) - Snapdragon laptops
5. **CPU Fallback** - Universal support

## Troubleshooting

### Check Intel GPU Available
```bash
# List Intel GPUs
lspci | grep -i intel | grep -i vga

# Check OpenVINO devices
python -c "from openvino.runtime import Core; print(Core().available_devices)"
```

### Common Issues

1. **No Intel GPU detected**
   - Update Intel drivers
   - Check BIOS for iGPU enabled
   
2. **OpenVINO errors**
   - Install level-zero drivers
   - Set `NEOReadDebugKeys=1` for debugging

3. **Performance issues**
   - Check thermal throttling
   - Increase TDP limit in BIOS

## Contributing

This module is part of the open-source Unicorn Execution Engine. Contributions welcome!

### Future Work
- [ ] INT8 quantization support
- [ ] Multi-GPU for Intel Arc
- [ ] Async/streaming inference
- [ ] SYCL/oneAPI integration

## License

MIT License - Magic Unicorn Unconventional Technology & Stuff Inc

## 💾 Pre-built Packages

### Models (Git LFS)
- `models/kokoro-v0_19.onnx` (311MB) - Kokoro TTS model
- `models/voices-v1.0.bin` (25MB) - Voice embeddings

### Wheels
- `wheels/unicorn_execution_engine-1.0.0-linux_x86_64_intel_igpu.whl`
- `wheels/onnxruntime_openvino-1.17.0-cp310-cp310-linux_x86_64.whl`

### Docker Images
```bash
# Intel iGPU optimized
docker pull magicunicorn/unicorn-execution-engine:kokoro-intel-igpu

# Run with GPU access
docker run --device /dev/dri -p 8880:8880 \
    magicunicorn/unicorn-execution-engine:kokoro-intel-igpu
```

## 🔧 Building from Source

### Intel iGPU Package
```bash
./build_intel_igpu.sh
```

Creates:
- Python wheels in `wheels/`
- Standalone package in `prebuilt/intel-igpu/`
- Docker image `unicorn-execution-engine:kokoro-intel-igpu`
- Distribution tarball

## Related Projects

- [Unicorn-Orator](https://github.com/Unicorn-Commander/Unicorn-Orator) - Full TTS platform using this module
- [HuggingFace Models](https://huggingface.co/magicunicorn/kokoro-tts-intel) - Pre-trained Kokoro models
- [OpenVINO](https://github.com/openvinotoolkit/openvino) - Intel's inference toolkit