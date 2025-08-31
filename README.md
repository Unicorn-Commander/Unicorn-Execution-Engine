# Unicorn Execution Engine

**Multi-Platform Hardware-Accelerated AI Execution Framework**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NPU Models](https://img.shields.io/badge/Models-HuggingFace-ff9800)](https://huggingface.co/magicunicorn)
[![Performance](https://img.shields.io/badge/Speedup-3x--220x-brightgreen)](https://github.com/Unicorn-Commander/Unicorn-Execution-Engine)

## 🚀 Overview

The Unicorn Execution Engine is a high-performance runtime for deploying AI models on specialized hardware accelerators including Intel iGPUs, AMD NPUs, and more. Developed by Magic Unicorn Inc., this engine achieves unprecedented performance through hardware-specific optimizations.

## ✨ Key Features

### Intel iGPU (OpenVINO)
- **3-5x Speedup** for Kokoro TTS vs CPU
- **15W Power Efficiency** for laptops
- **50+ Professional Voices** for TTS
- **Zero-Copy Shared Memory** architecture

### AMD NPU (MLIR-AIE2)
- **220x Speedup** for WhisperX vs CPU
- **Custom MLIR-AIE2 Kernels** for optimal utilization
- **INT8/INT4 Quantization** with minimal accuracy loss
- **16 TOPS INT8** performance on Phoenix NPU

## 📊 Performance Benchmarks

### Kokoro TTS v0.19 (Intel iGPU)
| Platform | Latency | Power | Speedup |
|----------|---------|-------|---------|
| Intel Iris Xe | 150ms | 15W | **3.0x** |
| Intel UHD | 250ms | 12W | **1.8x** |
| CPU (i7) | 450ms | 35W | Baseline |

### WhisperX Speech Recognition (AMD NPU)
| Model | CPU Time | NPU Time | Speedup | Accuracy |
|-------|----------|----------|---------|----------|
| Large-v3 | 59.4 min | 16.2 sec | **220x** | 99% |
| Large-v2 | 54.0 min | 18.0 sec | **180x** | 98% |
| Medium | 27.0 min | 14.4 sec | **112x** | 95% |

## 🛠️ Quick Start

### Intel iGPU - Kokoro TTS
```bash
# Install with OpenVINO support
pip install unicorn-execution-engine[intel-igpu]

# Or use Docker
docker pull magicunicorn/unicorn-execution-engine:kokoro-intel-igpu
```

```python
from tts.kokoro_intel_igpu import KokoroIntelTTS

# Initialize with Intel iGPU
tts = KokoroIntelTTS(device="igpu")

# Synthesize with 50+ voices
audio = tts.synthesize("Hello world!", voice="af_bella")
```

### AMD NPU - WhisperX
```bash
# Install with NPU support
pip install unicorn-execution-engine[amd-npu]
```

```python
from unicorn_engine import NPUWhisperX

# Load quantized model
model = NPUWhisperX.from_pretrained("magicunicorn/whisperx-large-v3-npu")

# Transcribe with 220x speedup
result = model.transcribe("meeting.wav")
```

## 🏗️ Architecture

```
Unicorn Execution Engine
├── TTS Module
│   ├── Kokoro v0.19 (Intel iGPU) ✅
│   ├── Whisper (AMD NPU) ✅
│   └── Bark (Planned)
├── LLM Module
│   ├── Llama (AMD NPU) 🚧
│   └── Mistral (NVIDIA) 📋
└── Vision Module
    ├── CLIP (Apple ANE) 📋
    └── SAM (Qualcomm) 📋
```

## 📦 Pre-built Models & Packages

### HuggingFace Models
- [kokoro-tts-intel](https://huggingface.co/magicunicorn/kokoro-tts-intel) - Kokoro TTS for Intel iGPU
- [whisperx-large-v3-npu](https://huggingface.co/magicunicorn/whisperx-large-v3-npu) - WhisperX for AMD NPU

### Docker Images
```bash
# Intel iGPU with Kokoro TTS
docker run --device /dev/dri -p 8880:8880 \
    magicunicorn/unicorn-execution-engine:kokoro-intel-igpu

# AMD NPU with WhisperX  
docker run --device /dev/accel -p 8881:8881 \
    magicunicorn/unicorn-execution-engine:whisperx-amd-npu
```

## 🔧 Platform-Specific Setup

### Intel iGPU Setup
```bash
# Install OpenVINO and drivers
sudo apt-get install intel-opencl-icd intel-level-zero-gpu level-zero
pip install openvino==2024.0.0 onnxruntime-openvino==1.17.0

# Verify Intel GPU
lspci | grep -i intel | grep -i vga
```

### AMD NPU Setup
```bash
# Install XRT and drivers
sudo apt-get install xrt amd-npu-driver
pip install pyxrt>=2.0.0

# Verify NPU
ls /dev/accel/accel0
```

## 💾 Model Files (Git LFS)

### Kokoro TTS
- `models/kokoro-v0_19.onnx` (311MB) - TTS model
- `models/voices-v1.0.bin` (25MB) - 50+ voice embeddings

### WhisperX
- `models/whisperx-large-v3.npumodel` (1.5GB) - Quantized INT8 model
- `models/whisperx-kernels.xclbin` (50MB) - Custom MLIR kernels

## 🚀 Advanced Features

### Multi-Hardware Pipeline
```python
from unicorn_engine import MultiPlatformEngine

engine = MultiPlatformEngine()

# Automatically selects best hardware
# Intel iGPU for TTS, AMD NPU for STT
pipeline = engine.create_pipeline([
    ("speech_recognition", "amd-npu"),
    ("text_synthesis", "intel-igpu")
])

result = await pipeline.process(audio_input)
```

### Custom Optimization
```python
# Intel iGPU optimization
from intel_igpu_module import IntelIGPUExecutor
executor = IntelIGPUExecutor()
executor.optimize_for_latency()

# AMD NPU quantization
from unicorn_engine import Quantizer
quantizer = Quantizer(target="npu", precision="int8")
```

## 📈 Roadmap

- [x] Intel iGPU support (OpenVINO)
- [x] AMD NPU support (MLIR-AIE2)
- [ ] NVIDIA GPU support (TensorRT)
- [ ] Apple Neural Engine support
- [ ] Qualcomm Hexagon DSP support
- [ ] Multi-device distribution

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development
```bash
# Clone repo
git clone https://github.com/Unicorn-Commander/Unicorn-Execution-Engine
cd Unicorn-Execution-Engine

# Install dev dependencies
pip install -e .[dev]

# Run tests
pytest tests/
```

## 📚 Documentation

- [Intel iGPU Guide](./docs/INTEL_IGPU.md) - Kokoro TTS optimization
- [AMD NPU Guide](./docs/AMD_NPU.md) - WhisperX acceleration
- [API Reference](./docs/API.md) - Complete API documentation
- [Benchmarks](./docs/BENCHMARKS.md) - Performance analysis

## 🏢 About Magic Unicorn Inc.

Magic Unicorn Inc. develops enterprise AI solutions optimized for edge deployment. The Unicorn Commander Suite provides complete AI infrastructure for on-premise deployments.

### Related Projects
- [Unicorn-Orator](https://github.com/Unicorn-Commander/Unicorn-Orator) - Full TTS platform
- [Meeting-Ops](https://github.com/Unicorn-Commander/Meeting-Ops) - AI meeting recorder
- [Unicorn Models](https://huggingface.co/magicunicorn) - Pre-optimized models

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Intel for OpenVINO and iGPU support
- AMD for NPU hardware and MLIR-AIE2
- OpenAI for original Whisper models
- The open-source community

## 📞 Contact

- **GitHub Issues**: [Report bugs](https://github.com/Unicorn-Commander/Unicorn-Execution-Engine/issues)
- **HuggingFace**: [Discussion forum](https://huggingface.co/magicunicorn)
- **Email**: support@magicunicorn.dev

---

**© 2025 Magic Unicorn Inc.** | Part of the Unicorn Commander Suite

⭐ Star us on GitHub if you find this useful!