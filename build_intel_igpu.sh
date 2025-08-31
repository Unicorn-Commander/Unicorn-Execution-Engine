#!/bin/bash
# Build script for Intel iGPU optimized wheel
# Unicorn Execution Engine - Kokoro TTS Module

echo "🦄 Building Unicorn Execution Engine - Intel iGPU Package"
echo "==========================================================="

# Clean previous builds
rm -rf build dist *.egg-info

# Create wheel with Intel iGPU dependencies
echo "Building wheel package..."
python3 -m pip wheel . --wheel-dir wheels/ \
    --global-option="--plat-name" \
    --global-option="linux_x86_64_intel_igpu"

# Download OpenVINO runtime if not present
if [ ! -f "wheels/onnxruntime_openvino-1.17.0-cp310-cp310-linux_x86_64.whl" ]; then
    echo "Downloading OpenVINO runtime wheel..."
    cd wheels
    pip download onnxruntime-openvino==1.17.0 --platform linux_x86_64 --python-version 310 --only-binary :all:
    cd ..
fi

# Create prebuilt package
echo "Creating prebuilt package..."
mkdir -p prebuilt/intel-igpu
cp -r tts prebuilt/intel-igpu/
cp -r models prebuilt/intel-igpu/
cp intel_igpu_module.py prebuilt/intel-igpu/

# Create standalone installer
cat > prebuilt/intel-igpu/install.sh << 'EOF'
#!/bin/bash
# Intel iGPU Installer for Kokoro TTS

echo "Installing Kokoro TTS with Intel iGPU support..."

# Check for Intel GPU
if lspci | grep -i intel | grep -i -E "vga|display" > /dev/null; then
    echo "✓ Intel GPU detected"
else
    echo "⚠️  Warning: Intel GPU not detected, will use CPU fallback"
fi

# Install dependencies
pip install onnxruntime-openvino==1.17.0
pip install numpy soundfile scipy

# Install Intel GPU drivers if needed
if ! command -v clinfo &> /dev/null; then
    echo "Installing Intel GPU drivers..."
    sudo apt-get update
    sudo apt-get install -y intel-opencl-icd intel-level-zero-gpu level-zero
fi

echo "✓ Installation complete!"
echo "Run: python3 -m tts.kokoro_intel_igpu"
EOF

chmod +x prebuilt/intel-igpu/install.sh

# Create Docker prebuild
echo "Creating Docker prebuild..."
cat > Dockerfile.intel-igpu << 'EOF'
FROM openvino/ubuntu22_runtime:2024.0.0

WORKDIR /app

# Copy models and code
COPY models/ /app/models/
COPY tts/ /app/tts/
COPY intel_igpu_module.py /app/

# Install Python dependencies
RUN pip install --no-cache-dir \
    onnxruntime-openvino==1.17.0 \
    numpy soundfile scipy fastapi uvicorn

# Expose port for API
EXPOSE 8880

# Run TTS server
CMD ["python3", "-m", "tts.kokoro_intel_igpu"]
EOF

# Build Docker image
echo "Building Docker image..."
docker build -f Dockerfile.intel-igpu -t unicorn-execution-engine:kokoro-intel-igpu .

# Create tarball for distribution
echo "Creating distribution package..."
tar -czf unicorn-execution-engine-kokoro-intel-igpu-v1.0.tar.gz \
    prebuilt/intel-igpu \
    models/*.onnx \
    models/*.bin \
    README.md

echo "✅ Build complete!"
echo ""
echo "Artifacts created:"
echo "  - wheels/ - Python wheel packages"
echo "  - prebuilt/intel-igpu/ - Standalone package"
echo "  - unicorn-execution-engine:kokoro-intel-igpu - Docker image"
echo "  - unicorn-execution-engine-kokoro-intel-igpu-v1.0.tar.gz - Distribution package"
echo ""
echo "To test locally:"
echo "  python3 -m tts.kokoro_intel_igpu"