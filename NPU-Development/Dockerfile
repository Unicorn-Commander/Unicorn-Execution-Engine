# Dockerfile for NPU Development

FROM ubuntu:22.04

# Set environment variables for XRT
ENV XILINX_XRT="/opt/xilinx/xrt"
ENV PATH="/opt/xilinx/xrt/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/xilinx/xrt/lib:${LD_LIBRARY_PATH}"
ENV PYTHONPATH="/opt/xilinx/xrt/python:${PYTHONPATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    python3 \
    python3-pip \
    python3-venv \
    libboost-all-dev \
    libudev-dev \
    libdrm-dev \
    libssl-dev \
    libffi-dev \
    pkg-config \
    dkms \
    bc \
    pciutils \
    libprotobuf-dev \
    libgoogle-glog-dev \
    libunwind-dev \
    libyaml-cpp-dev \
    opencl-headers \
    ocl-icd-opencl-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and install XRT .deb packages
RUN wget https://www.xilinx.com/bin/public/extension/mc/downloadFile/xrt_202320.2.15.225_22.04-amd64-xrt.deb -O /tmp/xrt.deb && \
    apt install -y /tmp/xrt.deb && \
    rm /tmp/xrt.deb

# Set up the working directory
WORKDIR /workspace

# Copy the project files into the container
COPY . .

# Install XDNA kernel driver
RUN git clone https://github.com/amd/xdna-driver.git /tmp/xdna-driver && \
    mkdir -p /tmp/xdna-driver/build && \
    cd /tmp/xdna-driver/build && \
    cmake .. && \
    make -j$(nproc) && \
    make install && \
    modprobe amdxdna && \
    echo "amdxdna" | tee /etc/modules-load.d/amdxdna.conf && \
    rm -rf /tmp/xdna-driver

# Create Python virtual environment and install dependencies
RUN python3 -m venv /root/npu-dev/npu_dev_env && \
    source /root/npu-dev/npu_dev_env/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    pip install \
        onnx==1.12.0 \
        onnxruntime>=1.22.0 \
        torch>=2.0.0 \
        transformers>=4.40.0 \
        numpy>=1.24.0 \
        scipy>=1.10.0 \
        librosa>=0.10.0 \
        soundfile>=0.12.0 \
        sounddevice>=0.4.0 \
        pydub>=0.25.0 \
        onnx>=1.15.0 \
        netron \
        psutil \
        matplotlib \
        jupyter \
        ipykernel && \
    pip install \
        black \
        flake8 \
        pytest \
        mypy \
        pre-commit

# Create environment setup script
RUN mkdir -p /root/npu-dev && \
    echo "#!/bin/bash" > /root/npu-dev/setup_npu_env.sh && \
    echo "" >> /root/npu-dev/setup_npu_env.sh && \
    echo "# NPU Development Environment Setup Script" >> /root/npu-dev/setup_npu_env.sh && \
    echo "# Source this script to set up your NPU development environment" >> /root/npu_dev_env.sh && \
    echo "" >> /root/npu-dev/setup_npu_env.sh && \
    echo "# Colors" >> /root/npu-dev/setup_npu_env.sh && \
    echo "GREEN='\033[0;32m'" >> /root/npu-dev/setup_npu_env.sh && \
    echo "BLUE='\033[0;34m'" >> /root/npu-dev/setup_npu_env.sh && \
    echo "NC='\033[0m'" >> /root/npu-dev/setup_npu_env.sh && \
    echo "" >> /root/npu-dev/setup_npu_env.sh && \
    echo "echo -e \"${BLUE}Setting up NPU Development Environment...${NC}\"" >> /root/npu-dev/setup_npu_env.sh && \
    echo "" >> /root/npu-dev/setup_npu_env.sh && \
    echo "# Get script directory" >> /root/npu-dev/setup_npu_env.sh && \
    echo "SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"" >> /root/npu-dev/setup_npu_env.sh && \
    echo "" >> /root/npu-dev/setup_npu_env.sh && \
    echo "# Set up XRT environment" >> /root/npu-dev/setup_npu_env.sh && \
    echo "if [[ -f \"/opt/xilinx/xrt/setup.sh\" ]]; then" >> /root/npu-dev/setup_npu_env.sh && \
    echo "    source /opt/xilinx/xrt/setup.sh" >> /root/npu-dev/setup_npu_env.sh && \
    echo "    echo -e \"${GREEN}\u2713 XRT environment loaded${NC}\"" >> /root/npu-dev/setup_npu_env.sh && \
    echo "else" >> /root/npu-dev/setup_npu_env.sh && \
    echo "    echo \"Warning: XRT not found at /opt/xilinx/xrt/setup.sh\"" >> /root/npu-dev/setup_npu_env.sh && \
    echo "fi" >> /root/npu-dev/setup_npu_env.sh && \
    echo "" >> /root/npu-dev/setup_npu_env.sh && \
    echo "# Set up MLIR-AIE environment" >> /root/npu-dev/setup_npu_env.sh && \
    echo "if [[ -d \"${SCRIPT_DIR}/mlir-aie\" ]]; then" >> /root/npu-dev/setup_npu_env.sh && \
    echo "    source \"${SCRIPT_DIR}/mlir-aie/ironenv/bin/activate\"" >> /root/npu-dev/setup_npu_env.sh && \
    echo "    source \"${SCRIPT_DIR}/mlir-aie/utils/env_setup.sh\"" >> /root/npu-dev/setup_npu_env.sh && \
    echo "    echo -e \"${GREEN}\u2713 MLIR-AIE environment loaded${NC}\"" >> /root/npu-dev/setup_npu_env.sh && \
    echo "else" >> /root/npu-dev/setup_npu_env.sh && \
    echo "    echo \"Warning: MLIR-AIE not found at ${SCRIPT_DIR}/mlir-aie\"" >> /root/npu-dev/setup_npu_env.sh && \
    echo "fi" >> /root/npu-dev/setup_npu_env.sh && \
    echo "" >> /root/npu-dev/setup_npu_env.sh && \
    echo "# Set up Python environment" >> /root/npu-dev/setup_npu_env.sh && \
    echo "if [[ -d \"${SCRIPT_DIR}/npu_dev_env\" ]]; then" >> /root/npu-dev/setup_npu_env.sh && \
    echo "    source \"${SCRIPT_DIR}/npu_dev_env/bin/activate\"" >> /root/npu-dev/setup_npu_env.sh && \
    echo "    echo -e \"${GREEN}\u2713 Python NPU dev environment activated${NC}\"" >> /root/npu-dev/setup_npu_env.sh && \
    echo "else" >> /root/npu-dev/setup_npu_env.sh && \
    echo "    echo \"Warning: Python environment not found at ${SCRIPT_DIR}/npu_dev_env\"" >> /root/npu-dev/setup_npu_env.sh && \
    echo "fi" >> /root/npu-dev/setup_npu_env.sh && \
    echo "" >> /root/npu-dev/setup_npu_env.sh && \
    echo "# Set environment variables" >> /root/npu-dev/setup_npu_env.sh && \
    echo "export NPU_DEV_ROOT=\"${SCRIPT_DIR}\"" >> /root/npu-dev/setup_npu_env.sh && \
    echo "export PYTHONPATH=\"${SCRIPT_DIR}:${PYTHONPATH}\"" >> /root/npu-dev/setup_npu_env.sh && \
    echo "" >> /root/npu-dev/setup_npu_env.sh && \
    echo "echo -e \"${GREEN}NPU Development Environment Ready!${NC}\"" >> /root/npu-dev/setup_npu_env.sh && \
    echo "echo \"NPU_DEV_ROOT: ${NPU_DEV_ROOT}\"" >> /root/npu-dev/setup_npu_env.sh && \
    chmod +x /root/npu-dev/setup_npu_env.sh

# Run verification script
RUN /bin/bash -c "source /root/npu-dev/setup_npu_env.sh && bash scripts/verify_essentials.sh"

# Set up the entrypoint
ENTRYPOINT ["/bin/bash", "-c", "source /root/npu-dev/setup_npu_env.sh && /bin/bash"]
CMD [""]