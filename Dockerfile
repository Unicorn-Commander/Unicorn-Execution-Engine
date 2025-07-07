
# Dockerfile for NPU Development

FROM xilinx/vitis-ai:latest

# Install system dependencies
RUN apt-get update && apt-get install -y     build-essential     cmake     git     curl     wget     python3     python3-pip     python3-venv     libboost-all-dev     libudev-dev     libdrm-dev     libssl-dev     libffi-dev     pkg-config     dkms     bc     && rm -rf /var/lib/apt/lists/*

# Set up the working directory
WORKDIR /workspace

# Copy the project files into the container
COPY . .

# Run the installation script
RUN /bin/bash -c "source /opt/xilinx/xrt/setup.sh && bash NPU-Development/scripts/install_npu_stack.sh"

# Set up the entrypoint
CMD ["/bin/bash"]

