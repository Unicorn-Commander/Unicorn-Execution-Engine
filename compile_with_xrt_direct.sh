#!/bin/bash
# Compile the NPU files directly with XRT support

cd llama.cpp

# Find the NPU source files
NPU_SOURCES="npu_stub.cpp npu_xrt_compute.cpp"

# Compile with XRT
echo "Compiling NPU modules with XRT..."
g++ -c npu_stub.cpp -o npu_stub_xrt.o \
    -I/opt/xilinx/xrt/include \
    -I./ggml/include \
    -I./include \
    -I./src \
    -DLLAMA_NPU_XRT_ENABLED \
    -fPIC -O3

g++ -c npu_xrt_compute.cpp -o npu_xrt_compute_xrt.o \
    -I/opt/xilinx/xrt/include \
    -I./ggml/include \
    -I./include \
    -I./src \
    -DLLAMA_NPU_XRT_ENABLED \
    -fPIC -O3

echo "✅ NPU modules compiled with XRT support"

# Now we need to relink llama-cli with these objects
# This is complex due to the build system, so the wrapper approach is simpler
