# 🧠 NPU Implementation Guide - AMD Phoenix NPU

**Status**: NPU Initialized ✅ | Kernels Loaded ✅ | Execution Pending ⏳

## 📊 Current NPU Status

### ✅ **What's Working**
1. **Hardware Detection**: AMD Phoenix NPU (16 TOPS) at `/dev/accel/accel0`
2. **Driver Loading**: All XRT libraries successfully loaded
   - `/opt/xilinx/xrt/lib/libxrt_core.so.2` ✅
   - `/opt/xilinx/xrt/lib/libxrt_coreutil.so.2` ✅
   - `/opt/xilinx/xrt/lib/libxrt_driver_xdna.so.2` ✅
3. **Kernel Binaries**: Pre-compiled kernels available
   - `attention_256_int8.bin` (5.5 KB)
   - `attention_512_int8.bin` (13.5 KB)
   - `attention_1024_int8.bin` (41.5 KB)
   - `attention_2048_int8.bin` (145.5 KB)
   - INT4 variants also available

### ⏳ **What's Needed for Execution**

## 1️⃣ **XRT C++ Wrapper Implementation**

### Required Headers
```cpp
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <experimental/xrt_xclbin.h>
```

### Basic Implementation Structure
```cpp
class NPUExecutor {
private:
    xrt::device device;
    xrt::uuid xclbin_uuid;
    xrt::kernel kernel;
    
public:
    NPUExecutor(int device_index = 0) {
        device = xrt::device(device_index);
    }
    
    void load_xclbin(const std::string& xclbin_path) {
        xrt::xclbin xclbin(xclbin_path);
        xclbin_uuid = device.load_xclbin(xclbin);
    }
    
    void load_kernel(const std::string& kernel_name) {
        kernel = xrt::kernel(device, xclbin_uuid, kernel_name);
    }
};
```

### Python Binding with pybind11
```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

PYBIND11_MODULE(npu_xrt, m) {
    py::class_<NPUExecutor>(m, "NPUExecutor")
        .def(py::init<int>(), py::arg("device_index") = 0)
        .def("load_xclbin", &NPUExecutor::load_xclbin)
        .def("load_kernel", &NPUExecutor::load_kernel)
        .def("execute", &NPUExecutor::execute);
}
```

## 2️⃣ **Buffer Management**

### NPU Buffer Allocation
```cpp
// Allocate NPU buffer
xrt::bo allocate_buffer(size_t size_bytes, xrt::bo::flags flags) {
    return xrt::bo(device, size_bytes, flags, kernel.group_id(0));
}

// Transfer data to NPU
void write_buffer(xrt::bo& buffer, const void* data, size_t size) {
    buffer.write(data);
    buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

// Read data from NPU
void read_buffer(xrt::bo& buffer, void* data, size_t size) {
    buffer.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    buffer.read(data);
}
```

## 3️⃣ **Kernel Execution Pipeline**

### Steps for NPU Kernel Execution
```cpp
void execute_attention(py::array_t<float> input_data) {
    // 1. Allocate buffers
    size_t input_size = input_data.size() * sizeof(float);
    xrt::bo input_buffer = allocate_buffer(input_size, xrt::bo::flags::normal);
    xrt::bo output_buffer = allocate_buffer(output_size, xrt::bo::flags::normal);
    
    // 2. Transfer input data
    write_buffer(input_buffer, input_data.data(), input_size);
    
    // 3. Set kernel arguments
    xrt::run run = kernel(input_buffer, output_buffer, seq_length, num_heads);
    
    // 4. Wait for completion
    run.wait();
    
    // 5. Read results
    std::vector<float> output(output_size / sizeof(float));
    read_buffer(output_buffer, output.data(), output_size);
    
    return output;
}
```

## 4️⃣ **Build Configuration**

### CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.12)
project(npu_xrt)

find_package(pybind11 REQUIRED)
find_package(XRT REQUIRED)

pybind11_add_module(npu_xrt npu_xrt.cpp)

target_include_directories(npu_xrt PRIVATE 
    ${XRT_INCLUDE_DIRS}
    /opt/xilinx/xrt/include
)

target_link_libraries(npu_xrt PRIVATE 
    ${XRT_LIBRARIES}
    xrt_coreutil
)
```

### Build Commands
```bash
mkdir build && cd build
cmake ..
make -j8
```

## 5️⃣ **Integration Plan**

### Phase 1: Basic Execution (1-2 days)
- [ ] Create C++ NPU executor class
- [ ] Implement buffer allocation/transfer
- [ ] Add kernel loading from binary
- [ ] Create Python bindings

### Phase 2: Attention Implementation (2-3 days)
- [ ] Map attention parameters to kernel arguments
- [ ] Handle multiple heads and sequences
- [ ] Implement KV cache support
- [ ] Add performance monitoring

### Phase 3: Full Integration (1-2 days)
- [ ] Integrate with existing pipeline
- [ ] Add error handling and fallback
- [ ] Benchmark NPU vs GPU performance
- [ ] Optimize data transfers

## 📊 **Expected Performance**

With NPU execution working:
- **Attention Computation**: 16 TOPS theoretical
- **Expected Speedup**: 2-3x for attention layers
- **Power Efficiency**: 5-10x better than GPU
- **Latency**: Sub-millisecond for small batches

## 🛠️ **Quick Start Commands**

```bash
# Install dependencies
sudo apt install cmake pybind11-dev

# Set XRT paths
export XRT_PATH=/opt/xilinx/xrt
export LD_LIBRARY_PATH=$XRT_PATH/lib:$LD_LIBRARY_PATH

# Build NPU wrapper
cd npu_xrt_wrapper
mkdir build && cd build
cmake .. -DPYTHON_EXECUTABLE=$(which python3)
make -j8

# Test
python3 -c "import npu_xrt; print('NPU XRT module loaded!')"
```

## 🎯 **Success Criteria**

1. **Load kernel binary** into NPU
2. **Execute attention computation** on real data
3. **Verify results** match CPU/GPU output
4. **Measure performance** > 10 GOPS
5. **Integrate** with main pipeline

---

*Once implemented, the NPU will provide 16 TOPS of compute specifically optimized for AI workloads, complementing our GPU's 8.9 TFLOPS for a powerful hybrid system!*