/*
 * NPU Kernel Loader Implementation
 * Uses real compiled XCLBIN kernels with XRT
 */

#include "npu_kernel_loader.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <chrono>
#include <filesystem>

// XRT includes
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

namespace fs = std::filesystem;

NPUKernelLoader::NPUKernelLoader() : initialized_(false) {
    setup_seq_len_kernels();
}

NPUKernelLoader::~NPUKernelLoader() {
    // XRT cleanup is automatic with smart pointers
}

void NPUKernelLoader::setup_seq_len_kernels() {
    // Map sequence lengths to kernel paths
    const std::string kernel_dir = "../npu_kernels_gemma3_4b/";
    
    seq_len_kernels_[128] = {
        "attention_128", 
        kernel_dir + "attention_gemma3_4b_128.xclbin",
        128, 1024, 16, false
    };
    
    seq_len_kernels_[256] = {
        "attention_256",
        kernel_dir + "attention_gemma3_4b_256.xclbin", 
        256, 1024, 16, false
    };
    
    seq_len_kernels_[512] = {
        "attention_512",
        kernel_dir + "attention_gemma3_4b_512.xclbin",
        512, 1024, 16, false
    };
    
    seq_len_kernels_[1024] = {
        "attention_1024",
        kernel_dir + "attention_gemma3_4b_1024.xclbin",
        1024, 1024, 16, false
    };
}

bool NPUKernelLoader::initialize(int device_id) {
    try {
        std::cout << "[NPU Kernel Loader] Initializing NPU device..." << std::endl;
        
        // Open NPU device
        device_ = std::make_unique<xrt::device>(device_id);
        
        // Get device name
        device_name_ = device_->get_info<xrt::info::device::name>();
        std::cout << "[NPU Kernel Loader] Device: " << device_name_ << std::endl;
        
        // Check if this is Phoenix NPU
        if (device_name_.find("Phoenix") == std::string::npos &&
            device_name_.find("NPU") == std::string::npos) {
            std::cerr << "[NPU Kernel Loader] Warning: Device may not be Phoenix NPU" << std::endl;
        }
        
        initialized_ = true;
        std::cout << "[NPU Kernel Loader] Initialization successful!" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[NPU Kernel Loader] Initialization failed: " << e.what() << std::endl;
        return false;
    }
}

bool NPUKernelLoader::load_kernel(const std::string& xclbin_path, const std::string& kernel_name) {
    if (!initialized_) {
        std::cerr << "[NPU Kernel Loader] Not initialized" << std::endl;
        return false;
    }
    
    try {
        // Check if file exists
        if (!fs::exists(xclbin_path)) {
            std::cerr << "[NPU Kernel Loader] XCLBIN not found: " << xclbin_path << std::endl;
            return false;
        }
        
        std::cout << "[NPU Kernel Loader] Loading kernel from: " << xclbin_path << std::endl;
        
        // Load XCLBIN
        auto xclbin = std::make_unique<xrt::xclbin>(xclbin_path);
        device_->register_xclbin(*xclbin);
        
        // Get kernel
        auto kernel = std::make_unique<xrt::kernel>(*device_, xclbin->get_uuid(), kernel_name);
        
        // Store for later use
        xclbins_[kernel_name] = std::move(xclbin);
        kernels_[kernel_name] = std::move(kernel);
        
        std::cout << "[NPU Kernel Loader] Kernel loaded: " << kernel_name << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[NPU Kernel Loader] Failed to load kernel: " << e.what() << std::endl;
        return false;
    }
}

xrt::kernel* NPUKernelLoader::get_attention_kernel(int seq_len) {
    // Find closest sequence length
    int best_seq_len = 0;
    for (const auto& [len, info] : seq_len_kernels_) {
        if (len >= seq_len && (best_seq_len == 0 || len < best_seq_len)) {
            best_seq_len = len;
        }
    }
    
    if (best_seq_len == 0) {
        std::cerr << "[NPU Kernel Loader] No kernel for seq_len=" << seq_len << std::endl;
        return nullptr;
    }
    
    auto& kernel_info = seq_len_kernels_[best_seq_len];
    
    // Load kernel if not already loaded
    if (!kernel_info.loaded) {
        if (load_kernel(kernel_info.xclbin_path, kernel_info.name)) {
            kernel_info.loaded = true;
        } else {
            return nullptr;
        }
    }
    
    auto it = kernels_.find(kernel_info.name);
    return (it != kernels_.end()) ? it->second.get() : nullptr;
}

xrt::bo* NPUKernelLoader::allocate_buffer(size_t size, int memory_bank) {
    if (!initialized_) return nullptr;
    
    try {
        // Use cacheable flag for better performance
        auto buffer = new xrt::bo(*device_, size, xrt::bo::flags::cacheable, memory_bank);
        return buffer;
    } catch (const std::exception& e) {
        std::cerr << "[NPU Kernel Loader] Buffer allocation failed: " << e.what() << std::endl;
        return nullptr;
    }
}

void NPUKernelLoader::free_buffer(xrt::bo* buffer) {
    delete buffer;
}

int NPUKernelLoader::execute_attention(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    bool is_causal
) {
    if (!initialized_) {
        std::cerr << "[NPU Kernel Loader] Not initialized" << std::endl;
        return -1;
    }
    
    // Get appropriate kernel
    xrt::kernel* kernel = get_attention_kernel(seq_len);
    if (!kernel) {
        std::cerr << "[NPU Kernel Loader] No kernel available for seq_len=" << seq_len << std::endl;
        return -1;
    }
    
    std::cout << "[NPU Kernel Loader] Executing attention: "
              << "batch=" << batch_size << ", heads=" << num_heads 
              << ", seq=" << seq_len << ", dim=" << head_dim << std::endl;
    
    try {
        // Calculate buffer sizes
        size_t qkv_size = batch_size * num_heads * seq_len * head_dim * sizeof(float);
        size_t out_size = batch_size * num_heads * seq_len * head_dim * sizeof(float);
        
        // Allocate NPU buffers
        auto q_bo = xrt::bo(*device_, qkv_size, xrt::bo::flags::cacheable, BANK_DMA);
        auto k_bo = xrt::bo(*device_, qkv_size, xrt::bo::flags::cacheable, BANK_DMA);
        auto v_bo = xrt::bo(*device_, qkv_size, xrt::bo::flags::cacheable, BANK_DMA);
        auto out_bo = xrt::bo(*device_, out_size, xrt::bo::flags::cacheable, BANK_DMA);
        
        // Map buffers for writing
        float* q_map = q_bo.map<float*>();
        float* k_map = k_bo.map<float*>();
        float* v_map = v_bo.map<float*>();
        
        // Copy input data
        memcpy(q_map, q, qkv_size);
        memcpy(k_map, k, qkv_size);
        memcpy(v_map, v, qkv_size);
        
        // Sync to device
        q_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        k_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        v_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        
        // Create run handle
        auto run = xrt::run(*kernel);
        
        // Set kernel arguments
        run.set_arg(0, q_bo);     // Q input
        run.set_arg(1, k_bo);     // K input
        run.set_arg(2, v_bo);     // V input
        run.set_arg(3, out_bo);   // Output
        run.set_arg(4, batch_size);
        run.set_arg(5, num_heads);
        run.set_arg(6, seq_len);
        run.set_arg(7, head_dim);
        run.set_arg(8, is_causal ? 1 : 0);
        
        // Execute kernel
        auto start = std::chrono::high_resolution_clock::now();
        run.start();
        run.wait();
        auto end = std::chrono::high_resolution_clock::now();
        
        // Sync output back
        out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        
        // Copy output data
        float* out_map = out_bo.map<float*>();
        memcpy(output, out_map, out_size);
        
        // Calculate execution time
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "[NPU Kernel Loader] Execution time: " << duration.count() / 1000.0 << " ms" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "[NPU Kernel Loader] Execution failed: " << e.what() << std::endl;
        return -1;
    }
}

std::string NPUKernelLoader::get_kernel_path(int seq_len) {
    auto it = seq_len_kernels_.find(seq_len);
    if (it != seq_len_kernels_.end()) {
        return it->second.xclbin_path;
    }
    return "";
}