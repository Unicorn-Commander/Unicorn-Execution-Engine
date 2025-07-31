/*
 * NPU Kernel Loader for Real Hardware Execution
 * Loads and manages compiled XCLBIN kernels
 */

#ifndef NPU_KERNEL_LOADER_H
#define NPU_KERNEL_LOADER_H

#include <stdint.h>
#include <string>
#include <memory>
#include <unordered_map>

// Forward declarations for XRT and GGML
namespace xrt {
    class device;
    class xclbin;
    class kernel;
    class bo;
}
struct ggml_tensor; // Forward-declare to avoid dependency

class NPUKernelLoader {
public:
    struct KernelInfo {
        std::string name;
        std::string xclbin_path;
        int seq_len;
        int hidden_dim;
        int num_heads;
        bool loaded;
    };

    NPUKernelLoader();
    ~NPUKernelLoader();

    // Initialize NPU device
    bool initialize(int device_id = 0);

    // Load kernel from XCLBIN
    bool load_kernel(const std::string& xclbin_path, const std::string& kernel_name);

    // Get kernel for specific sequence length
    xrt::kernel* get_attention_kernel(int seq_len);

    // Execute attention kernel, now accepting full tensor metadata
    int execute_attention(
        const struct ggml_tensor * q,
        const struct ggml_tensor * k,
        const struct ggml_tensor * v,
        struct ggml_tensor * output,
        int batch_size,
        int num_heads,
        int seq_len,
        int head_dim,
        bool is_causal
    );

    // Buffer management
    xrt::bo* allocate_buffer(size_t size, int memory_bank);
    void free_buffer(xrt::bo* buffer);

    // Get device info
    bool is_initialized() const { return initialized_; }
    const std::string& get_device_name() const { return device_name_; }

private:
    bool initialized_;
    std::string device_name_;
    std::unique_ptr<xrt::device> device_;
    std::unordered_map<std::string, std::unique_ptr<xrt::xclbin>> xclbins_;
    std::unordered_map<std::string, std::unique_ptr<xrt::kernel>> kernels_;
    std::unordered_map<int, KernelInfo> seq_len_kernels_;

    // Memory banks
    static constexpr int BANK_DMA = 131071;    // 0x1FFFF
    static constexpr int BANK_COMPUTE = 65536;  // 0x10000
    static constexpr int BANK_COMPUTE2 = 65537; // 0x10001

    // Implementation details
    struct Impl;
    std::unique_ptr<Impl> impl_;

    // Helper methods
    void setup_seq_len_kernels();
    std::string get_kernel_path(int seq_len);
};

#endif // NPU_KERNEL_LOADER_H