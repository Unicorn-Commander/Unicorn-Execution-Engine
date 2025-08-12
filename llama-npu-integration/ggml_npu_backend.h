/*
 * GGML NPU Backend Header
 * Integration interface for llama.cpp's GGML framework
 */

#ifndef GGML_NPU_BACKEND_H
#define GGML_NPU_BACKEND_H

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct ggml_backend;
struct ggml_tensor;

// Initialize GGML NPU backend
struct ggml_backend* ggml_backend_npu_init(int device_id);

// Free NPU backend
void ggml_backend_npu_free(struct ggml_backend* backend);

// Check if tensor should be offloaded to NPU
bool ggml_npu_should_offload(const struct ggml_tensor* tensor);

// Compute operation on NPU
int ggml_backend_npu_compute(
    struct ggml_backend* backend,
    struct ggml_tensor* tensor
);

// Get NPU capabilities
void ggml_backend_npu_get_caps(
    struct ggml_backend* backend,
    int* max_seq_len,
    int* max_batch_size,
    int* supports_int8
);

// Register NPU backend with GGML
int ggml_backend_register_npu(void);

// NPU-Vulkan bridge functions
int npu_vulkan_bridge_init(int enable_npu, int enable_vulkan, int verbose);

int npu_vulkan_bridge_submit_attention(
    const struct ggml_tensor* q,
    const struct ggml_tensor* k,
    const struct ggml_tensor* v,
    struct ggml_tensor* output
);

int npu_vulkan_bridge_submit_linear(
    float* input,
    float* output,
    int batch_size,
    int in_dim,
    int out_dim
);

void npu_vulkan_bridge_get_stats(
    uint64_t* npu_ops,
    uint64_t* vulkan_ops,
    uint64_t* npu_time_us,
    uint64_t* vulkan_time_us
);

void npu_vulkan_bridge_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif // GGML_NPU_BACKEND_H