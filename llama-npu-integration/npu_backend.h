/*
 * NPU Backend Header for llama.cpp
 * Public API for NPU acceleration
 */

#ifndef NPU_BACKEND_H
#define NPU_BACKEND_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// NPU device capabilities
typedef struct {
    int available;
    int num_tiles;
    int tops_int8;
    int max_seq_len;
    int max_batch_size;
    char name[256];
} npu_device_info_t;

// NPU buffer structure
typedef struct {
    void* data;
    size_t size;
    int device_id;
    int memory_bank;
} npu_buffer_t;

// NPU context
typedef struct npu_context npu_context_t;

// Core NPU functions
int npu_backend_init(void);
int npu_backend_available(void);
const npu_device_info_t* npu_backend_get_info(void);
void npu_backend_cleanup(void);

// Buffer management
npu_buffer_t* npu_allocate_buffer(size_t size, int memory_bank);
void npu_free_buffer(npu_buffer_t* buffer);

// Operation decisions
int npu_should_offload_attention(int seq_len, int num_heads, int head_dim);

// Forward declare the ggml_tensor struct to avoid pulling in the full ggml.h header.
// This allows us to pass tensor metadata through the backend without creating a hard dependency.
struct ggml_tensor;

// NPU kernels
int npu_attention_forward_int8(
    const struct ggml_tensor * q,
    const struct ggml_tensor * k,
    const struct ggml_tensor * v,
    struct ggml_tensor * output,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    int is_causal
);

// Performance monitoring
void npu_backend_get_stats(
    uint64_t* kernel_time_us,
    uint64_t* transfer_time_us,
    uint64_t* total_ops
);

// Quantization helpers
void quantize_fp32_to_int8(const float* input, int8_t* output, int n, float* scale);
void dequantize_int8_to_fp32(const int8_t* input, float* output, int n, float scale);

#ifdef __cplusplus
}
#endif

#endif // NPU_BACKEND_H