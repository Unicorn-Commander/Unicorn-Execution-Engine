/*
 * NPU Backend with Real Kernel Execution
 * Uses compiled XCLBIN kernels for actual NPU acceleration
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <memory>

#include "npu_kernel_loader.h"
#include "ggml.h" // Use the full GGML API for tensor structures

#ifdef __cplusplus
extern "C" {
#endif

// Include original header
#include "npu_backend.h"
#include "npu_backend_internal.h"

// Global NPU kernel loader
static std::unique_ptr<NPUKernelLoader> g_kernel_loader = nullptr;

// Global NPU context (simplified)
static npu_context_t* g_npu_ctx = NULL;

// Initialize NPU backend with real kernels
int npu_backend_init(void) {
    if (g_npu_ctx != NULL) {
        return 0; // Already initialized
    }

    printf("[NPU Backend] Initializing with real kernel support...\n");

    // Create context
    g_npu_ctx = (npu_context_t*)calloc(1, sizeof(npu_context_t));
    if (!g_npu_ctx) {
        return -1;
    }

    // Initialize kernel loader
    g_kernel_loader = std::make_unique<NPUKernelLoader>();
    if (!g_kernel_loader->initialize(0)) {
        printf("[NPU Backend] Failed to initialize kernel loader\n");
        free(g_npu_ctx);
        g_npu_ctx = NULL;
        g_kernel_loader.reset();
        return -1;
    }

    // Set device info
    g_npu_ctx->info.available = 1;
    g_npu_ctx->info.num_tiles = 20;  // Phoenix NPU: 4x5 topology
    g_npu_ctx->info.tops_int8 = 16;  // 16 TOPS
    g_npu_ctx->info.max_seq_len = 4096; // Kernels up to 4096
    g_npu_ctx->info.max_batch_size = 1;  // Current kernels support batch=1
    snprintf(g_npu_ctx->info.name, sizeof(g_npu_ctx->info.name),
             "%s", g_kernel_loader->get_device_name().c_str());

    printf("[NPU Backend] Initialized: %s\n", g_npu_ctx->info.name);
    printf("[NPU Backend] Real kernel execution enabled!\n");

    return 0;
}

// Check if NPU is available
int npu_backend_available(void) {
    return g_npu_ctx != NULL && g_kernel_loader != nullptr;
}

// Get NPU device info
const npu_device_info_t* npu_backend_get_info(void) {
    if (!g_npu_ctx) return NULL;
    return &g_npu_ctx->info;
}

// Check if operation should use NPU
int npu_should_offload_attention(int seq_len, int num_heads, int head_dim) {
    if (!npu_backend_available()) return 0;

    // Check if we have a kernel for this sequence length
    if (seq_len > g_npu_ctx->info.max_seq_len) return 0;

    // We have kernels for various sequence lengths
    if (g_kernel_loader->get_attention_kernel(seq_len) != nullptr) {
        // Check head dimensions (our kernels expect specific configs)
        if (head_dim == 64 || head_dim == 128) {
            return 1;  // Use NPU
        }
    }

    return 0;
}

// Allocate NPU buffer (using XRT)
npu_buffer_t* npu_allocate_buffer(size_t size, int memory_bank) {
    if (!g_kernel_loader) return nullptr;

    npu_buffer_t* buffer = (npu_buffer_t*)malloc(sizeof(npu_buffer_t));
    if (!buffer) return nullptr;

    buffer->size = size;
    buffer->memory_bank = memory_bank;
    buffer->device_id = 0;

    // Allocate through kernel loader
    xrt::bo* xrt_buffer = g_kernel_loader->allocate_buffer(size, memory_bank);
    if (!xrt_buffer) {
        free(buffer);
        return nullptr;
    }

    buffer->data = xrt_buffer;
    return buffer;
}

// Free NPU buffer
void npu_free_buffer(npu_buffer_t* buffer) {
    if (!buffer) return;

    if (buffer->data && g_kernel_loader) {
        g_kernel_loader->free_buffer((xrt::bo*)buffer->data);
    }

    free(buffer);
}

// NPU attention kernel with real execution
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
) {
    if (!npu_backend_available()) {
        printf("[NPU Backend] NPU not available\n");
        return -1;
    }

    if (!g_kernel_loader) {
        printf("[NPU Backend] Kernel loader not initialized\n");
        return -1;
    }

    // Check if we support this configuration
    if (batch_size != 1) {
        printf("[NPU Backend] Warning: Current kernels only support batch_size=1\n");
        return -1;
    }

    printf("[NPU Backend] Executing real NPU kernel: seq_len=%d, heads=%d, dim=%d\n",
           seq_len, num_heads, head_dim);

    // Execute on real NPU hardware, now passing the full tensors
    int result = g_kernel_loader->execute_attention(
        q, k, v, output,
        batch_size, num_heads, seq_len, head_dim, is_causal
    );

    if (result == 0) {
        // Update statistics
        g_npu_ctx->total_ops += 2LL * batch_size * num_heads * seq_len * seq_len * head_dim;
        g_npu_ctx->kernel_time_us += 1000;  // Actual time tracked by kernel loader

        // This message is now printed inside the kernel loader for better timing accuracy
        // printf("[NPU Backend] Real NPU execution completed successfully!\n");
    } else {
        printf("[NPU Backend] NPU execution failed, falling back might be needed\n");
    }

    return result;
}


// Get NPU performance stats
void npu_backend_get_stats(uint64_t* kernel_time_us, uint64_t* transfer_time_us, uint64_t* total_ops) {
    if (!g_npu_ctx) {
        *kernel_time_us = 0;
        *transfer_time_us = 0;
        *total_ops = 0;
        return;
    }
    
    *kernel_time_us = g_npu_ctx->kernel_time_us;
    *transfer_time_us = g_npu_ctx->transfer_time_us;
    *total_ops = g_npu_ctx->total_ops;
}

// Cleanup NPU backend
void npu_backend_cleanup(void) {
    if (!g_npu_ctx) return;
    
    printf("[NPU Backend] Cleaning up...\n");
    
    // Free buffers
    npu_free_buffer(g_npu_ctx->q_buffer);
    npu_free_buffer(g_npu_ctx->k_buffer);
    npu_free_buffer(g_npu_ctx->v_buffer);
    npu_free_buffer(g_npu_ctx->out_buffer);
    
    // Cleanup kernel loader
    g_kernel_loader.reset();
    
    free(g_npu_ctx);
    g_npu_ctx = NULL;
    
    printf("[NPU Backend] Cleaned up\n");
}

// INT8 quantization helper (keeping original for compatibility)
void quantize_fp32_to_int8(const float* input, int8_t* output, int n, float* scale) {
    float max_val = 0.0f;
    
    for (int i = 0; i < n; i++) {
        float abs_val = fabsf(input[i]);
        if (abs_val > max_val) max_val = abs_val;
    }
    
    *scale = max_val / 127.0f;
    if (*scale < 1e-6f) *scale = 1e-6f;
    
    float inv_scale = 1.0f / (*scale);
    for (int i = 0; i < n; i++) {
        int32_t q = (int32_t)roundf(input[i] * inv_scale);
        output[i] = (int8_t)(q > 127 ? 127 : q < -128 ? -128 : q);
    }
}

void dequantize_int8_to_fp32(const int8_t* input, float* output, int n, float scale) {
    for (int i = 0; i < n; i++) {
        output[i] = (float)input[i] * scale;
    }
}

#ifdef __cplusplus
}
#endif