/*
 * GGML NPU Backend Implementation
 * Integrates NPU backend with llama.cpp's GGML framework
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Use the full, real GGML API
#include "ggml.h"
#include "ggml-backend.h"

// Include NPU backend header
extern "C" {
#include "npu_backend.h"
}

// GGML NPU backend context
struct ggml_backend_npu_context {
    npu_context_t* npu_ctx;
    int device_id;
    char name[256];
};

// Re-introduce the ggml_backend struct definition
struct ggml_backend {
    const char* name;
    void* context;
};

// Forward declare the npu_context_t struct
typedef struct npu_context npu_context_t;

// Check if tensor operation should be offloaded to NPU
static bool ggml_npu_should_offload(const struct ggml_tensor* tensor) {
    if (!npu_backend_available()) {
        return false;
    }

    // Offload Flash Attention operations
    if (tensor->op == GGML_OP_FLASH_ATTN_EXT) {
        const struct ggml_tensor* q = tensor->src[0];
        int64_t seq_len = q->ne[2];
        int64_t num_heads = q->ne[3];
        int64_t head_dim = q->ne[1];

        return npu_should_offload_attention(seq_len, num_heads, head_dim);
    }

    // Add other potential offload operations here in the future

    return false;
}

// Initialize GGML NPU backend
extern "C" struct ggml_backend* ggml_backend_npu_init(int device_id) {
    printf("[GGML NPU] Initializing NPU backend...\n");

    if (npu_backend_init() != 0) {
        printf("[GGML NPU] Failed to initialize NPU\n");
        return NULL;
    }

    struct ggml_backend_npu_context* ctx =
        (struct ggml_backend_npu_context*)malloc(sizeof(struct ggml_backend_npu_context));
    if (!ctx) {
        return NULL;
    }

    ctx->device_id = device_id;
    snprintf(ctx->name, sizeof(ctx->name), "NPU:%d", device_id);

    struct ggml_backend* backend = (struct ggml_backend*)malloc(sizeof(struct ggml_backend));
    if (!backend) {
        free(ctx);
        return NULL;
    }

    backend->name = ctx->name;
    backend->context = ctx;

    const npu_device_info_t* info = npu_backend_get_info();
    if (info) {
        printf("[GGML NPU] Backend initialized: %s\n", info->name);
        printf("[GGML NPU]   Performance: %d TOPS (INT8)\n", info->tops_int8);
        printf("[GGML NPU]   Max seq len: %d\n", info->max_seq_len);
    }

    return backend;
}

// Compute attention on NPU
static int ggml_npu_compute_attention(
    const struct ggml_tensor* q,
    const struct ggml_tensor* k,
    const struct ggml_tensor* v,
    struct ggml_tensor* output,
    bool is_causal
) {
    // Extract dimensions from the Q tensor
    int64_t batch_size = q->ne[3];
    int64_t num_heads = q->ne[2];
    int64_t seq_len = q->ne[1];
    int64_t head_dim = q->ne[0];

    printf("[GGML NPU] Computing attention: batch=%ld, heads=%ld, seq_len=%ld, head_dim=%ld\n",
           (long)batch_size, (long)num_heads, (long)seq_len, (long)head_dim);

    // Ensure data is in FP32 format (NPU kernel handles this internally for now)
    if (q->type != GGML_TYPE_F32) {
        // In a real implementation, dequantization might be needed here if the kernel didn't support it.
    }

    // *** FIX: Pass the full tensor objects to the NPU backend ***
    // This provides the necessary metadata (strides, dimensions) for correct GQA-aware indexing.
    int result = npu_attention_forward_int8(
        q, k, v, output,
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        is_causal ? 1 : 0
    );

    return result;
}

// Main compute function for NPU backend
extern "C" int ggml_backend_npu_compute(
    struct ggml_backend* backend,
    struct ggml_tensor* tensor
) {
    if (!backend || !tensor) {
        return -1;
    }

    if (!ggml_npu_should_offload(tensor)) {
        return -2; // Signal that this operation is not handled by the NPU backend
    }

    printf("[GGML NPU] Offloading operation: %s (op=%s)\n",
           tensor->name, ggml_op_name(tensor->op));

    switch (tensor->op) {
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_npu_compute_attention(
                tensor->src[0],  // Q
                tensor->src[1],  // K
                tensor->src[2],  // V
                tensor,          // Output
                true             // is_causal
            );

        default:
            // This operation is not supported on the NPU
            return -2;
    }
}

// Get NPU backend capabilities
extern "C" void ggml_backend_npu_get_caps(
    struct ggml_backend* backend,
    int* max_seq_len,
    int* max_batch_size,
    int* supports_int8
) {
    if (!backend) return;
    
    const npu_device_info_t* info = npu_backend_get_info();
    if (!info) return;
    
    *max_seq_len = info->max_seq_len;
    *max_batch_size = info->max_batch_size;
    *supports_int8 = 1;  // NPU optimized for INT8
}

// Free NPU backend
extern "C" void ggml_backend_npu_free(struct ggml_backend* backend) {
    if (!backend) return;
    
    struct ggml_backend_npu_context* ctx = 
        (struct ggml_backend_npu_context*)backend->context;
    
    npu_backend_cleanup();
    
    free(ctx);
    free(backend);
    
    printf("[GGML NPU] Backend freed\n");
}

// Integration hook for llama.cpp
extern "C" int ggml_backend_register_npu(void) {
    printf("[GGML NPU] Registering NPU backend with GGML...\n");
    
    // In real integration:
    // 1. Register backend with GGML's backend registry
    // 2. Set up function pointers for all operations
    // 3. Configure memory allocation callbacks
    
    return 0;
}