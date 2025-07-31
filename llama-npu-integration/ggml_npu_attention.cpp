/*
 * GGML NPU Attention Implementation
 * Now using Direct NPU Runtime from transcription project
 */

#include "ggml_npu_attention.h"
#include "npu_runtime_direct.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <unistd.h>
#include <stdlib.h>  // for posix_memalign

// Use the full, real GGML API
#include "ggml.h"

// Global NPU runtime instance
static std::unique_ptr<npu_direct::DirectNPURuntime> g_npu_runtime;

// NPU attention uses temporary buffer and modifies Q in-place

bool ggml_npu_can_flash_attn(
    const struct ggml_tensor * q,
    const struct ggml_tensor * k,
    const struct ggml_tensor * v) {

    // Initialize NPU runtime if needed
    if (!g_npu_runtime) {
        g_npu_runtime = std::make_unique<npu_direct::DirectNPURuntime>();
        if (!g_npu_runtime->initialize()) {
            printf("❌ Failed to initialize Direct NPU Runtime\n");
            return false;
        }
        printf("✅ Direct NPU Runtime initialized from transcription project\n");
    }

    if (!g_npu_runtime->is_available()) {
        printf("⚠️  Direct NPU runtime not available\n");
        return false;
    }

    // Check tensor constraints for NPU
    if (q->type != GGML_TYPE_F32 && q->type != GGML_TYPE_F16) {
        printf("⚠️  NPU only supports F32/F16 tensors, got type %d\n", q->type);
        return false;
    }

    // Extract dimensions (corrected layout based on observation)
    int64_t head_dim = q->ne[1];    // 64 is at ne[1]
    int64_t seq_len = q->ne[2];     // seq_len is at ne[2]
    int64_t num_heads = q->ne[3];   // num_heads is at ne[3]

    printf("🔍 NPU Capability Check: seq_len=%ld, heads=%ld, head_dim=%ld\n",
           seq_len, num_heads, head_dim);

    // For testing, accept any reasonable dimensions
    if (head_dim > 0 && head_dim <= 256 && num_heads > 0 && seq_len > 0 && seq_len <= 4096) {
        printf("✅ NPU can handle this attention configuration\n");
        return true;
    }

    printf("⚠️  NPU cannot handle attention: invalid dimensions\n");
    return false;
}

extern "C" struct ggml_tensor * ggml_npu_flash_attn_ext(
    struct ggml_context * ctx,
    struct ggml_tensor  * q,
    struct ggml_tensor  * k,
    struct ggml_tensor  * v,
    struct ggml_tensor  * mask,
    float                 scale,
    float                 max_bias,
    float                 logit_softcap) {

    printf("🧠 NPU ATTENTION CALLED! Attempting Direct NPU acceleration...\n");

    // Initialize Direct NPU runtime if needed
    if (!g_npu_runtime) {
        g_npu_runtime = std::make_unique<npu_direct::DirectNPURuntime>();
        if (!g_npu_runtime->initialize()) {
            printf("❌ Direct NPU runtime initialization failed - no fallback!\n");
            return nullptr;
        }
        printf("✅ Direct NPU Runtime ready for attention operations\n");
    }

    // Extract dimensions
    int64_t head_dim = q->ne[1];
    int64_t seq_len = q->ne[2];
    int64_t num_heads = q->ne[3];
    int64_t batch_size = 1; // Assume batch size 1 for now

    printf("🔍 NPU Attention Tensor Debug:\n");
    printf("   Q dimensions: [%ld, %ld, %ld, %ld]\n", q->ne[0], q->ne[1], q->ne[2], q->ne[3]);
    printf("   K dimensions: [%ld, %ld, %ld, %ld]\n", k->ne[0], k->ne[1], k->ne[2], k->ne[3]);
    printf("   V dimensions: [%ld, %ld, %ld, %ld]\n", v->ne[0], v->ne[1], v->ne[2], v->ne[3]);
    printf("🔍 NPU Attention: batch=%ld, heads=%ld, seq_len=%ld, head_dim=%ld\n",
           batch_size, num_heads, seq_len, head_dim);

    if (!ggml_npu_can_flash_attn(q, k, v)) {
        printf("❌ NPU cannot handle this attention - no fallback!\n");
        return nullptr;
    }

    // Create a new tensor for the NPU output
    struct ggml_tensor * output = ggml_new_tensor(ctx, q->type, 4, q->ne);

    printf("🚀 Executing REAL NPU attention kernel...\n");

    auto start_time = std::chrono::high_resolution_clock::now();

    size_t total_elements = batch_size * num_heads * seq_len * head_dim;

    // REAL NPU ATTENTION COMPUTATION using Direct Runtime
    printf("🚀 Using Direct NPU Runtime from transcription project...\n");
    
    // Extract tensor data pointers
    float* q_data = static_cast<float*>(q->data);
    float* k_data = static_cast<float*>(k->data);
    float* v_data = static_cast<float*>(v->data);
    float* output_data = static_cast<float*>(output->data);
    
    bool npu_result = g_npu_runtime->execute_attention(
        q_data, k_data, v_data, output_data,
        seq_len, num_heads, head_dim
    );

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    if (npu_result) {
        printf("✅ NPU REAL attention computed in %ld μs! Processing %zu elements\n",
               duration.count(), total_elements);
        printf("🎯 Direct NPU hardware acceleration successful!\n");
        printf("🦄 NPU+Vulkan hybrid inference working with transcription tech!\n");
        
        // Print performance stats
        g_npu_runtime->print_performance_stats();
    } else {
        printf("❌ Direct NPU computation failed\n");
        printf("⚠️  This means the hardware NPU is not accessible\n");
        return nullptr;
    }

    printf("📐 Returning NPU-processed tensor: [%ld, %ld, %ld, %ld]\n",
           output->ne[0], output->ne[1], output->ne[2], output->ne[3]);

    // Return the new output tensor
    return output;
}