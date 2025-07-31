#include <vector>
#include <cstdlib> // For rand, RAND_MAX
#include <cmath>   // For sqrtf, expf, INFINITY, fabsf
#include <cstring> // For memset
#include <chrono>  // For std::chrono

// Include our NPU backend
extern "C" {
#include "npu_backend.h"
}

// Include ggml.h for ggml_tensor and ggml_context
#include "ggml.h"

// Test configuration
struct test_config {
    int seq_len;
    int batch_size;
    int num_heads;
    int head_dim;
    const char* description;
};

// Initialize test data
void init_test_data(float* data, size_t size, float scale = 1.0f) {
    for (size_t i = 0; i < size; i++) {
        data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
}

// Compute reference attention on CPU
void attention_reference(
    const ggml_tensor* q, const ggml_tensor* k, const ggml_tensor* v,
    ggml_tensor* output
) {
    const int64_t head_dim = q->ne[0];
    const int64_t seq_len = q->ne[1];
    const int64_t num_heads = q->ne[2];
    const int64_t num_kv_heads = k->ne[2]; // Assuming GQA

    const float* q_data = (const float*)q->data;
    const float* k_data = (const float*)k->data;
    const float* v_data = (const float*)v->data;
    float* out_data = (float*)output->data;

    const float scale = 1.0f / sqrtf((float)head_dim);

    for (int h = 0; h < num_heads; h++) {
        const int kv_head_idx = h / (num_heads / num_kv_heads);
        for (int i = 0; i < seq_len; i++) {
            std::vector<float> scores(seq_len);
            float max_score = -INFINITY;

            for (int j = 0; j < seq_len; j++) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    const size_t q_idx = (h * q->nb[2]) + (i * q->nb[1]) + (d * q->nb[0]);
                    const size_t k_idx = (kv_head_idx * k->nb[2]) + (j * k->nb[1]) + (d * k->nb[0]);
                    score += q_data[q_idx / sizeof(float)] * k_data[k_idx / sizeof(float)];
                }
                scores[j] = score * scale;
                // Causal mask (assuming causal attention for this test)
                if (j > i) scores[j] = -INFINITY;
                max_score = std::max(max_score, scores[j]);
            }

            float sum_exp = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                scores[j] = expf(scores[j] - max_score);
                sum_exp += scores[j];
            }
            const float inv_sum_exp = 1.0f / sum_exp;
            for (int j = 0; j < seq_len; j++) scores[j] *= inv_sum_exp;

            for (int d = 0; d < head_dim; d++) {
                float out_val = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    const size_t v_idx = (kv_head_idx * v->nb[2]) + (j * v->nb[1]) + (d * v->nb[0]);
                    out_val += scores[j] * v_data[v_idx / sizeof(float)];
                }
                const size_t out_idx = (h * output->nb[2]) + (i * output->nb[1]) + (d * output->nb[0]);
                out_data[out_idx / sizeof(float)] = out_val;
            }
        }
    }
}

// Test a specific configuration
bool test_configuration(const test_config& config) {
    printf("\n=== Testing: %s ===\n", config.description);
    printf("Configuration: batch=%d, heads=%d, seq_len=%d, head_dim=%d\n",
           config.batch_size, config.num_heads, config.seq_len, config.head_dim);

    // Check if NPU supports this configuration
    if (!npu_should_offload_attention(config.seq_len, config.num_heads, config.head_dim)) {
        printf("⚠️  Configuration not supported by NPU\n");
        return true;  // Not a failure, just skip
    }

    // GGML context for tensor allocation
    struct ggml_init_params params = { 16 * 1024 * 1024, NULL, false };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        printf("❌ Failed to initialize GGML context\n");
        return false;
    }

    // Create ggml_tensors for Q, K, V, and Output
    // Assuming GQA with num_kv_heads = num_heads for simplicity in this test
    const int num_kv_heads = config.num_heads; 

    struct ggml_tensor* q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, config.head_dim, config.seq_len, config.num_heads);
    struct ggml_tensor* k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, config.head_dim, config.seq_len, num_kv_heads);
    struct ggml_tensor* v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, config.head_dim, config.seq_len, num_kv_heads);
    struct ggml_tensor* output_npu = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, config.head_dim, config.seq_len, config.num_heads);
    struct ggml_tensor* output_ref = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, config.head_dim, config.seq_len, config.num_heads);

    if (!q || !k || !v || !output_npu || !output_ref) {
        printf("❌ Failed to create ggml_tensors\n");
        ggml_free(ctx);
        return false;
    }

    // Initialize with test data
    init_test_data((float*)q->data, ggml_nelements(q), 0.1f);
    init_test_data((float*)k->data, ggml_nelements(k), 0.1f);
    init_test_data((float*)v->data, ggml_nelements(v), 0.1f);
    memset(output_npu->data, 0, ggml_nbytes(output_npu));
    memset(output_ref->data, 0, ggml_nbytes(output_ref));

    // Run on NPU
    printf("Executing on NPU...\n");
    auto start_npu = std::chrono::high_resolution_clock::now();

    int npu_result = npu_attention_forward_int8(
        q, k, v, output_npu,
        config.batch_size, config.num_heads, config.seq_len, config.head_dim,
        0  // not causal for testing
    );

    auto end_npu = std::chrono::high_resolution_clock::now();
    double npu_time_ms = std::chrono::duration<double, std::milli>(end_npu - start_npu).count();

    if (npu_result != 0) {
        printf("❌ NPU execution failed with code %d\n", npu_result);
        ggml_free(ctx);
        return false;
    }

    printf("✅ NPU execution completed in %.2f ms\n", npu_time_ms);

    // Compute reference on CPU
    printf("Computing CPU reference...\n");
    auto start_cpu = std::chrono::high_resolution_clock::now();

    attention_reference(q, k, v, output_ref);

    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time_ms = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    printf("CPU reference completed in %.2f ms\n", cpu_time_ms);
    printf("Speedup: %.2fx\n", cpu_time_ms / npu_time_ms);

    // Compare results
    float max_error = 0.0f;
    float avg_error = 0.0f;
    size_t total_elements = ggml_nelements(output_npu);
    for (size_t i = 0; i < total_elements; i++) {
        float error = fabsf(((float*)output_npu->data)[i] - ((float*)output_ref->data)[i]);
        avg_error += error;
        if (error > max_error) max_error = error;
    }
    avg_error /= total_elements;

    printf("Accuracy:\n");
    printf("  Max error: %e\n", max_error);
    printf("  Avg error: %e\n", avg_error);

    bool passed = max_error < 0.01f;  // 1% tolerance for INT8
    printf("  Status: %s\n", passed ? "✅ PASSED" : "❌ FAILED");

    // Performance metrics
    int64_t flops = 2LL * config.batch_size * config.num_heads *
                    config.seq_len * config.seq_len * config.head_dim;
    double tflops = (double)flops / 1e12 / ((double)npu_time_ms / 1000.0);
    printf("Performance: %.2f TFLOPS\n", tflops);

    // Cleanup
    ggml_free(ctx);

    return passed;
}

// Test kernel availability
void test_kernel_availability() {
    printf("\n=== Kernel Availability Test ===\n");

    const npu_device_info_t* info = npu_backend_get_info();
    if (!info) {
        printf("❌ NPU not available\n");
        return;
    }

    printf("NPU Device: %s\n", info->name);
    printf("Capabilities:\n");
    printf("  - INT8 Performance: %d TOPS\n", info->tops_int8);
    printf("  - Max sequence length: %d\n", info->max_seq_len);
    printf("  - Max batch size: %d\n", info->max_batch_size);

    // Test which configurations are supported
    int test_seq_lens[] = {64, 128, 256, 512, 1024, 2048};
    int test_head_dims[] = {64, 128};
    int test_num_heads[] = {8, 16, 32};

    printf("\nSupported configurations:\n");
    printf("SeqLen | Heads | HeadDim | Supported\n");
    printf("-------|-------|---------|----------\n");

    for (int seq_len : test_seq_lens) {
        for (int num_heads : test_num_heads) {
            for (int head_dim : test_head_dims) {
                bool supported = npu_should_offload_attention(seq_len, num_heads, head_dim);
                printf("%6d | %5d | %7d | %s\n", 
                       seq_len, num_heads, head_dim,
                       supported ? "✅ Yes" : "❌ No");
            }
        }
    }
}

int main() {
    printf("🦄 Real NPU Kernel Test Suite\n");
    printf("=============================\n");

    // Initialize NPU backend
    printf("\nInitializing NPU backend...\n");
    if (npu_backend_init() != 0) {
        printf("❌ Failed to initialize NPU backend\n");
        return 1;
    }

    // Seed random generator
    srand(12345);

    // Test kernel availability
    test_kernel_availability();

    // Define test configurations
    std::vector<test_config> configs = {
        {128, 1, 16, 64, "Small context (128 tokens)"},
        {256, 1, 16, 64, "Medium context (256 tokens)"},
        {512, 1, 16, 64, "Large context (512 tokens)"},
        {1024, 1, 16, 64, "XL context (1024 tokens)"},

        // Different head configurations
        {256, 1, 8, 64, "8 heads, 256 context"},
        {256, 1, 32, 64, "32 heads, 256 context"},

        // Different head dimensions
        {256, 1, 16, 128, "128 head dimension"},
    };

    // Run tests
    int passed = 0;
    int total = 0;

    for (const auto& config : configs) {
        if (test_configuration(config)) {
            passed++;
        }
        total++;
    }

    // Summary
    printf("\n=== Test Summary ===\n");
    printf("Total tests: %d\n", total);
    printf("Passed: %d\n", passed);
    printf("Failed: %d\n", total - passed);

    // Performance summary
    uint64_t kernel_time, transfer_time, total_ops;
    npu_backend_get_stats(&kernel_time, &transfer_time, &total_ops);

    if (total_ops > 0) {
        printf("\nPerformance Statistics:\n");
        printf("  Total operations: %.2f GOPS\n", (double)total_ops / 1e9);
        printf("  Kernel time: %.2f ms\n", (double)kernel_time / 1000.0);
        printf("  Transfer time: %.2f ms\n", (double)transfer_time / 1000.0);
    }

    // Cleanup
    npu_backend_cleanup();

    printf("\n%s All tests completed!\n",
           passed == total ? "✅" : "⚠️");

    return (passed == total) ? 0 : 1;
}