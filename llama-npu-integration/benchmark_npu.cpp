/*
 * NPU Backend Benchmark Tool
 * Measures performance of NPU operations
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <chrono>
#include <vector>
#include <algorithm>

// Include our headers
#include "npu_backend.h"
#include "ggml_npu_backend.h"
#include "ggml.h" // Include ggml.h for ggml_tensor and ggml_context

// Benchmark configuration
struct benchmark_config {
    int warmup_runs;
    int benchmark_runs;
    std::vector<int> sequence_lengths;
    std::vector<int> batch_sizes;
    std::vector<int> head_counts;
    int head_dim;
    bool verbose;
};

// Benchmark result
struct benchmark_result {
    int seq_len;
    int batch_size;
    int num_heads;
    double mean_time_ms;
    double std_time_ms;
    double min_time_ms;
    double max_time_ms;
    double tflops;
    double gb_per_sec;
};

// Helper to allocate aligned memory (no longer needed for ggml_tensor data)
// void* aligned_alloc_wrapper(size_t alignment, size_t size) {
//     void* ptr = nullptr;
// #ifdef _WIN32
//     ptr = _aligned_malloc(size, alignment);
// #else
//     if (posix_memalign(&ptr, alignment, size) != 0) {
//         return nullptr;
//     }
// #endif
//     return ptr;
// }

// Helper to free aligned memory (no longer needed for ggml_tensor data)
// void aligned_free_wrapper(void* ptr) {
// #ifdef _WIN32
//     _aligned_free(ptr);
// #else
//     free(ptr);
// #endif
// }

// Initialize random tensor
void init_random_tensor(float* data, size_t size) {
    for (size_t i = 0; i < size; i++) {
        data[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
}

// Benchmark attention operation
benchmark_result benchmark_attention(
    const benchmark_config& config,
    int seq_len,
    int batch_size,
    int num_heads
) {
    benchmark_result result = {
        .seq_len = seq_len,
        .batch_size = batch_size,
        .num_heads = num_heads
    };

    // GGML context for tensor allocation
    struct ggml_init_params params = { 16 * 1024 * 1024, NULL, false };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        printf("Failed to initialize GGML context\n");
        return result;
    }

    // Create ggml_tensors for Q, K, V, and Output
    // Assuming Q, K, V, Output have dimensions [head_dim, seq_len, num_heads]
    // For GQA, K and V might have fewer heads (num_kv_heads)
    // For simplicity in this benchmark, we'll assume num_kv_heads == num_heads
    // In a real scenario, you'd need to pass num_kv_heads from the model.
    const int num_kv_heads = num_heads; // Simplified for benchmark

    struct ggml_tensor* q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, config.head_dim, seq_len, num_heads);
    struct ggml_tensor* k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, config.head_dim, seq_len, num_kv_heads);
    struct ggml_tensor* v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, config.head_dim, seq_len, num_kv_heads);
    struct ggml_tensor* output = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, config.head_dim, seq_len, num_heads);

    if (!q || !k || !v || !output) {
        printf("Failed to create ggml_tensors\n");
        ggml_free(ctx);
        return result;
    }

    // Fill with random data
    init_random_tensor((float*)q->data, ggml_nelements(q));
    init_random_tensor((float*)k->data, ggml_nelements(k));
    init_random_tensor((float*)v->data, ggml_nelements(v));

    // Warmup runs
    if (config.verbose) {
        printf("Warming up with %d runs...\n", config.warmup_runs);
    }

    for (int i = 0; i < config.warmup_runs; i++) {
        npu_attention_forward_int8(
            q, k, v, output,
            batch_size, num_heads, seq_len, config.head_dim,
            1  // causal
        );
    }

    // Benchmark runs
    std::vector<double> times_ms;
    times_ms.reserve(config.benchmark_runs);

    if (config.verbose) {
        printf("Running %d benchmark iterations...\n", config.benchmark_runs);
    }

    for (int i = 0; i < config.benchmark_runs; i++) {
        auto start = std::chrono::high_resolution_clock::now();

        int status = npu_attention_forward_int8(
            q, k, v, output,
            batch_size, num_heads, seq_len, config.head_dim,
            1  // causal
        );

        auto end = std::chrono::high_resolution_clock::now();

        if (status != 0) {
            printf("NPU operation failed with status %d\n", status);
            break;
        }

        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        times_ms.push_back(time_ms);
    }

    // Calculate statistics
    if (!times_ms.empty()) {
        // Mean
        double sum = 0.0;
        for (double t : times_ms) sum += t;
        result.mean_time_ms = sum / times_ms.size();

        // Standard deviation
        double variance = 0.0;
        for (double t : times_ms) {
            double diff = t - result.mean_time_ms;
            variance += diff * diff;
        }
        result.std_time_ms = sqrt(variance / times_ms.size());

        // Min/Max
        result.min_time_ms = *std::min_element(times_ms.begin(), times_ms.end());
        result.max_time_ms = *std::max_element(times_ms.begin(), times_ms.end());

        // Calculate FLOPS
        // Attention FLOPs: 2 * batch * heads * seq^2 * dim (for Q*K)
        //                + 2 * batch * heads * seq^2 * dim (for scores*V)
        int64_t flops = 4LL * batch_size * num_heads * seq_len * seq_len * config.head_dim;
        result.tflops = (flops / 1e12) / (result.mean_time_ms / 1000.0);

        // Calculate bandwidth
        // Memory: 3 input tensors + 1 output tensor
        int64_t bytes = 4LL * ggml_nelements(q) * sizeof(float);
        result.gb_per_sec = (bytes / 1e9) / (result.mean_time_ms / 1000.0);
    }

    // Cleanup
    ggml_free(ctx);

    return result;
}

// Benchmark NPU vs Vulkan decision making
void benchmark_decision_making(const benchmark_config& config) {
    printf("\n=== NPU Offload Decision Benchmark ===\n");
    printf("Testing which operations should use NPU vs Vulkan\n\n");

    std::vector<int> test_seq_lens = {32, 64, 128, 256, 512, 1024};
    std::vector<int> test_head_counts = {8, 16, 32, 64};

    printf("Seq_Len | Heads | Head_Dim | Should_Use_NPU | Compute_Intensity\n");
    printf("--------|-------|----------|----------------|------------------\n");

    for (int seq_len : test_seq_lens) {
        for (int num_heads : test_head_counts) {
            int should_offload = npu_should_offload_attention(
                seq_len, num_heads, config.head_dim
            );

            // Calculate compute intensity
            int64_t flops = 2LL * seq_len * seq_len * config.head_dim * num_heads;
            int64_t bytes = (int64_t)seq_len * config.head_dim * sizeof(float);
            float compute_intensity = (float)flops / bytes;

            printf("%7d | %5d | %8d | %14s | %17.1f\n",
                   seq_len, num_heads, config.head_dim,
                   should_offload ? "Yes" : "No",
                   compute_intensity);
        }
    }
}

// Print benchmark results
void print_results(const std::vector<benchmark_result>& results) {
    printf("\n=== Benchmark Results ===\n");
    printf("Batch | Heads | Seq_Len | Mean_Time(ms) | Std(ms) | Min(ms) | Max(ms) | TFLOPS | GB/s\n");
    printf("------|-------|---------|---------------|---------|---------|---------|--------|------\n");

    for (const auto& r : results) {
        printf("%5d | %5d | %7d | %13.3f | %7.3f | %7.3f | %7.3f | %6.2f | %5.1f\n",
               r.batch_size, r.num_heads, r.seq_len,
               r.mean_time_ms, r.std_time_ms, r.min_time_ms, r.max_time_ms,
               r.tflops, r.gb_per_sec);
    }
}

// Compare with theoretical performance
void analyze_performance() {
    printf("\n=== Performance Analysis ===\n");

    const npu_device_info_t* info = npu_backend_get_info();
    if (!info) {
        printf("NPU not available\n");
        return;
    }

    printf("NPU Device: %s\n", info->name);
    printf("Theoretical INT8 Performance: %d TOPS\n", info->tops_int8);
    printf("Number of Tiles: %d\n", info->num_tiles);

    // Get runtime statistics
    uint64_t kernel_time_us, transfer_time_us, total_ops;
    npu_backend_get_stats(&kernel_time_us, &transfer_time_us, &total_ops);

    if (total_ops > 0) {
        double kernel_time_s = (double)kernel_time_us / 1e6;
        double transfer_time_s = (double)transfer_time_us / 1e6;
        double achieved_tops = ((double)total_ops / 1e12) / kernel_time_s;

        printf("\nRuntime Statistics:\n");
        printf("  Total Operations: %.2f GOP\n", (double)total_ops / 1e9);
        printf("  Kernel Time: %.3f s\n", kernel_time_s);
        printf("  Transfer Time: %.3f s\n", transfer_time_s);
        printf("  Achieved Performance: %.2f TOPS\n", achieved_tops);
        printf("  Efficiency: %.1f%%\n", (achieved_tops / info->tops_int8) * 100.0);
    }
}

// Test NPU-Vulkan bridge
void test_bridge() {
    printf("\n=== Testing NPU-Vulkan Bridge ===\n");

    // Initialize bridge
    if (npu_vulkan_bridge_init(1, 1, 1) != 0) {
        printf("Failed to initialize bridge\n");
        return;
    }

    // Test attention submission
    int batch = 1, seq_len = 128, hidden_dim = 768;

    // GGML context for tensor allocation
    struct ggml_init_params params = { 16 * 1024 * 1024, NULL, false };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        printf("Failed to initialize GGML context for bridge test\n");
        return;
    }

    // Create ggml_tensors for Q, K, V, and Output
    const int num_heads = 12; // Example, adjust as needed
    const int head_dim = hidden_dim / num_heads;
    const int num_kv_heads = num_heads; // Simplified for benchmark

    struct ggml_tensor* q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, seq_len, num_heads);
    struct ggml_tensor* k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, seq_len, num_kv_heads);
    struct ggml_tensor* v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, seq_len, num_kv_heads);
    struct ggml_tensor* output = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, seq_len, num_heads);

    if (!q || !k || !v || !output) {
        printf("Failed to create ggml_tensors for bridge test\n");
        ggml_free(ctx);
        return;
    }

    // Fill with random data
    init_random_tensor((float*)q->data, ggml_nelements(q));
    init_random_tensor((float*)k->data, ggml_nelements(k));
    init_random_tensor((float*)v->data, ggml_nelements(v));

    int backend_used = npu_vulkan_bridge_submit_attention(
        q, k, v, output
    );

    printf("Attention operation submitted to: %s\n",
           backend_used > 0 ? "NPU" : "Vulkan");

    // Get statistics
    uint64_t npu_ops, vulkan_ops, npu_time, vulkan_time;
    npu_vulkan_bridge_get_stats(&npu_ops, &vulkan_ops, &npu_time, &vulkan_time);

    printf("Bridge Statistics:\n");
    printf("  NPU operations: %lu\n", npu_ops);
    printf("  Vulkan operations: %lu\n", vulkan_ops);
    printf("  NPU time: %.3f ms\n", (double)npu_time / 1000.0);
    printf("  Vulkan time: %.3f ms\n", (double)vulkan_time / 1000.0);

    ggml_free(ctx);
    npu_vulkan_bridge_cleanup();
}

int main(int argc, char** argv) {
    printf("🦄 NPU Backend Benchmark Tool\n");
    printf("============================= \n");

    // Initialize NPU
    if (npu_backend_init() != 0) {
        printf("Failed to initialize NPU backend\n");
        return 1;
    }

    // Check availability
    if (!npu_backend_available()) {
        printf("NPU is not available\n");
        npu_backend_cleanup();
        return 1;
    }

    // Configure benchmark
    benchmark_config config = {
        .warmup_runs = 5,
        .benchmark_runs = 20,
        .sequence_lengths = {64, 128, 256, 512},
        .batch_sizes = {1},
        .head_counts = {8, 16, 32},
        .head_dim = 64,
        .verbose = true
    };

    // Run benchmarks
    std::vector<benchmark_result> results;

    for (int batch : config.batch_sizes) {
        for (int heads : config.head_counts) {
            for (int seq_len : config.sequence_lengths) {
                printf("\nBenchmarking: batch=%d, heads=%d, seq_len=%d\n",
                       batch, heads, seq_len);

                auto result = benchmark_attention(config, seq_len, batch, heads);
                results.push_back(result);
            }
        }
    }

    // Print results
    print_results(results);

    // Additional tests
    benchmark_decision_making(config);
    analyze_performance();
    test_bridge();

    // Cleanup
    npu_backend_cleanup();

    printf("\n✅ Benchmark completed!\n");
    return 0;
}