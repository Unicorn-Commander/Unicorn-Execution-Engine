/*
 * NPU-Vulkan Bridge for llama.cpp
 * Coordinates workload distribution between Vulkan GPU and NPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

// Use the full, real GGML API
#include "ggml.h"

// Include NPU backend header
extern "C" {
#include "npu_backend.h"
}

// Bridge configuration
struct bridge_config {
    bool enable_npu;
    bool enable_vulkan;
    int npu_min_seq_len;
    int npu_max_seq_len;
    float npu_efficiency_threshold;
    bool verbose;
};

// Operation types
enum op_type {
    OP_LINEAR,      // Matrix multiplication
    OP_ATTENTION,   // Self/cross attention
    OP_LAYERNORM,   // Layer normalization
    OP_ACTIVATION,  // GELU, ReLU, etc.
    OP_EMBEDDING    // Token/position embeddings
};

// Work item for scheduling, now with tensor pointers
struct work_item {
    op_type type;
    const ggml_tensor* q_tensor;
    const ggml_tensor* k_tensor;
    const ggml_tensor* v_tensor;
    ggml_tensor* output_tensor;
    bool is_complete;
    std::atomic<bool> npu_eligible;
};

// Bridge context
class NPUVulkanBridge {
private:
    bridge_config config;
    npu_context_t* npu_ctx;

    // Work queues
    std::queue<work_item*> npu_queue;
    std::queue<work_item*> vulkan_queue;

    // Synchronization
    std::mutex npu_mutex;
    std::mutex vulkan_mutex;
    std::condition_variable npu_cv;
    std::condition_variable vulkan_cv;

    // Performance tracking
    std::atomic<uint64_t> npu_ops_completed;
    std::atomic<uint64_t> vulkan_ops_completed;
    std::atomic<uint64_t> npu_time_us;
    std::atomic<uint64_t> vulkan_time_us;

    // Worker threads
    std::thread npu_worker;
    std::thread vulkan_worker;
    std::atomic<bool> running;

public:
    NPUVulkanBridge(const bridge_config& cfg) : config(cfg), running(true) {
        npu_ops_completed = 0;
        vulkan_ops_completed = 0;
        npu_time_us = 0;
        vulkan_time_us = 0;

        // Initialize NPU if enabled
        if (config.enable_npu) {
            if (npu_backend_init() == 0) {
                printf("[Bridge] NPU backend initialized\n");
            } else {
                printf("[Bridge] NPU initialization failed, disabling NPU\n");
                config.enable_npu = false;
            }
        }

        // Start worker threads
        if (config.enable_npu) {
            npu_worker = std::thread(&NPUVulkanBridge::npu_worker_thread, this);
        }
        if (config.enable_vulkan) {
            vulkan_worker = std::thread(&NPUVulkanBridge::vulkan_worker_thread, this);
        }
    }

    ~NPUVulkanBridge() {
        // Stop workers
        running = false;

        // Wake up threads
        npu_cv.notify_all();
        vulkan_cv.notify_all();

        // Join threads
        if (npu_worker.joinable()) npu_worker.join();
        if (vulkan_worker.joinable()) vulkan_worker.join();

        // Cleanup NPU
        if (config.enable_npu) {
            npu_backend_cleanup();
        }
    }

    // Decide which backend to use for an operation
    bool should_use_npu(const work_item* item) {
        if (!config.enable_npu) return false;

        if (item->type == OP_ATTENTION) {
            int seq_len = item->q_tensor->ne[2];
            if (seq_len >= config.npu_min_seq_len && seq_len <= config.npu_max_seq_len) {
                return true;
            }
        }

        return false;
    }

    // Submit work item
    void submit_work(work_item* item) {
        bool use_npu = should_use_npu(item);
        item->npu_eligible = use_npu;

        if (use_npu) {
            std::lock_guard<std::mutex> lock(npu_mutex);
            npu_queue.push(item);
            npu_cv.notify_one();

            if (config.verbose) {
                printf("[Bridge] Submitted ATTENTION to NPU (seq_len=%lld)\n", item->q_tensor->ne[2]);
            }
        } else {
            std::lock_guard<std::mutex> lock(vulkan_mutex);
            vulkan_queue.push(item);
            vulkan_cv.notify_one();

            if (config.verbose) {
                printf("[Bridge] Submitted %s to Vulkan\n", op_type_str(item->type));
            }
        }
    }

    // Wait for work completion
    void wait_for_completion(work_item* item) {
        while (!item->is_complete) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    // Get performance statistics
    void get_stats(uint64_t* npu_ops, uint64_t* vulkan_ops,
                   uint64_t* npu_us, uint64_t* vulkan_us) {
        *npu_ops = npu_ops_completed.load();
        *vulkan_ops = vulkan_ops_completed.load();
        *npu_us = npu_time_us.load();
        *vulkan_us = vulkan_time_us.load();
    }

private:
    // NPU worker thread
    void npu_worker_thread() {
        printf("[Bridge] NPU worker started\n");

        while (running) {
            work_item* item = nullptr;

            {
                std::unique_lock<std::mutex> lock(npu_mutex);
                npu_cv.wait(lock, [this] { return !npu_queue.empty() || !running; });

                if (!running) break;
                if (!npu_queue.empty()) {
                    item = npu_queue.front();
                    npu_queue.pop();
                }
            }

            if (item) {
                auto start = std::chrono::high_resolution_clock::now();
                process_on_npu(item);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

                npu_time_us += duration.count();
                npu_ops_completed++;
                item->is_complete = true;
            }
        }

        printf("[Bridge] NPU worker stopped\n");
    }

    // Vulkan worker thread (simulated)
    void vulkan_worker_thread() {
        printf("[Bridge] Vulkan worker started\n");

        while (running) {
            work_item* item = nullptr;

            {
                std::unique_lock<std::mutex> lock(vulkan_mutex);
                vulkan_cv.wait(lock, [this] { return !vulkan_queue.empty() || !running; });

                if (!running) break;
                if (!vulkan_queue.empty()) {
                    item = vulkan_queue.front();
                    vulkan_queue.pop();
                }
            }

            if (item) {
                auto start = std::chrono::high_resolution_clock::now();
                process_on_vulkan(item);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

                vulkan_time_us += duration.count();
                vulkan_ops_completed++;
                item->is_complete = true;
            }
        }

        printf("[Bridge] Vulkan worker stopped\n");
    }

    // Process attention on NPU
    void process_on_npu(work_item* item) {
        if (item->type == OP_ATTENTION) {
            const ggml_tensor* q = item->q_tensor;
            const ggml_tensor* k = item->k_tensor;
            const ggml_tensor* v = item->v_tensor;
            ggml_tensor* output = item->output_tensor;

            // Extract dimensions for the call
            int batch_size = q->ne[3];
            int num_heads = q->ne[2];
            int seq_len = q->ne[1];
            int head_dim = q->ne[0];

            // Call NPU attention with full tensor objects
            npu_attention_forward_int8(
                q, k, v, output,
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                1  // is_causal = true
            );
        }
    }

    // Process on Vulkan (simulated for now)
    void process_on_vulkan(work_item* item) {
        // In a real implementation, this would dispatch to a Vulkan compute shader.
        // For now, we simulate work by sleeping.
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    // Helper to convert op type to string
    const char* op_type_str(op_type type) {
        switch (type) {
            case OP_LINEAR: return "LINEAR";
            case OP_ATTENTION: return "ATTENTION";
            case OP_LAYERNORM: return "LAYERNORM";
            case OP_ACTIVATION: return "ACTIVATION";
            case OP_EMBEDDING: return "EMBEDDING";
            default: return "UNKNOWN";
        }
    }
};

// C interface for llama.cpp integration
extern "C" {

static NPUVulkanBridge* g_bridge = nullptr;

int npu_vulkan_bridge_init(int enable_npu, int enable_vulkan, int verbose) {
    if (g_bridge) {
        return 0;  // Already initialized
    }

    bridge_config config = {
        .enable_npu = enable_npu != 0,
        .enable_vulkan = enable_vulkan != 0,
        .npu_min_seq_len = 64,
        .npu_max_seq_len = 4096,
        .npu_efficiency_threshold = 0.7f,
        .verbose = verbose != 0
    };

    g_bridge = new NPUVulkanBridge(config);

    printf("[Bridge] NPU-Vulkan bridge initialized\n");
    printf("[Bridge]   NPU: %s\n", config.enable_npu ? "enabled" : "disabled");
    printf("[Bridge]   Vulkan: %s\n", config.enable_vulkan ? "enabled" : "disabled");

    return 0;
}

int npu_vulkan_bridge_submit_attention(
    const struct ggml_tensor* q,
    const struct ggml_tensor* k,
    const struct ggml_tensor* v,
    struct ggml_tensor* output
) {
    if (!g_bridge) return -1;

    work_item* item = new work_item{
        .type = OP_ATTENTION,
        .q_tensor = q,
        .k_tensor = k,
        .v_tensor = v,
        .output_tensor = output,
        .is_complete = false
    };

    g_bridge->submit_work(item);
    g_bridge->wait_for_completion(item);

    bool used_npu = item->npu_eligible.load();
    delete item;

    return used_npu ? 1 : 0;  // Return which backend was used
}

// This function is now a placeholder as linear ops are not offloaded in this simplified bridge
int npu_vulkan_bridge_submit_linear(
    float* input,
    float* output,
    int batch_size,
    int in_dim,
    int out_dim
) {
    if (!g_bridge) return -1;
    // Linear ops are currently handled by the main GGML backend (e.g., Vulkan)
    return 0;
}

void npu_vulkan_bridge_get_stats(
    uint64_t* npu_ops,
    uint64_t* vulkan_ops,
    uint64_t* npu_time_us,
    uint64_t* vulkan_time_us
) {
    if (!g_bridge) {
        *npu_ops = 0;
        *vulkan_ops = 0;
        *npu_time_us = 0;
        *vulkan_time_us = 0;
        return;
    }

    g_bridge->get_stats(npu_ops, vulkan_ops, npu_time_us, vulkan_time_us);
}

void npu_vulkan_bridge_cleanup(void) {
    if (g_bridge) {
        delete g_bridge;
        g_bridge = nullptr;
        printf("[Bridge] NPU-Vulkan bridge cleaned up\n");
    }
}

} // extern "C"
