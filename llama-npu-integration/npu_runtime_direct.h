/*
 * Direct NPU Runtime Header
 * Based on transcription project's proven NPU access
 */

#pragma once

#include <memory>
#include <cstdint>

namespace npu_direct {

/**
 * DirectNPURuntime - Real hardware NPU access
 * 
 * Features from transcription project:
 * - Direct IOCTL interface to /dev/accel/accel0
 * - No XRT dependencies
 * - Real NPU buffer management
 * - Hardware context creation
 * - Performance tracking
 */
class DirectNPURuntime {
public:
    DirectNPURuntime();
    ~DirectNPURuntime();
    
    // Initialization
    bool initialize();
    bool is_available();
    
    // Attention execution
    bool execute_attention(
        const float* q_data, const float* k_data, const float* v_data,
        float* output, int seq_len, int num_heads, int head_dim);
    
    // Performance monitoring
    void print_performance_stats();
    
    // Non-copyable
    DirectNPURuntime(const DirectNPURuntime&) = delete;
    DirectNPURuntime& operator=(const DirectNPURuntime&) = delete;
    
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace npu_direct