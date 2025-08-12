/*
 * NPU Backend Internal Definitions
 * Shared structures between implementation files
 */

#ifndef NPU_BACKEND_INTERNAL_H
#define NPU_BACKEND_INTERNAL_H

#include "npu_backend.h"

// NPU context structure (internal)
struct npu_context {
    void* device;      // xrt::device or placeholder
    void* kernel;      // xrt::kernel or placeholder
    npu_device_info_t info;
    
    // Pre-allocated buffers for attention
    npu_buffer_t* q_buffer;
    npu_buffer_t* k_buffer;
    npu_buffer_t* v_buffer;
    npu_buffer_t* out_buffer;
    
    // Performance counters
    uint64_t kernel_time_us;
    uint64_t transfer_time_us;
    uint64_t total_ops;
};

#endif // NPU_BACKEND_INTERNAL_H