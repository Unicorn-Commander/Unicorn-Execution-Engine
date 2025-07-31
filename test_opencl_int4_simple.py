#!/usr/bin/env python3.13
"""
Simple OpenCL INT4 Test
Verify INT4 quantization provides expected speedup
"""

import torch
import numpy as np
import pyopencl as cl
import time
import os

def test_int4_concept():
    """Test if INT4 operations provide expected speedup"""
    print("🦄 OpenCL INT4 Concept Test")
    print("=" * 60)
    
    # Setup OpenCL
    os.environ['PYOPENCL_CTX'] = '0'
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    
    print(f"Device: {device.name}")
    print(f"Compute units: {device.max_compute_units}")
    
    # Simple INT4 vs FP32 kernel comparison
    kernel_source = """
// FP32 GEMM kernel (baseline)
__kernel void gemm_fp32(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M, const int N, const int K
) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// Simulated INT4 GEMM kernel
__kernel void gemm_int4_simulated(
    __global const uchar* A_packed,   // INT4 packed
    __global const float* B,          
    __global float* C,
    __global const float* scale,
    const int M, const int N, const int K
) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    float s = scale[0];
    
    // Process 8 INT4 values at a time (4 bytes = 8 INT4 values)
    for (int k = 0; k < K; k += 8) {
        // Load 4 bytes = 8 INT4 values
        uchar4 packed = vload4(0, A_packed + (row * K + k) / 2);
        
        // Unpack and compute
        #pragma unroll
        for (int i = 0; i < 8 && k + i < K; i++) {
            int byte_idx = i / 2;
            int nibble = i % 2;
            
            uchar byte_val = (byte_idx == 0) ? packed.x :
                           (byte_idx == 1) ? packed.y :
                           (byte_idx == 2) ? packed.z : packed.w;
            
            int int4_val = (nibble == 0) ? (byte_val & 0xF) - 8 : 
                                          ((byte_val >> 4) & 0xF) - 8;
            
            float a_val = (float)int4_val * s;
            sum += a_val * B[(k + i) * N + col];
        }
    }
    
    C[row * N + col] = sum;
}
"""
    
    program = cl.Program(ctx, kernel_source).build()
    
    # Test sizes
    M, N, K = 2048, 2048, 2048
    
    print(f"\nMatrix size: {M}x{N}x{K}")
    print(f"FP32 size: {M*K*4/(1024**2):.1f} MB")
    print(f"INT4 size: {M*K*0.5/(1024**2):.1f} MB (8x reduction)")
    
    # Create test data
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    
    # Quantize A to INT4
    scale = np.abs(A).max() / 7.0
    A_int4 = np.round(A / scale).clip(-8, 7).astype(np.int8)
    
    # Pack INT4 (2 values per byte)
    A_packed = np.zeros((M, K//2), dtype=np.uint8)
    for i in range(0, K, 2):
        val1 = A_int4[:, i] + 8
        val2 = A_int4[:, i+1] + 8 if i+1 < K else 8
        A_packed[:, i//2] = (val1 & 0xF) | ((val2 & 0xF) << 4)
    
    # Create buffers
    mf = cl.mem_flags
    A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    A_packed_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A_packed)
    B_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    C_buf_fp32 = cl.Buffer(ctx, mf.WRITE_ONLY, size=M*N*4)
    C_buf_int4 = cl.Buffer(ctx, mf.WRITE_ONLY, size=M*N*4)
    scale_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, 
                         hostbuf=np.array([scale], dtype=np.float32))
    
    # Benchmark FP32
    print("\n📊 Benchmarking FP32 GEMM...")
    times_fp32 = []
    for _ in range(5):
        event = program.gemm_fp32(queue, (M, N), None,
                                 A_buf, B_buf, C_buf_fp32,
                                 np.int32(M), np.int32(N), np.int32(K))
        event.wait()
        exec_time = (event.profile.end - event.profile.start) * 1e-9
        times_fp32.append(exec_time)
    
    time_fp32 = min(times_fp32)
    
    # Benchmark INT4
    print("📊 Benchmarking INT4 GEMM...")
    times_int4 = []
    for _ in range(5):
        event = program.gemm_int4_simulated(queue, (M, N), None,
                                          A_packed_buf, B_buf, C_buf_int4, scale_buf,
                                          np.int32(M), np.int32(N), np.int32(K))
        event.wait()
        exec_time = (event.profile.end - event.profile.start) * 1e-9
        times_int4.append(exec_time)
    
    time_int4 = min(times_int4)
    
    # Calculate speedup
    speedup = time_fp32 / time_int4
    
    # Verify correctness
    C_fp32 = np.empty((M, N), dtype=np.float32)
    C_int4 = np.empty((M, N), dtype=np.float32)
    cl.enqueue_copy(queue, C_fp32, C_buf_fp32).wait()
    cl.enqueue_copy(queue, C_int4, C_buf_int4).wait()
    
    # Check accuracy
    rel_error = np.abs(C_fp32 - C_int4).mean() / np.abs(C_fp32).mean()
    
    print("\n📈 Results:")
    print(f"  FP32 time: {time_fp32*1000:.1f}ms")
    print(f"  INT4 time: {time_int4*1000:.1f}ms")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  Relative error: {rel_error:.4f}")
    
    # Calculate effective GFLOPS
    gflops_fp32 = (2 * M * N * K) / (time_fp32 * 1e9)
    gflops_int4 = (2 * M * N * K) / (time_int4 * 1e9)
    
    print(f"\n⚡ Performance:")
    print(f"  FP32: {gflops_fp32:.1f} GFLOPS")
    print(f"  INT4: {gflops_int4:.1f} GFLOPS (effective)")
    
    # Memory bandwidth
    mem_read_fp32 = (M*K + K*N) * 4  # bytes
    mem_read_int4 = (M*K*0.5 + K*N*4)  # INT4 A + FP32 B
    
    bandwidth_fp32 = mem_read_fp32 / (time_fp32 * 1e9)
    bandwidth_int4 = mem_read_int4 / (time_int4 * 1e9)
    
    print(f"\n💾 Memory bandwidth:")
    print(f"  FP32: {bandwidth_fp32:.1f} GB/s")
    print(f"  INT4: {bandwidth_int4:.1f} GB/s")
    
    # Project to transformer layer
    print("\n🦄 Projecting to transformer layer:")
    # Assuming GEMM is 80% of layer time
    layer_time_fp32 = 0.125  # 125ms baseline
    gemm_fraction = 0.8
    
    projected_layer_time = layer_time_fp32 * (1 - gemm_fraction + gemm_fraction / speedup)
    projected_speedup = layer_time_fp32 / projected_layer_time
    projected_tokens_per_sec = 1.0 / (projected_layer_time * 42)
    
    print(f"  Projected layer time: {projected_layer_time*1000:.1f}ms")
    print(f"  Projected speedup: {projected_speedup:.1f}x")
    print(f"  Projected speed: {projected_tokens_per_sec:.2f} tokens/sec")
    
    if projected_tokens_per_sec >= 21.0:
        print(f"  🎯 INT4 alone could achieve target!")
    elif projected_tokens_per_sec >= 10.0:
        print(f"  🔥 Very promising! Small optimizations needed.")
    elif projected_tokens_per_sec >= 5.0:
        print(f"  ⚡ Good progress! Combined with other opts will work.")
    else:
        print(f"  🔧 More optimization needed beyond INT4.")
        
    return speedup, projected_tokens_per_sec


if __name__ == "__main__":
    speedup, tokens_per_sec = test_int4_concept()
    
    print("\n" + "="*60)
    print("🏁 CONCLUSION:")
    if speedup >= 5.0:
        print(f"✅ INT4 provides {speedup:.1f}x speedup - confirms our analysis!")
        print(f"📋 Next steps:")
        print(f"   1. Optimize INT4 kernel further (register blocking, shared memory)")
        print(f"   2. Implement INT4 attention kernel")
        print(f"   3. Add NPU integration for remaining speedup")
    else:
        print(f"⚠️ INT4 speedup {speedup:.1f}x lower than expected")
        print(f"📋 Need to investigate:")
        print(f"   1. Memory bandwidth limitations")
        print(f"   2. Kernel optimization opportunities")
        print(f"   3. Hardware-specific features")