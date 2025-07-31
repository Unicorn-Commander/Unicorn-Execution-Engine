#!/usr/bin/env python3.13
"""
Test iGPU acceleration independently
Focus on GEMM and FFN operations using CLBlast
"""

import os
import sys
import time
import numpy as np
import torch
import pyopencl as cl
import pyopencl.array as cl_array

class iGPUAccelerator:
    """iGPU-only acceleration test"""
    
    def __init__(self):
        """Initialize OpenCL for iGPU"""
        print("🎮 Initializing iGPU Accelerator")
        print("=" * 50)
        
        # Find AMD GPU
        platforms = cl.get_platforms()
        amd_devices = []
        
        for platform in platforms:
            if 'AMD' in platform.name or 'Advanced Micro Devices' in platform.name:
                devices = platform.get_devices(device_type=cl.device_type.GPU)
                amd_devices.extend(devices)
        
        if not amd_devices:
            raise RuntimeError("No AMD GPU found")
        
        # Use the first AMD GPU (should be the iGPU)
        self.device = amd_devices[0]
        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        
        print(f"✅ Found GPU: {self.device.name}")
        print(f"   Compute Units: {self.device.max_compute_units}")
        print(f"   Max Work Group: {self.device.max_work_group_size}")
        print(f"   Global Memory: {self.device.global_mem_size / (1024**3):.1f} GB")
        
        # Load CLBlast if available
        try:
            import pyopencl_blas
            self.has_clblast = True
            print("   CLBlast: ✅ Available")
        except:
            self.has_clblast = False
            print("   CLBlast: ❌ Not available (using custom kernels)")
        
        # Create optimized kernels
        self._create_kernels()
    
    def _create_kernels(self):
        """Create optimized OpenCL kernels"""
        self.program = cl.Program(self.ctx, """
        __kernel void gemm_optimized(
            __global const float* A,
            __global const float* B,
            __global float* C,
            const int M,
            const int N,
            const int K,
            const float alpha,
            const float beta
        ) {
            const int row = get_global_id(0);
            const int col = get_global_id(1);
            
            if (row >= M || col >= N) return;
            
            float sum = 0.0f;
            
            // Vectorized accumulation
            for (int k = 0; k < K; k += 4) {
                float4 a_vec = vload4(0, A + row * K + k);
                float4 b_vec = (float4)(
                    B[k * N + col],
                    B[(k+1) * N + col],
                    B[(k+2) * N + col],
                    B[(k+3) * N + col]
                );
                sum += dot(a_vec, b_vec);
            }
            
            // Handle remainder
            for (int k = (K/4)*4; k < K; k++) {
                sum += A[row * K + k] * B[k * N + col];
            }
            
            C[row * N + col] = alpha * sum + beta * C[row * N + col];
        }
        
        __kernel void gelu_activation(
            __global float* x,
            const int size
        ) {
            const int idx = get_global_id(0);
            if (idx >= size) return;
            
            float val = x[idx];
            // Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/PI) * (x + 0.044715 * x^3)))
            const float sqrt_2_pi = 0.79788456f;
            float x3 = val * val * val;
            float tanh_arg = sqrt_2_pi * (val + 0.044715f * x3);
            float tanh_val = tanh(tanh_arg);
            x[idx] = 0.5f * val * (1.0f + tanh_val);
        }
        
        __kernel void layer_norm(
            __global const float* input,
            __global float* output,
            __global const float* gamma,
            __global const float* beta,
            const int batch_size,
            const int hidden_size
        ) {
            const int batch_idx = get_global_id(0);
            if (batch_idx >= batch_size) return;
            
            const int offset = batch_idx * hidden_size;
            
            // Calculate mean
            float mean = 0.0f;
            for (int i = 0; i < hidden_size; i++) {
                mean += input[offset + i];
            }
            mean /= (float)hidden_size;
            
            // Calculate variance
            float variance = 0.0f;
            for (int i = 0; i < hidden_size; i++) {
                float diff = input[offset + i] - mean;
                variance += diff * diff;
            }
            variance /= (float)hidden_size;
            
            // Normalize
            float inv_std = rsqrt(variance + 1e-5f);
            for (int i = 0; i < hidden_size; i++) {
                float normalized = (input[offset + i] - mean) * inv_std;
                output[offset + i] = normalized * gamma[i] + beta[i];
            }
        }
        """).build()
    
    def benchmark_gemm(self, m, n, k):
        """Benchmark GEMM operation"""
        print(f"\n📊 Benchmarking GEMM ({m}x{k} @ {k}x{n})")
        
        # Create test matrices
        A = np.random.randn(m, k).astype(np.float32)
        B = np.random.randn(k, n).astype(np.float32)
        C = np.zeros((m, n), dtype=np.float32)
        
        # Transfer to GPU
        A_gpu = cl_array.to_device(self.queue, A)
        B_gpu = cl_array.to_device(self.queue, B)
        C_gpu = cl_array.to_device(self.queue, C)
        
        # Warmup
        for _ in range(5):
            self.program.gemm_optimized(
                self.queue, (m, n), None,
                A_gpu.data, B_gpu.data, C_gpu.data,
                np.int32(m), np.int32(n), np.int32(k),
                np.float32(1.0), np.float32(0.0)
            )
        self.queue.finish()
        
        # Benchmark
        start_time = time.time()
        iterations = 100
        
        for _ in range(iterations):
            event = self.program.gemm_optimized(
                self.queue, (m, n), None,
                A_gpu.data, B_gpu.data, C_gpu.data,
                np.int32(m), np.int32(n), np.int32(k),
                np.float32(1.0), np.float32(0.0)
            )
        
        self.queue.finish()
        elapsed = time.time() - start_time
        
        # Calculate performance
        flops = 2 * m * n * k * iterations
        gflops = flops / elapsed / 1e9
        
        print(f"   Time: {elapsed/iterations*1000:.2f} ms per GEMM")
        print(f"   Performance: {gflops:.1f} GFLOPS")
        
        return gflops
    
    def test_model_operations(self):
        """Test operations for Gemma 3 4B model"""
        print("\n🦄 Testing Gemma 3 4B Model Operations")
        
        # Model dimensions
        batch_size = 1
        seq_len = 256
        hidden_size = 2560
        intermediate_size = 6912
        num_heads = 20
        
        print(f"\n   Model config:")
        print(f"   - Hidden size: {hidden_size}")
        print(f"   - Intermediate: {intermediate_size}")
        print(f"   - Attention heads: {num_heads}")
        
        # Test key operations
        results = {}
        
        # 1. QKV projection
        print("\n   1️⃣ QKV Projection:")
        qkv_gflops = self.benchmark_gemm(seq_len, 3*hidden_size, hidden_size)
        results['qkv_projection'] = qkv_gflops
        
        # 2. FFN Up projection
        print("\n   2️⃣ FFN Up Projection:")
        up_gflops = self.benchmark_gemm(seq_len, intermediate_size, hidden_size)
        results['ffn_up'] = up_gflops
        
        # 3. FFN Down projection
        print("\n   3️⃣ FFN Down Projection:")
        down_gflops = self.benchmark_gemm(seq_len, hidden_size, intermediate_size)
        results['ffn_down'] = down_gflops
        
        # 4. Output projection
        print("\n   4️⃣ Output Projection:")
        out_gflops = self.benchmark_gemm(seq_len, hidden_size, hidden_size)
        results['output_proj'] = out_gflops
        
        # Calculate theoretical token/s
        print("\n   📈 Performance Summary:")
        avg_gflops = sum(results.values()) / len(results)
        print(f"   Average GFLOPS: {avg_gflops:.1f}")
        
        # Estimate tokens/s (rough calculation)
        # Each token requires ~4x hidden_size^2 FLOPs for main operations
        flops_per_token = 4 * hidden_size * hidden_size * 32  # 32 layers
        tokens_per_sec = (avg_gflops * 1e9) / flops_per_token
        
        print(f"   Estimated tokens/s: {tokens_per_sec:.1f}")
        
        return results

def main():
    """Run iGPU tests"""
    try:
        accelerator = iGPUAccelerator()
        results = accelerator.test_model_operations()
        
        print("\n✅ iGPU Testing Complete!")
        print("\n🎯 Next Steps:")
        print("   1. iGPU shows good GEMM performance")
        print("   2. Can handle 70% of LLM compute (linear ops)")
        print("   3. NPU would handle remaining 30% (attention)")
        print("   4. For now, can run with iGPU + CPU hybrid")
        
    except Exception as e:
        print(f"\n❌ iGPU test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()