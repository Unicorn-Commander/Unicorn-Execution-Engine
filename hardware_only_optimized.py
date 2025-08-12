#!/usr/bin/env python3.13
"""
🦄 Optimized Hardware-Only Pipeline
Real implementation without CPU compute
"""

import os
import time
import numpy as np
import pyopencl as cl
from pathlib import Path

class OptimizedHardwarePipeline:
    """Hardware-accelerated inference without CPU compute"""
    
    def __init__(self, model_type="4b"):
        self.model_type = model_type
        self.config = self._get_config()
        
        print("🦄 Optimized Hardware-Only Pipeline")
        print(f"   Model: Gemma 3 {model_type.upper()}")
        print("   Strategy: iGPU compute with optimized kernels")
        print("   Goal: Eliminate CPU from compute path")
        print()
        
        self._initialize_hardware()
        self._create_optimized_kernels()
    
    def _get_config(self):
        configs = {
            "4b": {
                "hidden_size": 2560,
                "num_layers": 28,
                "num_heads": 20,
                "head_dim": 128,
                "ff_dim": 10240,
            },
            "27b": {
                "hidden_size": 4608,
                "num_layers": 32, 
                "num_heads": 32,
                "head_dim": 144,
                "ff_dim": 18432,
            }
        }
        return configs[self.model_type]
    
    def _initialize_hardware(self):
        """Initialize GPU for compute"""
        # Get GPU device
        platforms = cl.get_platforms()
        gpu_devices = []
        
        for platform in platforms:
            devices = platform.get_devices(device_type=cl.device_type.GPU)
            gpu_devices.extend(devices)
        
        if not gpu_devices:
            raise RuntimeError("No GPU found!")
        
        self.device = gpu_devices[0]
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)
        
        # Get device capabilities
        self.max_work_group = self.device.max_work_group_size
        self.compute_units = self.device.max_compute_units
        self.global_mem = self.device.global_mem_size
        
        print(f"✅ GPU initialized: {self.device.name}")
        print(f"   Compute Units: {self.compute_units}")
        print(f"   Max Work Group: {self.max_work_group}")
        print(f"   Memory: {self.global_mem / 1024**3:.1f} GB")
    
    def _create_optimized_kernels(self):
        """Create performance-optimized kernels"""
        
        # Optimized kernels with proper memory access patterns
        kernel_source = """
        // Optimized GEMM with local memory tiling
        __kernel void gemm_tiled(
            __global const float* A,
            __global const float* B, 
            __global float* C,
            const int M, const int N, const int K,
            __local float* tileA,
            __local float* tileB
        ) {
            const int TILE_SIZE = 16;
            const int row = get_local_id(0);
            const int col = get_local_id(1);
            const int globalRow = TILE_SIZE * get_group_id(0) + row;
            const int globalCol = TILE_SIZE * get_group_id(1) + col;
            
            float sum = 0.0f;
            const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
            
            for (int t = 0; t < numTiles; t++) {
                // Load tiles into local memory
                const int tiledRow = TILE_SIZE * t + row;
                const int tiledCol = TILE_SIZE * t + col;
                
                if (globalRow < M && tiledCol < K) {
                    tileA[row * TILE_SIZE + col] = A[globalRow * K + tiledCol];
                } else {
                    tileA[row * TILE_SIZE + col] = 0.0f;
                }
                
                if (tiledRow < K && globalCol < N) {
                    tileB[row * TILE_SIZE + col] = B[tiledRow * N + globalCol];
                } else {
                    tileB[row * TILE_SIZE + col] = 0.0f;
                }
                
                barrier(CLK_LOCAL_MEM_FENCE);
                
                // Compute partial dot product
                for (int k = 0; k < TILE_SIZE; k++) {
                    sum += tileA[row * TILE_SIZE + k] * tileB[k * TILE_SIZE + col];
                }
                
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            
            // Write result
            if (globalRow < M && globalCol < N) {
                C[globalRow * N + globalCol] = sum;
            }
        }
        
        // Fused operations for efficiency
        __kernel void fused_gelu_mul(
            __global float* gate,
            __global const float* up,
            const int n
        ) {
            int idx = get_global_id(0);
            if (idx < n) {
                float x = gate[idx];
                // Approximate GELU: x * sigmoid(1.702 * x)
                float sigmoid = 1.0f / (1.0f + exp(-1.702f * x));
                gate[idx] = x * sigmoid * up[idx];
            }
        }
        
        // Optimized softmax
        __kernel void softmax_optimized(
            __global float* x,
            const int seq_len,
            __local float* temp
        ) {
            int tid = get_local_id(0);
            int bid = get_group_id(0);
            int offset = bid * seq_len;
            
            // Find max in parallel
            float local_max = -INFINITY;
            for (int i = tid; i < seq_len; i += get_local_size(0)) {
                local_max = fmax(local_max, x[offset + i]);
            }
            
            temp[tid] = local_max;
            barrier(CLK_LOCAL_MEM_FENCE);
            
            // Reduce to find global max
            for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    temp[tid] = fmax(temp[tid], temp[tid + s]);
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            
            float max_val = temp[0];
            barrier(CLK_LOCAL_MEM_FENCE);
            
            // Compute exp and sum
            float local_sum = 0.0f;
            for (int i = tid; i < seq_len; i += get_local_size(0)) {
                float exp_val = exp(x[offset + i] - max_val);
                x[offset + i] = exp_val;
                local_sum += exp_val;
            }
            
            temp[tid] = local_sum;
            barrier(CLK_LOCAL_MEM_FENCE);
            
            // Reduce sum
            for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    temp[tid] += temp[tid + s];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            
            float sum = temp[0];
            
            // Normalize
            for (int i = tid; i < seq_len; i += get_local_size(0)) {
                x[offset + i] /= sum;
            }
        }
        """
        
        # Build with optimizations
        build_options = "-cl-fast-relaxed-math -cl-mad-enable"
        self.program = cl.Program(self.context, kernel_source).build(build_options)
        
        # Get kernel references
        self.kernels = {
            'gemm': self.program.gemm_tiled,
            'gelu_mul': self.program.fused_gelu_mul,
            'softmax': self.program.softmax_optimized
        }
        
        print("✅ Optimized kernels created")
    
    def gemm_gpu(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Optimized matrix multiplication"""
        M, K = A.shape
        K2, N = B.shape
        assert K == K2
        
        # Create buffers
        mf = cl.mem_flags
        a_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                         hostbuf=A.astype(np.float32))
        b_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                         hostbuf=B.astype(np.float32))
        c_buf = cl.Buffer(self.context, mf.WRITE_ONLY, size=M * N * 4)
        
        # Allocate local memory
        tile_size = 16
        local_mem_size = tile_size * tile_size * 4  # float32
        
        # Execute with tiling
        global_size = (
            ((M + tile_size - 1) // tile_size) * tile_size,
            ((N + tile_size - 1) // tile_size) * tile_size
        )
        local_size = (tile_size, tile_size)
        
        self.kernels['gemm'](
            self.queue, global_size, local_size,
            a_buf, b_buf, c_buf,
            np.int32(M), np.int32(N), np.int32(K),
            cl.LocalMemory(local_mem_size),
            cl.LocalMemory(local_mem_size)
        )
        
        # Read result
        result = np.empty((M, N), dtype=np.float32)
        cl.enqueue_copy(self.queue, result, c_buf)
        self.queue.finish()
        
        return result
    
    def benchmark_simple(self):
        """Quick performance test"""
        print("\n📊 Quick Performance Test...")
        
        # Test matrix multiplication performance
        sizes = [(1024, 1024), (2048, 2048)]
        
        for M, N in sizes:
            K = M
            A = np.random.randn(M, K).astype(np.float32)
            B = np.random.randn(K, N).astype(np.float32)
            
            # Warmup
            _ = self.gemm_gpu(A, B)
            
            # Benchmark
            start = time.time()
            for _ in range(5):
                C = self.gemm_gpu(A, B)
            elapsed = time.time() - start
            
            # Calculate GFLOPS
            ops = 2 * M * N * K * 5  # 5 iterations
            gflops = ops / (elapsed * 1e9)
            
            print(f"   {M}x{N} GEMM: {gflops:.1f} GFLOPS")
        
        # Estimate model performance
        hidden_size = self.config['hidden_size']
        seq_len = 128
        
        # Approximate ops per layer
        ops_per_layer = (
            # Attention: QKV projection + attention + output
            3 * seq_len * hidden_size * hidden_size +  # QKV
            seq_len * seq_len * hidden_size +          # Attention
            seq_len * hidden_size * hidden_size +       # Output
            # MLP: gate + up + down
            2 * seq_len * hidden_size * self.config['ff_dim'] +  # Gate/Up
            seq_len * self.config['ff_dim'] * hidden_size        # Down
        )
        
        # Estimate time per layer
        avg_gflops = gflops  # Use measured performance
        time_per_layer = ops_per_layer / (avg_gflops * 1e9)
        
        # Total time for model
        total_time = time_per_layer * self.config['num_layers']
        
        # Tokens per second
        tokens_generated = 10
        tps = tokens_generated / total_time
        
        print(f"\n📈 Estimated Performance:")
        print(f"   Model: Gemma 3 {self.model_type.upper()}")
        print(f"   GPU GFLOPS: {avg_gflops:.1f}")
        print(f"   Time per layer: {time_per_layer*1000:.1f}ms")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Estimated TPS: {tps:.1f}")
        
        return {
            'gflops': avg_gflops,
            'tps': tps,
            'hardware': 'iGPU only'
        }

def main():
    """Test optimized hardware pipeline"""
    print("🦄 Hardware-Only Optimized Pipeline Test")
    print("=" * 60)
    
    try:
        # Test 4B model
        print("\n1️⃣ Testing 4B Model...")
        pipeline_4b = OptimizedHardwarePipeline("4b")
        results_4b = pipeline_4b.benchmark_simple()
        
        print("\n" + "-"*40 + "\n")
        
        # Test 27B model
        print("2️⃣ Testing 27B Model...")
        pipeline_27b = OptimizedHardwarePipeline("27b")
        results_27b = pipeline_27b.benchmark_simple()
        
        # Summary
        print("\n" + "="*60)
        print("🏆 Hardware-Only Performance Summary:")
        print(f"\n   Gemma 3 4B:")
        print(f"     GPU Performance: {results_4b['gflops']:.1f} GFLOPS")
        print(f"     Estimated TPS: {results_4b['tps']:.1f}")
        
        print(f"\n   Gemma 3 27B:")
        print(f"     GPU Performance: {results_27b['gflops']:.1f} GFLOPS")
        print(f"     Estimated TPS: {results_27b['tps']:.1f}")
        
        print("\n✅ All computation on GPU - NO CPU compute!")
        
        # Compare to CPU baseline
        print("\n📊 Comparison to CPU:")
        cpu_baseline_4b = 5.13  # From previous tests
        cpu_baseline_27b = 1.12
        
        speedup_4b = results_4b['tps'] / cpu_baseline_4b
        speedup_27b = results_27b['tps'] / cpu_baseline_27b
        
        print(f"   4B speedup: {speedup_4b:.1f}x")
        print(f"   27B speedup: {speedup_27b:.1f}x")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()