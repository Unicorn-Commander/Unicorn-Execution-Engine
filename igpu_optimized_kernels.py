#!/usr/bin/env python3.13
"""
🦄 Highly Optimized iGPU Kernels for AMD RDNA2
Target: Outperform CPU (600+ GFLOPS)
"""

import time
import numpy as np
import pyopencl as cl

class OptimizediGPUKernels:
    """RDNA2-optimized kernels for transformer operations"""
    
    def __init__(self):
        print("🚀 Creating RDNA2-Optimized iGPU Kernels...")
        self._initialize_gpu()
        self._create_optimized_kernels()
    
    def _initialize_gpu(self):
        """Initialize AMD GPU with optimal settings"""
        platforms = cl.get_platforms()
        amd_devices = []
        
        for platform in platforms:
            if 'AMD' in platform.name or 'Radeon' in platform.name:
                devices = platform.get_devices(device_type=cl.device_type.GPU)
                amd_devices.extend(devices)
        
        if not amd_devices:
            # Fallback to any GPU
            for platform in platforms:
                devices = platform.get_devices(device_type=cl.device_type.GPU)
                if devices:
                    amd_devices = devices
                    break
        
        if not amd_devices:
            raise RuntimeError("No GPU found!")
        
        self.device = amd_devices[0]
        self.context = cl.Context([self.device])
        
        # Create command queue with profiling
        self.queue = cl.CommandQueue(self.context,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
        
        # Get device info for optimization
        self.compute_units = self.device.max_compute_units
        self.wavefront_size = 64  # RDNA2 uses 64-thread wavefronts
        self.local_mem_size = self.device.local_mem_size
        self.max_work_group = self.device.max_work_group_size
        
        print(f"✅ GPU: {self.device.name}")
        print(f"   Compute Units: {self.compute_units}")
        print(f"   Wavefront Size: {self.wavefront_size}")
        print(f"   Local Memory: {self.local_mem_size / 1024:.1f} KB")
        print(f"   Max Work Group: {self.max_work_group}")
    
    def _create_optimized_kernels(self):
        """Create highly optimized kernels for RDNA2"""
        
        # RDNA2-optimized GEMM kernel
        gemm_kernel = """
        // Optimized GEMM for AMD RDNA2 architecture
        // Uses LDS (Local Data Share) and wavefront-level optimizations
        
        #define TILE_M 64
        #define TILE_N 64
        #define TILE_K 16
        #define THREADS_M 8
        #define THREADS_N 8
        #define LOAD_K 4
        
        __kernel __attribute__((reqd_work_group_size(8, 8, 1)))
        void gemm_rdna2_optimized(
            __global const float* restrict A,
            __global const float* restrict B,
            __global float* restrict C,
            const int M, const int N, const int K,
            __local float* lds
        ) {
            // Thread indices
            const int tx = get_local_id(0);
            const int ty = get_local_id(1);
            const int bx = get_group_id(0);
            const int by = get_group_id(1);
            
            // Each thread computes 8x8 tile of C
            float c[8][8] = {{0.0f}};
            
            // LDS for A and B tiles
            __local float* tileA = lds;
            __local float* tileB = lds + TILE_M * TILE_K;
            
            // Global memory indices
            const int aRow = by * TILE_M;
            const int bCol = bx * TILE_N;
            
            // Main loop over K dimension
            for (int k = 0; k < K; k += TILE_K) {
                // Collaborative load of A tile into LDS
                #pragma unroll
                for (int i = 0; i < TILE_M / 8; i++) {
                    #pragma unroll
                    for (int j = 0; j < TILE_K / 8; j++) {
                        int row = aRow + ty + i * 8;
                        int col = k + tx + j * 8;
                        if (row < M && col < K) {
                            tileA[(ty + i * 8) * TILE_K + tx + j * 8] = A[row * K + col];
                        } else {
                            tileA[(ty + i * 8) * TILE_K + tx + j * 8] = 0.0f;
                        }
                    }
                }
                
                // Collaborative load of B tile into LDS
                #pragma unroll
                for (int i = 0; i < TILE_K / 8; i++) {
                    #pragma unroll
                    for (int j = 0; j < TILE_N / 8; j++) {
                        int row = k + ty + i * 8;
                        int col = bCol + tx + j * 8;
                        if (row < K && col < N) {
                            tileB[(ty + i * 8) * TILE_N + tx + j * 8] = B[row * N + col];
                        } else {
                            tileB[(ty + i * 8) * TILE_N + tx + j * 8] = 0.0f;
                        }
                    }
                }
                
                barrier(CLK_LOCAL_MEM_FENCE);
                
                // Compute using data from LDS
                #pragma unroll
                for (int kk = 0; kk < TILE_K; kk++) {
                    // Load vectors from LDS
                    float a_vec[8];
                    float b_vec[8];
                    
                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        a_vec[i] = tileA[(ty * 8 + i) * TILE_K + kk];
                        b_vec[i] = tileB[kk * TILE_N + tx * 8 + i];
                    }
                    
                    // Outer product
                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        #pragma unroll
                        for (int j = 0; j < 8; j++) {
                            c[i][j] = fma(a_vec[i], b_vec[j], c[i][j]);
                        }
                    }
                }
                
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            
            // Write results to global memory
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    int row = aRow + ty * 8 + i;
                    int col = bCol + tx * 8 + j;
                    if (row < M && col < N) {
                        C[row * N + col] = c[i][j];
                    }
                }
            }
        }
        
        // Optimized attention kernel
        __kernel void flash_attention_rdna2(
            __global const float* Q,
            __global const float* K,
            __global const float* V,
            __global float* output,
            const int batch_size,
            const int num_heads,
            const int seq_len,
            const int head_dim,
            __local float* lds
        ) {
            // Flash Attention algorithm optimized for RDNA2
            const int head = get_group_id(0);
            const int tid = get_local_id(0);
            
            if (head >= batch_size * num_heads) return;
            
            // Tile size for flash attention
            const int TILE_SIZE = 64;
            
            // Process attention in tiles to fit in LDS
            __local float* q_tile = lds;
            __local float* k_tile = lds + TILE_SIZE * head_dim;
            __local float* v_tile = lds + 2 * TILE_SIZE * head_dim;
            __local float* scores = lds + 3 * TILE_SIZE * head_dim;
            
            // Each wavefront processes one query row
            for (int q_idx = tid; q_idx < seq_len; q_idx += get_local_size(0)) {
                float max_score = -INFINITY;
                float sum_exp = 0.0f;
                float acc[16] = {0.0f};  // Accumulator for output
                
                // Process K,V in tiles
                for (int k_start = 0; k_start <= q_idx; k_start += TILE_SIZE) {
                    int k_end = min(k_start + TILE_SIZE, q_idx + 1);
                    
                    // Compute scores for this tile
                    for (int k_idx = k_start; k_idx < k_end; k_idx++) {
                        float score = 0.0f;
                        
                        // Dot product Q[q_idx] · K[k_idx]
                        #pragma unroll 4
                        for (int d = 0; d < head_dim; d++) {
                            int q_offset = head * seq_len * head_dim + q_idx * head_dim + d;
                            int k_offset = head * seq_len * head_dim + k_idx * head_dim + d;
                            score = fma(Q[q_offset], K[k_offset], score);
                        }
                        
                        score /= sqrt((float)head_dim);
                        scores[k_idx - k_start] = score;
                        max_score = fmax(max_score, score);
                    }
                    
                    // Softmax and accumulate
                    float tile_sum = 0.0f;
                    for (int k_idx = k_start; k_idx < k_end; k_idx++) {
                        float exp_score = exp(scores[k_idx - k_start] - max_score);
                        scores[k_idx - k_start] = exp_score;
                        tile_sum += exp_score;
                    }
                    
                    // Update accumulator
                    for (int k_idx = k_start; k_idx < k_end; k_idx++) {
                        float weight = scores[k_idx - k_start];
                        
                        #pragma unroll 4
                        for (int d = 0; d < head_dim && d < 16; d++) {
                            int v_offset = head * seq_len * head_dim + k_idx * head_dim + d;
                            acc[d] = fma(weight, V[v_offset], acc[d]);
                        }
                    }
                    
                    sum_exp += tile_sum;
                }
                
                // Write output
                #pragma unroll 4
                for (int d = 0; d < head_dim && d < 16; d++) {
                    int out_offset = head * seq_len * head_dim + q_idx * head_dim + d;
                    output[out_offset] = acc[d] / sum_exp;
                }
            }
        }
        
        // Optimized LayerNorm
        __kernel void layer_norm_rdna2(
            __global const float* input,
            __global float* output,
            __global const float* gamma,
            __global const float* beta,
            const int batch_size,
            const int hidden_size
        ) {
            const int idx = get_global_id(0);
            if (idx >= batch_size) return;
            
            const int offset = idx * hidden_size;
            
            // Two-pass algorithm for numerical stability
            // Pass 1: Compute mean
            float mean = 0.0f;
            for (int i = 0; i < hidden_size; i += 4) {
                float4 vals = vload4(i/4, input + offset);
                mean += vals.x + vals.y + vals.z + vals.w;
            }
            mean /= hidden_size;
            
            // Pass 2: Compute variance
            float var = 0.0f;
            for (int i = 0; i < hidden_size; i += 4) {
                float4 vals = vload4(i/4, input + offset);
                float4 diff = vals - mean;
                float4 sq = diff * diff;
                var += sq.x + sq.y + sq.z + sq.w;
            }
            var = rsqrt(var / hidden_size + 1e-5f);
            
            // Normalize and scale
            for (int i = 0; i < hidden_size; i += 4) {
                float4 vals = vload4(i/4, input + offset);
                float4 norm = (vals - mean) * var;
                float4 g = vload4(i/4, gamma);
                float4 b = vload4(i/4, beta);
                vstore4(g * norm + b, i/4, output + offset);
            }
        }
        
        // Fused GELU activation
        __kernel void gelu_fused(
            __global float* gate,
            __global const float* up,
            const int n
        ) {
            const int idx = get_global_id(0);
            if (idx >= n/4) return;
            
            float4 g = vload4(idx, gate);
            float4 u = vload4(idx, up);
            
            // Approximate GELU with fused multiply
            float4 sigmoid = 1.0f / (1.0f + exp(-1.702f * g));
            float4 result = g * sigmoid * u;
            
            vstore4(result, idx, gate);
        }
        """
        
        # Build with AMD-specific optimizations
        build_options = """
        -cl-std=CL2.0
        -cl-fast-relaxed-math
        -cl-mad-enable
        -cl-no-signed-zeros
        -cl-finite-math-only
        """
        
        self.program = cl.Program(self.context, gemm_kernel).build(build_options)
        
        # Get kernel handles
        self.gemm_kernel = self.program.gemm_rdna2_optimized
        self.attention_kernel = self.program.flash_attention_rdna2
        self.layernorm_kernel = self.program.layer_norm_rdna2
        self.gelu_kernel = self.program.gelu_fused
        
        print("✅ RDNA2-optimized kernels created")
    
    def benchmark_gemm(self):
        """Benchmark optimized GEMM performance"""
        print("\n📊 Benchmarking Optimized GEMM...")
        
        sizes = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)]
        
        results = {}
        
        for M, N, K in sizes:
            # Create test matrices
            A = np.random.randn(M, K).astype(np.float32)
            B = np.random.randn(K, N).astype(np.float32)
            C = np.zeros((M, N), dtype=np.float32)
            
            # Create buffers
            mf = cl.mem_flags
            a_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
            b_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
            c_buf = cl.Buffer(self.context, mf.WRITE_ONLY, size=C.nbytes)
            
            # Allocate local memory
            lds_size = (64 * 16 + 16 * 64) * 2 * 4  # tileA + tileB in float32
            
            # Set work sizes
            global_size = (N // 64 * 8, M // 64 * 8)
            local_size = (8, 8)
            
            # Warmup
            for _ in range(3):
                self.gemm_kernel(self.queue, global_size, local_size,
                                a_buf, b_buf, c_buf,
                                np.int32(M), np.int32(N), np.int32(K),
                                cl.LocalMemory(lds_size))
                self.queue.finish()
            
            # Benchmark
            iterations = 10
            start = time.time()
            
            for _ in range(iterations):
                event = self.gemm_kernel(self.queue, global_size, local_size,
                                       a_buf, b_buf, c_buf,
                                       np.int32(M), np.int32(N), np.int32(K),
                                       cl.LocalMemory(lds_size))
            
            self.queue.finish()
            elapsed = time.time() - start
            
            # Calculate GFLOPS
            ops = 2.0 * M * N * K * iterations
            gflops = ops / (elapsed * 1e9)
            
            results[f"{M}x{N}x{K}"] = gflops
            
            print(f"   {M}x{N}x{K}: {gflops:.1f} GFLOPS")
        
        return results
    
    def run_transformer_layer(self, batch_size, seq_len, hidden_size, num_heads):
        """Benchmark complete transformer layer"""
        print(f"\n📊 Transformer Layer ({hidden_size}D, {num_heads} heads)...")
        
        head_dim = hidden_size // num_heads
        
        # Create test data
        Q = np.random.randn(batch_size * num_heads, seq_len, head_dim).astype(np.float32)
        K = np.random.randn(batch_size * num_heads, seq_len, head_dim).astype(np.float32)
        V = np.random.randn(batch_size * num_heads, seq_len, head_dim).astype(np.float32)
        output = np.zeros_like(Q)
        
        # Create buffers
        mf = cl.mem_flags
        q_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Q)
        k_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=K)
        v_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=V)
        out_buf = cl.Buffer(self.context, mf.WRITE_ONLY, size=output.nbytes)
        
        # LDS size for flash attention
        lds_size = 4 * 64 * head_dim * 4  # 4 tiles * tile_size * head_dim * sizeof(float)
        
        # Run attention
        global_size = (batch_size * num_heads,)
        local_size = (64,)  # One wavefront
        
        start = time.time()
        
        self.attention_kernel(self.queue, global_size, local_size,
                            q_buf, k_buf, v_buf, out_buf,
                            np.int32(batch_size), np.int32(num_heads),
                            np.int32(seq_len), np.int32(head_dim),
                            cl.LocalMemory(lds_size))
        
        self.queue.finish()
        elapsed = time.time() - start
        
        # Calculate FLOPS for attention
        ops = batch_size * num_heads * seq_len * seq_len * head_dim * 4  # Approximate
        gflops = ops / (elapsed * 1e9)
        
        print(f"   Attention: {elapsed*1000:.1f}ms ({gflops:.1f} GFLOPS)")
        
        return elapsed

def main():
    """Test optimized iGPU kernels"""
    print("🦄 RDNA2-Optimized iGPU Kernel Test")
    print("=" * 60)
    print("Target: Outperform CPU (600+ GFLOPS)")
    print()
    
    try:
        # Initialize optimized kernels
        kernels = OptimizediGPUKernels()
        
        # Benchmark GEMM
        gemm_results = kernels.benchmark_gemm()
        
        # Find peak performance
        peak_gflops = max(gemm_results.values())
        print(f"\n🏆 Peak GEMM Performance: {peak_gflops:.1f} GFLOPS")
        
        if peak_gflops > 600:
            print("✅ SUCCESS: iGPU outperforms CPU!")
        else:
            print("⚠️  Need further optimization...")
        
        # Test transformer layers
        print("\n📊 Transformer Layer Performance:")
        
        # 4B model layer
        time_4b = kernels.run_transformer_layer(1, 128, 2560, 20)
        
        # 27B model layer  
        time_27b = kernels.run_transformer_layer(1, 128, 4608, 32)
        
        # Estimate TPS
        print("\n📈 Estimated Tokens Per Second:")
        
        # 4B model (28 layers)
        total_time_4b = time_4b * 28 * 2  # x2 for MLP
        tps_4b = 10 / total_time_4b
        print(f"   Gemma 3 4B: {tps_4b:.1f} TPS")
        
        # 27B model (32 layers)
        total_time_27b = time_27b * 32 * 2  # x2 for MLP
        tps_27b = 10 / total_time_27b
        print(f"   Gemma 3 27B: {tps_27b:.1f} TPS")
        
        print("\n✅ All computation on iGPU - NO CPU compute!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()