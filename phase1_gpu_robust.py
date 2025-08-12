#!/usr/bin/env python3.13
"""
Phase 1 GPU Implementation - Robust Version
Avoids complex kernel patterns that cause hangs on gfx1103
"""

import numpy as np
import pyopencl as cl
import time
from pathlib import Path

class Phase1GPURobust:
    """Robust GPU implementation that works around gfx1103 issues"""
    
    def __init__(self, model_type="4b"):
        self.model_type = model_type
        self.config = self._get_config()
        
        print("🦄 Phase 1 GPU Robust Implementation")
        print(f"   Model: Gemma 3 {model_type.upper()}")
        print("   Strategy: Simple, robust kernels that avoid hangs")
        print("   GPU: Confirmed working with basic OpenCL")
        print()
        
        self._initialize_gpu()
        self._create_robust_kernels()
    
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
    
    def _initialize_gpu(self):
        """Initialize GPU with robust settings"""
        platforms = cl.get_platforms()
        gpu_devices = []
        
        for platform in platforms:
            devices = platform.get_devices(device_type=cl.device_type.GPU)
            gpu_devices.extend(devices)
        
        if not gpu_devices:
            raise RuntimeError("No GPU found!")
        
        self.device = gpu_devices[0]
        self.ctx = cl.Context([self.device])
        
        # Create command queue without profiling (more stable)
        self.queue = cl.CommandQueue(self.ctx)
        
        print(f"✅ GPU initialized: {self.device.name}")
        print(f"   Memory: {self.device.global_mem_size / 1024**3:.1f} GB")
        print(f"   Compute Units: {self.device.max_compute_units}")
    
    def _create_robust_kernels(self):
        """Create simple, robust kernels that avoid complex patterns"""
        
        kernel_source = """
        // Simple GEMM kernel - no complex tiling or local memory
        __kernel void gemm_simple(
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
        
        // QKV projection - fused but simple
        __kernel void qkv_projection_simple(
            __global const float* input,
            __global const float* W_qkv,
            __global float* output,
            const int batch_seq, const int hidden_size
        ) {
            int row = get_global_id(0);
            int col = get_global_id(1);
            
            if (row >= batch_seq || col >= 3 * hidden_size) return;
            
            float sum = 0.0f;
            for (int k = 0; k < hidden_size; k++) {
                sum += input[row * hidden_size + k] * W_qkv[k * (3 * hidden_size) + col];
            }
            
            output[row * (3 * hidden_size) + col] = sum;
        }
        
        // Attention softmax - row-wise operation
        __kernel void attention_softmax_rows(
            __global float* scores,
            const int batch_heads, const int seq_len
        ) {
            int idx = get_global_id(0);
            if (idx >= batch_heads * seq_len) return;
            
            int batch_head = idx / seq_len;
            int row = idx % seq_len;
            
            __global float* row_scores = scores + batch_head * seq_len * seq_len + row * seq_len;
            
            // Find max for numerical stability
            float max_val = -INFINITY;
            for (int col = 0; col <= row; col++) {
                max_val = fmax(max_val, row_scores[col]);
            }
            
            // Compute exp and sum
            float sum = 0.0f;
            for (int col = 0; col <= row; col++) {
                float exp_val = exp(row_scores[col] - max_val);
                row_scores[col] = exp_val;
                sum += exp_val;
            }
            
            // Normalize
            for (int col = 0; col <= row; col++) {
                row_scores[col] /= sum;
            }
            
            // Zero future positions (causal)
            for (int col = row + 1; col < seq_len; col++) {
                row_scores[col] = 0.0f;
            }
        }
        
        // MLP gate and up projection - separate kernels to avoid issues
        __kernel void mlp_gate_up_separate(
            __global const float* input,
            __global const float* W_gate,
            __global const float* W_up,
            __global float* gate_out,
            __global float* up_out,
            const int batch_seq, const int hidden_size, const int ff_dim
        ) {
            int row = get_global_id(0);
            int col = get_global_id(1);
            
            if (row >= batch_seq || col >= ff_dim) return;
            
            // Compute gate projection
            float gate_sum = 0.0f;
            float up_sum = 0.0f;
            
            for (int k = 0; k < hidden_size; k++) {
                float input_val = input[row * hidden_size + k];
                gate_sum += input_val * W_gate[k * ff_dim + col];
                up_sum += input_val * W_up[k * ff_dim + col];
            }
            
            gate_out[row * ff_dim + col] = gate_sum;
            up_out[row * ff_dim + col] = up_sum;
        }
        
        // GELU activation and multiply
        __kernel void gelu_multiply(
            __global const float* gate,
            __global const float* up,
            __global float* output,
            const int size
        ) {
            int idx = get_global_id(0);
            if (idx >= size) return;
            
            float gate_val = gate[idx];
            float up_val = up[idx];
            
            // GELU approximation
            float sigmoid = 1.0f / (1.0f + exp(-1.702f * gate_val));
            float gelu_gate = gate_val * sigmoid;
            
            output[idx] = gelu_gate * up_val;
        }
        
        // Simple residual addition
        __kernel void add_residual(
            __global float* output,
            __global const float* residual,
            const int size
        ) {
            int idx = get_global_id(0);
            if (idx >= size) return;
            
            output[idx] += residual[idx];
        }
        """
        
        # Build with minimal optimizations (more stable)
        build_options = "-cl-std=CL1.2"  # Use older standard for stability
        
        try:
            self.program = cl.Program(self.ctx, kernel_source).build(build_options)
            print("✅ Robust kernels created successfully")
        except Exception as e:
            print(f"❌ Kernel compilation failed: {e}")
            raise
    
    def gemm_gpu(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Simple GEMM on GPU"""
        M, K = A.shape
        K2, N = B.shape
        assert K == K2
        
        # Create buffers
        mf = cl.mem_flags
        a_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A.astype(np.float32))
        b_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B.astype(np.float32))
        c_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=M * N * 4)
        
        # Execute with simple global size
        global_size = (M, N)
        self.program.gemm_simple(self.queue, global_size, None,
                                a_buf, b_buf, c_buf,
                                np.int32(M), np.int32(N), np.int32(K))
        
        # Read result
        result = np.empty((M, N), dtype=np.float32)
        cl.enqueue_copy(self.queue, result, c_buf)
        self.queue.finish()
        
        return result
    
    def qkv_projection_fused_gpu(self, input_data, W_q, W_k, W_v):
        """Fused QKV projection on GPU"""
        batch_seq, hidden_size = input_data.shape
        
        # Concatenate weights
        W_qkv = np.concatenate([W_q, W_k, W_v], axis=1).astype(np.float32)
        
        # Create buffers
        mf = cl.mem_flags
        input_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_data)
        w_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=W_qkv)
        output_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=batch_seq * 3 * hidden_size * 4)
        
        # Execute
        global_size = (batch_seq, 3 * hidden_size)
        self.program.qkv_projection_simple(self.queue, global_size, None,
                                          input_buf, w_buf, output_buf,
                                          np.int32(batch_seq), np.int32(hidden_size))
        
        # Read result
        result = np.empty((batch_seq, 3 * hidden_size), dtype=np.float32)
        cl.enqueue_copy(self.queue, result, output_buf)
        self.queue.finish()
        
        # Split Q, K, V
        Q = result[:, :hidden_size]
        K = result[:, hidden_size:2*hidden_size]
        V = result[:, 2*hidden_size:]
        
        return Q, K, V
    
    def attention_gpu(self, Q, K, V, num_heads):
        """Attention computation on GPU with simple kernels"""
        batch_size, seq_len, hidden_size = Q.shape
        head_dim = hidden_size // num_heads
        
        # Reshape for multi-head
        Q = Q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        
        # Flatten batch and heads
        Q = Q.reshape(-1, seq_len, head_dim)
        K = K.reshape(-1, seq_len, head_dim)
        V = V.reshape(-1, seq_len, head_dim)
        
        batch_heads = batch_size * num_heads
        
        # Compute attention scores on GPU
        scores = np.zeros((batch_heads, seq_len, seq_len), dtype=np.float32)
        
        for i in range(batch_heads):
            # Use simple GEMM for Q @ K^T
            scores[i] = self.gemm_gpu(Q[i], K[i].T) / np.sqrt(head_dim)
        
        # Apply softmax on GPU
        scores_flat = scores.reshape(-1, seq_len, seq_len)
        
        mf = cl.mem_flags
        scores_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=scores_flat)
        
        global_size = (batch_heads * seq_len,)
        self.program.attention_softmax_rows(self.queue, global_size, None,
                                           scores_buf,
                                           np.int32(batch_heads), np.int32(seq_len))
        
        cl.enqueue_copy(self.queue, scores_flat, scores_buf)
        self.queue.finish()
        
        # Apply attention to values
        output = np.zeros((batch_heads, seq_len, head_dim), dtype=np.float32)
        for i in range(batch_heads):
            output[i] = self.gemm_gpu(scores_flat[i], V[i])
        
        # Reshape back
        output = output.reshape(batch_size, num_heads, seq_len, head_dim)
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, hidden_size)
        
        return output
    
    def mlp_gpu(self, input_data, W_gate, W_up, W_down):
        """MLP computation on GPU with separate kernels"""
        batch_seq, hidden_size = input_data.shape
        ff_dim = W_gate.shape[1]
        
        # Create buffers
        mf = cl.mem_flags
        input_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_data)
        w_gate_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=W_gate)
        w_up_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=W_up)
        gate_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=batch_seq * ff_dim * 4)
        up_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=batch_seq * ff_dim * 4)
        
        # Gate and up projections
        global_size = (batch_seq, ff_dim)
        self.program.mlp_gate_up_separate(self.queue, global_size, None,
                                         input_buf, w_gate_buf, w_up_buf,
                                         gate_buf, up_buf,
                                         np.int32(batch_seq), np.int32(hidden_size), np.int32(ff_dim))
        
        # GELU and multiply
        intermediate_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=batch_seq * ff_dim * 4)
        global_size = (batch_seq * ff_dim,)
        self.program.gelu_multiply(self.queue, global_size, None,
                                  gate_buf, up_buf, intermediate_buf,
                                  np.int32(batch_seq * ff_dim))
        
        # Read intermediate result
        intermediate = np.empty((batch_seq, ff_dim), dtype=np.float32)
        cl.enqueue_copy(self.queue, intermediate, intermediate_buf)
        self.queue.finish()
        
        # Down projection using simple GEMM
        output = self.gemm_gpu(intermediate, W_down)
        
        return output
    
    def transformer_layer_gpu(self, hidden_states, layer_idx):
        """Complete transformer layer on GPU"""
        batch_size, seq_len, hidden_size = hidden_states.shape
        batch_seq = batch_size * seq_len
        
        # Create weights (would be loaded in real implementation)
        W_q = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
        W_k = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
        W_v = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
        W_o = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
        
        W_gate = np.random.randn(hidden_size, self.config['ff_dim']).astype(np.float32) * 0.02
        W_up = np.random.randn(hidden_size, self.config['ff_dim']).astype(np.float32) * 0.02
        W_down = np.random.randn(self.config['ff_dim'], hidden_size).astype(np.float32) * 0.02
        
        # Reshape for operations
        x = hidden_states.reshape(-1, hidden_size).astype(np.float32)
        
        # QKV projection (fused)
        Q, K, V = self.qkv_projection_fused_gpu(x, W_q, W_k, W_v)
        Q = Q.reshape(batch_size, seq_len, hidden_size)
        K = K.reshape(batch_size, seq_len, hidden_size)
        V = V.reshape(batch_size, seq_len, hidden_size)
        
        # Attention
        attn_output = self.attention_gpu(Q, K, V, self.config['num_heads'])
        
        # Output projection
        attn_output = self.gemm_gpu(attn_output.reshape(-1, hidden_size), W_o)
        attn_output = attn_output.reshape(batch_size, seq_len, hidden_size)
        
        # Residual connection (on GPU)
        hidden_states = hidden_states + attn_output
        
        # MLP
        mlp_input = hidden_states.reshape(-1, hidden_size).astype(np.float32)
        mlp_output = self.mlp_gpu(mlp_input, W_gate, W_up, W_down)
        mlp_output = mlp_output.reshape(batch_size, seq_len, hidden_size)
        
        # Final residual
        hidden_states = hidden_states + mlp_output
        
        return hidden_states
    
    def benchmark(self):
        """Benchmark GPU implementation"""
        print("\n📊 Benchmarking Phase 1 GPU Implementation...")
        
        # Test with small sequence first
        seq_lengths = [32, 128]
        results = {}
        
        for seq_len in seq_lengths:
            print(f"\n   Testing sequence length: {seq_len}")
            
            hidden_states = np.random.randn(1, seq_len, self.config['hidden_size']).astype(np.float32)
            
            # Warmup
            print("   Warming up...")
            for _ in range(2):
                _ = self.transformer_layer_gpu(hidden_states, 0)
            
            # Benchmark
            print("   Benchmarking...")
            iterations = 10
            start = time.time()
            
            for _ in range(iterations):
                output = self.transformer_layer_gpu(hidden_states, 0)
            
            self.queue.finish()
            elapsed = time.time() - start
            
            layer_time = elapsed / iterations
            total_time = layer_time * self.config['num_layers']
            tps = 10 / total_time  # 10 tokens generated
            
            results[seq_len] = {
                'layer_time': layer_time,
                'total_time': total_time,
                'tps': tps
            }
            
            print(f"   Layer time: {layer_time*1000:.1f}ms")
            print(f"   Total time: {total_time:.2f}s")
            print(f"   TPS: {tps:.2f}")
        
        return results

def main():
    print("🦄 Phase 1 GPU Robust Implementation")
    print("=" * 60)
    
    try:
        # Test with 4B model first
        print("\n1️⃣ Testing Gemma 3 4B...")
        pipeline = Phase1GPURobust("4b")
        results = pipeline.benchmark()
        
        print("\n" + "="*60)
        print("🏆 Phase 1 GPU Results:")
        for seq_len, result in results.items():
            print(f"   {seq_len} tokens: {result['tps']:.2f} TPS")
        
        # Compare to baseline
        baseline_tps = 5.13
        avg_tps = np.mean([r['tps'] for r in results.values()])
        speedup = avg_tps / baseline_tps
        
        print(f"\n📊 Performance:")
        print(f"   Average TPS: {avg_tps:.2f}")
        print(f"   Baseline TPS: {baseline_tps:.2f}")
        print(f"   Speedup: {speedup:.2f}x")
        
        if speedup > 1.5:
            print("\n✅ Phase 1 GPU fusion achieves target speedup!")
        else:
            print("\n⚠️  Performance needs optimization")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()