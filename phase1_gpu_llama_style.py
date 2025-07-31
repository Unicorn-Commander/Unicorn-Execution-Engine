#!/usr/bin/env python3.13
"""
Phase 1 GPU Implementation - llama.cpp Style
Simple, stable kernels that avoid complex patterns
"""

import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import numpy as np
import pyopencl as cl
import time
from pathlib import Path

class Phase1GPULlamaStyle:
    """GPU implementation following llama.cpp's simple kernel approach"""
    
    def __init__(self, model_type="4b"):
        self.model_type = model_type
        self.config = self._get_config()
        
        print("🦄 Phase 1 GPU - llama.cpp Style")
        print(f"   Model: Gemma 3 {model_type.upper()}")
        print("   Strategy: Simple kernels, no complex fusion")
        print("   HSA Override: 11.0.0")
        print()
        
        self._initialize_gpu()
        self._create_simple_kernels()
    
    def _get_config(self):
        configs = {
            "4b": {
                "hidden_size": 2560,
                "num_layers": 28,
                "num_heads": 20,
                "head_dim": 128,
                "ff_dim": 10240,
            }
        }
        return configs[self.model_type]
    
    def _initialize_gpu(self):
        """Initialize GPU with conservative settings"""
        platforms = cl.get_platforms()
        gpu_devices = []
        
        for platform in platforms:
            devices = platform.get_devices(device_type=cl.device_type.GPU)
            gpu_devices.extend(devices)
        
        self.device = gpu_devices[0]
        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.ctx)
        
        print(f"✅ GPU: {self.device.name}")
        print(f"   Memory: {self.device.global_mem_size / 1024**3:.1f} GB")
    
    def _create_simple_kernels(self):
        """Create simple kernels like llama.cpp"""
        
        kernel_source = """
        // Simple row-wise operations to avoid complex memory patterns
        
        // Basic vector addition
        __kernel void vec_add(
            __global const float* a,
            __global const float* b,
            __global float* c,
            const int n
        ) {
            int i = get_global_id(0);
            if (i < n) {
                c[i] = a[i] + b[i];
            }
        }
        
        // Row-wise matrix-vector multiplication
        __kernel void mat_vec_mul(
            __global const float* mat,  // [rows, cols]
            __global const float* vec,  // [cols]
            __global float* out,        // [rows]
            const int rows,
            const int cols
        ) {
            int row = get_global_id(0);
            if (row >= rows) return;
            
            float sum = 0.0f;
            for (int col = 0; col < cols; col++) {
                sum += mat[row * cols + col] * vec[col];
            }
            out[row] = sum;
        }
        
        // Simple GEMV (for transformer ops)
        __kernel void gemv_simple(
            __global const float* A,
            __global const float* x,
            __global float* y,
            const int M,
            const int N
        ) {
            int i = get_global_id(0);
            if (i >= M) return;
            
            float sum = 0.0f;
            for (int j = 0; j < N; j++) {
                sum += A[i * N + j] * x[j];
            }
            y[i] = sum;
        }
        
        // Row-wise softmax (stable)
        __kernel void softmax_rows(
            __global float* x,
            const int rows,
            const int cols
        ) {
            int row = get_global_id(0);
            if (row >= rows) return;
            
            __global float* row_data = x + row * cols;
            
            // Find max
            float max_val = row_data[0];
            for (int i = 1; i < cols; i++) {
                max_val = fmax(max_val, row_data[i]);
            }
            
            // Exp and sum
            float sum = 0.0f;
            for (int i = 0; i < cols; i++) {
                row_data[i] = exp(row_data[i] - max_val);
                sum += row_data[i];
            }
            
            // Normalize
            for (int i = 0; i < cols; i++) {
                row_data[i] /= sum;
            }
        }
        
        // GELU activation (element-wise)
        __kernel void gelu_activation(
            __global float* x,
            const int n
        ) {
            int i = get_global_id(0);
            if (i >= n) return;
            
            float val = x[i];
            float sigmoid = 1.0f / (1.0f + exp(-1.702f * val));
            x[i] = val * sigmoid;
        }
        
        // Element-wise multiply
        __kernel void elem_multiply(
            __global const float* a,
            __global const float* b,
            __global float* c,
            const int n
        ) {
            int i = get_global_id(0);
            if (i < n) {
                c[i] = a[i] * b[i];
            }
        }
        """
        
        # Build with minimal options
        self.program = cl.Program(self.ctx, kernel_source).build()
        print("✅ Simple kernels created (llama.cpp style)")
    
    def gemv_gpu(self, matrix, vector):
        """Simple matrix-vector multiplication"""
        M, N = matrix.shape
        
        mf = cl.mem_flags
        mat_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=matrix)
        vec_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vector)
        out_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=M * 4)
        
        self.program.gemv_simple(self.queue, (M,), None,
                                mat_buf, vec_buf, out_buf,
                                np.int32(M), np.int32(N))
        
        result = np.empty(M, dtype=np.float32)
        cl.enqueue_copy(self.queue, result, out_buf)
        self.queue.finish()
        
        return result
    
    def transformer_layer_simple(self, hidden_states, layer_idx):
        """Transformer layer with simple operations"""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Create weights
        W_qkv = np.random.randn(hidden_size, 3 * hidden_size).astype(np.float32) * 0.02
        W_o = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
        W_mlp = np.random.randn(hidden_size, self.config['ff_dim'] * 2).astype(np.float32) * 0.02
        W_proj = np.random.randn(self.config['ff_dim'], hidden_size).astype(np.float32) * 0.02
        
        # Process sequence by sequence (simple, stable)
        outputs = []
        
        for b in range(batch_size):
            seq_outputs = []
            
            for s in range(seq_len):
                # Get single token
                token = hidden_states[b, s, :].astype(np.float32)
                
                # QKV projection (single GEMV)
                qkv = self.gemv_gpu(W_qkv.T, token)
                
                # Split Q, K, V
                q = qkv[:hidden_size]
                k = qkv[hidden_size:2*hidden_size]
                v = qkv[2*hidden_size:]
                
                # Simple self-attention (for single token)
                # In real implementation, would accumulate K,V cache
                attn_out = v  # Simplified
                
                # Output projection
                attn_out = self.gemv_gpu(W_o.T, attn_out)
                
                # Residual
                token = token + attn_out
                
                # MLP
                mlp_out = self.gemv_gpu(W_mlp.T, token)
                
                # Split and apply GELU to first half
                gate = mlp_out[:self.config['ff_dim']].copy()
                up = mlp_out[self.config['ff_dim']:]
                
                # GELU on GPU
                mf = cl.mem_flags
                gate_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=gate)
                self.program.gelu_activation(self.queue, (len(gate),), None,
                                           gate_buf, np.int32(len(gate)))
                cl.enqueue_copy(self.queue, gate, gate_buf)
                
                # Element-wise multiply
                activated = gate * up
                
                # Project down
                mlp_final = self.gemv_gpu(W_proj.T, activated)
                
                # Final residual
                token = token + mlp_final
                
                seq_outputs.append(token)
            
            outputs.append(np.stack(seq_outputs))
        
        return np.stack(outputs)
    
    def benchmark_simple(self):
        """Benchmark simple operations"""
        print("\n📊 Benchmarking Simple GPU Operations...")
        
        # Test GEMV performance
        M, N = 2560, 2560
        matrix = np.random.randn(M, N).astype(np.float32)
        vector = np.random.randn(N).astype(np.float32)
        
        # Warmup
        for _ in range(5):
            _ = self.gemv_gpu(matrix.T, vector)
        
        # Benchmark
        iterations = 100
        start = time.time()
        for _ in range(iterations):
            _ = self.gemv_gpu(matrix.T, vector)
        elapsed = time.time() - start
        
        ops = 2 * M * N * iterations
        gflops = ops / (elapsed * 1e9)
        print(f"   GEMV Performance: {gflops:.1f} GFLOPS")
        
        # Test layer performance
        print("\n   Testing transformer layer...")
        hidden_states = np.random.randn(1, 32, self.config['hidden_size']).astype(np.float32)
        
        start = time.time()
        output = self.transformer_layer_simple(hidden_states, 0)
        layer_time = time.time() - start
        
        print(f"   Single layer time: {layer_time*1000:.1f}ms")
        
        # Estimate TPS
        total_time = layer_time * self.config['num_layers']
        tps = 10 / total_time
        
        print(f"   Estimated TPS: {tps:.2f}")
        
        return {
            'gemv_gflops': gflops,
            'layer_time': layer_time,
            'tps': tps
        }

def main():
    print("🦄 Phase 1 GPU - llama.cpp Style Test")
    print("=" * 60)
    
    try:
        # Ensure HSA override is set
        print(f"HSA Override: {os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'Not set')}")
        
        pipeline = Phase1GPULlamaStyle("4b")
        results = pipeline.benchmark_simple()
        
        print("\n" + "="*60)
        print("🏆 Results:")
        print(f"   GEMV Performance: {results['gemv_gflops']:.1f} GFLOPS")
        print(f"   Layer Time: {results['layer_time']*1000:.1f}ms")
        print(f"   Estimated TPS: {results['tps']:.2f}")
        
        baseline_tps = 5.13
        speedup = results['tps'] / baseline_tps
        print(f"\n   vs CPU Baseline: {speedup:.2f}x")
        
        if results['tps'] > baseline_tps:
            print("\n✅ Simple GPU approach works!")
        else:
            print("\n⚠️  Still slower than CPU")
            print("   Consider using CLBlast for better performance")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()