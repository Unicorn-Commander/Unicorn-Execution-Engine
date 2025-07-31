#!/usr/bin/env python3.13
"""
Magic Unicorn OpenCL INT4 Implementation
Alternative to HIP WMMA using OpenCL with INT4 quantization
"""

import torch
import numpy as np
import pyopencl as cl
import time
from typing import Dict, Tuple, Optional
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MagicUnicornOpenCLINT4:
    """OpenCL-based INT4 quantized inference engine"""
    
    def __init__(self):
        logger.info("🦄 Magic Unicorn OpenCL INT4 Engine")
        logger.info("Alternative implementation using OpenCL with INT4 quantization")
        
        self.ctx = None
        self.queue = None
        self.program = None
        
        # Setup OpenCL
        self.setup_opencl()
        
        # Compile INT4 kernels
        self.compile_int4_kernels()
        
        # Performance stats
        self.stats = {
            'qkv_time': 0,
            'attn_time': 0,
            'ffn_time': 0,
            'quant_time': 0,
            'total_time': 0
        }
        
    def setup_opencl(self):
        """Initialize OpenCL context for iGPU"""
        # Find AMD GPU
        platforms = cl.get_platforms()
        gpu_device = None
        
        for platform in platforms:
            devices = platform.get_devices(device_type=cl.device_type.GPU)
            for device in devices:
                if 'gfx1103' in device.name.lower():
                    gpu_device = device
                    break
            if gpu_device:
                break
                
        if not gpu_device:
            raise RuntimeError("No compatible AMD GPU found")
            
        self.ctx = cl.Context([gpu_device])
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        
        logger.info(f"✓ OpenCL initialized on: {gpu_device.name}")
        logger.info(f"  Max compute units: {gpu_device.max_compute_units}")
        logger.info(f"  Global memory: {gpu_device.global_mem_size / (1024**3):.1f} GB")
        
    def compile_int4_kernels(self):
        """Compile optimized INT4 OpenCL kernels"""
        kernel_source = """
// INT4 GEMM kernel with 2x2 register blocking for RDNA3
__kernel void gemm_int4_blocked(
    __global const uchar* A_packed,  // INT4 packed weights (2 values per byte)
    __global const float* B,         // FP32 input 
    __global float* C,               // FP32 output
    __global const float* scales,    // Quantization scales
    const int M, const int N, const int K
) {
    const int bx = get_group_id(0);
    const int by = get_group_id(1);
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    const int TILE = 16;
    
    // Shared memory for tile
    __local float As[TILE][TILE];
    __local float Bs[TILE][TILE];
    
    // Register blocking 2x2
    float c[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    
    const int row = by * TILE + ty;
    const int col = bx * TILE + tx;
    
    // Load quantization scale
    float scale = scales[by * TILE + ty];
    
    // Main GEMM loop with INT4 unpacking
    for (int k = 0; k < K; k += TILE) {
        // Load and unpack INT4 weights
        if ((row < M) && (k + tx < K)) {
            int packed_idx = (row * K + k + tx) / 2;
            uchar packed = A_packed[packed_idx];
            
            // Unpack INT4 (lower 4 bits for even index, upper for odd)
            int int4_val;
            if ((k + tx) % 2 == 0) {
                int4_val = (packed & 0xF) - 8;  // Sign extend from 4-bit
            } else {
                int4_val = ((packed >> 4) & 0xF) - 8;
            }
            
            As[ty][tx] = (float)int4_val * scale;
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load input tile
        if ((k + ty < K) && (col < N)) {
            Bs[ty][tx] = B[(k + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute with 2x2 register blocking
        #pragma unroll
        for (int i = 0; i < TILE; i++) {
            float a0 = As[ty * 2][i];
            float a1 = As[ty * 2 + 1][i];
            float b0 = Bs[i][tx * 2];
            float b1 = Bs[i][tx * 2 + 1];
            
            c[0][0] += a0 * b0;
            c[0][1] += a0 * b1;
            c[1][0] += a1 * b0;
            c[1][1] += a1 * b1;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write results
    if ((row * 2 < M) && (col * 2 < N)) {
        C[(row * 2) * N + col * 2] = c[0][0];
        if (col * 2 + 1 < N) C[(row * 2) * N + col * 2 + 1] = c[0][1];
        if (row * 2 + 1 < M) {
            C[(row * 2 + 1) * N + col * 2] = c[1][0];
            if (col * 2 + 1 < N) C[(row * 2 + 1) * N + col * 2 + 1] = c[1][1];
        }
    }
}

// Optimized INT4 attention kernel
__kernel void attention_int4_sdp(
    __global const float* Q,
    __global const uchar* K_packed,
    __global const uchar* V_packed,
    __global float* output,
    __global const float* k_scale,
    __global const float* v_scale,
    const int seq_len,
    const int head_dim,
    const int num_heads
) {
    const int head = get_global_id(0);
    const int pos = get_global_id(1);
    
    if (head >= num_heads || pos >= seq_len) return;
    
    const int head_offset = head * head_dim;
    float max_score = -INFINITY;
    
    // Compute attention scores with INT4 K
    __local float scores[256];  // Assuming max seq_len 256
    
    for (int i = 0; i <= pos; i++) {
        float score = 0.0f;
        
        // Dot product Q[pos] * K[i] with INT4 unpacking
        for (int d = 0; d < head_dim; d++) {
            float q_val = Q[pos * num_heads * head_dim + head_offset + d];
            
            // Unpack INT4 K value
            int k_idx = i * num_heads * head_dim + head_offset + d;
            int packed_idx = k_idx / 2;
            uchar packed = K_packed[packed_idx];
            
            int k_int4 = (k_idx % 2 == 0) ? (packed & 0xF) - 8 : ((packed >> 4) & 0xF) - 8;
            float k_val = (float)k_int4 * k_scale[head];
            
            score += q_val * k_val;
        }
        
        score /= sqrt((float)head_dim);
        scores[i] = score;
        max_score = fmax(max_score, score);
    }
    
    // Softmax
    float sum = 0.0f;
    for (int i = 0; i <= pos; i++) {
        scores[i] = exp(scores[i] - max_score);
        sum += scores[i];
    }
    
    for (int i = 0; i <= pos; i++) {
        scores[i] /= sum;
    }
    
    // Weighted sum with INT4 V
    for (int d = 0; d < head_dim; d++) {
        float out = 0.0f;
        
        for (int i = 0; i <= pos; i++) {
            // Unpack INT4 V value
            int v_idx = i * num_heads * head_dim + head_offset + d;
            int packed_idx = v_idx / 2;
            uchar packed = V_packed[packed_idx];
            
            int v_int4 = (v_idx % 2 == 0) ? (packed & 0xF) - 8 : ((packed >> 4) & 0xF) - 8;
            float v_val = (float)v_int4 * v_scale[head];
            
            out += scores[i] * v_val;
        }
        
        output[pos * num_heads * head_dim + head_offset + d] = out;
    }
}

// Fast INT4 packing kernel
__kernel void pack_to_int4(
    __global const float* input,
    __global uchar* output,
    __global float* scale,
    const int size
) {
    const int gid = get_global_id(0);
    const int pair_idx = gid;
    
    if (pair_idx * 2 >= size) return;
    
    // Find scale (simplified - in practice use per-channel)
    float max_val = 0.0f;
    for (int i = pair_idx * 256; i < min((pair_idx + 1) * 256, size); i++) {
        max_val = fmax(max_val, fabs(input[i]));
    }
    
    float quant_scale = max_val / 7.0f;
    if (pair_idx == 0) scale[0] = quant_scale;
    
    // Pack two INT4 values
    float val1 = input[pair_idx * 2];
    float val2 = (pair_idx * 2 + 1 < size) ? input[pair_idx * 2 + 1] : 0.0f;
    
    int int4_1 = clamp((int)round(val1 / quant_scale), -8, 7) + 8;
    int int4_2 = clamp((int)round(val2 / quant_scale), -8, 7) + 8;
    
    output[pair_idx] = (int4_1 & 0xF) | ((int4_2 & 0xF) << 4);
}
"""
        
        self.program = cl.Program(self.ctx, kernel_source).build(
            options=['-cl-fast-relaxed-math', '-cl-mad-enable']
        )
        
        logger.info("✓ INT4 OpenCL kernels compiled successfully")
        
    def quantize_weights_int4(self, weights: Dict[str, torch.Tensor]) -> Dict[str, Tuple[cl.Buffer, cl.Buffer, Tuple]]:
        """Quantize weights to INT4 and upload to GPU"""
        logger.info("Quantizing weights to INT4...")
        
        cl_weights = {}
        
        for name, tensor in weights.items():
            # Flatten and convert to numpy
            shape = tensor.shape
            flat = tensor.flatten().numpy().astype(np.float32)
            
            # Find scale
            scale = np.abs(flat).max() / 7.0
            
            # Quantize to INT4
            int4_vals = np.round(flat / scale).clip(-8, 7).astype(np.int8)
            
            # Pack INT4 (2 values per byte)
            packed_size = (len(int4_vals) + 1) // 2
            packed = np.zeros(packed_size, dtype=np.uint8)
            
            for i in range(0, len(int4_vals), 2):
                val1 = int4_vals[i] + 8  # Make unsigned
                val2 = int4_vals[i + 1] + 8 if i + 1 < len(int4_vals) else 8
                packed[i // 2] = (val1 & 0xF) | ((val2 & 0xF) << 4)
                
            # Create OpenCL buffers
            mf = cl.mem_flags
            packed_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=packed)
            scale_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, 
                                 hostbuf=np.array([scale], dtype=np.float32))
            
            cl_weights[name] = (packed_buf, scale_buf, shape)
            
        logger.info(f"✓ Quantized {len(cl_weights)} weights to INT4")
        return cl_weights
        
    def gemm_int4(self, input_buf: cl.Buffer, weight_packed: cl.Buffer, 
                  weight_scale: cl.Buffer, output_buf: cl.Buffer,
                  M: int, N: int, K: int):
        """Execute INT4 GEMM kernel"""
        global_size = ((N + 15) // 16 * 8, (M + 15) // 16 * 8)  # 2x2 register blocking
        local_size = (8, 8)
        
        event = self.program.gemm_int4_blocked(
            self.queue, global_size, local_size,
            weight_packed, input_buf, output_buf, weight_scale,
            np.int32(M), np.int32(N), np.int32(K)
        )
        
        return event
        
    def forward_layer_int4(self, x: torch.Tensor, weights: Dict[str, Tuple[cl.Buffer, cl.Buffer, Tuple]]) -> torch.Tensor:
        """Forward pass through transformer layer with INT4 weights"""
        batch_size, seq_len, hidden_size = x.shape
        mf = cl.mem_flags
        
        # Upload input
        x_np = x.numpy().astype(np.float32)
        x_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=x_np)
        
        # QKV projections with INT4
        qkv_size = hidden_size * 3
        qkv_buf = cl.Buffer(self.ctx, mf.READ_WRITE, size=batch_size * seq_len * qkv_size * 4)
        
        # Combine QKV projections
        start = time.time()
        events = []
        
        for i, proj in enumerate(['q_proj', 'k_proj', 'v_proj']):
            if proj in weights:
                packed_buf, scale_buf, shape = weights[proj]
                
                # Output offset for this projection
                offset = i * hidden_size * batch_size * seq_len * 4
                output_region = cl.Buffer(self.ctx, mf.READ_WRITE, size=1)
                
                event = self.gemm_int4(
                    x_buf, packed_buf, scale_buf, qkv_buf,
                    shape[0], batch_size * seq_len, shape[1]
                )
                events.append(event)
                
        # Wait for QKV completion
        for event in events:
            event.wait()
            
        self.stats['qkv_time'] = time.time() - start
        
        # Simplified attention (using FP32 for now)
        # In production, would use INT4 attention kernel
        start = time.time()
        qkv_np = np.empty((batch_size, seq_len, qkv_size), dtype=np.float32)
        cl.enqueue_copy(self.queue, qkv_np, qkv_buf).wait()
        
        q = torch.from_numpy(qkv_np[:, :, :hidden_size])
        k = torch.from_numpy(qkv_np[:, :, hidden_size:2*hidden_size])
        v = torch.from_numpy(qkv_np[:, :, 2*hidden_size:])
        
        # Simple attention computation
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(hidden_size)
        
        # Causal mask
        if seq_len > 1:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            scores.masked_fill_(mask, float('-inf'))
            
        attn_weights = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)
        
        self.stats['attn_time'] = time.time() - start
        
        # Output projection with INT4
        start = time.time()
        attn_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, 
                            hostbuf=attn_out.numpy().astype(np.float32))
        out_buf = cl.Buffer(self.ctx, mf.READ_WRITE, size=x_np.nbytes)
        
        if 'o_proj' in weights:
            packed_buf, scale_buf, shape = weights['o_proj']
            event = self.gemm_int4(
                attn_buf, packed_buf, scale_buf, out_buf,
                shape[0], batch_size * seq_len, shape[1]
            )
            event.wait()
            
        # Residual connection
        cl.enqueue_copy(self.queue, x_np, out_buf).wait()
        x_residual = x + torch.from_numpy(x_np)
        
        # FFN with INT4
        ffn_start = time.time()
        
        # Gate and up projections
        intermediate_size = int(hidden_size * 2.625)
        gate_buf = cl.Buffer(self.ctx, mf.READ_WRITE, size=batch_size * seq_len * intermediate_size * 4)
        up_buf = cl.Buffer(self.ctx, mf.READ_WRITE, size=batch_size * seq_len * intermediate_size * 4)
        
        x_res_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                             hostbuf=x_residual.numpy().astype(np.float32))
        
        # Execute gate and up projections
        for buf, proj in [(gate_buf, 'gate_proj'), (up_buf, 'up_proj')]:
            if proj in weights:
                packed_buf, scale_buf, shape = weights[proj]
                event = self.gemm_int4(
                    x_res_buf, packed_buf, scale_buf, buf,
                    shape[0], batch_size * seq_len, shape[1]
                )
                event.wait()
                
        # Apply SiLU activation and multiply (in numpy for simplicity)
        gate_np = np.empty((batch_size, seq_len, intermediate_size), dtype=np.float32)
        up_np = np.empty((batch_size, seq_len, intermediate_size), dtype=np.float32)
        
        cl.enqueue_copy(self.queue, gate_np, gate_buf).wait()
        cl.enqueue_copy(self.queue, up_np, up_buf).wait()
        
        # SiLU(gate) * up
        gate_torch = torch.from_numpy(gate_np)
        up_torch = torch.from_numpy(up_np)
        hidden = torch.nn.functional.silu(gate_torch) * up_torch
        
        # Down projection with INT4
        hidden_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                              hostbuf=hidden.numpy().astype(np.float32))
        final_buf = cl.Buffer(self.ctx, mf.READ_WRITE, size=x_np.nbytes)
        
        if 'down_proj' in weights:
            packed_buf, scale_buf, shape = weights['down_proj']
            event = self.gemm_int4(
                hidden_buf, packed_buf, scale_buf, final_buf,
                shape[0], batch_size * seq_len, shape[1]
            )
            event.wait()
            
        # Final residual
        final_np = np.empty_like(x_np)
        cl.enqueue_copy(self.queue, final_np, final_buf).wait()
        
        output = x_residual + torch.from_numpy(final_np)
        
        self.stats['ffn_time'] = time.time() - ffn_start
        self.stats['total_time'] = time.time() - start
        
        return output
        
    def benchmark_int4_performance(self):
        """Benchmark INT4 performance"""
        logger.info("\n🚀 INT4 OpenCL Performance Benchmark")
        logger.info("=" * 50)
        
        # Test configurations
        test_configs = [
            (1, 32, 2560, "Small context"),
            (1, 128, 2560, "Medium context"),
            (1, 256, 2560, "Large context"),
        ]
        
        # Create dummy weights
        hidden_size = 2560
        intermediate_size = int(hidden_size * 2.625)
        
        weights = {
            'q_proj': torch.randn(hidden_size, hidden_size),
            'k_proj': torch.randn(hidden_size, hidden_size),
            'v_proj': torch.randn(hidden_size, hidden_size),
            'o_proj': torch.randn(hidden_size, hidden_size),
            'gate_proj': torch.randn(intermediate_size, hidden_size),
            'up_proj': torch.randn(intermediate_size, hidden_size),
            'down_proj': torch.randn(hidden_size, intermediate_size),
        }
        
        # Quantize weights
        cl_weights = self.quantize_weights_int4(weights)
        
        for batch_size, seq_len, hidden, desc in test_configs:
            logger.info(f"\nTesting {desc}: batch={batch_size}, seq={seq_len}")
            
            # Create test input
            x = torch.randn(batch_size, seq_len, hidden)
            
            # Warmup
            for _ in range(2):
                _ = self.forward_layer_int4(x, cl_weights)
                
            # Benchmark
            times = []
            for _ in range(5):
                start = time.time()
                output = self.forward_layer_int4(x, cl_weights)
                times.append(time.time() - start)
                
            avg_time = np.mean(times)
            min_time = np.min(times)
            
            # Calculate performance
            tokens_per_sec = 1.0 / (min_time * 42)  # 42 layers
            speedup_vs_fp32 = 0.125 / min_time  # Assuming 125ms FP32 baseline
            
            logger.info(f"  Layer time: {min_time*1000:.1f}ms (avg: {avg_time*1000:.1f}ms)")
            logger.info(f"  Speed: {tokens_per_sec:.3f} tokens/sec")
            logger.info(f"  Speedup vs FP32: {speedup_vs_fp32:.1f}x")
            logger.info(f"  Component breakdown:")
            logger.info(f"    QKV: {self.stats['qkv_time']*1000:.1f}ms")
            logger.info(f"    Attention: {self.stats['attn_time']*1000:.1f}ms")
            logger.info(f"    FFN: {self.stats['ffn_time']*1000:.1f}ms")
            
            if tokens_per_sec >= 21.0:
                logger.info(f"  🎯 TARGET ACHIEVED!")
            elif speedup_vs_fp32 >= 5.0:
                logger.info(f"  🔥 Significant speedup! Getting close to target.")
                
        return tokens_per_sec


def main():
    """Test INT4 OpenCL implementation"""
    engine = MagicUnicornOpenCLINT4()
    engine.benchmark_int4_performance()


if __name__ == "__main__":
    main()