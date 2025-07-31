#!/usr/bin/env python3
"""
Create NPU attention kernel using mlir-aie Python API
"""

import os
import sys
import argparse
import numpy as np

# Add mlir-aie to Python path
sys.path.insert(0, '/home/ucadmin/npu-dev/mlir-aie/python')

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *

def create_attention_kernel(seq_len=256, hidden_dim=2560, num_heads=20, head_dim=128):
    """Create attention kernel MLIR using Python API"""
    
    with mlir_mod_ctx() as ctx:
        @device(AIEDevice.npu1_4col)
        def attention_device():
            # Use simpler matrix multiplication approach instead of full attention
            # This follows the pattern from mlir-aie matrix multiplication examples
            
            # Memory tiles
            memtile_0_1 = tile(0, 1)
            memtile_1_1 = tile(1, 1)
            
            # Compute tiles
            core_0_2 = tile(0, 2)
            
            # Use standard matrix sizes that work with NPU
            M = min(seq_len, 256)  # Limit to avoid memory issues
            K = min(head_dim, 128)
            N = min(head_dim, 128)
            
            # Define matrix types
            a_ty = np.ndarray[(M, K), np.dtype[np.float32]]
            b_ty = np.ndarray[(K, N), np.dtype[np.float32]]
            c_ty = np.ndarray[(M, N), np.dtype[np.float32]]
            
            # External function declarations (these would be provided as kernel objects)
            zero_func = external_func("zero_f32", inputs=[c_ty])
            matmul_func = external_func("matmul_f32_f32", inputs=[a_ty, b_ty, c_ty])
            
            # Shim tile for host interface
            shim_tile = tile(0, 0)
            
            # Object FIFOs for data movement
            inA = object_fifo("inA", shim_tile, memtile_0_1, 2, a_ty)
            memA = object_fifo("memA", memtile_0_1, core_0_2, 2, a_ty)
            object_fifo_link(inA, memA)
            
            inB = object_fifo("inB", shim_tile, memtile_1_1, 2, b_ty)
            memB = object_fifo("memB", memtile_1_1, core_0_2, 2, b_ty)
            object_fifo_link(inB, memB)
            
            memC = object_fifo("memC", core_0_2, memtile_0_1, 2, c_ty)
            outC = object_fifo("outC", memtile_0_1, shim_tile, 2, c_ty)
            object_fifo_link(memC, outC)
            
            # Core computation
            @core(core_0_2, f"gemma3_attention_{M}x{K}x{N}.o")
            def attention_core():
                # Initialize output
                elem_out = memC.acquire(ObjectFifoPort.Produce, 1)
                zero_func(elem_out)
                
                # Compute matrix multiplication (simplified attention)
                elem_in_a = memA.acquire(ObjectFifoPort.Consume, 1)
                elem_in_b = memB.acquire(ObjectFifoPort.Consume, 1)
                matmul_func(elem_in_a, elem_in_b, elem_out)
                memA.release(ObjectFifoPort.Consume, 1)
                memB.release(ObjectFifoPort.Consume, 1)
                
                memC.release(ObjectFifoPort.Produce, 1)
            
            # Runtime sequence for DMA operations
            @runtime_sequence(
                np.ndarray[(M*K,), np.dtype[np.float32]],
                np.ndarray[(K*N,), np.dtype[np.float32]],
                np.ndarray[(M*N,), np.dtype[np.float32]]
            )
            def sequence(A, B, C):
                # DMA operations
                npu_dma_memcpy_nd(
                    metadata=inA,
                    bd_id=0,
                    mem=A,
                    sizes=[1, 1, M, K],
                    strides=[M*K, M*K, K, 1]
                )
                npu_dma_memcpy_nd(
                    metadata=inB,
                    bd_id=1,
                    mem=B,
                    sizes=[1, 1, K, N],
                    strides=[K*N, K*N, N, 1]
                )
                npu_dma_memcpy_nd(
                    metadata=outC,
                    bd_id=2,
                    mem=C,
                    sizes=[1, 1, M, N],
                    strides=[M*N, M*N, N, 1]
                )
                dma_wait(outC)
    
    return ctx.module

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=2560)
    parser.add_argument("--num-heads", type=int, default=20)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--output", type=str, default="attention_kernel.mlir")
    
    args = parser.parse_args()
    
    print(f"🚀 Creating NPU attention kernel:")
    print(f"   Sequence length: {args.seq_len}")
    print(f"   Hidden dimension: {args.hidden_dim}")
    print(f"   Number of heads: {args.num_heads}")
    print(f"   Head dimension: {args.head_dim}")
    
    # Create kernel
    module = create_attention_kernel(
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        head_dim=args.head_dim
    )
    
    # Write MLIR output
    with open(args.output, 'w') as f:
        f.write(str(module))
    
    print(f"✅ MLIR written to: {args.output}")

if __name__ == "__main__":
    main()