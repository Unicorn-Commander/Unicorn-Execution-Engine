#!/usr/bin/env python3.13
"""
Fixed MLP Fusion Test - Proper dimension handling for 2*ff_dim
"""

import numpy as np
import pyopencl as cl
import time
from pathlib import Path

def test_mlp_fusion_fixed():
    """Test the fixed MLP fusion kernels"""
    
    # Initialize OpenCL
    platforms = cl.get_platforms()
    gpu_devices = []
    for platform in platforms:
        devices = platform.get_devices(device_type=cl.device_type.GPU)
        gpu_devices.extend(devices)
    
    if not gpu_devices:
        raise RuntimeError("No GPU found!")
    
    ctx = cl.Context([gpu_devices[0]])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    
    print(f"Using GPU: {gpu_devices[0].name}")
    
    # Load kernels
    kernel_path = Path("/home/ucadmin/Development/Unicorn-Execution-Engine/kernels/phase1/mlp_fusion_fixed.cl")
    with open(kernel_path, 'r') as f:
        kernel_source = f.read()
    
    build_options = "-cl-std=CL2.0 -cl-fast-relaxed-math -cl-mad-enable"
    program = cl.Program(ctx, kernel_source).build(build_options)
    
    # Test dimensions
    test_configs = [
        {"name": "Small", "M": 128, "hidden_size": 256, "ff_dim": 1024},
        {"name": "4B-like", "M": 128, "hidden_size": 2560, "ff_dim": 10240},
    ]
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing {config['name']} configuration:")
        print(f"M={config['M']}, hidden_size={config['hidden_size']}, ff_dim={config['ff_dim']}")
        
        M = config['M']
        hidden_size = config['hidden_size']
        ff_dim = config['ff_dim']
        
        # Create test data
        np.random.seed(42)
        input_data = np.random.randn(M, hidden_size).astype(np.float32)
        W_gate = np.random.randn(hidden_size, ff_dim).astype(np.float32) * 0.02
        W_up = np.random.randn(hidden_size, ff_dim).astype(np.float32) * 0.02
        W_down = np.random.randn(ff_dim, hidden_size).astype(np.float32) * 0.02
        
        # CPU reference
        print("\nComputing CPU reference...")
        gate_ref = np.dot(input_data, W_gate)
        up_ref = np.dot(input_data, W_up)
        
        # GELU approximation
        sigmoid = 1.0 / (1.0 + np.exp(-1.702 * gate_ref))
        gelu_gate_ref = gate_ref * sigmoid
        
        # Element-wise multiply and down projection
        intermediate_ref = gelu_gate_ref * up_ref
        output_ref = np.dot(intermediate_ref, W_down)
        
        # Test three-kernel approach
        print("\nTesting three-kernel approach...")
        
        # Create buffers
        mf = cl.mem_flags
        input_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_data)
        W_gate_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=W_gate)
        W_up_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=W_up)
        W_down_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=W_down)
        
        # Intermediate buffers
        gate_up_buf = cl.Buffer(ctx, mf.READ_WRITE, size=M * 2 * ff_dim * 4)  # 2*ff_dim
        intermediate_buf = cl.Buffer(ctx, mf.READ_WRITE, size=M * ff_dim * 4)
        output_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=M * hidden_size * 4)
        
        # Kernel 1: Gate + Up projection
        print("  Running gate_up_fused_fixed...")
        global_size = (M, 2 * ff_dim)
        local_size = None
        
        event1 = program.gate_up_fused_fixed(
            queue, global_size, local_size,
            input_buf, W_gate_buf, W_up_buf, gate_up_buf,
            np.int32(M), np.int32(hidden_size), np.int32(ff_dim)
        )
        event1.wait()
        
        # Verify intermediate result
        gate_up_result = np.empty((M, 2 * ff_dim), dtype=np.float32)
        cl.enqueue_copy(queue, gate_up_result, gate_up_buf)
        
        print(f"  Gate+Up output shape: {gate_up_result.shape}")
        print(f"  Gate values (first 5): {gate_up_result[0, :5]}")
        print(f"  Up values (first 5): {gate_up_result[0, ff_dim:ff_dim+5]}")
        
        # Kernel 2: GELU + Multiply
        print("  Running gelu_multiply_fixed...")
        global_size = (M * ff_dim,)
        local_size = None
        
        event2 = program.gelu_multiply_fixed(
            queue, global_size, local_size,
            gate_up_buf, intermediate_buf,
            np.int32(M), np.int32(ff_dim)
        )
        event2.wait()
        
        # Kernel 3: Down projection
        print("  Running down_projection_fixed...")
        global_size = (M, hidden_size)
        
        event3 = program.down_projection_fixed(
            queue, global_size, local_size,
            intermediate_buf, W_down_buf, output_buf,
            np.int32(M), np.int32(ff_dim), np.int32(hidden_size)
        )
        event3.wait()
        
        # Get result
        output_gpu = np.empty((M, hidden_size), dtype=np.float32)
        cl.enqueue_copy(queue, output_gpu, output_buf)
        
        # Verify correctness
        max_error = np.max(np.abs(output_gpu - output_ref))
        mean_error = np.mean(np.abs(output_gpu - output_ref))
        print(f"\nCorrectness check (three-kernel):")
        print(f"  Max error: {max_error:.6e}")
        print(f"  Mean error: {mean_error:.6e}")
        print(f"  Pass: {max_error < 1e-3}")
        
        # Test single-kernel approach (if LDS permits)
        # lds_required = 2 * 16 * 16 * 4  # Rough estimate
        # if lds_required < gpu_devices[0].local_mem_size:
        #     print("\nTesting single-kernel approach...")
            
        #     output_buf_single = cl.Buffer(ctx, mf.WRITE_ONLY, size=M * hidden_size * 4)
            
        #     global_size = (M, hidden_size)
        #     local_size = (16, 16) if M >= 16 and hidden_size >= 16 else None
            
        #     event_single = program.mlp_fused_single(
        #         queue, global_size, local_size,
        #         input_buf, W_gate_buf, W_up_buf, W_down_buf, output_buf_single,
        #         cl.LocalMemory(lds_required),
        #         np.int32(M), np.int32(hidden_size), np.int32(ff_dim)
        #     )
        #     event_single.wait()
            
        #     output_single = np.empty((M, hidden_size), dtype=np.float32)
        #     cl.enqueue_copy(queue, output_single, output_buf_single)
            
        #     max_error_single = np.max(np.abs(output_single - output_ref))
        #     print(f"  Single kernel max error: {max_error_single:.6e}")
        #     print(f"  Single kernel pass: {max_error_single < 1e-3}")
        
        # Timing comparison
        if max_error < 1e-3:
            print("\n📊 Performance comparison:")
            
            # Time three-kernel approach
            iterations = 20
            queue.finish()
            
            start = time.time()
            for _ in range(iterations):
                program.gate_up_fused_fixed(queue, (M, 2*ff_dim), None,
                    input_buf, W_gate_buf, W_up_buf, gate_up_buf,
                    np.int32(M), np.int32(hidden_size), np.int32(ff_dim))
                program.gelu_multiply_fixed(queue, (M*ff_dim,), None,
                    gate_up_buf, intermediate_buf,
                    np.int32(M), np.int32(ff_dim))
                program.down_projection_fixed(queue, (M, hidden_size), None,
                    intermediate_buf, W_down_buf, output_buf,
                    np.int32(M), np.int32(ff_dim), np.int32(hidden_size))
            queue.finish()
            
            three_kernel_time = (time.time() - start) / iterations * 1000
            
            print(f"  Three-kernel time: {three_kernel_time:.2f} ms")
            
            # Calculate GFLOPS
            ops = 2 * M * hidden_size * ff_dim * 2 + 2 * M * ff_dim * hidden_size  # Approximate
            gflops = ops / (three_kernel_time / 1000) / 1e9
            print(f"  Effective GFLOPS: {gflops:.1f}")

if __name__ == "__main__":
    test_mlp_fusion_fixed()