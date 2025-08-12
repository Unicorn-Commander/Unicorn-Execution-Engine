#!/usr/bin/env python3.13
"""
Pipeline Debug Helper - Isolate segfault in MLP fusion
"""

import numpy as np
import pyopencl as cl
import time
import traceback
from pathlib import Path

class PipelineDebugger:
    """Debug OpenCL pipeline segfaults"""
    
    def __init__(self):
        # Initialize OpenCL
        platforms = cl.get_platforms()
        gpu_devices = []
        for platform in platforms:
            devices = platform.get_devices(device_type=cl.device_type.GPU)
            gpu_devices.extend(devices)
        
        if not gpu_devices:
            raise RuntimeError("No GPU found!")
        
        self.device = gpu_devices[0]
        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.ctx, 
            properties=cl.command_queue_properties.PROFILING_ENABLE)
        
        print(f"🔧 Debug Helper - GPU: {self.device.name}")
        print(f"   Max Memory: {self.device.global_mem_size / 1024**3:.1f} GB")
        print(f"   Local Memory: {self.device.local_mem_size / 1024:.1f} KB")
    
    def verify_buffer(self, buf, expected_size, name):
        """Verify a buffer can be read safely"""
        try:
            test_data = np.zeros(expected_size, dtype=np.float32)
            cl.enqueue_copy(self.queue, test_data, buf)
            self.queue.finish()
            print(f"✓ {name} buffer readable ({expected_size} elements)")
            return True
        except Exception as e:
            print(f"✗ {name} buffer error: {e}")
            return False
    
    def test_safe_mlp(self):
        """Test the safe MLP implementation"""
        print("\n" + "="*50)
        print("Testing Safe MLP Implementation")
        
        # Load safe kernels
        kernel_path = Path("/home/ucadmin/Development/Unicorn-Execution-Engine/kernels/phase1/mlp_safe.cl")
        with open(kernel_path, 'r') as f:
            kernel_source = f.read()
        
        build_options = "-cl-std=CL2.0"
        program = cl.Program(self.ctx, kernel_source).build(build_options)
        
        # Test dimensions - start small
        M = 1
        hidden_size = 16
        ff_dim = 32
        
        print(f"Dimensions: M={M}, hidden_size={hidden_size}, ff_dim={ff_dim}")
        
        # Create test data
        np.random.seed(42)
        input_data = np.random.randn(M, hidden_size).astype(np.float32)
        W_gate = np.random.randn(hidden_size, ff_dim).astype(np.float32) * 0.02
        W_up = np.random.randn(hidden_size, ff_dim).astype(np.float32) * 0.02
        W_down = np.random.randn(ff_dim, hidden_size).astype(np.float32) * 0.02
        
        # Create buffers
        mf = cl.mem_flags
        input_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_data)
        W_gate_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=W_gate)
        W_up_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=W_up)
        W_down_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=W_down)
        output_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=M * hidden_size * 4)
        
        # Verify all buffers
        self.verify_buffer(input_buf, M * hidden_size, "input")
        self.verify_buffer(W_gate_buf, hidden_size * ff_dim, "W_gate")
        self.verify_buffer(W_up_buf, hidden_size * ff_dim, "W_up")
        self.verify_buffer(W_down_buf, ff_dim * hidden_size, "W_down")
        
        print("\n1️⃣ Testing debug kernel...")
        try:
            program.debug_mlp_memory(
                self.queue, (1,), None,
                input_buf, W_gate_buf, W_up_buf, W_down_buf, output_buf,
                np.int32(M), np.int32(hidden_size), np.int32(ff_dim)
            )
            self.queue.finish()
            print("✓ Debug kernel passed")
        except Exception as e:
            print(f"✗ Debug kernel failed: {e}")
            return False
        
        print("\n2️⃣ Testing safe single kernel...")
        try:
            global_size = (M * hidden_size,)
            program.mlp_safe_single(
                self.queue, global_size, None,
                input_buf, W_gate_buf, W_up_buf, W_down_buf, output_buf,
                np.int32(M), np.int32(hidden_size), np.int32(ff_dim)
            )
            self.queue.finish()
            
            # Get result
            output_gpu = np.empty((M, hidden_size), dtype=np.float32)
            cl.enqueue_copy(self.queue, output_gpu, output_buf)
            
            print("✓ Safe single kernel passed")
            print(f"  Output shape: {output_gpu.shape}")
            print(f"  Output range: [{output_gpu.min():.3f}, {output_gpu.max():.3f}]")
            return True
            
        except Exception as e:
            print(f"✗ Safe single kernel failed: {e}")
            traceback.print_exc()
            return False
    
    def test_three_kernel_safe(self):
        """Test the conservative three-kernel approach"""
        print("\n" + "="*50)
        print("Testing Safe Three-Kernel Approach")
        
        # Dimensions
        M = 1
        hidden_size = 16
        ff_dim = 32
        
        # Load kernels
        kernel_path = Path("/home/ucadmin/Development/Unicorn-Execution-Engine/kernels/phase1/mlp_safe.cl")
        with open(kernel_path, 'r') as f:
            kernel_source = f.read()
        
        program = cl.Program(self.ctx, kernel_source).build()
        
        # Create test data
        np.random.seed(42)
        input_data = np.random.randn(M, hidden_size).astype(np.float32)
        W_gate = np.random.randn(hidden_size, ff_dim).astype(np.float32) * 0.02
        W_up = np.random.randn(hidden_size, ff_dim).astype(np.float32) * 0.02
        W_down = np.random.randn(ff_dim, hidden_size).astype(np.float32) * 0.02
        
        # Create buffers with explicit size calculation
        mf = cl.mem_flags
        input_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_data)
        W_gate_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=W_gate)
        W_up_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=W_up)
        W_down_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=W_down)
        
        # Separate buffers for gate and up (avoid 2*ff_dim complexity)
        gate_buf = cl.Buffer(self.ctx, mf.READ_WRITE, size=M * ff_dim * 4)
        up_buf = cl.Buffer(self.ctx, mf.READ_WRITE, size=M * ff_dim * 4)
        intermediate_buf = cl.Buffer(self.ctx, mf.READ_WRITE, size=M * ff_dim * 4)
        output_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=M * hidden_size * 4)
        
        try:
            print("1️⃣ Gate and Up projections...")
            program.gate_up_safe(
                self.queue, (M, ff_dim), None,
                input_buf, W_gate_buf, W_up_buf, gate_buf, up_buf,
                np.int32(M), np.int32(hidden_size), np.int32(ff_dim)
            )
            self.queue.finish()
            print("✓ Gate+Up kernel passed")
            
            print("2️⃣ GELU multiply...")
            program.gelu_multiply_safe(
                self.queue, (M * ff_dim,), None,
                gate_buf, up_buf, intermediate_buf,
                np.int32(M), np.int32(ff_dim)
            )
            self.queue.finish()
            print("✓ GELU multiply kernel passed")
            
            print("3️⃣ Down projection...")
            program.down_projection_safe(
                self.queue, (M, hidden_size), None,
                intermediate_buf, W_down_buf, output_buf,
                np.int32(M), np.int32(ff_dim), np.int32(hidden_size)
            )
            self.queue.finish()
            print("✓ Down projection kernel passed")
            
            # Get final result
            output_gpu = np.empty((M, hidden_size), dtype=np.float32)
            cl.enqueue_copy(self.queue, output_gpu, output_buf)
            
            print("✓ Three-kernel approach completed successfully")
            print(f"  Output shape: {output_gpu.shape}")
            print(f"  Output range: [{output_gpu.min():.3f}, {output_gpu.max():.3f}]")
            
            return True
            
        except Exception as e:
            print(f"✗ Three-kernel approach failed: {e}")
            traceback.print_exc()
            return False
    
    def compare_with_pipeline_conditions(self):
        """Test under conditions similar to the full pipeline"""
        print("\n" + "="*50)
        print("Testing Pipeline-Like Conditions")
        
        # Use larger, more realistic dimensions
        M = 128  # Typical sequence length
        hidden_size = 2560  # 4B model
        ff_dim = 10240  # 4B model
        
        print(f"Large dimensions: M={M}, hidden_size={hidden_size}, ff_dim={ff_dim}")
        
        # Calculate memory usage
        input_size = M * hidden_size * 4
        weights_size = (hidden_size * ff_dim * 3 + ff_dim * hidden_size) * 4
        buffers_size = M * (2 * ff_dim + ff_dim + hidden_size) * 4
        total_mb = (input_size + weights_size + buffers_size) / 1024 / 1024
        
        print(f"Memory usage: {total_mb:.1f} MB")
        
        if total_mb > self.device.global_mem_size / 1024 / 1024 * 0.8:
            print("⚠️  Memory usage might be too high, reducing dimensions...")
            M = 64
            hidden_size = 1280
            ff_dim = 5120
        
        # Test the safe single kernel with large dimensions
        return self.test_safe_mlp_large(M, hidden_size, ff_dim)
    
    def test_safe_mlp_large(self, M, hidden_size, ff_dim):
        """Test safe MLP with larger dimensions"""
        kernel_path = Path("/home/ucadmin/Development/Unicorn-Execution-Engine/kernels/phase1/mlp_safe.cl")
        with open(kernel_path, 'r') as f:
            kernel_source = f.read()
        
        program = cl.Program(self.ctx, kernel_source).build()
        
        # Create test data
        np.random.seed(42)
        input_data = np.random.randn(M, hidden_size).astype(np.float32)
        W_gate = np.random.randn(hidden_size, ff_dim).astype(np.float32) * 0.02
        W_up = np.random.randn(hidden_size, ff_dim).astype(np.float32) * 0.02
        W_down = np.random.randn(ff_dim, hidden_size).astype(np.float32) * 0.02
        
        # Create buffers
        mf = cl.mem_flags
        input_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_data)
        W_gate_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=W_gate)
        W_up_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=W_up)
        W_down_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=W_down)
        output_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=M * hidden_size * 4)
        
        try:
            print("Testing with large dimensions...")
            global_size = (M * hidden_size,)
            
            program.mlp_safe_single(
                self.queue, global_size, None,
                input_buf, W_gate_buf, W_up_buf, W_down_buf, output_buf,
                np.int32(M), np.int32(hidden_size), np.int32(ff_dim)
            )
            self.queue.finish()
            
            output_gpu = np.empty((M, hidden_size), dtype=np.float32)
            cl.enqueue_copy(self.queue, output_gpu, output_buf)
            
            print("✓ Large dimension test passed")
            return True
            
        except Exception as e:
            print(f"✗ Large dimension test failed: {e}")
            return False

def main():
    """Run comprehensive MLP debugging"""
    debugger = PipelineDebugger()
    
    print("🔍 MLP Fusion Segfault Debugging")
    print("=" * 60)
    
    # Test 1: Safe implementation with small dimensions
    success1 = debugger.test_safe_mlp()
    
    # Test 2: Three-kernel safe approach
    success2 = debugger.test_three_kernel_safe()
    
    # Test 3: Pipeline-like conditions
    success3 = debugger.compare_with_pipeline_conditions()
    
    print("\n" + "="*60)
    print("🏁 Debug Summary:")
    print(f"   Safe single kernel (small): {'✓' if success1 else '✗'}")
    print(f"   Safe three kernels: {'✓' if success2 else '✗'}")
    print(f"   Large dimensions: {'✓' if success3 else '✗'}")
    
    if all([success1, success2, success3]):
        print("\n✅ All safe implementations work!")
        print("   → Use mlp_safe.cl kernels in your pipeline")
        print("   → The issue is likely in the original kernel indexing")
    else:
        print("\n❌ Still having issues...")
        print("   → The problem might be deeper (GPU driver, memory)")

if __name__ == "__main__":
    main()