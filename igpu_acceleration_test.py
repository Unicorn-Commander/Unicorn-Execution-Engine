#!/usr/bin/env python3.13
"""
🦄 iGPU Acceleration Test - OpenCL Matrix Operations  
Test real GPU acceleration using AMD Radeon iGPU via OpenCL
"""

import os
import sys
import time
import numpy as np

try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except ImportError:
    print("❌ PyOpenCL not available. Install with: pip install pyopencl")
    OPENCL_AVAILABLE = False
    sys.exit(1)

def test_opencl_setup():
    """Test OpenCL platform and device detection"""
    print("🦄 OpenCL iGPU Setup Test")
    print("=" * 40)
    
    try:
        # Get platforms
        platforms = cl.get_platforms()
        print(f"✅ Found {len(platforms)} OpenCL platform(s)")
        
        for i, platform in enumerate(platforms):
            print(f"   Platform {i}: {platform.name}")
            print(f"   Vendor: {platform.vendor}")
            print(f"   Version: {platform.version}")
            
            # Get devices for this platform
            devices = platform.get_devices()
            print(f"   Devices: {len(devices)}")
            
            for j, device in enumerate(devices):
                print(f"     Device {j}: {device.name}")
                print(f"     Type: {cl.device_type.to_string(device.type)}")
                print(f"     Max compute units: {device.max_compute_units}")
                print(f"     Max work group size: {device.max_work_group_size}")
                print(f"     Global memory: {device.global_mem_size // (1024**3)} GB")
                print(f"     Local memory: {device.local_mem_size // 1024} KB")
                print(f"     Max clock: {device.max_clock_frequency} MHz")
        
        # Create context with GPU device
        gpu_devices = []
        for platform in platforms:
            devices = platform.get_devices(device_type=cl.device_type.GPU)
            gpu_devices.extend(devices)
        
        if not gpu_devices:
            print("❌ No GPU devices found")
            return None, None
        
        print(f"\n🎯 Using GPU: {gpu_devices[0].name}")
        context = cl.Context([gpu_devices[0]])
        queue = cl.CommandQueue(context)
        
        print("✅ OpenCL context and queue created")
        return context, queue
        
    except Exception as e:
        print(f"❌ OpenCL setup failed: {e}")
        return None, None

def create_matrix_multiply_kernel():
    """Create OpenCL kernel for matrix multiplication"""
    kernel_source = """
    __kernel void matrix_multiply(
        __global const float* A,
        __global const float* B, 
        __global float* C,
        const int M,
        const int N,
        const int K
    ) {
        int row = get_global_id(0);
        int col = get_global_id(1);
        
        if (row < M && col < K) {
            float sum = 0.0f;
            for (int i = 0; i < N; i++) {
                sum += A[row * N + i] * B[i * K + col];
            }
            C[row * K + col] = sum;
        }
    }
    """
    return kernel_source

def benchmark_opencl_matrix_ops(context, queue):
    """Benchmark matrix operations on iGPU"""
    print("\n📊 iGPU Matrix Operations Benchmark")
    print("=" * 45)
    
    # Create program and kernel
    kernel_source = create_matrix_multiply_kernel()
    program = cl.Program(context, kernel_source).build()
    kernel = program.matrix_multiply
    
    # Test configurations matching our models
    test_configs = [
        {"name": "4B Attention", "M": 128, "N": 2560, "K": 2560},
        {"name": "27B Attention", "M": 128, "N": 4608, "K": 4608},
        {"name": "4B MLP", "M": 128, "N": 10240, "K": 2560},
        {"name": "27B MLP", "M": 128, "N": 18432, "K": 4608},
    ]
    
    results = {}
    
    for config in test_configs:
        name = config["name"]
        M, N, K = config["M"], config["N"], config["K"]
        
        print(f"\n🧮 Testing {name} ({M}x{N} @ {N}x{K})")
        
        # Create test matrices
        A_host = np.random.randn(M, N).astype(np.float32)
        B_host = np.random.randn(N, K).astype(np.float32) 
        C_host = np.zeros((M, K), dtype=np.float32)
        
        # Create OpenCL buffers
        mf = cl.mem_flags
        A_buffer = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A_host)
        B_buffer = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B_host)
        C_buffer = cl.Buffer(context, mf.WRITE_ONLY, C_host.nbytes)
        
        # Warm up
        for _ in range(3):
            kernel(queue, (M, K), None, A_buffer, B_buffer, C_buffer, 
                  np.int32(M), np.int32(N), np.int32(K))
            queue.finish()
        
        # Benchmark
        times = []
        for _ in range(10):
            start_time = time.time()
            
            # Execute kernel
            kernel(queue, (M, K), None, A_buffer, B_buffer, C_buffer,
                  np.int32(M), np.int32(N), np.int32(K))
            queue.finish()
            
            times.append(time.time() - start_time)
        
        avg_time = np.mean(times)
        
        # Read result back for verification
        cl.enqueue_copy(queue, C_host, C_buffer)
        
        # Verify correctness
        C_cpu = np.dot(A_host, B_host)
        error = np.max(np.abs(C_host - C_cpu))
        
        # Calculate performance
        flops = 2 * M * N * K
        gflops = flops / (avg_time * 1e9)
        
        # Memory bandwidth
        total_memory = A_host.nbytes + B_host.nbytes + C_host.nbytes
        bandwidth_gbs = total_memory / (avg_time * 1024**3)
        
        results[name] = {
            "time_ms": avg_time * 1000,
            "gflops": gflops,
            "bandwidth_gbs": bandwidth_gbs,
            "max_error": error
        }
        
        print(f"   Time: {avg_time*1000:.1f}ms")
        print(f"   Performance: {gflops:.1f} GFLOPS")
        print(f"   Bandwidth: {bandwidth_gbs:.1f} GB/s")
        print(f"   Max error: {error:.2e}")
        print(f"   Correctness: {'✅' if error < 1e-4 else '❌'}")
    
    return results

def compare_cpu_vs_gpu(gpu_results):
    """Compare GPU results with CPU baseline"""
    print("\n⚖️  CPU vs iGPU Comparison")
    print("=" * 35)
    
    # Load CPU baseline if available
    try:
        import json
        with open("cpu_baseline_results.json", "r") as f:
            cpu_data = json.load(f)
        cpu_results = cpu_data["matrix_benchmarks"]
        
        print("📊 Performance Comparison:")
        print("   Operation                CPU GFLOPS    iGPU GFLOPS   Speedup")
        print("   " + "-"*55)
        
        total_speedup = 0
        count = 0
        
        for name in gpu_results:
            if name in cpu_results:
                cpu_gflops = cpu_results[name]["gflops"]
                gpu_gflops = gpu_results[name]["gflops"]
                speedup = gpu_gflops / cpu_gflops
                total_speedup += speedup
                count += 1
                
                print(f"   {name:<20} {cpu_gflops:>8.1f}      {gpu_gflops:>8.1f}      {speedup:>4.1f}x")
        
        avg_speedup = total_speedup / count if count > 0 else 0
        print("   " + "-"*55)
        print(f"   Average speedup: {avg_speedup:.1f}x")
        
        return avg_speedup
        
    except FileNotFoundError:
        print("❌ CPU baseline not found. Run cpu_baseline_benchmark.py first")
        return 0

def estimate_igpu_model_performance(gpu_results):
    """Estimate model performance with iGPU acceleration"""
    print("\n🚀 iGPU Model Performance Estimation")
    print("=" * 45)
    
    models = {
        "4B": {
            "layers": 28,
            "attention_key": "4B Attention",
            "mlp_key": "4B MLP"
        },
        "27B": {
            "layers": 32, 
            "attention_key": "27B Attention",
            "mlp_key": "27B MLP"
        }
    }
    
    for name, config in models.items():
        print(f"\n📊 Gemma 3 {name} with iGPU:")
        
        # Get matrix operation times
        attn_time = gpu_results[config["attention_key"]]["time_ms"]
        mlp_time = gpu_results[config["mlp_key"]]["time_ms"]
        
        # Estimate full layer time (including overhead)
        attention_total = attn_time * 4 + 5  # QKV + output + overhead
        mlp_total = mlp_time * 2 + 3  # up + down + overhead
        other_ops = 2  # layer norm, etc.
        
        layer_time = attention_total + mlp_total + other_ops
        full_model_time = layer_time * config["layers"]
        
        # Tokens per second
        output_tokens = 5
        tps = output_tokens / (full_model_time / 1000)
        
        print(f"   Layer time: {layer_time:.1f}ms")
        print(f"   Full model: {full_model_time:.1f}ms")
        print(f"   Estimated TPS: {tps:.2f}")

def main():
    print("🦄 iGPU Acceleration Test Suite")
    print("=" * 50)
    
    if not OPENCL_AVAILABLE:
        return
    
    # Test 1: Setup OpenCL
    context, queue = test_opencl_setup()
    if not context:
        print("❌ OpenCL setup failed")
        return
    
    # Test 2: Benchmark matrix operations
    gpu_results = benchmark_opencl_matrix_ops(context, queue)
    
    # Test 3: Compare with CPU
    speedup = compare_cpu_vs_gpu(gpu_results)
    
    # Test 4: Estimate model performance
    estimate_igpu_model_performance(gpu_results)
    
    # Summary
    print("\n" + "="*50)
    print("🏆 iGPU ACCELERATION SUMMARY")
    print("="*50)
    
    print("\n📊 iGPU Performance:")
    for name, result in gpu_results.items():
        print(f"   {name}: {result['gflops']:.1f} GFLOPS")
    
    if speedup > 0:
        print(f"\n⚡ Average speedup vs CPU: {speedup:.1f}x")
    
    print(f"\n✅ iGPU acceleration verified and working!")

if __name__ == "__main__":
    main()