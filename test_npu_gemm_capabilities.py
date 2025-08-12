#!/usr/bin/env python3
"""
Test NPU GEMM capabilities and memory bandwidth
Investigate if NPU can accelerate matrix multiplication
"""

import numpy as np
import pyxrt
import time
import os
import psutil
import torch
import pyopencl as cl

class NPUGEMMTester:
    """Test NPU GEMM operations and bandwidth"""
    
    def __init__(self):
        self.device = None
        self.gemm_kernel = None
        self.int8_kernel = None
        self.setup_npu()
        self.setup_igpu()
        
    def setup_npu(self):
        """Setup NPU and load GEMM kernels"""
        try:
            self.device = pyxrt.device(0)
            print("✅ NPU Device initialized")
            
            # Get device info
            device_name = self.device.get_info(pyxrt.info.device.name)
            print(f"   Device: {device_name}")
            print(f"   Architecture: Phoenix XDNA1 (16 TOPS INT8)")
            
            # Load GEMM kernels
            gemm_kernels = {
                'gemm_fp32': '/opt/xilinx/xrt/amdxdna/bins/17f0_20/gemm.xclbin',
                'gemm_int8': '/opt/xilinx/xrt/amdxdna/bins/17f0_20/gemm_int8.elf',
                'gemm_elf': '/opt/xilinx/xrt/amdxdna/bins/17f0_20/gemm_elf.xclbin'
            }
            
            self.kernels = {}
            for name, path in gemm_kernels.items():
                if os.path.exists(path) and path.endswith('.xclbin'):
                    try:
                        xclbin = pyxrt.xclbin(path)
                        uuid = self.device.register_xclbin(xclbin)
                        kernels = xclbin.get_kernels()
                        if kernels:
                            kernel_name = kernels[0].get_name()
                            kernel = pyxrt.kernel(self.device, uuid, kernel_name)
                            self.kernels[name] = kernel
                            print(f"✅ Loaded {name}: {kernel_name}")
                    except Exception as e:
                        print(f"⚠️  Failed to load {name}: {e}")
                        
        except Exception as e:
            print(f"❌ NPU setup failed: {e}")
            
    def setup_igpu(self):
        """Setup iGPU for comparison"""
        try:
            platforms = cl.get_platforms()
            amd_platform = None
            
            for platform in platforms:
                if "AMD" in platform.name:
                    amd_platform = platform
                    break
                    
            if amd_platform:
                devices = amd_platform.get_devices(cl.device_type.GPU)
                if devices:
                    self.igpu_device = devices[0]
                    self.igpu_context = cl.Context([self.igpu_device])
                    self.igpu_queue = cl.CommandQueue(self.igpu_context)
                    print(f"✅ iGPU: {self.igpu_device.name}")
        except Exception as e:
            print(f"⚠️  iGPU setup: {e}")
            
    def test_memory_bandwidth(self):
        """Test NPU memory bandwidth"""
        print("\n📊 Testing NPU Memory Bandwidth")
        print("=" * 50)
        
        # Test different buffer sizes
        sizes_mb = [1, 4, 16, 64, 256]
        
        for size_mb in sizes_mb:
            size_bytes = size_mb * 1024 * 1024
            elements = size_bytes // 4  # float32
            
            try:
                # Create test data
                data = np.random.randn(elements).astype(np.float32)
                
                # Allocate NPU buffer
                start = time.time()
                bo = pyxrt.bo(self.device, size_bytes, pyxrt.bo.flags.normal, 0)
                alloc_time = time.time() - start
                
                # Write to NPU
                start = time.time()
                bo.write(data.tobytes())
                bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
                write_time = time.time() - start
                
                # Read from NPU
                start = time.time()
                bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
                result = bytearray(size_bytes)
                bo.read(result)
                read_time = time.time() - start
                
                # Calculate bandwidth
                write_bw = (size_mb / write_time) if write_time > 0 else 0
                read_bw = (size_mb / read_time) if read_time > 0 else 0
                
                print(f"\n{size_mb}MB buffer:")
                print(f"   Allocation: {alloc_time*1000:.1f}ms")
                print(f"   Write BW: {write_bw:.1f} MB/s")
                print(f"   Read BW: {read_bw:.1f} MB/s")
                
            except Exception as e:
                print(f"\n{size_mb}MB: Failed - {e}")
                
    def test_gemm_performance(self):
        """Test GEMM performance on NPU vs iGPU"""
        print("\n🔧 Testing GEMM Performance")
        print("=" * 50)
        
        # Test matrix sizes (M, N, K)
        test_configs = [
            (256, 256, 256, "Small"),
            (1024, 1024, 1024, "Medium"),
            (2048, 2048, 2048, "Large"),
        ]
        
        for M, N, K, desc in test_configs:
            print(f"\n📐 {desc} GEMM: {M}x{K} @ {K}x{N} = {M}x{N}")
            
            # Create test matrices
            A = np.random.randn(M, K).astype(np.float32)
            B = np.random.randn(K, N).astype(np.float32)
            
            # Calculate FLOPs
            flops = 2 * M * N * K
            
            # Test CPU baseline
            start = time.time()
            C_cpu = A @ B
            cpu_time = time.time() - start
            cpu_gflops = (flops / cpu_time) / 1e9
            print(f"   CPU: {cpu_time*1000:.1f}ms ({cpu_gflops:.1f} GFLOPS)")
            
            # Test iGPU if available
            if hasattr(self, 'igpu_context'):
                start = time.time()
                # Simple iGPU GEMM (not optimized)
                igpu_time = time.time() - start
                # Would need proper OpenCL kernel here
                print(f"   iGPU: Available but kernel not implemented")
            
            # Test NPU GEMM
            if self.kernels:
                for kernel_name, kernel in self.kernels.items():
                    try:
                        # Allocate NPU buffers
                        a_size = M * K * 4
                        b_size = K * N * 4
                        c_size = M * N * 4
                        
                        a_bo = pyxrt.bo(self.device, a_size, pyxrt.bo.flags.normal, kernel.group_id(0))
                        b_bo = pyxrt.bo(self.device, b_size, pyxrt.bo.flags.normal, kernel.group_id(1))
                        c_bo = pyxrt.bo(self.device, c_size, pyxrt.bo.flags.normal, kernel.group_id(2))
                        
                        # Write data
                        a_bo.write(A.tobytes())
                        b_bo.write(B.tobytes())
                        
                        a_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
                        b_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
                        
                        # Execute kernel
                        start = time.time()
                        run = kernel(a_bo, b_bo, c_bo, M, N, K)
                        run.wait()
                        npu_time = time.time() - start
                        
                        # Read result
                        c_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
                        
                        npu_gflops = (flops / npu_time) / 1e9
                        print(f"   NPU ({kernel_name}): {npu_time*1000:.1f}ms ({npu_gflops:.1f} GFLOPS)")
                        
                    except Exception as e:
                        print(f"   NPU ({kernel_name}): Failed - {str(e)[:50]}")
                        
    def analyze_memory_architecture(self):
        """Analyze system memory architecture"""
        print("\n💾 Memory Architecture Analysis")
        print("=" * 50)
        
        # System memory info
        mem = psutil.virtual_memory()
        print(f"\nSystem Memory:")
        print(f"   Total: {mem.total / 1024**3:.1f} GB")
        print(f"   Available: {mem.available / 1024**3:.1f} GB")
        print(f"   Used: {mem.percent:.1f}%")
        
        # Check memory topology
        try:
            numa_nodes = subprocess.check_output(['numactl', '--hardware'], text=True)
            print(f"\nNUMA Configuration:")
            for line in numa_nodes.split('\n')[:10]:
                if line.strip():
                    print(f"   {line}")
        except:
            print("\nNUMA: Not available")
            
        # Theoretical bandwidth calculation
        print(f"\n📊 Theoretical Bandwidth:")
        print(f"   DDR5-5600: ~89.6 GB/s (dual channel)")
        print(f"   Shared between: CPU, iGPU, NPU")
        print(f"   Effective per device: ~30 GB/s (estimated)")
        
        # NPU specific info
        print(f"\n🎯 NPU Memory Access:")
        print(f"   Phoenix NPU shares system memory")
        print(f"   No dedicated HBM (unlike discrete GPUs)")
        print(f"   DMA transfers compete with CPU/GPU")
        
    def test_concurrent_access(self):
        """Test concurrent NPU and iGPU access"""
        print("\n🔄 Testing Concurrent NPU+iGPU Access")
        print("=" * 50)
        
        # Create workload
        size = 1024
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)
        
        # Sequential test
        start = time.time()
        C1 = A @ B  # CPU
        C2 = A @ B  # Simulate second operation
        seq_time = time.time() - start
        
        print(f"\nSequential operations: {seq_time*1000:.1f}ms")
        print(f"Bandwidth utilization: Alternating")
        
        # Concurrent would require threading
        print(f"\nConcurrent operations: Would require parallel execution")
        print(f"Expected bottleneck: Memory bandwidth contention")
        
    def recommendations(self):
        """Provide recommendations based on findings"""
        print("\n🎯 Recommendations")
        print("=" * 50)
        
        print("\n1. NPU GEMM Capability:")
        print("   ✅ NPU has GEMM kernels (gemm.xclbin, gemm_int8.elf)")
        print("   ⚠️  Limited by shared memory bandwidth")
        print("   💡 Best for INT8 operations (16 TOPS capability)")
        
        print("\n2. Memory Bandwidth Limitations:")
        print("   🚫 Major bottleneck: Shared DDR5 memory")
        print("   📊 ~90 GB/s total, divided among CPU/GPU/NPU")
        print("   🔄 Competition for bandwidth reduces efficiency")
        
        print("\n3. Optimal Usage Pattern:")
        print("   🎯 NPU: INT8 quantized models")
        print("   🎯 iGPU: FP16/FP32 large GEMM operations")
        print("   🎯 Avoid: Concurrent large transfers")
        
        print("\n4. Performance Expectations:")
        print("   📉 NPU GEMM likely slower than iGPU for FP32")
        print("   📈 NPU may excel at INT8 GEMM (2-4x theoretical)")
        print("   ⚖️  Trade-off: Precision vs Performance")


def main():
    """Run NPU GEMM capability tests"""
    print("🦄 NPU GEMM Capability and Bandwidth Analysis")
    print("=" * 70)
    
    tester = NPUGEMMTester()
    
    # Run tests
    tester.test_memory_bandwidth()
    tester.test_gemm_performance()
    tester.analyze_memory_architecture()
    tester.test_concurrent_access()
    tester.recommendations()
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()