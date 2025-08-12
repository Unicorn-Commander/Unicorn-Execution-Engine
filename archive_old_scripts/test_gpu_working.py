#!/usr/bin/env python3
"""
Test GPU Working - Verify GPU is actually being used
Monitor GPU usage while running compute
"""

import numpy as np
import time
import logging
import subprocess
import threading

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPUMonitor:
    """Monitor GPU usage in background"""
    def __init__(self):
        self.running = False
        self.max_gpu = 0.0
        self.max_vram = 0.0
        self.samples = []
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
        
    def stop(self):
        self.running = False
        self.thread.join()
        return self.max_gpu, self.max_vram, self.samples
        
    def _monitor(self):
        while self.running:
            try:
                result = subprocess.run(['radeontop', '-d', '-', '-l', '1'], 
                                      capture_output=True, text=True, timeout=0.5)
                if result.stdout:
                    import re
                    gpu_match = re.search(r'gpu\s+(\d+\.\d+)%', result.stdout)
                    vram_match = re.search(r'vram\s+(\d+\.\d+)%\s+(\d+\.\d+)mb', result.stdout)
                    
                    if gpu_match:
                        gpu_pct = float(gpu_match.group(1))
                        self.max_gpu = max(self.max_gpu, gpu_pct)
                        
                    if vram_match:
                        vram_pct = float(vram_match.group(1))
                        vram_mb = float(vram_match.group(2))
                        self.max_vram = max(self.max_vram, vram_mb)
                        
                    self.samples.append((gpu_pct if gpu_match else 0, vram_mb if vram_match else 0))
            except:
                pass
            time.sleep(0.1)

def test_gpu_compute():
    """Test if GPU compute is actually working"""
    logger.info("🧪 TESTING GPU COMPUTE - MONITORING USAGE")
    
    # Initialize Vulkan
    from real_vulkan_matrix_compute import VulkanMatrixCompute
    
    vulkan = VulkanMatrixCompute()
    if not vulkan.initialize():
        logger.error("❌ Failed to initialize Vulkan")
        return False
    
    logger.info("✅ Vulkan initialized successfully")
    
    # Test 1: Matrix multiplication
    logger.info("\n📊 TEST 1: Matrix Multiplication")
    size = 2048
    
    A = np.random.randn(size, size).astype(np.float32)
    B = np.random.randn(size, size).astype(np.float32)
    
    # Start GPU monitoring
    monitor = GPUMonitor()
    monitor.start()
    
    logger.info(f"   Running {size}x{size} matrix multiply on GPU...")
    start_time = time.time()
    
    # Run multiple times to see sustained usage
    for i in range(10):
        result = vulkan.compute_matrix_multiply(A, B)
        if i == 0:
            logger.info(f"   Result shape: {result.shape}")
    
    gpu_time = time.time() - start_time
    
    # Stop monitoring
    max_gpu, max_vram, samples = monitor.stop()
    
    # CPU comparison
    logger.info("   Running on CPU for comparison...")
    cpu_start = time.time()
    cpu_result = np.matmul(A, B)
    cpu_time = time.time() - cpu_start
    
    # Results
    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
    error = np.mean(np.abs(result - cpu_result))
    
    logger.info(f"\n   ✅ TEST 1 RESULTS:")
    logger.info(f"      GPU time: {gpu_time:.3f}s")
    logger.info(f"      CPU time: {cpu_time:.3f}s")
    logger.info(f"      Speedup: {speedup:.2f}x")
    logger.info(f"      Max GPU usage: {max_gpu:.1f}%")
    logger.info(f"      Max VRAM: {max_vram:.1f} MB")
    logger.info(f"      Numerical error: {error:.6f}")
    
    gpu_working = max_gpu > 10.0  # Should see >10% GPU usage
    
    # Test 2: Transformer-like operations
    logger.info("\n📊 TEST 2: Transformer Operations (Q/K/V)")
    
    batch = 8
    seq_len = 512
    hidden = 768
    
    hidden_states = np.random.randn(batch * seq_len, hidden).astype(np.float32)
    q_weight = np.random.randn(hidden, hidden).astype(np.float32)
    k_weight = np.random.randn(hidden, hidden).astype(np.float32) 
    v_weight = np.random.randn(hidden, hidden).astype(np.float32)
    
    monitor2 = GPUMonitor()
    monitor2.start()
    
    logger.info("   Running fused Q/K/V projection on GPU...")
    qkv_start = time.time()
    
    # Test fused operation
    q, k, v = vulkan.compute_fused_qkv_projection(hidden_states, q_weight, k_weight, v_weight)
    
    qkv_time = time.time() - qkv_start
    max_gpu2, max_vram2, _ = monitor2.stop()
    
    logger.info(f"\n   ✅ TEST 2 RESULTS:")
    logger.info(f"      QKV time: {qkv_time:.3f}s")
    logger.info(f"      Output shapes: Q={q.shape}, K={k.shape}, V={v.shape}")
    logger.info(f"      Max GPU usage: {max_gpu2:.1f}%")
    logger.info(f"      Max VRAM: {max_vram2:.1f} MB")
    
    # Test 3: Memory usage
    logger.info("\n📊 TEST 3: GPU Memory Usage")
    logger.info(f"   Allocated buffers: {len(vulkan.allocated_buffers)}")
    logger.info(f"   Memory usage: {vulkan.memory_usage_mb:.1f} MB")
    logger.info(f"   Buffer pool sizes: {list(vulkan.buffer_pool.keys())}")
    
    # Cleanup
    vulkan.cleanup()
    
    # Final verdict
    logger.info("\n🎯 FINAL RESULTS:")
    if gpu_working:
        logger.info("   ✅ GPU COMPUTE IS WORKING!")
        logger.info(f"   ✅ Peak GPU usage: {max(max_gpu, max_gpu2):.1f}%")
        logger.info(f"   ✅ Peak VRAM: {max(max_vram, max_vram2):.1f} MB")
    else:
        logger.info("   ❌ GPU COMPUTE NOT WORKING - Using CPU fallback")
        logger.info("   ❌ No significant GPU usage detected")
    
    return gpu_working

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("GPU COMPUTE VERIFICATION TEST")
    logger.info("="*60)
    
    success = test_gpu_compute()
    
    if success:
        logger.info("\n🎉 GPU acceleration confirmed!")
        logger.info("💡 Next step: Load model weights to GPU memory")
    else:
        logger.info("\n❌ GPU acceleration not working")
        logger.info("💡 Check Vulkan drivers and shader compilation")