#!/usr/bin/env python3
"""
Force GPU execution test - verify Vulkan shaders actually run on GPU
"""

import time
import logging
import subprocess
import threading
from real_vulkan_matrix_compute import VulkanMatrixCompute
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPUMonitor:
    """Monitor GPU usage in real-time"""
    
    def __init__(self):
        self.max_gpu = 0
        self.monitoring = False
        
    def start(self):
        """Start monitoring in background thread"""
        self.monitoring = True
        self.max_gpu = 0
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()
        
    def stop(self):
        """Stop monitoring and return max GPU usage"""
        self.monitoring = False
        self.thread.join()
        return self.max_gpu
        
    def _monitor_loop(self):
        """Monitor GPU usage continuously"""
        while self.monitoring:
            try:
                result = subprocess.run(
                    ['radeontop', '-d', '-', '-l', '1'], 
                    capture_output=True, 
                    text=True,
                    timeout=0.5
                )
                
                for line in result.stdout.split('\n'):
                    if 'gpu' in line:
                        try:
                            # Extract GPU percentage
                            parts = line.split(',')
                            for part in parts:
                                if 'gpu' in part:
                                    gpu_str = part.split('gpu')[1].split('%')[0].strip()
                                    gpu_usage = float(gpu_str)
                                    self.max_gpu = max(self.max_gpu, gpu_usage)
                                    if gpu_usage > 0:
                                        logger.info(f"🎮 GPU ACTIVE: {gpu_usage}%")
                                    break
                        except:
                            pass
            except:
                pass
            time.sleep(0.1)


def test_real_gpu_execution():
    """Test if Vulkan operations actually execute on GPU"""
    
    logger.info("🔥 Testing REAL GPU execution...")
    
    # Initialize Vulkan
    vulkan = VulkanMatrixCompute()
    if not vulkan.initialize():
        logger.error("Failed to initialize Vulkan")
        return
        
    # Test sizes that should definitely trigger GPU
    sizes = [1024, 2048, 4096, 8192]
    
    for size in sizes:
        logger.info(f"\n📊 Testing {size}x{size} matrix multiply...")
        
        # Create test matrices
        a = np.random.randn(size, size).astype(np.float32)
        b = np.random.randn(size, size).astype(np.float32)
        
        # Start GPU monitoring
        monitor = GPUMonitor()
        monitor.start()
        
        # Run multiple iterations to ensure GPU usage is captured
        logger.info("Running GPU operations...")
        for i in range(10):
            start = time.time()
            
            # This SHOULD execute on GPU
            result = vulkan.matrix_multiply(a, b)
            
            elapsed = time.time() - start
            gflops = (2 * size ** 3) / (elapsed * 1e9)
            logger.info(f"  Iteration {i}: {elapsed*1000:.1f}ms, {gflops:.1f} GFLOPS")
            
        # Stop monitoring and get max GPU usage
        max_gpu = monitor.stop()
        
        logger.info(f"✅ Max GPU usage: {max_gpu}%")
        
        if max_gpu > 10:
            logger.info("🎉 GPU IS BEING USED!")
        else:
            logger.error("❌ GPU NOT ACTIVE - Operations may be on CPU!")
            
    # Test persistent buffers
    logger.info("\n🔥 Testing persistent buffer operations...")
    
    # Create persistent weight buffer
    weight = np.random.randn(4096, 4096).astype(np.float32)
    persistent_weight = vulkan.create_persistent_buffer(weight)
    
    monitor = GPUMonitor()
    monitor.start()
    
    # Run many operations
    for i in range(50):
        input_data = np.random.randn(512, 4096).astype(np.float32)
        result = vulkan.compute_matrix_multiply_persistent(
            input_data, persistent_weight, weight.shape
        )
        
    max_gpu = monitor.stop()
    logger.info(f"Persistent buffer max GPU: {max_gpu}%")
    
    
def verify_shader_execution():
    """Verify shaders are compiled for GPU, not CPU"""
    
    logger.info("\n🔍 Verifying shader compilation...")
    
    # Check if shaders exist
    import os
    shader_files = [
        'matrix_multiply.spv',
        'gate_up_silu_mul.spv',
        'down_proj.spv'
    ]
    
    for shader in shader_files:
        if os.path.exists(shader):
            logger.info(f"✅ Found compiled shader: {shader}")
            
            # Check shader info
            result = subprocess.run(
                ['spirv-dis', shader], 
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Look for GPU-specific instructions
                if 'OpCapability Shader' in result.stdout:
                    logger.info(f"  ✅ Shader compiled for GPU execution")
                else:
                    logger.warning(f"  ⚠️ Shader may not be GPU-optimized")
        else:
            logger.warning(f"❌ Shader not found: {shader}")


if __name__ == "__main__":
    # First verify shaders
    verify_shader_execution()
    
    # Then test GPU execution
    test_real_gpu_execution()