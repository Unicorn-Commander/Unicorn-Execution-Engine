#!/usr/bin/env python3
"""
Test Real GPU Inference - Verify VRAM usage and performance
"""

import subprocess
import time
import logging
import threading

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def monitor_gpu(duration=10):
    """Monitor GPU usage for specified duration"""
    logger.info("🔍 Monitoring GPU usage...")
    
    measurements = []
    start_time = time.time()
    
    while time.time() - start_time < duration:
        try:
            result = subprocess.run(['radeontop', '-d', '-', '-l', '1'], 
                                  capture_output=True, text=True, timeout=1)
            if result.stdout:
                # Parse the output
                import re
                gpu_match = re.search(r'gpu\s+(\d+\.\d+)%', result.stdout)
                vram_match = re.search(r'vram\s+(\d+\.\d+)%\s+(\d+\.\d+)mb', result.stdout)
                gtt_match = re.search(r'gtt\s+(\d+\.\d+)%\s+(\d+\.\d+)mb', result.stdout)
                
                if gpu_match and vram_match and gtt_match:
                    measurements.append({
                        'time': time.time() - start_time,
                        'gpu': float(gpu_match.group(1)),
                        'vram_pct': float(vram_match.group(1)),
                        'vram_mb': float(vram_match.group(2)),
                        'gtt_pct': float(gtt_match.group(1)),
                        'gtt_mb': float(gtt_match.group(2))
                    })
        except Exception as e:
            logger.error(f"Error monitoring GPU: {e}")
        
        time.sleep(0.5)
    
    return measurements

def test_real_gpu_inference():
    """Test inference with real GPU monitoring"""
    logger.info("🚀 Testing Real GPU Inference")
    
    # Start GPU monitoring in background
    monitor_thread = threading.Thread(target=lambda: monitor_results.extend(monitor_gpu(15)))
    monitor_results = []
    monitor_thread.start()
    
    # Give monitoring time to start
    time.sleep(1)
    
    # Import and test
    logger.info("Loading Vulkan compute engine...")
    from real_vulkan_matrix_compute import VulkanMatrixCompute
    import numpy as np
    
    vulkan = VulkanMatrixCompute()
    vulkan.initialize()
    
    logger.info("Running GPU compute tests...")
    
    # Test different sizes to stress GPU
    sizes = [1024, 2048, 4096, 8192]
    
    for size in sizes:
        logger.info(f"\n🧪 Testing {size}x{size} matrix multiplication")
        
        # Create large matrices
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)
        
        # Run multiple times to see sustained usage
        for i in range(3):
            start = time.time()
            result = vulkan.matrix_multiply(A, B)
            duration = time.time() - start
            
            logger.info(f"   Iteration {i+1}: {duration:.3f}s")
            
            # Small delay between iterations
            time.sleep(0.1)
    
    # Clean up
    vulkan.cleanup()
    
    # Wait for monitoring to complete
    monitor_thread.join()
    
    # Analyze results
    logger.info("\n📊 GPU Usage Analysis:")
    
    if monitor_results:
        max_gpu = max(m['gpu'] for m in monitor_results)
        max_vram = max(m['vram_mb'] for m in monitor_results)
        max_gtt = max(m['gtt_mb'] for m in monitor_results)
        avg_gpu = sum(m['gpu'] for m in monitor_results) / len(monitor_results)
        
        logger.info(f"   Peak GPU usage: {max_gpu:.1f}%")
        logger.info(f"   Average GPU usage: {avg_gpu:.1f}%")
        logger.info(f"   Peak VRAM usage: {max_vram:.1f} MB ({max_vram/1024:.1f} GB)")
        logger.info(f"   Peak GTT usage: {max_gtt:.1f} MB")
        
        # Check if we're actually using the GPU
        if max_gpu > 5:
            logger.info("   ✅ GPU is being used!")
        else:
            logger.warning("   ⚠️ GPU usage is too low - may be using CPU fallback")
        
        if max_vram > 2000:  # More than 2GB
            logger.info("   ✅ VRAM is being utilized!")
        else:
            logger.warning("   ⚠️ VRAM usage is too low")
    else:
        logger.error("   ❌ No GPU measurements collected")
    
    # Now test with the pipeline
    logger.info("\n🔥 Testing Pure Hardware Pipeline...")
    
    try:
        from pure_hardware_pipeline import PureHardwarePipeline
        
        pipeline = PureHardwarePipeline()
        logger.info("Pipeline initialized")
        
        # Try to generate something
        prompt = "Hello"
        logger.info(f"Generating response for: '{prompt}'")
        
        start = time.time()
        # This might fail if model loading isn't working
        tokens = pipeline.generate(prompt, max_tokens=10)
        duration = time.time() - start
        
        logger.info(f"Generated in {duration:.3f}s")
        logger.info(f"Response: {tokens}")
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        logger.info("This is expected if model loading isn't fully implemented yet")

if __name__ == "__main__":
    test_real_gpu_inference()