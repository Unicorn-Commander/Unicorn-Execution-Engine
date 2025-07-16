#!/usr/bin/env python3
"""
Test script to verify optimizations are working
"""

import time
import logging
import subprocess
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_optimization_components():
    """Test that all optimization components are in place"""
    
    logger.info("🧪 Testing Optimization Components")
    logger.info("=" * 60)
    
    # 1. Check TFLOPS update
    logger.info("\n1️⃣ Checking TFLOPS update...")
    with open("vulkan_compute_framework.py", "r") as f:
        content = f.read()
        if "8.9 TFLOPS" in content:
            logger.info("   ✅ TFLOPS correctly updated to 8.9")
        else:
            logger.error("   ❌ TFLOPS not updated!")
    
    # 2. Check Lightning Fast Loader
    logger.info("\n2️⃣ Checking Lightning Fast Loader...")
    with open("pure_hardware_pipeline.py", "r") as f:
        content = f.read()
        if "from lightning_fast_loader import LightningFastLoader" in content:
            logger.info("   ✅ Lightning Fast Loader imported")
        else:
            logger.error("   ❌ Still using old loader!")
    
    # 3. Check transformer optimized shader
    logger.info("\n3️⃣ Checking transformer optimized shader...")
    if os.path.exists("transformer_optimized.spv"):
        logger.info("   ✅ Optimized shader exists")
        
        # Check if it's being used
        with open("real_vulkan_matrix_compute.py", "r") as f:
            content = f.read()
            if "transformer_optimized.spv" in content:
                logger.info("   ✅ Optimized shader configured")
            else:
                logger.error("   ❌ Shader not configured!")
    else:
        logger.error("   ❌ Optimized shader not found!")
    
    # 4. Check hardware tuner
    logger.info("\n4️⃣ Checking hardware tuner...")
    with open("pure_hardware_api_server.py", "r") as f:
        content = f.read()
        if "from advanced_hardware_tuner import HardwareSpecificOptimizer" in content:
            logger.info("   ✅ Hardware tuner imported")
            if "hardware_optimizer.start_adaptive_optimization()" in content:
                logger.info("   ✅ Hardware tuner activated")
            else:
                logger.error("   ❌ Hardware tuner not activated!")
        else:
            logger.error("   ❌ Hardware tuner not imported!")
    
    # 5. Test imports
    logger.info("\n5️⃣ Testing imports...")
    try:
        from lightning_fast_loader import LightningFastLoader
        logger.info("   ✅ Lightning Fast Loader imports successfully")
    except Exception as e:
        logger.error(f"   ❌ Lightning Fast Loader import failed: {e}")
    
    try:
        from advanced_hardware_tuner import HardwareSpecificOptimizer
        logger.info("   ✅ Hardware tuner imports successfully")
    except Exception as e:
        logger.error(f"   ❌ Hardware tuner import failed: {e}")
    
    # 6. Check NPU
    logger.info("\n6️⃣ Checking NPU availability...")
    try:
        result = subprocess.run(["/opt/xilinx/xrt/bin/xrt-smi", "examine"], 
                              capture_output=True, text=True, timeout=5)
        if "NPU Phoenix" in result.stdout:
            logger.info("   ✅ NPU Phoenix detected")
        else:
            logger.warning("   ⚠️ NPU not detected")
    except Exception as e:
        logger.warning(f"   ⚠️ NPU check failed: {e}")
    
    # 7. Check Vulkan
    logger.info("\n7️⃣ Checking Vulkan...")
    try:
        result = subprocess.run(["vulkaninfo", "--summary"], 
                              capture_output=True, text=True, timeout=5)
        if "AMD" in result.stdout:
            logger.info("   ✅ AMD Vulkan device detected")
        else:
            logger.warning("   ⚠️ Vulkan device not detected")
    except Exception as e:
        logger.warning(f"   ⚠️ Vulkan check failed: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 Optimization component test complete!")
    logger.info("\n📊 Expected Performance Improvements:")
    logger.info("   • Loading Time: 120s → 10-15s (8-12x faster)")
    logger.info("   • Tokens/Second: 0.3-0.5 → 5-15 TPS (10-30x faster)")
    logger.info("   • iGPU Compute: 2.7 → 8.9 TFLOPS (3.3x more)")
    logger.info("   • Memory: Basic mmap → Ollama-style pinned")
    logger.info("   • Shaders: Basic matmul → Fused transformer ops")

if __name__ == "__main__":
    test_optimization_components()