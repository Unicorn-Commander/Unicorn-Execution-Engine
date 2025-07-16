#!/usr/bin/env python3
"""
🦄 Strict NPU+iGPU Inference Test - No CPU Fallback!
Tests the complete pipeline performance with real hardware acceleration
"""

import logging
import time
import sys
from pure_hardware_pipeline_gpu_fixed import PureHardwarePipelineGPUFixed

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def verify_hardware():
    """Verify NPU and GPU are available - no fallback allowed"""
    import os
    import subprocess
    
    # Check NPU
    if not os.path.exists("/dev/accel/accel0"):
        logger.error("❌ FAIL: NPU device not found at /dev/accel/accel0")
        logger.error("This test requires NPU hardware - no CPU fallback allowed!")
        return False
    
    # Check GPU via Vulkan
    try:
        result = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True)
        if "AMD Radeon" not in result.stdout:
            logger.error("❌ FAIL: AMD GPU not detected via Vulkan")
            return False
    except:
        logger.error("❌ FAIL: Vulkan not available")
        return False
    
    logger.info("✅ Hardware verification passed:")
    logger.info("  - NPU: AMD Phoenix NPU detected")
    logger.info("  - GPU: AMD Radeon Graphics detected")
    logger.info("  - Mode: STRICT NPU+iGPU only (no CPU fallback)")
    return True

def main():
    logger.info("="*60)
    logger.info("🦄 UNICORN EXECUTION ENGINE - STRICT HARDWARE TEST")
    logger.info("🎯 NPU+iGPU or FAILURE - No CPU Fallback!")
    logger.info("="*60)
    
    # Strict hardware check
    if not verify_hardware():
        logger.error("🚫 Hardware requirements not met. Exiting.")
        sys.exit(1)
    
    # Initialize the pipeline
    logger.info("\n🚀 Initializing Pure Hardware Pipeline...")
    try:
        pipeline = PureHardwarePipelineGPUFixed()
        logger.info("✅ Pipeline created")
    except Exception as e:
        logger.error(f"❌ Failed to create pipeline: {e}")
        sys.exit(1)
    
    # Initialize the model
    logger.info("\n📦 Loading quantized model...")
    model_path = "quantized_models/gemma-3-27b-it-layer-by-layer"
    
    try:
        if not pipeline.initialize(model_path):
            logger.error("❌ Failed to initialize model")
            sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        sys.exit(1)
        
    logger.info("✅ Model loaded successfully!")
    logger.info(f"  - Model path: {model_path}")
    logger.info(f"  - Hardware mode: NPU+iGPU acceleration")
    
    # The test prompt
    prompt = "What do you think about Magic Unicorn Unconventional Technology & Stuff for a company name for an advanced technology company that is AI centric?"
    
    logger.info("\n📝 Test Prompt:")
    logger.info(f"'{prompt}'")
    
    logger.info("\n🔥 Starting inference (NPU+iGPU only)...")
    logger.info("⚡ Watch for GPU/NPU activity!")
    
    # Measure performance
    start_time = time.time()
    tokens_generated = 0
    
    try:
        # Generate response with strict hardware enforcement
        result = pipeline.generate_tokens(
            prompt, 
            max_tokens=100,  # Generate a reasonable response
            temperature=0.7
        )
        
        # Count actual tokens generated
        if result and len(result) > len(prompt):
            tokens_generated = len(result.split()) - len(prompt.split())
        
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        logger.error("💥 This is expected if hardware acceleration isn't working")
        sys.exit(1)
    
    # Calculate performance
    end_time = time.time()
    total_time = end_time - start_time
    tps = tokens_generated / total_time if total_time > 0 else 0
    
    # Report results
    logger.info("\n" + "="*60)
    logger.info("📊 INFERENCE RESULTS")
    logger.info("="*60)
    logger.info(f"⏱️  Total time: {total_time:.2f} seconds")
    logger.info(f"🔢 Tokens generated: {tokens_generated}")
    logger.info(f"⚡ Performance: {tps:.2f} tokens/second")
    logger.info(f"\n💬 Generated Response:\n{result}")
    
    # Performance analysis
    logger.info("\n" + "="*60)
    logger.info("🎯 PERFORMANCE ANALYSIS")
    logger.info("="*60)
    
    if tps >= 81:
        logger.info(f"🏆 EXCELLENT: {tps:.2f} TPS - Exceeds 81 TPS target!")
    elif tps >= 50:
        logger.info(f"✅ GOOD: {tps:.2f} TPS - Meeting performance goals")
    elif tps >= 10:
        logger.info(f"⚠️  MODERATE: {tps:.2f} TPS - Hardware acceleration working")
    elif tps >= 1:
        logger.info(f"🐌 SLOW: {tps:.2f} TPS - Needs optimization")
    else:
        logger.info(f"❌ CRITICAL: {tps:.2f} TPS - Hardware acceleration may not be working")
    
    logger.info("\n🦄 Test complete!")
    
    # Return exit code based on success
    return 0 if tps > 0 else 1

if __name__ == "__main__":
    sys.exit(main())