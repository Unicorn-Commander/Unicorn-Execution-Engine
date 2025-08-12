#!/usr/bin/env python3
"""
Test the BEAST MODE pipeline that should achieve 100+ TPS
"""

import logging
import time
from vulkan_beast_mode_shaders import VulkanBeastModeShaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("🦄🔥 Testing BEAST MODE with Magic Unicorn prompt...")
    
    # Initialize the 100+ TPS beast mode pipeline
    pipeline = VulkanBeastModeShaders()
    
    # Load model
    model_path = "quantized_models/gemma-3-27b-it-layer-by-layer"
    logger.info("Loading model with BEAST MODE optimizations...")
    
    if not pipeline.initialize(model_path):
        logger.error("Failed to initialize BEAST MODE pipeline")
        return
    
    logger.info("✅ BEAST MODE activated!")
    
    # Test prompt
    prompt = "Magic Unicorn Unconventional Technology & Stuff is a groundbreaking Applied AI company that"
    logger.info(f"Prompt: '{prompt}'")
    
    # Monitor GPU
    logger.info("GPU should spike to near 100% utilization...")
    
    # Generate tokens
    logger.info("Generating response...")
    start_time = time.time()
    
    try:
        # Test the pipeline's text generation
        result = pipeline.generate(prompt, max_tokens=50)
        
        elapsed = time.time() - start_time
        tps = 50 / elapsed if elapsed > 0 else 0
        
        logger.info(f"✅ Generated response in {elapsed:.1f}s")
        logger.info(f"⚡ Performance: {tps:.1f} TPS")
        logger.info(f"🎯 Target: 100+ TPS (Beast Mode)")
        
        if tps >= 100:
            logger.info("🎉🔥 BEAST MODE ACHIEVED! 100+ TPS!")
        elif tps >= 81:
            logger.info("🎉 TARGET ACHIEVED! 81+ TPS!")
        elif tps >= 50:
            logger.info("✅ Good performance! Getting closer to 100 TPS")
        else:
            logger.info(f"⚠️ Performance below target: {tps:.1f} TPS")
        
        logger.info(f"\nGenerated text:\n{result}")
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    logger.info("Cleaning up...")
    pipeline.cleanup()

if __name__ == "__main__":
    main()