#!/usr/bin/env python3
"""
Quick test of Magic Unicorn prompt with the working pipeline
"""

import logging
import time
from vulkan_kernel_optimized_pipeline import VulkanOptimizedPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("🦄 Testing Magic Unicorn Unconventional Technology & Stuff...")
    
    # Initialize the proven 11.1 TPS pipeline
    pipeline = VulkanOptimizedPipeline()
    
    # Load model
    model_path = "quantized_models/gemma-3-27b-it-layer-by-layer"
    logger.info("Loading model...")
    
    if not pipeline.initialize(model_path):
        logger.error("Failed to initialize pipeline")
        return
    
    logger.info("✅ Model loaded successfully!")
    
    # Test prompt
    prompt = "Magic Unicorn Unconventional Technology & Stuff is a groundbreaking Applied AI company that"
    logger.info(f"Prompt: '{prompt}'")
    
    # Generate tokens
    logger.info("Generating response...")
    start_time = time.time()
    
    try:
        # Tokenize (simplified for test)
        input_ids = [ord(c) % 1000 for c in prompt]
        
        # Generate
        generated_ids = pipeline.generate_tokens(
            input_ids, 
            max_tokens=50,
            temperature=0.7,
            top_p=0.9
        )
        
        elapsed = time.time() - start_time
        tps = len(generated_ids) / elapsed if elapsed > 0 else 0
        
        logger.info(f"✅ Generated {len(generated_ids)} tokens in {elapsed:.1f}s")
        logger.info(f"⚡ Performance: {tps:.1f} TPS")
        
        # Simple detokenization
        response = ''.join([chr((t % 94) + 33) for t in generated_ids])
        logger.info(f"Generated text: {response}")
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    pipeline.cleanup()

if __name__ == "__main__":
    main()