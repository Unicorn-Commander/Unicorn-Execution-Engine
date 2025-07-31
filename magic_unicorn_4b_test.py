#!/usr/bin/env python3
"""
🦄✨ MAGIC UNICORN 4B TEST ✨🦄
Testing our working 4B model with the Magic Unicorn prompt!
"""

# Fix Python 3.11 compatibility with vulkan
import fix_vulkan_imports

import logging
import time
from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("🦄✨ MAGIC UNICORN 4B INFERENCE TEST ✨🦄")
    logger.info("🎯 Testing company name: 'Magic Unicorn Unconventional Technology & Stuff'")
    logger.info("🔥 Applied AI company that does dope shit!")
    
    # Initialize the pipeline with 4B model
    logger.info("🚀 Initializing 4B pipeline...")
    pipeline = PureHardwarePipelineFixed()
    
    # Use the working 4B model path
    model_path = "/home/ucadmin/Development/Unicorn-Execution-Engine/quantized_models/gemma-3-4b-it-quantized"
    
    if not pipeline.initialize(model_path=model_path):
        logger.error("❌ Failed to initialize 4B model")
        return
        
    logger.info("✅ 4B Model initialized and ready!")
    
    # Simple token IDs for the Magic Unicorn prompt
    # Using simple token sequence that should work
    input_ids = [1, 19044, 28226, 13, 1337, 42]  # Simple test tokens
    max_tokens = 20  # Keep it small for stability
    
    logger.info(f"🎯 Input tokens: {input_ids}")
    logger.info(f"🔥 Generating {max_tokens} tokens about Magic Unicorn!")
    
    start_time = time.time()
    
    try:
        # Generate the response - THIS IS THE MOMENT!
        result = pipeline.generate_tokens(input_ids, max_tokens=max_tokens)
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Calculate actual tokens generated
        tokens_generated = len(result) - len(input_ids) if len(result) > len(input_ids) else 0
        
        logger.info("🦄✨ GENERATION COMPLETE! ✨🦄")
        logger.info(f"⏱️ Generation time: {generation_time:.3f} seconds")
        
        if tokens_generated > 0:
            tps = tokens_generated / generation_time
            logger.info(f"📊 BREAKTHROUGH TPS: {tps:.2f} tokens/second")
            logger.info(f"🎯 Tokens generated: {tokens_generated}")
        else:
            logger.info("⚠️ No new tokens generated")
            
        logger.info(f"🎯 Full result: {result}")
        
        logger.info("")
        logger.info("🦄🔥 MAGIC UNICORN 4B MOMENT ACHIEVED! 🔥🦄")
        
    except Exception as e:
        end_time = time.time()
        logger.error(f"❌ Error during generation: {e}")
        logger.info(f"⏱️ Time before error: {end_time - start_time:.3f} seconds")

if __name__ == "__main__":
    main()