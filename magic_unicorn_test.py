#!/usr/bin/env python3
"""
🦄✨ MAGIC UNICORN UNCONVENTIONAL TECHNOLOGY & STUFF ✨🦄
The moment of truth - Real inference on real hardware with real model!
"""

import logging
import time
from pure_hardware_pipeline_gpu_fixed import PureHardwarePipelineGPUFixed

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("🦄✨ MAGIC UNICORN INFERENCE TEST ✨🦄")
    logger.info("🎯 Testing company name: 'Magic Unicorn Unconventional Technology & Stuff'")
    logger.info("🔥 Applied AI company that does dope shit!")
    
    # Initialize the pipeline
    logger.info("🚀 Initializing GPU pipeline...")
    pipeline = PureHardwarePipelineGPUFixed()
    
    # Initialize the model
    logger.info("🔥 Initializing model...")
    model_path = "quantized_models/gemma-3-27b-it-layer-by-layer"
    
    if not pipeline.initialize(model_path):
        logger.error("Failed to initialize")
        return
        
    logger.info("✅ Model initialized and ready!")
    
    # The epic prompt
    prompt = "Magic Unicorn Unconventional Technology & Stuff is a groundbreaking Applied AI company that"
    
    logger.info(f"📝 Prompt: '{prompt}'")
    logger.info("🎯 Generating response... WATCH THAT GPU SPIKE! 🔥")
    
    start_time = time.time()
    
    # Generate the response - THIS IS THE MOMENT!
    result = pipeline.generate_tokens(prompt, max_tokens=50)
    
    end_time = time.time()
    generation_time = end_time - start_time
    
    logger.info("🦄✨ GENERATION COMPLETE! ✨🦄")
    logger.info(f"⏱️ Generation time: {generation_time:.2f} seconds")
    logger.info(f"📊 Tokens per second: {50 / generation_time:.2f} TPS")
    logger.info("🎯 Generated response:")
    logger.info(f"'{result}'")
    
    logger.info("")
    logger.info("🦄🔥 MAGIC UNICORN MOMENT ACHIEVED! 🔥🦄")

if __name__ == "__main__":
    main()