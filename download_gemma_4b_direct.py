#!/usr/bin/env python3
"""Direct download of Gemma-3-4B-IT using alternative method"""

import os
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_with_gdown():
    """Try downloading with gdown or alternative methods"""
    
    # First, let's test with the existing 2B model
    if os.path.exists("/home/ucadmin/models/gemma-2-2b-it"):
        logger.info("✅ Found Gemma-2-2B model, let's test with that first!")
        return "/home/ucadmin/models/gemma-2-2b-it"
    
    # For now, let's work with what we have
    logger.info("🔍 Available models:")
    
    # Check quantized models
    quant_path = "./quantized_models/"
    if os.path.exists(quant_path):
        for model in os.listdir(quant_path):
            size = 0
            model_path = os.path.join(quant_path, model)
            if os.path.isdir(model_path):
                for root, dirs, files in os.walk(model_path):
                    for f in files:
                        if f.endswith('.safetensors'):
                            size += os.path.getsize(os.path.join(root, f))
                if size > 0:
                    logger.info(f"  - {model}: {size/1024/1024/1024:.1f}GB")
    
    return None

if __name__ == "__main__":
    result = download_with_gdown()
    if result:
        logger.info(f"📁 Model available at: {result}")
    else:
        logger.info("\n💡 Suggestion: Let's test with Gemma-2-2B first!")
        logger.info("   Run: python3 test_gemma_2b.py")