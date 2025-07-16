#!/usr/bin/env python3
"""
Simple test for Gemma 3n E4B acceleration with minimal features
"""

import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_simple_acceleration():
    """Test basic model loading and inference without full acceleration"""
    logger.info("🦄 Testing Simple Gemma 3n E4B Acceleration")
    
    try:
        # Load model
        logger.info("📥 Loading Gemma 3n E4B model...")
        tokenizer = AutoTokenizer.from_pretrained("./models/gemma-3n-e4b-it", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            "./models/gemma-3n-e4b-it",
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        logger.info("✅ Model loaded successfully")
        
        # Test inference
        prompt = "Hello, I'm Aaron. Please tell me about yourself."
        inputs = tokenizer(prompt, return_tensors="pt")
        
        logger.info("🚀 Starting inference...")
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        
        logger.info(f"✅ Response: {response}")
        logger.info("🎯 Simple acceleration test complete!")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_acceleration()