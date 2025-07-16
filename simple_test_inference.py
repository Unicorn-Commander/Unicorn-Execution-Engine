#!/usr/bin/env python3
"""
Simple Test Inference - Working Real Model Test
Basic inference without complex sampling to validate the framework
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_real_inference():
    """Test real inference with Gemma 3 4B"""
    logger.info("🦄 REAL MODEL INFERENCE TEST")
    logger.info("🎯 Testing with Gemma 3 4B-IT")
    logger.info("=" * 50)
    
    try:
        # Load tokenizer
        logger.info("📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("./models/gemma-3-4b-it")
        
        # Load model (CPU for stability)
        logger.info("📦 Loading model...")
        start_load = time.time()
        
        model = AutoModelForCausalLM.from_pretrained(
            "./models/gemma-3-4b-it",
            torch_dtype=torch.float16,
            device_map="cpu",  # Use CPU for stability
            low_cpu_mem_usage=True
        )
        model.eval()
        
        load_time = time.time() - start_load
        logger.info(f"✅ Model loaded in {load_time:.1f}s")
        
        # Test prompts
        test_prompts = [
            "Hello, I am",
            "The weather today is",
            "Artificial intelligence"
        ]
        
        for prompt in test_prompts:
            logger.info(f"\\n🧪 Testing: '{prompt}'")
            
            # Tokenize
            start_time = time.time()
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # Generate with greedy decoding (most stable)
            with torch.no_grad():
                generation_start = time.time()
                
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=20,  # Short generation for testing
                    do_sample=False,    # Greedy decoding - most stable
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )
                
                generation_time = time.time() - generation_start
            
            # Decode
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            new_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            total_time = time.time() - start_time
            output_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
            tps = output_tokens / generation_time if generation_time > 0 else 0
            
            logger.info(f"   📝 Generated: '{new_text}'")
            logger.info(f"   📊 {output_tokens} tokens in {generation_time:.2f}s")
            logger.info(f"   🚀 {tps:.1f} tokens/second")
        
        logger.info(f"\\n✅ REAL INFERENCE WORKING!")
        logger.info(f"🎯 Framework validated with actual model execution")
        return True
        
    except Exception as e:
        logger.error(f"❌ Real inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_real_inference()