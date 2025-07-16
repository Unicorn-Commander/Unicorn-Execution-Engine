#!/usr/bin/env python3
"""
Simple Gemma 3 27B Performance Test
Focus on actual 27B model loading and TPS measurement
"""

import os
import sys
import time
import torch
import psutil
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gemma27b_performance():
    """Test Gemma 3 27B performance"""
    logger.info("🦄 Gemma 3 27B Performance Test")
    logger.info("=" * 50)
    
    # Check available models
    model_paths = [
        "./quantized_models/gemma-3-27b-it-memory-efficient",  # 30.8GB
        "./models/gemma-3-27b-it"  # Original 102GB
    ]
    
    # System info
    memory = psutil.virtual_memory()
    logger.info(f"💾 Available RAM: {memory.available / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB")
    
    for model_path in model_paths:
        if not os.path.exists(model_path):
            logger.info(f"❌ Not found: {model_path}")
            continue
            
        logger.info(f"\n🧪 Testing: {model_path}")
        logger.info("-" * 40)
        
        try:
            start_time = time.time()
            
            # Load tokenizer
            logger.info("🔤 Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Track memory before model loading
            memory_before = psutil.virtual_memory()
            
            # Load model
            logger.info("🧠 Loading model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Model info
            param_count = sum(p.numel() for p in model.parameters())
            load_time = time.time() - start_time
            memory_after = psutil.virtual_memory()
            memory_used = (memory_before.available - memory_after.available) / (1024**3)
            
            logger.info(f"✅ Model loaded: {param_count:,} parameters ({param_count/1e9:.1f}B)")
            logger.info(f"⏱️ Load time: {load_time:.1f}s")
            logger.info(f"💾 Memory used: {memory_used:.1f}GB")
            
            # Verify this is actually 27B
            if param_count < 20e9:
                logger.error(f"❌ This is not a 27B model! ({param_count/1e9:.1f}B parameters)")
                continue
            
            # Quick inference test
            test_prompts = [
                "Explain quantum computing",
                "What is artificial intelligence?",
                "How do neural networks work?"
            ]
            
            total_tokens = 0
            total_time = 0
            
            for i, prompt in enumerate(test_prompts):
                logger.info(f"🔮 Test {i+1}/3: {prompt[:30]}...")
                
                try:
                    # Tokenize
                    inputs = tokenizer(prompt, return_tensors="pt")
                    input_length = inputs.input_ids.shape[1]
                    
                    # Generate
                    gen_start = time.time()
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=20,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    gen_time = time.time() - gen_start
                    
                    # Decode
                    generated_text = tokenizer.decode(
                        outputs[0][input_length:], 
                        skip_special_tokens=True
                    )
                    
                    actual_tokens = len(tokenizer.encode(generated_text))
                    tps = actual_tokens / gen_time if gen_time > 0 else 0
                    
                    logger.info(f"   ✅ {actual_tokens} tokens in {gen_time:.2f}s = {tps:.2f} TPS")
                    logger.info(f"   📝 Output: {generated_text[:50]}...")
                    
                    total_tokens += actual_tokens
                    total_time += gen_time
                    
                except Exception as e:
                    logger.error(f"   ❌ Inference failed: {e}")
                
                time.sleep(0.5)
            
            # Summary
            if total_time > 0:
                avg_tps = total_tokens / total_time
                logger.info(f"\n📊 PERFORMANCE SUMMARY:")
                logger.info(f"   🚀 Average TPS: {avg_tps:.2f}")
                logger.info(f"   📊 Total tokens: {total_tokens}")
                logger.info(f"   ⏱️ Total time: {total_time:.2f}s")
                logger.info(f"   🧠 Model size: {param_count/1e9:.1f}B parameters")
                logger.info(f"   💾 Memory usage: {memory_used:.1f}GB")
                
                # Performance assessment
                if avg_tps > 1.0:
                    logger.info(f"   ✅ Good performance!")
                elif avg_tps > 0.5:
                    logger.info(f"   ⚠️ Moderate performance")
                else:
                    logger.info(f"   ❌ Low performance")
            
            # Cleanup
            del model
            del tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return avg_tps if total_time > 0 else 0
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            continue
    
    logger.error("❌ No successful tests")
    return 0

def main():
    """Main function"""
    tps = test_gemma27b_performance()
    
    if tps > 0:
        logger.info(f"\n🎯 FINAL RESULT: {tps:.2f} TPS with Gemma 3 27B")
        return 0
    else:
        logger.error("❌ No successful performance measurement")
        return 1

if __name__ == "__main__":
    sys.exit(main())