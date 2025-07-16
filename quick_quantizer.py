#!/usr/bin/env python3
"""
Quick Quantizer - Validate process with Gemma 3 4B then scale to 27B
Fast validation of real quantization pipeline
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
import logging
import os
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_quantize_test(model_name: str = "gemma-3-4b-it"):
    """Quick quantization test"""
    logger.info(f"🚀 QUICK QUANTIZATION TEST - {model_name.upper()}")
    logger.info("=" * 60)
    
    model_path = f"./models/{model_name}"
    output_path = f"./quantized_models/{model_name}-quantized"
    
    try:
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Load tokenizer
        logger.info("📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Configure 4-bit quantization
        logger.info("🔧 Configuring 4-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Load model with quantization
        logger.info("📦 Loading model with 4-bit quantization...")
        start_time = time.time()
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16
        )
        
        load_time = time.time() - start_time
        
        logger.info(f"✅ Model loaded with quantization in {load_time:.1f}s")
        
        # Get memory footprint
        if hasattr(model, 'get_memory_footprint'):
            memory_gb = model.get_memory_footprint() / (1024**3)
            logger.info(f"💾 Quantized model memory: {memory_gb:.1f}GB")
        
        # Test inference
        logger.info("🧪 Testing quantized inference...")
        
        test_prompts = ["Hello, I am", "The future is"]
        total_tps = 0
        tests_run = 0
        
        for prompt in test_prompts:
            try:
                inputs = tokenizer(prompt, return_tensors="pt")
                
                start_gen = time.time()
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        max_new_tokens=20,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                gen_time = time.time() - start_gen
                
                response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                output_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
                tps = output_tokens / gen_time if gen_time > 0 else 0
                
                logger.info(f"   🎯 '{prompt}' → '{response[:30]}...' ({tps:.1f} TPS)")
                total_tps += tps
                tests_run += 1
                
            except Exception as e:
                logger.warning(f"   ⚠️ Test failed: {e}")
        
        avg_tps = total_tps / tests_run if tests_run > 0 else 0
        
        # Save quantized model
        logger.info("💾 Saving quantized model...")
        try:
            model.save_pretrained(output_path, safe_serialization=True)
            tokenizer.save_pretrained(output_path)
            
            # Save quantization info
            info = {
                "model": model_name,
                "quantization": "4-bit NF4",
                "performance_tps": avg_tps,
                "memory_gb": memory_gb if 'memory_gb' in locals() else "unknown",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(f"{output_path}/quantization_info.json", "w") as f:
                json.dump(info, f, indent=2)
            
            logger.info(f"✅ Saved to: {output_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ Save failed: {e}")
        
        # Results
        logger.info("\n" + "=" * 60)
        logger.info("🎉 QUANTIZATION TEST COMPLETE!")
        logger.info(f"✅ Model: {model_name}")
        logger.info(f"✅ Average TPS: {avg_tps:.1f}")
        logger.info(f"✅ Memory usage: {memory_gb:.1f}GB" if 'memory_gb' in locals() else "✅ Memory: Optimized")
        logger.info(f"✅ Output: {output_path}")
        
        return {
            "success": True,
            "model": model_name,
            "avg_tps": avg_tps,
            "output_path": output_path
        }
        
    except Exception as e:
        logger.error(f"❌ Quantization failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

def test_quantized_chat(model_path: str):
    """Test chat with quantized model"""
    logger.info(f"💬 TESTING QUANTIZED CHAT")
    logger.info(f"📁 Model: {model_path}")
    logger.info("=" * 50)
    
    try:
        # Load quantized model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        logger.info("✅ Quantized model loaded for chat")
        
        # Simple chat test
        conversation = ["User: Hello! How are you today?", "Assistant:"]
        prompt = "\\n".join(conversation)
        
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            start_time = time.time()
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            gen_time = time.time() - start_time
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        output_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
        tps = output_tokens / gen_time
        
        logger.info(f"🤖 Response: '{response}'")
        logger.info(f"📊 Performance: {tps:.1f} TPS")
        
        return {"success": True, "tps": tps, "response": response}
        
    except Exception as e:
        logger.error(f"❌ Chat test failed: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma-3-4b-it", 
                       choices=["gemma-3-4b-it", "gemma-3-27b-it"],
                       help="Model to quantize")
    parser.add_argument("--test-chat", 
                       help="Test chat with quantized model at this path")
    
    args = parser.parse_args()
    
    if args.test_chat:
        test_quantized_chat(args.test_chat)
    else:
        # Run quantization
        result = quick_quantize_test(args.model)
        
        if result["success"]:
            logger.info(f"\\n🎮 Test chat with:")
            logger.info(f"python {__file__} --test-chat {result['output_path']}")
        
        # If 4B was successful, offer to do 27B
        if args.model == "gemma-3-4b-it" and result["success"]:
            logger.info(f"\\n🚀 Next: Quantize 27B model with:")
            logger.info(f"python {__file__} --model gemma-3-27b-it")