#!/usr/bin/env python3
"""
Quick Multimodal Test - Resume quantization with progress tracking
"""
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
import time
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_multimodal_test():
    """Quick test to resume/continue multimodal quantization"""
    logger.info("🚀 RESUMING MULTIMODAL QUANTIZATION")
    logger.info("🎯 Gemma 3 27B with Vision Capabilities")
    logger.info("=" * 50)
    
    model_path = "./models/gemma-3-27b-it"
    output_path = "./quantized_models/gemma-3-27b-it-multimodal"
    
    try:
        # Check if already exists
        if os.path.exists(output_path) and os.listdir(output_path):
            logger.info(f"✅ Found existing quantized model at {output_path}")
            logger.info("🧪 Testing existing model...")
            
            try:
                processor = AutoProcessor.from_pretrained(output_path)
                model = AutoModelForCausalLM.from_pretrained(output_path, device_map="auto")
                
                # Quick test
                inputs = processor(text="Hello, how are you?", return_tensors="pt")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=20)
                response = processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                
                logger.info(f"✅ Existing model works: '{response}'")
                return {"success": True, "path": output_path, "status": "existing"}
                
            except Exception as e:
                logger.warning(f"⚠️ Existing model test failed: {e}")
                logger.info("🔄 Will create fresh quantization...")
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Load processor first (fast)
        logger.info("📥 Loading processor...")
        processor = AutoProcessor.from_pretrained(model_path)
        logger.info("✅ Processor loaded")
        
        # Configure quantization
        logger.info("🔧 Configuring 4-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Load model with progress
        logger.info("📦 Loading multimodal model with quantization...")
        logger.info("⏱️ This will take 10-15 minutes for 30.5B parameters...")
        logger.info("📊 Progress will be shown as checkpoint shards load...")
        
        start_time = time.time()
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16
        )
        
        load_time = time.time() - start_time
        logger.info(f"✅ Model loaded in {load_time/60:.1f} minutes!")
        
        # Quick test
        logger.info("🧪 Testing quantized model...")
        inputs = processor(text="The future of AI will be", return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        response = processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        logger.info(f"✅ Quantized model works: '{response}'")
        
        # Save model
        logger.info("💾 Saving quantized model...")
        model.save_pretrained(output_path, safe_serialization=True)
        processor.save_pretrained(output_path)
        
        # Create info file
        info = {
            "model": "gemma-3-27b-it-multimodal",
            "quantization": "4-bit NF4",
            "capabilities": ["text", "vision", "multimodal"],
            "load_time_minutes": load_time / 60,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        import json
        with open(f"{output_path}/model_info.json", "w") as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"✅ MULTIMODAL QUANTIZATION COMPLETE!")
        logger.info(f"📁 Saved to: {output_path}")
        logger.info(f"⏱️ Total time: {load_time/60:.1f} minutes")
        
        return {
            "success": True, 
            "path": output_path, 
            "load_time_minutes": load_time/60,
            "status": "new"
        }
        
    except Exception as e:
        logger.error(f"❌ Quantization failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    result = quick_multimodal_test()
    
    if result["success"]:
        print(f"\n🎉 SUCCESS! Multimodal model ready at:")
        print(f"📁 {result['path']}")
        print(f"\n🎮 Test with:")
        print(f"python terminal_chat.py --model {result['path']}")
        
        if result["status"] == "new":
            print(f"\n⏱️ Quantization took: {result['load_time_minutes']:.1f} minutes")
    else:
        print(f"\n❌ Failed: {result.get('error', 'Unknown error')}")