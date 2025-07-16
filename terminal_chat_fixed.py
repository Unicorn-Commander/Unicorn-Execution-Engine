#!/usr/bin/env python3
"""
Fixed Terminal Chat for Unicorn Execution Engine
Works with NPU-boosted models and proper generation
"""
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import time
import logging
import argparse
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Terminal chat with Unicorn models")
    parser.add_argument("--model", default="./quantized_models/gemma-3-4b-it-npu-boosted", 
                       help="Model path")
    args = parser.parse_args()
    
    logger.info("🦄 UNICORN EXECUTION ENGINE - TERMINAL CHAT")
    logger.info(f"🎯 Loading model: {args.model}")
    logger.info("=" * 60)
    
    try:
        # Check model path
        if not os.path.exists(args.model):
            logger.error(f"❌ Model not found: {args.model}")
            logger.info("Available models:")
            quantized_dir = "./quantized_models"
            if os.path.exists(quantized_dir):
                for model_dir in os.listdir(quantized_dir):
                    logger.info(f"   - {quantized_dir}/{model_dir}")
            return
        
        # Load model
        logger.info("📦 Loading model...")
        start_time = time.time()
        
        # Use AutoProcessor for multimodal models
        try:
            processor = AutoProcessor.from_pretrained(args.model)
            logger.info("✅ Using AutoProcessor (multimodal)")
        except:
            # Fallback to tokenizer only
            from transformers import AutoTokenizer
            processor = AutoTokenizer.from_pretrained(args.model)
            logger.info("✅ Using AutoTokenizer (text-only)")
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        model.eval()
        
        load_time = time.time() - start_time
        
        # Get model info
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✅ Model loaded successfully!")
        logger.info(f"📊 Parameters: {num_params/1e9:.1f}B") 
        logger.info(f"⏱️ Load time: {load_time:.1f}s")
        
        # Chat interface
        print("\n" + "=" * 60)
        print("💬 CHAT WITH UNICORN MODEL")
        print("Type 'quit' or 'exit' to end chat")
        print("=" * 60)
        
        while True:
            try:
                # Get user input
                user_input = input("\n🤔 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Format prompt properly
                if hasattr(processor, 'apply_chat_template'):
                    # Use chat template if available
                    messages = [{"role": "user", "content": user_input}]
                    prompt = processor.apply_chat_template(messages, tokenize=False)
                else:
                    # Simple prompt format
                    prompt = f"User: {user_input}\nAssistant:"
                
                # Generate response
                print("🤖 Assistant: ", end="", flush=True)
                
                if hasattr(processor, 'tokenizer'):
                    # AutoProcessor
                    inputs = processor(text=prompt, return_tensors="pt")
                else:
                    # AutoTokenizer
                    inputs = processor(prompt, return_tensors="pt", truncation=True, max_length=1500)
                
                start_gen = time.time()
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=processor.tokenizer.eos_token_id if hasattr(processor, 'tokenizer') else processor.eos_token_id,
                        eos_token_id=processor.tokenizer.eos_token_id if hasattr(processor, 'tokenizer') else processor.eos_token_id,
                        repetition_penalty=1.1
                    )
                
                gen_time = time.time() - start_gen
                
                # Decode response
                if hasattr(processor, 'decode'):
                    # AutoProcessor
                    full_response = processor.decode(outputs[0], skip_special_tokens=True)
                else:
                    # AutoTokenizer
                    full_response = processor.decode(outputs[0], skip_special_tokens=True)
                
                # Extract assistant response
                if "Assistant:" in full_response:
                    response = full_response.split("Assistant:")[-1].strip()
                else:
                    # Extract new tokens only
                    input_length = inputs['input_ids'].shape[1] if 'input_ids' in inputs else inputs.input_ids.shape[1]
                    new_tokens = outputs[0][input_length:]
                    if hasattr(processor, 'decode'):
                        response = processor.decode(new_tokens, skip_special_tokens=True)
                    else:
                        response = processor.decode(new_tokens, skip_special_tokens=True)
                
                # Clean up response
                response = response.replace("\nUser:", "").replace("\nAssistant:", "").strip()
                
                if response:
                    print(response)
                    
                    # Calculate performance
                    output_tokens = outputs.shape[1] - (inputs['input_ids'].shape[1] if 'input_ids' in inputs else inputs.input_ids.shape[1])
                    tps = output_tokens / gen_time if gen_time > 0 else 0
                    
                    print(f"\n📊 {output_tokens} tokens • {tps:.1f} tok/s • {gen_time:.1f}s")
                else:
                    print("[No response generated]")
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Continuing chat...")
                continue
        
    except Exception as e:
        logger.error(f"❌ Failed to start chat: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
