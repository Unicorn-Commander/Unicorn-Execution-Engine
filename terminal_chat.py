#!/usr/bin/env python3
"""
Terminal Chat Interface for Unicorn Execution Engine
Real model inference with terminal-based chat interface
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Terminal chat with real model"""
    logger.info("🦄 UNICORN EXECUTION ENGINE - TERMINAL CHAT")
    logger.info("🎯 Loading Gemma 3 4B for real inference")
    logger.info("=" * 60)
    
    try:
        # Load model
        logger.info("📦 Loading model...")
        start_time = time.time()
        
        tokenizer = AutoTokenizer.from_pretrained("./models/gemma-3-4b-it")
        
        # Configure tokenizer for Gemma
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            "./models/gemma-3-4b-it",
            torch_dtype=torch.float16,
            device_map="cpu",  # Use CPU for stability
            low_cpu_mem_usage=True
        )
        model.eval()
        
        load_time = time.time() - start_time
        
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✅ Model loaded successfully!")
        logger.info(f"📊 Parameters: {num_params/1e9:.1f}B")
        logger.info(f"⏱️ Load time: {load_time:.1f}s")
        
        # Chat interface
        print("\\n" + "=" * 60)
        print("💬 CHAT WITH GEMMA 3 4B")
        print("Type 'quit' or 'exit' to end chat")
        print("=" * 60)
        
        conversation = []
        
        while True:
            try:
                # Get user input
                user_input = input("\\n🤔 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Build conversation context for Gemma 3 format
                context_parts = ["<start_of_turn>user"]
                
                # Add recent conversation history (last 4 exchanges)
                for msg in conversation[-6:]:
                    if msg['role'] == 'User':
                        context_parts.append(f"{msg['content']}<end_of_turn>")
                        context_parts.append("<start_of_turn>model")
                    else:
                        context_parts.append(f"{msg['content']}<end_of_turn>")
                        context_parts.append("<start_of_turn>user")
                
                # Add current user message
                context_parts.append(f"{user_input}<end_of_turn>")
                context_parts.append("<start_of_turn>model")
                
                prompt = "\\n".join(context_parts)
                
                # Generate response
                print("🤖 Assistant: ", end="", flush=True)
                
                inputs = tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=1500,
                    padding=True,
                    add_special_tokens=True
                )
                
                start_gen = time.time()
                
                with torch.no_grad():
                    # Use greedy decoding to avoid sampling issues
                    outputs = model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=150,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                        use_cache=True
                    )
                
                gen_time = time.time() - start_gen
                
                # Decode response
                full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract assistant response for Gemma 3 format
                if "<start_of_turn>model" in full_response:
                    parts = full_response.split("<start_of_turn>model")
                    response = parts[-1].strip()
                else:
                    response = full_response[len(prompt):].strip()
                
                # Clean up response
                if "<end_of_turn>" in response:
                    response = response.split("<end_of_turn>")[0].strip()
                if "<start_of_turn>" in response:
                    response = response.split("<start_of_turn>")[0].strip()
                
                # Remove any remaining tokens
                response = response.replace("<start_of_turn>model", "").replace("<end_of_turn>", "").strip()
                
                if response:
                    print(response)
                    
                    # Calculate performance
                    output_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
                    tps = output_tokens / gen_time if gen_time > 0 else 0
                    
                    print(f"\\n📊 {output_tokens} tokens • {tps:.1f} tok/s • {gen_time:.1f}s")
                    
                    # Update conversation history
                    conversation.append({"role": "User", "content": user_input})
                    conversation.append({"role": "Assistant", "content": response})
                else:
                    print("[No response generated]")
                
            except KeyboardInterrupt:
                print("\\n\\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\\n❌ Error: {e}")
                print("Continuing chat...")
                continue
        
    except Exception as e:
        logger.error(f"❌ Failed to start chat: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()