#!/usr/bin/env python3
"""
Minimal Working Inference Engine
Real model inference with basic optimizations for immediate testing
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import logging
import argparse
import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MinimalInferenceEngine:
    """Minimal working inference engine for immediate testing"""
    
    def __init__(self, model_path: str = "./models/gemma-3-4b-it"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """Load model with basic optimizations"""
        logger.info(f"🚀 Loading model from {self.model_path}")
        logger.info(f"🎯 Target device: {self.device}")
        
        start_time = time.time()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with memory-efficient settings
        logger.info("📦 Loading model weights...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,  # Use FP16 for memory efficiency
            device_map="auto",          # Auto device mapping
            low_cpu_mem_usage=True,     # Reduce CPU memory usage
            trust_remote_code=True      # Allow custom model code
        )
        
        # Set to eval mode
        self.model.eval()
        
        load_time = time.time() - start_time
        
        # Get model info
        num_params = sum(p.numel() for p in self.model.parameters())
        memory_usage = psutil.virtual_memory()
        
        logger.info(f"✅ Model loaded successfully!")
        logger.info(f"📊 Parameters: {num_params/1e9:.1f}B")
        logger.info(f"💾 Memory usage: {(memory_usage.total - memory_usage.available)/1e9:.1f}GB")
        logger.info(f"⏱️ Load time: {load_time:.1f}s")
        
        return True
    
    def generate_text(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7):
        """Generate text with the loaded model"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info(f"🎯 Generating text for: \"{prompt[:50]}...\"")
        
        start_time = time.time()
        
        # Tokenize input
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = inputs.to(self.device)
        
        input_length = inputs.shape[1]
        
        # Generate with optimized settings
        with torch.no_grad():
            generation_start = time.time()
            
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,  # Enable KV cache for faster generation
                num_return_sequences=1,
                repetition_penalty=1.1
            )
            
            generation_time = time.time() - generation_start
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new tokens
        new_text = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        total_time = time.time() - start_time
        output_length = outputs.shape[1] - input_length
        
        # Calculate performance metrics
        tokens_per_second = output_length / generation_time if generation_time > 0 else 0
        time_to_first_token = generation_time / output_length if output_length > 0 else generation_time
        
        logger.info(f"✅ Generation complete!")
        logger.info(f"📊 Input tokens: {input_length}")
        logger.info(f"📊 Output tokens: {output_length}")
        logger.info(f"🚀 Tokens/second: {tokens_per_second:.1f}")
        logger.info(f"⚡ Time to first token: {time_to_first_token*1000:.1f}ms")
        logger.info(f"⏱️ Total time: {total_time:.1f}s")
        
        return {
            "prompt": prompt,
            "generated_text": new_text,
            "full_text": generated_text,
            "metrics": {
                "input_tokens": input_length,
                "output_tokens": output_length,
                "tokens_per_second": tokens_per_second,
                "time_to_first_token_ms": time_to_first_token * 1000,
                "total_time_s": total_time,
                "generation_time_s": generation_time
            }
        }
    
    def interactive_chat(self):
        """Interactive chat interface"""
        logger.info("🦄 UNICORN EXECUTION ENGINE - Interactive Chat")
        logger.info("💬 Type 'quit' or 'exit' to stop")
        logger.info("=" * 50)
        
        conversation_history = []
        
        while True:
            try:
                user_input = input("\\n🤔 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    logger.info("👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Simple conversation formatting for Gemma
                if conversation_history:
                    # Include recent context
                    context = "\\n".join(conversation_history[-4:])  # Last 2 exchanges
                    prompt = f"{context}\\nUser: {user_input}\\nAssistant:"
                else:
                    prompt = f"User: {user_input}\\nAssistant:"
                
                # Generate response
                result = self.generate_text(prompt, max_tokens=150, temperature=0.7)
                response = result["generated_text"].strip()
                
                # Clean up response
                if response.startswith("Assistant:"):
                    response = response[10:].strip()
                
                # Stop at next "User:" if model continues conversation
                if "\\nUser:" in response:
                    response = response.split("\\nUser:")[0].strip()
                
                print(f"🤖 Assistant: {response}")
                
                # Update conversation history
                conversation_history.append(f"User: {user_input}")
                conversation_history.append(f"Assistant: {response}")
                
                # Show performance metrics
                metrics = result["metrics"]
                print(f"📊 {metrics['tokens_per_second']:.1f} tok/s, {metrics['time_to_first_token_ms']:.0f}ms TTFT")
                
            except KeyboardInterrupt:
                logger.info("\\n👋 Interrupted by user. Goodbye!")
                break
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                continue

def main():
    parser = argparse.ArgumentParser(description="Minimal Inference Engine")
    parser.add_argument("--model", default="./models/gemma-3-4b-it", help="Model path")
    parser.add_argument("--prompt", help="Single prompt to generate")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = MinimalInferenceEngine(args.model)
    
    # Load model
    engine.load_model()
    
    if args.interactive:
        # Interactive mode
        engine.interactive_chat()
    elif args.prompt:
        # Single prompt mode
        result = engine.generate_text(args.prompt, args.max_tokens, args.temperature)
        print(f"\\n🤖 Generated text:\\n{result['generated_text']}")
        print(f"\\n📊 Performance: {result['metrics']['tokens_per_second']:.1f} tok/s")
    else:
        # Default test
        test_prompts = [
            "The future of AI will be",
            "Explain quantum computing in simple terms:",
            "Write a short poem about mountains:"
        ]
        
        for prompt in test_prompts:
            print(f"\\n{'='*60}")
            result = engine.generate_text(prompt, max_tokens=80)
            print(f"Prompt: {prompt}")
            print(f"Response: {result['generated_text']}")
            print(f"Performance: {result['metrics']['tokens_per_second']:.1f} tok/s")

if __name__ == "__main__":
    main()