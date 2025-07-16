#!/usr/bin/env python3
"""
Simple Qwen API Server
Uses the working Qwen 2.5 7B model for real inference
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI API Models
class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model to use")
    messages: List[Dict[str, str]] = Field(..., description="Messages")
    max_tokens: Optional[int] = Field(default=200, description="Maximum tokens")
    temperature: Optional[float] = Field(default=0.7, description="Temperature")
    top_p: Optional[float] = Field(default=0.9, description="Top-p sampling")
    stream: Optional[bool] = Field(default=False, description="Stream response")
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "unicorn-execution-engine"

class SimpleQwenManager:
    """Simple Qwen model manager using the working 7B model"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        
        # Use the working 7B model
        self.model_path = "./models/qwen2.5-7b-instruct"
        self.model_id = "qwen2.5-7b-instruct"
        
        # Check if model exists
        if not os.path.exists(self.model_path):
            logger.error(f"❌ Model not found: {self.model_path}")
            return
        
        logger.info(f"✅ Model found: {self.model_path}")
        self.load_model()
    
    def load_model(self):
        """Load the Qwen 2.5 7B model"""
        try:
            logger.info("📥 Loading Qwen 2.5 7B model...")
            
            # Load tokenizer
            logger.info("   🔤 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Load model with proper settings
            logger.info("   🧠 Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Get model info
            param_count = sum(p.numel() for p in self.model.parameters())
            logger.info(f"   ✅ Loaded Qwen 2.5 7B: {param_count:,} parameters")
            
            self.model_loaded = True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            self.model_loaded = False
    
    def generate_response(self, messages: List[Dict[str, str]], 
                         max_tokens: int = 200, temperature: float = 0.7, 
                         top_p: float = 0.9, stop: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate response using the model"""
        
        if not self.model_loaded:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Convert messages to prompt
        prompt = self.messages_to_prompt(messages)
        
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_length = inputs.input_ids.shape[1]
            
            # Generate response
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    use_cache=True
                )
            
            generation_time = time.time() - start_time
            
            # Decode response
            response_text = self.tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True
            )
            
            # Clean up response
            response_text = response_text.strip()
            
            # Handle stop sequences
            if stop:
                for stop_seq in stop:
                    if stop_seq in response_text:
                        response_text = response_text.split(stop_seq)[0]
                        break
            
            # Calculate tokens
            prompt_tokens = input_length
            completion_tokens = len(self.tokenizer.encode(response_text))
            total_tokens = prompt_tokens + completion_tokens
            
            # Create response
            response = {
                "id": f"chatcmpl-{int(time.time())}-qwen",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.model_id,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                },
                "performance": {
                    "generation_time": generation_time,
                    "tokens_per_second": completion_tokens / generation_time if generation_time > 0 else 0
                }
            }
            
            logger.info(f"✅ Generated {completion_tokens} tokens in {generation_time:.2f}s "
                       f"({completion_tokens/generation_time:.1f} TPS)")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    
    def messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to Qwen prompt format"""
        prompt_parts = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        
        # Add assistant start for generation
        prompt_parts.append("<|im_start|>assistant\n")
        
        return "\n".join(prompt_parts)
    
    def get_available_models(self) -> List[ModelInfo]:
        """Get available models"""
        if self.model_loaded:
            return [ModelInfo(
                id=self.model_id,
                created=int(time.time()),
                owned_by="unicorn-execution-engine"
            )]
        return []

# Initialize FastAPI app
app = FastAPI(
    title="Simple Qwen API Server",
    description="Simple working Qwen 2.5 API server",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model manager
model_manager = SimpleQwenManager()

@app.get("/")
async def root():
    return {
        "message": "Simple Qwen API Server",
        "model_loaded": model_manager.model_loaded,
        "model": model_manager.model_id if model_manager.model_loaded else None
    }

@app.get("/v1/models")
async def list_models():
    models = model_manager.get_available_models()
    return {
        "object": "list",
        "data": [model.dict() for model in models]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        if not model_manager.model_loaded:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        response = model_manager.generate_response(
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Chat completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model_manager.model_loaded else "unhealthy",
        "model_loaded": model_manager.model_loaded,
        "model": model_manager.model_id if model_manager.model_loaded else None
    }

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    
    args = parser.parse_args()
    
    logger.info("🦄 Starting Simple Qwen API Server")
    logger.info(f"🧠 Model loaded: {model_manager.model_loaded}")
    logger.info(f"🌐 Server URL: http://{args.host}:{args.port}")
    
    uvicorn.run(
        "qwen_simple_api_server:app",
        host=args.host,
        port=args.port,
        reload=False
    )

if __name__ == "__main__":
    main()