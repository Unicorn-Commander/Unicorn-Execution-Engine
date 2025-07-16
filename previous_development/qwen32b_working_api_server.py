#!/usr/bin/env python3
"""
Qwen 2.5 32B Working API Server
Real model inference with transformers library
"""

import os
import sys
import json
import time
import asyncio
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
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
    max_tokens: Optional[int] = Field(default=512, description="Maximum tokens")
    temperature: Optional[float] = Field(default=0.7, description="Temperature")
    top_p: Optional[float] = Field(default=0.9, description="Top-p sampling")
    stream: Optional[bool] = Field(default=False, description="Stream response")
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "unicorn-execution-engine"

class Qwen32BRealModelManager:
    """Real Qwen 2.5 32B model manager with actual inference"""
    
    def __init__(self):
        self.models = {}
        self.current_model = None
        self.current_tokenizer = None
        self.model_configs = {
            "qwen2.5-32b-instruct": {
                "path": "./models/qwen2.5-32b-instruct",
                "name": "Qwen 2.5 32B Instruct",
                "parameters": "32B",
                "context_length": 32768,
                "description": "Qwen 2.5 32B with real model inference"
            }
        }
        
        # Check available models
        self.available_models = {}
        for model_id, config in self.model_configs.items():
            if os.path.exists(config["path"]):
                self.available_models[model_id] = config
                logger.info(f"✅ Available model: {model_id}")
            else:
                logger.warning(f"❌ Model not found: {model_id}")
        
        if not self.available_models:
            logger.error("❌ No Qwen 2.5 32B models found!")
            return
        
        # Load default model
        self.load_default_model()
    
    def load_default_model(self):
        """Load the first available model"""
        if self.available_models:
            default_model = list(self.available_models.keys())[0]
            logger.info(f"🔄 Loading default model: {default_model}")
            self.load_model(default_model)
    
    def load_model(self, model_id: str) -> bool:
        """Load a specific Qwen 2.5 32B model"""
        if model_id not in self.available_models:
            logger.error(f"❌ Model not available: {model_id}")
            return False
        
        if self.current_model and model_id in self.models:
            logger.info(f"✅ Model {model_id} already loaded")
            self.current_model = self.models[model_id]["model"]
            self.current_tokenizer = self.models[model_id]["tokenizer"]
            return True
        
        try:
            config = self.available_models[model_id]
            logger.info(f"📥 Loading {config['name']}...")
            
            # Load tokenizer
            logger.info("   🔤 Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                config["path"],
                trust_remote_code=True
            )
            
            # Load model
            logger.info("   🧠 Loading model...")
            model = AutoModelForCausalLM.from_pretrained(
                config["path"],
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Store model
            self.models[model_id] = {
                "model": model,
                "tokenizer": tokenizer,
                "config": config
            }
            
            self.current_model = model
            self.current_tokenizer = tokenizer
            
            # Get model info
            param_count = sum(p.numel() for p in model.parameters())
            logger.info(f"   ✅ Loaded {config['name']}: {param_count:,} parameters")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load {model_id}: {e}")
            return False
    
    def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models"""
        models = []
        for model_id, config in self.available_models.items():
            models.append(ModelInfo(
                id=model_id,
                created=int(time.time()),
                owned_by="unicorn-execution-engine"
            ))
        return models
    
    def generate_response(self, messages: List[Dict[str, str]], model_id: str, 
                         max_tokens: int = 512, temperature: float = 0.7, 
                         top_p: float = 0.9, stop: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate response using real model inference"""
        
        # Load model if needed
        if not self.current_model or model_id not in self.models:
            if not self.load_model(model_id):
                raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        # Convert messages to prompt
        prompt = self.messages_to_prompt(messages)
        
        try:
            # Tokenize input
            inputs = self.current_tokenizer(prompt, return_tensors="pt")
            input_length = inputs.input_ids.shape[1]
            
            # Generate response
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.current_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.current_tokenizer.eos_token_id,
                    eos_token_id=self.current_tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    use_cache=True
                )
            
            generation_time = time.time() - start_time
            
            # Decode response
            response_text = self.current_tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True
            )
            
            # Handle stop sequences
            if stop:
                for stop_seq in stop:
                    if stop_seq in response_text:
                        response_text = response_text.split(stop_seq)[0]
                        break
            
            # Calculate tokens
            prompt_tokens = input_length
            completion_tokens = len(self.current_tokenizer.encode(response_text))
            total_tokens = prompt_tokens + completion_tokens
            
            # Create response
            response = {
                "id": f"chatcmpl-{int(time.time())}-qwen32b",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_id,
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
                    "tokens_per_second": completion_tokens / generation_time if generation_time > 0 else 0,
                    "hardware": "Real Qwen 2.5 32B inference"
                }
            }
            
            logger.info(f"✅ Generated {completion_tokens} tokens in {generation_time:.2f}s "
                       f"({completion_tokens/generation_time:.1f} TPS)")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
    
    def messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI messages to Qwen prompt format"""
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

# Initialize FastAPI app
app = FastAPI(
    title="Qwen 2.5 32B Real API Server",
    description="Real model inference for Qwen 2.5 32B",
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
model_manager = Qwen32BRealModelManager()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Qwen 2.5 32B Real API Server",
        "version": "1.0.0",
        "inference": "Real model inference",
        "available_models": len(model_manager.available_models),
        "docs": "/docs"
    }

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatible)"""
    models = model_manager.get_available_models()
    return {
        "object": "list",
        "data": [model.dict() for model in models]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Create chat completion (OpenAI compatible)"""
    try:
        # Validate model
        if request.model not in model_manager.available_models:
            raise HTTPException(
                status_code=404, 
                detail=f"Model {request.model} not found. Available: {list(model_manager.available_models.keys())}"
            )
        
        # Generate response
        response = model_manager.generate_response(
            messages=request.messages,
            model_id=request.model,
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

@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    """Get specific model info"""
    if model_id not in model_manager.available_models:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    config = model_manager.available_models[model_id]
    return {
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "unicorn-execution-engine",
        "name": config["name"],
        "parameters": config["parameters"],
        "context_length": config["context_length"],
        "description": config["description"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": int(time.time()),
        "loaded_models": len(model_manager.models),
        "available_models": len(model_manager.available_models),
        "inference_type": "Real model inference"
    }

@app.get("/metrics")
async def get_metrics():
    """Get performance metrics"""
    return {
        "server": "qwen32b-real-api",
        "models_loaded": len(model_manager.models),
        "models_available": len(model_manager.available_models),
        "inference_type": "Real Qwen 2.5 32B",
        "hardware": "CPU/GPU inference",
        "timestamp": int(time.time())
    }

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen 2.5 32B Real API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    logger.info("🦄 Starting Qwen 2.5 32B Real API Server")
    logger.info(f"📊 Available models: {len(model_manager.available_models)}")
    logger.info(f"🧠 Inference: Real model inference")
    logger.info(f"🌐 Server URL: http://{args.host}:{args.port}")
    
    # Print available models
    for model_id, config in model_manager.available_models.items():
        logger.info(f"   📦 {model_id}: {config['name']}")
    
    # Start server
    uvicorn.run(
        "qwen32b_working_api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()