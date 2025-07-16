#!/usr/bin/env python3
"""
Pure Hardware API Server FINAL - With Proper GPU Memory Allocation
Model loads to VRAM/GTT using Vulkan, not system RAM
OpenAI v1 API compatible
"""

import os
import sys
import asyncio
import time
import logging
import traceback
import psutil
import subprocess
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import json
import uuid
import numpy as np

# Import our fixed pipeline
from pure_hardware_pipeline_final import PureHardwarePipelineFinal

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pydantic models for OpenAI API
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="gemma-3-27b-pure-hardware", description="Model to use")
    messages: List[ChatMessage] = Field(..., description="List of messages")
    max_tokens: Optional[int] = Field(default=100, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=0.7, description="Sampling temperature")
    top_p: Optional[float] = Field(default=0.9, description="Top-p sampling")
    stream: Optional[bool] = Field(default=False, description="Stream the response")

# Global state
pipeline = None
model_loaded = False

# Simple tokenizer (replace with real one later)
class SimpleTokenizer:
    def __init__(self):
        self.vocab_size = 32000
        
    def encode(self, text: str) -> List[int]:
        # Basic character-level encoding
        tokens = []
        for char in text:
            token = ord(char) % self.vocab_size
            tokens.append(token)
        return tokens[:100]  # Limit length
    
    def decode(self, tokens: List[int]) -> str:
        # Basic decoding
        chars = []
        for token in tokens:
            if token < 128:
                chars.append(chr(token))
            else:
                chars.append('?')
        return ''.join(chars)

tokenizer = SimpleTokenizer()

# FastAPI app
app = FastAPI(
    title="Pure Hardware Gemma 27B API FINAL",
    description="OpenAI v1 compatible API with proper VRAM/GTT allocation via Vulkan",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_gpu_memory_stats():
    """Get current GPU memory usage"""
    stats = {}
    
    try:
        # VRAM
        result = subprocess.run(['rocm-smi', '--showmeminfo', 'vram'], 
                              capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'Total Memory' in line and 'GPU[0]' in line:
                stats['vram_total_gb'] = int(line.split(':')[-1].strip()) / (1024**3)
            elif 'Used Memory' in line and 'GPU[0]' in line:
                stats['vram_used_gb'] = int(line.split(':')[-1].strip()) / (1024**3)
        
        # GTT
        result = subprocess.run(['rocm-smi', '--showmeminfo', 'gtt'], 
                              capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'Total Memory' in line and 'GPU[0]' in line:
                stats['gtt_total_gb'] = int(line.split(':')[-1].strip()) / (1024**3)
            elif 'Used Memory' in line and 'GPU[0]' in line:
                stats['gtt_used_gb'] = int(line.split(':')[-1].strip()) / (1024**3)
    except:
        pass
    
    return stats

async def load_model():
    """Load model using fixed pipeline"""
    global pipeline, model_loaded
    
    try:
        logger.info("🚀 STARTING MODEL LOADING WITH PROPER GPU MEMORY")
        logger.info("🎮 Using Vulkan for VRAM/GTT allocation")
        logger.info("🧠 Using NPU kernels for attention")
        logger.info("⚡ No PyTorch/ROCm dependencies!")
        
        model_path = "./quantized_models/gemma-3-27b-it-layer-by-layer"
        if not Path(model_path).exists():
            logger.error(f"❌ Model path not found: {model_path}")
            return False
        
        # Initialize pipeline
        pipeline = PureHardwarePipelineFinal()
        
        if not pipeline.initialize(model_path):
            logger.error("❌ Failed to initialize pipeline")
            return False
        
        model_loaded = True
        
        logger.info("🎉 MODEL LOADING COMPLETE!")
        
        # Show memory distribution
        logger.info(f"📊 Model Memory Distribution:")
        logger.info(f"   VRAM layers: {len(pipeline.vram_layers)}")
        logger.info(f"   GTT layers: {len(pipeline.gtt_layers)}")
        logger.info(f"   CPU layers: {len(pipeline.cpu_layers)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        logger.error(traceback.format_exc())
        return False

async def generate_response(prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
    """Generate response using the model"""
    
    if not model_loaded or not pipeline:
        raise RuntimeError("Model not loaded")
    
    logger.info(f"🚀 Generating response for prompt: {prompt[:50]}...")
    
    start_time = time.time()
    
    try:
        # Encode prompt
        input_ids = tokenizer.encode(prompt)
        logger.info(f"   Encoded to {len(input_ids)} tokens")
        
        # Generate tokens
        generated_tokens = pipeline.generate_tokens(
            input_ids, 
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Decode response
        response_text = tokenizer.decode(generated_tokens)
        
        total_time = time.time() - start_time
        tps = len(generated_tokens) / total_time if total_time > 0 else 0
        
        logger.info(f"✅ Generated {len(generated_tokens)} tokens in {total_time:.2f}s ({tps:.1f} TPS)")
        
        return {
            "response": response_text,
            "tokens_generated": len(generated_tokens),
            "generation_time": total_time,
            "tokens_per_second": tps
        }
        
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Generation failed: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    await load_model()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    gpu_stats = get_gpu_memory_stats()
    
    distribution = {}
    if pipeline:
        distribution = {
            "vram_layers": len(pipeline.vram_layers),
            "gtt_layers": len(pipeline.gtt_layers),
            "cpu_layers": len(pipeline.cpu_layers),
            "total_layers": 62
        }
    
    return {
        "status": "healthy" if model_loaded else "loading",
        "model_loaded": model_loaded,
        "framework": "Pure Hardware (Vulkan + NPU)",
        "memory_stats": {
            "process_rss_gb": memory_info.rss / (1024**3),
            "vram_used_gb": gpu_stats.get('vram_used_gb', 0),
            "vram_total_gb": gpu_stats.get('vram_total_gb', 0),
            "gtt_used_gb": gpu_stats.get('gtt_used_gb', 0),
            "gtt_total_gb": gpu_stats.get('gtt_total_gb', 0),
        },
        "model_distribution": distribution
    }

@app.get("/v1/models")
async def list_models():
    """List available models"""
    models = [
        {
            "id": "gemma-3-27b-pure-hardware",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "pure-hardware-vulkan",
            "description": "Gemma 3 27B with proper VRAM/GTT allocation"
        }
    ]
    return {"object": "list", "data": models}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI v1 compatible chat completions"""
    try:
        if not model_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Extract prompt from messages
        prompt = ""
        for message in request.messages:
            prompt += f"{message.role}: {message.content}\n"
        prompt += "assistant: "
        
        # Generate response
        result = await generate_response(
            prompt, 
            request.max_tokens, 
            request.temperature
        )
        
        # Format as OpenAI response
        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result["response"]
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt),
                "completion_tokens": result["tokens_generated"],
                "total_tokens": len(prompt) + result["tokens_generated"]
            },
            "system_fingerprint": "vulkan-npu-pure-hardware"
        }
        
        if request.stream:
            # Streaming not implemented yet
            return response
        else:
            return response
        
    except Exception as e:
        logger.error(f"❌ Chat completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Pure Hardware Gemma 27B API Server FINAL",
        "version": "2.0.0",
        "features": [
            "✅ Model loads to VRAM/GTT using Vulkan",
            "✅ NPU kernels for attention",
            "✅ No PyTorch/ROCm dependencies",
            "✅ OpenAI v1 compatible API"
        ],
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    print("🦄 PURE HARDWARE GEMMA 27B API SERVER FINAL")
    print("=" * 60)
    print("🚀 Model loads to VRAM/GTT using Vulkan")
    print("🧠 NPU kernels for attention computation")
    print("🎮 Direct GPU acceleration without ML frameworks")
    print("⚡ No PyTorch/ROCm dependencies!")
    print("📡 Server: http://localhost:8008")
    print("🛑 Press Ctrl+C to stop")
    print()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8008,
        log_level="info"
    )