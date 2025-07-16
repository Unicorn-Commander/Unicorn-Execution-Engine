#!/usr/bin/env python3
"""
Gemma 3n E4B Real OpenAI API Server - ZERO SIMULATION
Uses actual model weights with NPU+iGPU acceleration
FAILS HARD if real components not available
"""

import os
import sys
import time
import logging
import asyncio
import uvicorn
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from gemma3n_e4b_real_implementation import Gemma3nE4BRealImplementation

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI API Models
class Message(BaseModel):
    role: str = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model to use for completion")
    messages: List[Message] = Field(..., description="List of messages")
    max_tokens: Optional[int] = Field(default=100, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for sampling")
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0, description="Top-p for nucleus sampling")
    stream: Optional[bool] = Field(default=False, description="Stream response")

class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage

class Gemma3nE4BRealAPIServer:
    """Real OpenAI API server for Gemma 3n E4B"""
    
    def __init__(self):
        self.model_id = "gemma-3n-e4b-it"
        self.implementation = None
        self.ready = False
        
        # Performance metrics
        self.metrics = {
            "requests_total": 0,
            "requests_successful": 0,
            "requests_failed": 0,
            "tokens_generated": 0,
            "total_inference_time": 0.0,
            "average_tokens_per_second": 0.0,
            "start_time": time.time()
        }
        
    async def initialize(self):
        """Initialize real implementation - FAIL HARD if not possible"""
        logger.info("🚀 Initializing REAL Gemma 3n E4B API Server...")
        
        try:
            # Initialize real implementation (will fail hard if hardware not available)
            self.implementation = Gemma3nE4BRealImplementation()
            
            # Configure real hardware
            self.implementation.configure_npu_acceleration()
            self.implementation.configure_igpu_acceleration()
            
            # Load real model
            self.implementation.load_real_model()
            
            # Optimize performance
            self.implementation.optimize_performance()
            
            self.ready = True
            logger.info("✅ REAL API SERVER READY")
            logger.info("   🦄 Model: Gemma 3n E4B MatFormer")
            logger.info("   🔥 Hardware: NPU Phoenix + AMD Radeon 780M")
            logger.info("   💾 Memory: 96GB HMA unified architecture")
            
        except Exception as e:
            logger.error(f"❌ CRITICAL: Real API server initialization failed: {e}")
            raise RuntimeError(f"Real implementation failed: {e}")
            
    def convert_messages_to_prompt(self, messages: List[Message]) -> str:
        """Convert OpenAI messages to Gemma 3n E4B prompt format"""
        prompt_parts = []
        
        for message in messages:
            role = message.role.lower()
            content = message.content.strip()
            
            if role == "system":
                prompt_parts.append(f"<|system|>\n{content}\n<|end|>")
            elif role == "user":
                prompt_parts.append(f"<|user|>\n{content}\n<|end|>")
            elif role == "assistant":
                prompt_parts.append(f"<|assistant|>\n{content}\n<|end|>")
        
        # Add assistant prompt for completion
        prompt_parts.append("<|assistant|>\n")
        
        return "\n".join(prompt_parts)
        
    async def generate_completion(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Generate completion using REAL implementation"""
        if not self.ready or not self.implementation:
            raise HTTPException(
                status_code=503,
                detail="Real model not ready. Check hardware and model loading."
            )
            
        # Convert messages to prompt
        prompt = self.convert_messages_to_prompt(request.messages)
        
        # Track metrics
        self.metrics["requests_total"] += 1
        
        try:
            # Use REAL inference (will fail if not working)
            result = self.implementation.real_inference(
                prompt=prompt,
                max_tokens=request.max_tokens
            )
            
            # Verify we got real results
            if not result.get("real_inference", False):
                raise HTTPException(
                    status_code=500,
                    detail="CRITICAL: Simulation detected - real inference required"
                )
                
            # Update metrics with real data
            self.metrics["requests_successful"] += 1
            self.metrics["tokens_generated"] += result["tokens_generated"]
            self.metrics["total_inference_time"] += result["generation_time"]
            
            if self.metrics["requests_successful"] > 0:
                self.metrics["average_tokens_per_second"] = (
                    self.metrics["tokens_generated"] / self.metrics["total_inference_time"]
                )
                
            return result
            
        except Exception as e:
            self.metrics["requests_failed"] += 1
            logger.error(f"Real inference failed: {e}")
            raise HTTPException(status_code=500, detail=f"Real inference failed: {e}")

# Create FastAPI app
app = FastAPI(
    title="Gemma 3n E4B Real OpenAI API Server",
    description="Real NPU+iGPU accelerated API server - NO SIMULATION",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize API server
api_server = Gemma3nE4BRealAPIServer()

@app.on_event("startup")
async def startup_event():
    """Startup event - initialize real implementation"""
    try:
        await api_server.initialize()
    except Exception as e:
        logger.error(f"❌ STARTUP FAILED: {e}")
        # Don't start server if real implementation failed
        raise e

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    if not api_server.ready:
        raise HTTPException(status_code=503, detail="Real model not ready")
        
    return {
        "status": "healthy",
        "model_ready": api_server.ready,
        "model_id": api_server.model_id,
        "real_implementation": True,
        "hardware": {
            "npu_detected": api_server.implementation.hardware.npu_detected,
            "igpu_detected": api_server.implementation.hardware.igpu_detected,
            "hma_total_gb": api_server.implementation.hardware.hma_total_gb
        },
        "metrics": api_server.metrics,
        "timestamp": time.time()
    }

@app.get("/v1/models")
async def list_models():
    """List available models"""
    return {
        "object": "list",
        "data": [{
            "id": api_server.model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "google",
            "permission": [],
            "real_implementation": True
        }]
    }

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create chat completion using REAL model"""
    
    # Validate request
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")
        
    if request.model != api_server.model_id:
        raise HTTPException(
            status_code=400,
            detail=f"Model {request.model} not supported. Use {api_server.model_id}"
        )
        
    # Generate using REAL implementation
    result = await api_server.generate_completion(request)
    
    # Create OpenAI-compatible response
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())
    
    response = ChatCompletionResponse(
        id=completion_id,
        created=created_time,
        model=request.model,
        choices=[ChatCompletionChoice(
            index=0,
            message=Message(
                role="assistant",
                content=result["response"]
            ),
            finish_reason="stop"
        )],
        usage=ChatCompletionUsage(
            prompt_tokens=result["input_tokens"],
            completion_tokens=result["tokens_generated"],
            total_tokens=result["input_tokens"] + result["tokens_generated"]
        )
    )
    
    # Log real performance
    logger.info(f"✅ Real completion: {result['tokens_per_second']:.1f} TPS")
    
    return response

@app.get("/v1/metrics")
async def get_metrics():
    """Get real performance metrics"""
    
    uptime = time.time() - api_server.metrics["start_time"]
    
    return {
        "metrics": api_server.metrics,
        "real_implementation": True,
        "hardware_status": {
            "npu_utilization": api_server.implementation.hardware.npu_utilization if api_server.ready else 0,
            "igpu_memory_gb": api_server.implementation.hardware.igpu_memory_gb if api_server.ready else 0,
            "hma_total_gb": api_server.implementation.hardware.hma_total_gb if api_server.ready else 0
        },
        "timestamp": time.time(),
        "uptime": uptime
    }

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "type": "real_implementation_error",
                "code": exc.status_code
            }
        }
    )

def main():
    """Main function to run the REAL API server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gemma 3n E4B Real OpenAI API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    logger.info("🦄 Starting Gemma 3n E4B REAL OpenAI API Server")
    logger.info("=" * 60)
    logger.info("🔥 ZERO SIMULATION - Real hardware acceleration only")
    logger.info(f"🌐 Server URL: http://{args.host}:{args.port}")
    logger.info(f"📚 Documentation: http://{args.host}:{args.port}/docs")
    logger.info("=" * 60)
    
    # Start server
    uvicorn.run(
        "gemma3n_e4b_real_api_server:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=False
    )

if __name__ == "__main__":
    main()