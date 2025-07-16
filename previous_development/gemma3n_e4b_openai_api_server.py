#!/usr/bin/env python3
"""
Gemma 3n E4B OpenAI API Server
Production-ready OpenAI v1 compatible API server with Unicorn loader integration
"""

import os
import sys
import json
import time
import logging
import asyncio
from typing import Dict, List, Optional, Any, AsyncGenerator
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

# Import our Unicorn loader
from gemma3n_e4b_unicorn_loader import (
    Gemma3nE4BUnicornLoader, 
    ModelConfig, 
    HardwareConfig, 
    InferenceConfig,
    InferenceMode,
    LoaderState
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI API Models
class Message(BaseModel):
    role: str = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")
    name: Optional[str] = Field(None, description="Name of the sender")

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model to use for completion")
    messages: List[Message] = Field(..., description="List of messages")
    max_tokens: Optional[int] = Field(default=512, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for sampling")
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0, description="Top-p for nucleus sampling")
    top_k: Optional[int] = Field(default=50, ge=1, le=100, description="Top-k for sampling")
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")
    repetition_penalty: Optional[float] = Field(default=1.1, ge=0.0, le=2.0, description="Repetition penalty")
    stop: Optional[List[str]] = Field(default=None, max_items=4, description="Stop sequences")
    stream: Optional[bool] = Field(default=False, description="Stream response")
    n: Optional[int] = Field(default=1, ge=1, le=1, description="Number of completions")
    user: Optional[str] = Field(default=None, description="User identifier")

class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str
    logprobs: Optional[Any] = None

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
    system_fingerprint: Optional[str] = None

class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None

class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]
    usage: Optional[ChatCompletionUsage] = None
    system_fingerprint: Optional[str] = None

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "unicorn-execution-engine"
    permission: List[Any] = []
    root: Optional[str] = None
    parent: Optional[str] = None

class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]

class ErrorResponse(BaseModel):
    error: Dict[str, Any]

class Gemma3nE4BAPIServer:
    """OpenAI v1 compatible API server for Gemma 3n E4B"""
    
    def __init__(self, model_path: str = "./models/gemma-3n-e4b-it"):
        self.model_path = model_path
        self.model_id = "gemma-3n-e4b-it"
        self.system_fingerprint = "fp_unicorn_" + str(uuid.uuid4())[:8]
        
        # Configure model and hardware
        self.model_config = ModelConfig(
            model_path=model_path,
            elastic_enabled=True,
            quantization_enabled=True,
            mix_n_match_enabled=True
        )
        
        self.hardware_config = HardwareConfig(
            npu_enabled=True,
            igpu_enabled=True,
            hma_enabled=True,
            turbo_mode=True,
            zero_copy_enabled=True
        )
        
        # Initialize Unicorn loader
        self.loader = None
        self.loader_ready = False
        
        # Performance metrics
        self.metrics = {
            "requests_total": 0,
            "requests_successful": 0,
            "requests_failed": 0,
            "tokens_generated": 0,
            "total_inference_time": 0.0,
            "average_tokens_per_second": 0.0,
            "active_elastic_params": 0,
            "memory_usage": 0
        }
        
        # Request tracking
        self.active_requests = {}
        
        # Initialize loader synchronously for now
        # Will be initialized properly when the server starts
    
    async def initialize_loader(self):
        """Initialize the Unicorn loader asynchronously"""
        try:
            logger.info("🚀 Initializing Gemma 3n E4B Unicorn Loader...")
            
            # Initialize loader
            self.loader = Gemma3nE4BUnicornLoader(self.model_config, self.hardware_config)
            
            # Load model
            if self.loader.load_model():
                self.loader_ready = True
                logger.info("✅ Unicorn loader initialized and ready!")
                
                # Update metrics safely
                try:
                    status = self.loader.get_status()
                    if isinstance(status, dict):
                        self.metrics["active_elastic_params"] = status.get("active_elastic_params", 0)
                        
                        memory_status = status.get("memory_status", {})
                        if isinstance(memory_status, dict) and "allocation_stats" in memory_status:
                            memory_stats = memory_status.get("allocation_stats", {})
                            if isinstance(memory_stats, dict):
                                self.metrics["memory_usage"] = memory_stats.get("total_allocated", 0)
                    else:
                        logger.warning(f"Loader returned non-dict status: {type(status)}")
                except Exception as status_error:
                    logger.warning(f"Could not update metrics: {status_error}")
                    # Continue anyway - loader is still functional
                
            else:
                logger.error("❌ Failed to load model")
                self.loader_ready = False
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize loader: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.loader_ready = False
    
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
    
    def create_inference_config(self, request: ChatCompletionRequest) -> InferenceConfig:
        """Create inference configuration from request"""
        
        # Determine inference mode based on request parameters
        if request.temperature > 1.0:
            mode = InferenceMode.PERFORMANCE
        elif request.temperature < 0.3:
            mode = InferenceMode.EFFICIENCY
        else:
            mode = InferenceMode.BALANCED
        
        return InferenceConfig(
            mode=mode,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            elastic_scaling=True,
            dynamic_allocation=True
        )
    
    async def generate_completion(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Generate completion using Unicorn loader"""
        
        if not self.loader_ready or not self.loader:
            raise HTTPException(
                status_code=503,
                detail="Model not ready. Please wait for initialization to complete."
            )
        
        # Convert messages to prompt
        prompt = self.convert_messages_to_prompt(request.messages)
        
        # Create inference config
        inference_config = self.create_inference_config(request)
        
        # Generate response
        start_time = time.time()
        
        try:
            # Generate using Unicorn loader
            result = self.loader.generate(prompt, inference_config)
            
            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])
            
            # Update metrics
            self.metrics["requests_successful"] += 1
            self.metrics["tokens_generated"] += result["tokens_generated"]
            self.metrics["total_inference_time"] += result["inference_time"]
            self.metrics["active_elastic_params"] = result["elastic_params_active"]
            self.metrics["memory_usage"] = result["memory_usage"]
            
            # Calculate average TPS
            if self.metrics["requests_successful"] > 0:
                self.metrics["average_tokens_per_second"] = (
                    self.metrics["tokens_generated"] / self.metrics["total_inference_time"]
                )
            
            return result
            
        except Exception as e:
            self.metrics["requests_failed"] += 1
            raise HTTPException(status_code=500, detail=str(e))
    
    async def stream_completion(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        """Stream completion using Unicorn loader"""
        
        if not self.loader_ready or not self.loader:
            error_chunk = ChatCompletionStreamResponse(
                id=f"chatcmpl-{uuid.uuid4()}",
                created=int(time.time()),
                model=request.model,
                choices=[ChatCompletionStreamChoice(
                    index=0,
                    delta={"role": "assistant", "content": ""},
                    finish_reason="error"
                )]
            )
            yield f"data: {error_chunk.json()}\n\n"
            return
        
        # Generate completion
        result = await self.generate_completion(request)
        
        if "error" in result:
            error_chunk = ChatCompletionStreamResponse(
                id=f"chatcmpl-{uuid.uuid4()}",
                created=int(time.time()),
                model=request.model,
                choices=[ChatCompletionStreamChoice(
                    index=0,
                    delta={"role": "assistant", "content": ""},
                    finish_reason="error"
                )]
            )
            yield f"data: {error_chunk.json()}\n\n"
            return
        
        # Stream the response
        completion_id = f"chatcmpl-{uuid.uuid4()}"
        created_time = int(time.time())
        generated_text = result["generated_text"]
        
        # Send initial chunk
        initial_chunk = ChatCompletionStreamResponse(
            id=completion_id,
            created=created_time,
            model=request.model,
            choices=[ChatCompletionStreamChoice(
                index=0,
                delta={"role": "assistant", "content": ""},
                finish_reason=None
            )]
        )
        yield f"data: {initial_chunk.json()}\n\n"
        
        # Stream content in chunks
        chunk_size = 5  # Words per chunk
        words = generated_text.split()
        
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_content = " ".join(chunk_words)
            
            if i + chunk_size < len(words):
                chunk_content += " "
            
            chunk = ChatCompletionStreamResponse(
                id=completion_id,
                created=created_time,
                model=request.model,
                choices=[ChatCompletionStreamChoice(
                    index=0,
                    delta={"content": chunk_content},
                    finish_reason=None
                )]
            )
            yield f"data: {chunk.json()}\n\n"
            
            # Small delay for streaming effect
            await asyncio.sleep(0.05)
        
        # Send final chunk
        final_chunk = ChatCompletionStreamResponse(
            id=completion_id,
            created=created_time,
            model=request.model,
            choices=[ChatCompletionStreamChoice(
                index=0,
                delta={},
                finish_reason="stop"
            )],
            usage=ChatCompletionUsage(
                prompt_tokens=len(request.messages),
                completion_tokens=result["tokens_generated"],
                total_tokens=len(request.messages) + result["tokens_generated"]
            )
        )
        yield f"data: {final_chunk.json()}\n\n"
        yield "data: [DONE]\n\n"

# Initialize API server
api_server = Gemma3nE4BAPIServer()

# Create FastAPI app
app = FastAPI(
    title="Gemma 3n E4B OpenAI API Server",
    description="OpenAI v1 compatible API server for Gemma 3n E4B with Unicorn Execution Engine",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Track request
    request_id = str(uuid.uuid4())
    api_server.active_requests[request_id] = {
        "method": request.method,
        "url": str(request.url),
        "start_time": start_time
    }
    
    # Process request
    response = await call_next(request)
    
    # Log completion
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url} - {response.status_code} - {process_time:.3f}s")
    
    # Update metrics
    api_server.metrics["requests_total"] += 1
    
    # Clean up request tracking
    api_server.active_requests.pop(request_id, None)
    
    return response

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    health_status = {
        "status": "healthy" if api_server.loader_ready else "loading",
        "model_ready": api_server.loader_ready,
        "model_id": api_server.model_id,
        "system_fingerprint": api_server.system_fingerprint,
        "metrics": api_server.metrics,
        "active_requests": len(api_server.active_requests)
    }
    
    if api_server.loader and api_server.loader_ready:
        loader_status = api_server.loader.get_status()
        health_status["loader_state"] = loader_status["state"]
        health_status["components"] = loader_status["components"]
    
    return health_status

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Gemma 3n E4B OpenAI API Server",
        "version": "1.0.0",
        "model": api_server.model_id,
        "ready": api_server.loader_ready,
        "documentation": "/docs"
    }

# Models endpoint
@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """List available models"""
    
    models = [
        ModelInfo(
            id=api_server.model_id,
            created=int(time.time()),
            owned_by="unicorn-execution-engine"
        )
    ]
    
    return ModelsResponse(data=models)

# Chat completions endpoint
@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create chat completion"""
    
    # Validate request
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")
    
    if request.model != api_server.model_id:
        raise HTTPException(
            status_code=400, 
            detail=f"Model {request.model} not found. Available: {api_server.model_id}"
        )
    
    # Handle streaming
    if request.stream:
        return StreamingResponse(
            api_server.stream_completion(request),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    
    # Generate completion
    result = await api_server.generate_completion(request)
    
    # Create response
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
                content=result["generated_text"]
            ),
            finish_reason="stop"
        )],
        usage=ChatCompletionUsage(
            prompt_tokens=len(request.messages),
            completion_tokens=result["tokens_generated"],
            total_tokens=len(request.messages) + result["tokens_generated"]
        ),
        system_fingerprint=api_server.system_fingerprint
    )
    
    return response

# Performance metrics endpoint
@app.get("/v1/metrics")
async def get_metrics():
    """Get performance metrics"""
    
    metrics = api_server.metrics.copy()
    
    if api_server.loader and api_server.loader_ready:
        loader_status = api_server.loader.get_status()
        metrics.update({
            "loader_metrics": loader_status.get("performance_metrics", {}),
            "active_elastic_params": loader_status.get("active_elastic_params", 0),
            "components_status": loader_status.get("components", {}),
            "memory_status": loader_status.get("memory_status", {})
        })
    
    return {
        "metrics": metrics,
        "timestamp": time.time(),
        "uptime": time.time() - api_server.metrics.get("start_time", time.time())
    }

# Model optimization endpoint
@app.post("/v1/optimize")
async def optimize_model():
    """Optimize model performance"""
    
    if not api_server.loader_ready or not api_server.loader:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    try:
        optimization_results = api_server.loader.optimize_performance()
        
        return {
            "status": "success",
            "optimization_results": optimization_results,
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Error handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error={
                "message": exc.detail,
                "type": "invalid_request_error",
                "code": exc.status_code
            }
        ).dict()
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    api_server.metrics["start_time"] = time.time()
    logger.info("🚀 Gemma 3n E4B OpenAI API Server starting...")
    
    # Initialize the loader
    await api_server.initialize_loader()

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("🛑 Gemma 3n E4B OpenAI API Server shutting down...")
    
    if api_server.loader:
        api_server.loader.shutdown()

def main():
    """Main function to run the API server"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Gemma 3n E4B OpenAI API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--log-level", default="info", help="Log level")
    parser.add_argument("--model-path", default="./models/gemma-3n-e4b-it", help="Path to model")
    
    args = parser.parse_args()
    
    # Update model path
    api_server.model_path = args.model_path
    
    logger.info("🦄 Starting Gemma 3n E4B OpenAI API Server")
    logger.info(f"🌐 Server URL: http://{args.host}:{args.port}")
    logger.info(f"📚 Documentation: http://{args.host}:{args.port}/docs")
    logger.info(f"🔧 Model: {api_server.model_id}")
    logger.info(f"💾 Model path: {args.model_path}")
    
    # Start server
    uvicorn.run(
        "gemma3n_e4b_openai_api_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
        reload=False,
        access_log=True
    )

if __name__ == "__main__":
    main()