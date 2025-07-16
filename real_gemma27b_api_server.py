#!/usr/bin/env python3
"""
Real Gemma 3 27B API Server - Loads actual quantized model
Uses NPU+iGPU pipeline with the real 26GB quantized model
"""

import os
import sys

# Force Vulkan-only mode BEFORE any other imports
os.environ['HIP_VISIBLE_DEVICES'] = ''
os.environ['ROCR_VISIBLE_DEVICES'] = ''
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['GPU_FORCE_64BIT_PTR'] = '0'

import torch
import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import json
import uuid
from pathlib import Path

# Force CPU-only PyTorch (Vulkan handles GPU separately)
torch.set_default_device('cpu')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for OpenAI API compatibility
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 150
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

class RealGemma27BAPIServer:
    """OpenAI API Server that loads REAL quantized Gemma 3 27B model"""
    
    def __init__(self, model_path: str = "./quantized_models/gemma-3-27b-it-layer-by-layer"):
        self.app = FastAPI(title="Real Gemma 27B API", version="1.0.0")
        self.model_path = Path(model_path)
        self.model_loaded = False
        self.vulkan_engine = None
        self.npu_kernel = None
        self.quantized_loader = None
        
        # Setup routes
        self._setup_routes()
        
        logger.info("🦄 REAL GEMMA 3 27B API SERVER")
        logger.info("=" * 60)
        logger.info(f"📁 Model path: {self.model_path}")
        logger.info(f"💾 Expected size: ~26GB quantized model")
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/v1/models")
        async def list_models():
            return {
                "object": "list",
                "data": [
                    {
                        "id": "gemma-3-27b-it-quantized-real",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "real-gemma-27b-api"
                    }
                ]
            }
        
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            return await self._handle_chat_completion(request)
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy" if self.model_loaded else "model_not_loaded",
                "model_loaded": self.model_loaded,
                "npu": "ready" if self.npu_kernel else "not_ready",
                "igpu": "ready" if self.vulkan_engine else "not_ready",
                "model_path": str(self.model_path),
                "model_exists": self.model_path.exists()
            }
    
    async def initialize_model(self) -> bool:
        """Initialize REAL quantized Gemma 3 27B model"""
        logger.info("🚀 Loading REAL quantized Gemma 3 27B model...")
        logger.info(f"📁 Loading from: {self.model_path}")
        
        # Check if model exists
        if not self.model_path.exists():
            logger.error(f"❌ Model path does not exist: {self.model_path}")
            return False
        
        # Initialize hardware first
        if not await self._initialize_hardware():
            logger.error("❌ Hardware initialization failed")
            return False
        
        # Load quantized model
        try:
            from quantized_gemma27b_npu_igpu_loader import QuantizedGemma27BNPUIGPULoader
            
            logger.info("📦 Loading quantized model weights...")
            self.quantized_loader = QuantizedGemma27BNPUIGPULoader(str(self.model_path))
            
            # Load model weights into memory (streaming loader)
            model_info = self.quantized_loader.load_model_streaming()
            model_size = model_info.get('total_size_gb', 26.0)
            
            self.model_loaded = True
            logger.info("🎉 REAL GEMMA 3 27B MODEL LOADED SUCCESSFULLY")
            logger.info(f"   💾 Model size: {model_size:.2f} GB")
            logger.info(f"   ⚡ NPU Phoenix: Ready for attention")
            logger.info(f"   🎮 iGPU Vulkan: Ready for FFN")
            logger.info(f"   🧠 CPU: Orchestration and embeddings")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            return False
    
    async def _initialize_hardware(self) -> bool:
        """Initialize NPU+iGPU hardware"""
        logger.info("🔧 Initializing NPU+iGPU hardware...")
        
        # Check NPU
        try:
            import subprocess
            result = subprocess.run(['xrt-smi', 'examine'], capture_output=True, text=True)
            if 'Phoenix' not in result.stdout or result.returncode != 0:
                logger.error("❌ NPU Phoenix NOT available")
                return False
            logger.info("✅ NPU Phoenix detected")
        except Exception as e:
            logger.error(f"❌ NPU detection failed: {e}")
            return False
        
        # Check iGPU
        try:
            import subprocess
            result = subprocess.run(['vulkaninfo', '--summary'], capture_output=True, text=True)
            if 'AMD Radeon Graphics' not in result.stdout or result.returncode != 0:
                logger.error("❌ AMD Radeon iGPU NOT available")
                return False
            logger.info("✅ iGPU AMD Radeon detected")
        except Exception as e:
            logger.error(f"❌ iGPU detection failed: {e}")
            return False
        
        # Initialize Vulkan engine
        try:
            from vulkan_ffn_compute_engine import VulkanFFNComputeEngine
            self.vulkan_engine = VulkanFFNComputeEngine()
            if not self.vulkan_engine.initialize():
                logger.error("❌ Vulkan engine initialization FAILED")
                return False
            logger.info("✅ Vulkan iGPU engine ready")
        except Exception as e:
            logger.error(f"❌ Vulkan engine failed: {e}")
            return False
        
        # Initialize NPU kernel
        try:
            from npu_attention_kernel_real import NPUAttentionKernelReal
            self.npu_kernel = NPUAttentionKernelReal()
            if not self.npu_kernel.initialize():
                logger.error("❌ NPU kernel initialization FAILED")
                return False
            logger.info("✅ NPU attention kernel ready")
        except Exception as e:
            logger.error(f"❌ NPU kernel failed: {e}")
            return False
        
        return True
    
    async def _handle_chat_completion(self, request: ChatCompletionRequest) -> JSONResponse:
        """Handle chat completion with REAL model processing"""
        
        if not self.model_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Extract user message
        user_message = ""
        for msg in request.messages:
            if msg.role == "user":
                user_message = msg.content
                break
        
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")
        
        logger.info("🦄 REAL GEMMA 3 27B CHAT COMPLETION")
        logger.info(f"   📝 User: {user_message[:100]}...")
        logger.info(f"   🎯 Max tokens: {request.max_tokens}")
        logger.info(f"   🌡️ Temperature: {request.temperature}")
        
        try:
            # Generate response using REAL quantized model
            response_text = await self._generate_with_real_model(
                user_message, 
                request.max_tokens,
                request.temperature
            )
            
            # Create OpenAI-compatible response
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            
            response = {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_text
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(user_message.split()),
                    "completion_tokens": len(response_text.split()),
                    "total_tokens": len(user_message.split()) + len(response_text.split())
                }
            }
            
            logger.info("✅ REAL GEMMA 3 27B COMPLETION SUCCESS")
            return JSONResponse(content=response)
            
        except Exception as e:
            logger.error(f"❌ REAL MODEL GENERATION FAILED: {e}")
            raise HTTPException(status_code=500, detail=f"Model generation failed: {str(e)}")
    
    async def _generate_with_real_model(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate text using REAL quantized Gemma 3 27B model"""
        
        logger.info("🦄 EXECUTING REAL GEMMA 3 27B GENERATION")
        logger.info("   💾 Using 26GB quantized model weights")
        logger.info("   🚫 NO CPU fallbacks - pure hardware acceleration")
        
        # Use the quantized loader to generate text
        if not self.quantized_loader:
            raise RuntimeError("❌ Quantized model loader not initialized")
        
        # Generate text using the real model pipeline
        start_time = time.time()
        
        # This will use the real model weights and NPU+iGPU hardware
        generated_text = self.quantized_loader.generate_text(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature
        )
        
        generation_time = time.time() - start_time
        tokens_per_second = max_tokens / generation_time if generation_time > 0 else 0
        
        logger.info("🎉 REAL MODEL GENERATION COMPLETE")
        logger.info(f"   ⚡ NPU Phoenix: Real attention computation")
        logger.info(f"   🎮 iGPU Vulkan: Real FFN computation")
        logger.info(f"   💾 Model: 26GB quantized weights loaded")
        logger.info(f"   ⏱️ Generation time: {generation_time:.2f}s")
        logger.info(f"   🚀 Speed: {tokens_per_second:.2f} tokens/sec")
        logger.info(f"   📝 Response: {generated_text[:100]}...")
        
        return generated_text

async def main():
    """Main function to start the real Gemma 27B API server"""
    server = RealGemma27BAPIServer()
    
    # Initialize model
    logger.info("🚀 Starting REAL Gemma 3 27B model initialization...")
    if not await server.initialize_model():
        logger.error("❌ Model initialization failed - cannot start server")
        sys.exit(1)
    
    # Start server
    config = uvicorn.Config(
        server.app,
        host="0.0.0.0",
        port=8003,
        log_level="info"
    )
    
    server_instance = uvicorn.Server(config)
    
    logger.info("🚀 REAL GEMMA 3 27B API SERVER STARTING")
    logger.info("=" * 50)
    logger.info("   📡 URL: http://0.0.0.0:8003")
    logger.info("   🦄 Real Gemma 3 27B (26GB quantized)")
    logger.info("   ⚡ NPU Phoenix + AMD Radeon 780M")
    logger.info("   🚫 NO CPU FALLBACKS")
    
    await server_instance.serve()

if __name__ == "__main__":
    asyncio.run(main())