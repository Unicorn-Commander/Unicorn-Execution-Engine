#!/usr/bin/env python3
"""
Magic Unicorn Streaming Inference Server
Real-time token generation with WebSocket streaming
"""

import asyncio
import websockets
import json
import time
import logging
import threading
import queue
from typing import Dict, List, Optional, AsyncGenerator, Any
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

# Add project path
import sys
sys.path.append('/home/ucadmin/Development/Unicorn-Execution-Engine')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StreamEventType(Enum):
    """Types of streaming events"""
    STREAM_START = "stream_start"
    TOKEN_GENERATED = "token_generated"
    STREAM_COMPLETE = "stream_complete"
    ERROR = "error"
    METRICS = "metrics"

@dataclass
class StreamEvent:
    """Streaming event data structure"""
    event_type: StreamEventType
    stream_id: str
    timestamp: float
    data: Any
    token_index: Optional[int] = None
    metrics: Optional[Dict[str, float]] = None

@dataclass
class StreamRequest:
    """Streaming inference request"""
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: List[str] = None
    stream_id: Optional[str] = None

class StreamingInferenceEngine:
    """
    🦄 Magic Unicorn Streaming Inference Engine
    
    Features:
    - Real-time token generation
    - WebSocket streaming
    - Concurrent request handling
    - Performance monitoring
    - Backpressure management
    """
    
    def __init__(self, model_path: str):
        """Initialize streaming inference engine"""
        
        self.model_path = model_path
        self.pipeline = None
        self.active_streams: Dict[str, asyncio.Queue] = {}
        self.stream_metrics: Dict[str, Dict[str, float]] = {}
        
        # Threading for pipeline execution
        self.inference_thread = None
        self.inference_queue = queue.Queue()
        self.is_running = False
        
        # Performance tracking
        self.total_tokens_generated = 0
        self.total_inference_time = 0.0
        self.server_start_time = time.time()
        
        logger.info("🦄 Magic Unicorn Streaming Engine initializing...")
    
    async def initialize(self):
        """Initialize the inference pipeline"""
        
        try:
            # Import and initialize pipeline
            from magic_unicorn_integrated_pipeline import MagicUnicornPipeline, MagicUnicornConfig, PipelineMode
            
            # Create a configuration for the streaming pipeline
            config = MagicUnicornConfig(
                model_path=self.model_path,
                mode=PipelineMode.PERFORMANCE, # Streaming typically prioritizes performance
                use_zero_copy=True,
                use_speculative_decoding=True,
                use_int4_quantization=True,
                use_flash_attention=True,
                use_streaming=False, # Avoid recursive streaming server initialization
                max_sequence_length=2048, # Use a reasonable default
                target_tps=10.0,
                max_memory_gb=8.0
            )

            self.pipeline = MagicUnicornPipeline(config)
            
            # Initialize the pipeline asynchronously
            await self.pipeline.initialize()
            
            # Start inference thread
            self.is_running = True
            self.inference_thread = threading.Thread(
                target=self._inference_worker, daemon=True
            )
            self.inference_thread.start()
            
            logger.info("✅ Streaming engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Streaming engine initialization failed: {e}")
            return False
    
    def _inference_worker(self):
        """Background worker for inference processing"""
        
        logger.info("🔄 Inference worker started")
        
        while self.is_running:
            try:
                # Get next request
                request_data = self.inference_queue.get(timeout=1.0)
                if request_data is None:
                    break
                
                stream_id, request, result_queue = request_data
                
                # Process streaming inference
                asyncio.run(self._process_streaming_request(stream_id, request, result_queue))
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Inference worker error: {e}")
    
    async def _process_streaming_request(self, stream_id: str, 
                                       request: StreamRequest, 
                                       result_queue: asyncio.Queue):
        """Process streaming inference request"""
        
        try:
            # Send stream start event
            start_event = StreamEvent(
                event_type=StreamEventType.STREAM_START,
                stream_id=stream_id,
                timestamp=time.time(),
                data={
                    'prompt': request.prompt,
                    'max_tokens': request.max_tokens,
                    'temperature': request.temperature
                }
            )
            await result_queue.put(start_event)
            
            # Initialize metrics
            stream_start_time = time.time()
            self.stream_metrics[stream_id] = {
                'start_time': stream_start_time,
                'tokens_generated': 0,
                'total_time': 0.0,
                'average_latency': 0.0,
                'tokens_per_second': 0.0
            }
            
            # Generate tokens one by one
            generated_text = ""
            
            for token_index in range(request.max_tokens):
                # Simulate token generation (replace with actual pipeline call)
                token_start_time = time.time()
                
                # TODO: Replace with actual pipeline.generate_token() call
                await asyncio.sleep(0.1)  # Simulate generation time
                new_token = f"token_{token_index} "
                
                token_end_time = time.time()
                token_latency = token_end_time - token_start_time
                
                generated_text += new_token
                
                # Update metrics
                self.stream_metrics[stream_id]['tokens_generated'] += 1
                self.stream_metrics[stream_id]['total_time'] = token_end_time - stream_start_time
                self.stream_metrics[stream_id]['average_latency'] = (
                    self.stream_metrics[stream_id]['total_time'] / 
                    self.stream_metrics[stream_id]['tokens_generated']
                )
                self.stream_metrics[stream_id]['tokens_per_second'] = (
                    self.stream_metrics[stream_id]['tokens_generated'] / 
                    self.stream_metrics[stream_id]['total_time']
                )
                
                # Send token event
                token_event = StreamEvent(
                    event_type=StreamEventType.TOKEN_GENERATED,
                    stream_id=stream_id,
                    timestamp=token_end_time,
                    data={
                        'token': new_token,
                        'generated_text': generated_text,
                        'token_latency': token_latency
                    },
                    token_index=token_index,
                    metrics=self.stream_metrics[stream_id].copy()
                )
                await result_queue.put(token_event)
                
                # Check for stop sequences
                if request.stop_sequences:
                    for stop_seq in request.stop_sequences:
                        if stop_seq in generated_text:
                            logger.info(f"🛑 Stop sequence detected: {stop_seq}")
                            break
                    else:
                        continue
                    break
            
            # Send completion event
            completion_event = StreamEvent(
                event_type=StreamEventType.STREAM_COMPLETE,
                stream_id=stream_id,
                timestamp=time.time(),
                data={
                    'generated_text': generated_text,
                    'total_tokens': self.stream_metrics[stream_id]['tokens_generated'],
                    'total_time': self.stream_metrics[stream_id]['total_time'],
                    'tokens_per_second': self.stream_metrics[stream_id]['tokens_per_second']
                },
                metrics=self.stream_metrics[stream_id].copy()
            )
            await result_queue.put(completion_event)
            
            # Update global metrics
            self.total_tokens_generated += self.stream_metrics[stream_id]['tokens_generated']
            self.total_inference_time += self.stream_metrics[stream_id]['total_time']
            
        except Exception as e:
            # Send error event
            error_event = StreamEvent(
                event_type=StreamEventType.ERROR,
                stream_id=stream_id,
                timestamp=time.time(),
                data={'error': str(e)}
            )
            await result_queue.put(error_event)
    
    async def start_stream(self, request: StreamRequest) -> str:
        """Start new streaming inference"""
        
        # Generate stream ID
        stream_id = request.stream_id or str(uuid.uuid4())
        
        # Create result queue
        result_queue = asyncio.Queue()
        self.active_streams[stream_id] = result_queue
        
        # Submit to inference worker
        self.inference_queue.put((stream_id, request, result_queue))
        
        logger.info(f"🚀 Started stream: {stream_id}")
        return stream_id
    
    async def get_stream_events(self, stream_id: str) -> AsyncGenerator[StreamEvent, None]:
        """Get streaming events for a stream"""
        
        if stream_id not in self.active_streams:
            raise ValueError(f"Stream not found: {stream_id}")
        
        result_queue = self.active_streams[stream_id]
        
        try:
            while True:
                # Wait for next event
                event = await result_queue.get()
                yield event
                
                # Clean up completed streams
                if event.event_type in [StreamEventType.STREAM_COMPLETE, StreamEventType.ERROR]:
                    del self.active_streams[stream_id]
                    if stream_id in self.stream_metrics:
                        del self.stream_metrics[stream_id]
                    break
                    
        except Exception as e:
            logger.error(f"❌ Stream error: {e}")
            # Clean up on error
            if stream_id in self.active_streams:
                del self.active_streams[stream_id]
            if stream_id in self.stream_metrics:
                del self.stream_metrics[stream_id]
    
    def get_server_metrics(self) -> Dict[str, Any]:
        """Get server-wide performance metrics"""
        
        uptime = time.time() - self.server_start_time
        overall_tps = self.total_tokens_generated / max(self.total_inference_time, 0.001)
        
        return {
            'uptime_seconds': uptime,
            'total_tokens_generated': self.total_tokens_generated,
            'total_inference_time': self.total_inference_time,
            'overall_tokens_per_second': overall_tps,
            'active_streams': len(self.active_streams),
            'average_stream_tps': overall_tps,
            'server_status': 'running' if self.is_running else 'stopped'
        }
    
    async def shutdown(self):
        """Shutdown the streaming engine"""
        
        logger.info("🔌 Shutting down streaming engine...")
        
        self.is_running = False
        
        # Stop inference worker
        if self.inference_thread:
            self.inference_queue.put(None)  # Signal shutdown
            self.inference_thread.join(timeout=5.0)
        
        # Close active streams
        for stream_id in list(self.active_streams.keys()):
            if stream_id in self.active_streams:
                del self.active_streams[stream_id]
        
        logger.info("✅ Streaming engine shutdown complete")

class MagicUnicornStreamingServer:
    """
    🦄 Magic Unicorn WebSocket Streaming Server
    """
    
    def __init__(self, host: str = "localhost", port: int = 8765, 
                 model_path: str = None):
        """Initialize streaming server"""
        
        self.host = host
        self.port = port
        self.model_path = model_path or "/home/ucadmin/Development/Unicorn-Execution-Engine/quantized_models/gemma-3-4b-it-quantized"
        
        self.engine = StreamingInferenceEngine(self.model_path)
        self.connected_clients: Dict[str, websockets.WebSocketServerProtocol] = {}
        
        logger.info(f"🦄 Magic Unicorn Streaming Server initializing on {host}:{port}")
    
    async def handle_websocket(self, websocket, path):
        """Handle WebSocket connection"""
        
        client_id = str(uuid.uuid4())
        self.connected_clients[client_id] = websocket
        
        logger.info(f"🔗 Client connected: {client_id}")
        
        try:
            await websocket.send(json.dumps({
                'type': 'connection_established',
                'client_id': client_id,
                'server_info': {
                    'name': 'Magic Unicorn Streaming Server',
                    'version': '1.0',
                    'capabilities': ['streaming_inference', 'real_time_metrics']
                }
            }))
            
            async for message in websocket:
                await self.handle_message(websocket, client_id, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"🔌 Client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"❌ WebSocket error: {e}")
        finally:
            if client_id in self.connected_clients:
                del self.connected_clients[client_id]
    
    async def handle_message(self, websocket, client_id: str, message: str):
        """Handle WebSocket message"""
        
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'start_stream':
                await self.handle_start_stream(websocket, client_id, data)
            elif message_type == 'get_metrics':
                await self.handle_get_metrics(websocket, client_id)
            else:
                await websocket.send(json.dumps({
                    'type': 'error',
                    'error': f'Unknown message type: {message_type}'
                }))
                
        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                'type': 'error',
                'error': 'Invalid JSON message'
            }))
        except Exception as e:
            await websocket.send(json.dumps({
                'type': 'error',
                'error': str(e)
            }))
    
    async def handle_start_stream(self, websocket, client_id: str, data: Dict):
        """Handle streaming inference request"""
        
        try:
            # Parse request
            request = StreamRequest(
                prompt=data['prompt'],
                max_tokens=data.get('max_tokens', 50),
                temperature=data.get('temperature', 0.7),
                top_p=data.get('top_p', 0.9),
                stop_sequences=data.get('stop_sequences'),
                stream_id=data.get('stream_id')
            )
            
            # Start stream
            stream_id = await self.engine.start_stream(request)
            
            # Stream events to client
            async for event in self.engine.get_stream_events(stream_id):
                event_dict = asdict(event)
                event_dict['event_type'] = event.event_type.value
                
                await websocket.send(json.dumps({
                    'type': 'stream_event',
                    'event': event_dict
                }))
                
        except Exception as e:
            await websocket.send(json.dumps({
                'type': 'error',
                'error': str(e)
            }))
    
    async def handle_get_metrics(self, websocket, client_id: str):
        """Handle metrics request"""
        
        try:
            metrics = self.engine.get_server_metrics()
            
            await websocket.send(json.dumps({
                'type': 'metrics',
                'data': metrics
            }))
            
        except Exception as e:
            await websocket.send(json.dumps({
                'type': 'error',
                'error': str(e)
            }))
    
    async def start_server(self):
        """Start the streaming server"""
        
        logger.info("🚀 Starting Magic Unicorn Streaming Server...")
        
        # Initialize inference engine
        if not await self.engine.initialize():
            raise Exception("Failed to initialize inference engine")
        
        # Start WebSocket server
        server = await websockets.serve(
            self.handle_websocket, 
            self.host, 
            self.port,
            ping_interval=20,
            ping_timeout=10
        )
        
        logger.info(f"🦄 Magic Unicorn Streaming Server running on ws://{self.host}:{self.port}")
        logger.info("🎯 Ready for real-time streaming inference!")
        
        return server
    
    async def shutdown(self):
        """Shutdown the server"""
        
        logger.info("🔌 Shutting down Magic Unicorn Streaming Server...")
        
        # Shutdown inference engine
        await self.engine.shutdown()
        
        # Close client connections
        for client_id, websocket in self.connected_clients.items():
            try:
                await websocket.close()
            except:
                pass
        
        logger.info("✅ Server shutdown complete")

async def main():
    """Main entry point"""
    
    logger.info("🦄✨ MAGIC UNICORN STREAMING SERVER ✨🦄")
    logger.info("=" * 70)
    
    # Create server
    server_instance = MagicUnicornStreamingServer(
        host="localhost",
        port=8765
    )
    
    try:
        # Start server
        server = await server_instance.start_server()
        
        # Run indefinitely
        await server.wait_closed()
        
    except KeyboardInterrupt:
        logger.info("🛑 Received shutdown signal")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
    finally:
        await server_instance.shutdown()

if __name__ == "__main__":
    asyncio.run(main())