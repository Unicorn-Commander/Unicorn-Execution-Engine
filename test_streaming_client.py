#!/usr/bin/env python3
"""
Test client for Magic Unicorn Streaming Server
"""

import asyncio
import websockets
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_streaming_client():
    """Test the streaming inference server"""
    
    logger.info("🧪 Testing Magic Unicorn Streaming Client...")
    
    uri = "ws://localhost:8765"
    
    try:
        async with websockets.connect(uri) as websocket:
            
            # Wait for connection established
            response = await websocket.recv()
            connection_data = json.loads(response)
            logger.info(f"✅ Connected: {connection_data}")
            
            # Send streaming request
            request = {
                'type': 'start_stream',
                'prompt': 'What is the capital of France?',
                'max_tokens': 20,
                'temperature': 0.7,
                'stream_id': 'test_stream_001'
            }
            
            logger.info(f"📤 Sending request: {request['prompt']}")
            await websocket.send(json.dumps(request))
            
            # Receive streaming events
            logger.info("📥 Receiving streaming events:")
            
            while True:
                try:
                    response = await websocket.recv()
                    data = json.loads(response)
                    
                    if data['type'] == 'stream_event':
                        event = data['event']
                        event_type = event['event_type']
                        
                        if event_type == 'stream_start':
                            logger.info("🚀 Stream started")
                            
                        elif event_type == 'token_generated':
                            token = event['data']['token']
                            latency = event['data']['token_latency']
                            metrics = event.get('metrics', {})
                            tps = metrics.get('tokens_per_second', 0)
                            
                            logger.info(f"🦄 Token: '{token.strip()}' (latency: {latency*1000:.1f}ms, TPS: {tps:.1f})")
                            
                        elif event_type == 'stream_complete':
                            final_text = event['data']['generated_text']
                            total_tokens = event['data']['total_tokens']
                            total_time = event['data']['total_time']
                            final_tps = event['data']['tokens_per_second']
                            
                            logger.info("🎉 Stream completed!")
                            logger.info(f"📝 Generated text: {final_text}")
                            logger.info(f"📊 Final metrics: {total_tokens} tokens in {total_time:.2f}s ({final_tps:.1f} TPS)")
                            break
                            
                        elif event_type == 'error':
                            error = event['data']['error']
                            logger.error(f"❌ Stream error: {error}")
                            break
                    
                    elif data['type'] == 'error':
                        logger.error(f"❌ Server error: {data['error']}")
                        break
                        
                except websockets.exceptions.ConnectionClosed:
                    logger.info("🔌 Connection closed")
                    break
            
            # Test metrics request
            logger.info("📊 Requesting server metrics...")
            await websocket.send(json.dumps({'type': 'get_metrics'}))
            
            response = await websocket.recv()
            metrics_data = json.loads(response)
            
            if metrics_data['type'] == 'metrics':
                metrics = metrics_data['data']
                logger.info("📈 Server metrics:")
                for key, value in metrics.items():
                    logger.info(f"   {key}: {value}")
            
    except ConnectionRefusedError:
        logger.error("❌ Cannot connect to server. Is the streaming server running?")
        logger.info("💡 Start the server with: python3 streaming_inference_server.py")
    except Exception as e:
        logger.error(f"❌ Client error: {e}")

async def main():
    """Main entry point"""
    
    logger.info("🦄 Magic Unicorn Streaming Test Client")
    logger.info("=" * 50)
    
    await test_streaming_client()

if __name__ == "__main__":
    asyncio.run(main())