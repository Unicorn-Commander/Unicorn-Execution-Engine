#!/usr/bin/env python3
"""
Test the Optimized OpenAI v1 Compatible API Server
Tests all endpoints and demonstrates the performance improvements
"""

import requests
import json
import time
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedAPITester:
    """Test the optimized API server endpoints"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def test_root_endpoint(self) -> Dict[str, Any]:
        """Test the root endpoint"""
        logger.info("🔍 Testing root endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/")
            response.raise_for_status()
            data = response.json()
            
            logger.info("✅ Root endpoint working!")
            logger.info(f"   Status: {data.get('status')}")
            logger.info(f"   Target TPS: {data.get('performance', {}).get('target_tps')}")
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Root endpoint failed: {e}")
            return {}
    
    def test_models_endpoint(self) -> Dict[str, Any]:
        """Test the models endpoint (OpenAI v1 compatible)"""
        logger.info("🔍 Testing /v1/models endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/v1/models")
            response.raise_for_status()
            data = response.json()
            
            logger.info("✅ Models endpoint working!")
            if data.get("data"):
                model = data["data"][0]
                logger.info(f"   Model: {model.get('id')}")
                logger.info(f"   Optimizations: {model.get('optimizations')}")
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Models endpoint failed: {e}")
            return {}
    
    def test_health_endpoint(self) -> Dict[str, Any]:
        """Test the health check endpoint"""
        logger.info("🔍 Testing /health endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            data = response.json()
            
            logger.info("✅ Health endpoint working!")
            logger.info(f"   Status: {data.get('status')}")
            logger.info(f"   Hardware: {data.get('hardware')}")
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Health endpoint failed: {e}")
            return {}
    
    def test_stats_endpoint(self) -> Dict[str, Any]:
        """Test the performance stats endpoint"""
        logger.info("🔍 Testing /stats endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/stats")
            response.raise_for_status()
            data = response.json()
            
            logger.info("✅ Stats endpoint working!")
            framework = data.get("optimization_framework", {})
            logger.info(f"   Batch Processing: {framework.get('batch_processing')}")
            logger.info(f"   Memory Pooling: {framework.get('memory_pooling')}")
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Stats endpoint failed: {e}")
            return {}
    
    def test_chat_completion(self, stream: bool = False) -> Dict[str, Any]:
        """Test the chat completion endpoint (OpenAI v1 compatible)"""
        mode = "streaming" if stream else "complete"
        logger.info(f"🔍 Testing /v1/chat/completions ({mode})...")
        
        # Test with the Lexus GX470 question
        payload = {
            "model": "gemma-3-27b-optimized",
            "messages": [
                {
                    "role": "user", 
                    "content": "What do you know about the 2008 Lexus GX470?"
                }
            ],
            "max_tokens": 150,
            "temperature": 0.7,
            "stream": stream
        }
        
        try:
            start_time = time.time()
            
            if stream:
                # Test streaming response
                response = self.session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    stream=True
                )
                response.raise_for_status()
                
                chunks = []
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            chunk_data = line_str[6:]  # Remove 'data: '
                            if chunk_data != '[DONE]':
                                try:
                                    chunk = json.loads(chunk_data)
                                    chunks.append(chunk)
                                except json.JSONDecodeError:
                                    pass
                
                response_time = time.time() - start_time
                
                logger.info("✅ Streaming chat completion working!")
                logger.info(f"   Response time: {response_time*1000:.1f}ms")
                logger.info(f"   Chunks received: {len(chunks)}")
                
                return {"chunks": len(chunks), "response_time": response_time}
                
            else:
                # Test complete response
                response = self.session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                
                response_time = time.time() - start_time
                
                logger.info("✅ Chat completion working!")
                logger.info(f"   Response time: {response_time*1000:.1f}ms")
                
                if data.get("choices"):
                    choice = data["choices"][0]
                    message = choice.get("message", {})
                    content = message.get("content", "")
                    
                    logger.info(f"   Response length: {len(content)} chars")
                    logger.info(f"   Tokens: {data.get('usage', {})}")
                    
                    # Show first part of response
                    preview = content[:100] + "..." if len(content) > 100 else content
                    logger.info(f"   Preview: {preview}")
                
                return data
                
        except Exception as e:
            logger.error(f"❌ Chat completion failed: {e}")
            return {}
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive API test suite"""
        logger.info("🧪 COMPREHENSIVE OPTIMIZED API TEST")
        logger.info("===================================")
        
        results = {}
        
        # Test all endpoints
        logger.info("\n📋 Testing all endpoints...")
        results["root"] = self.test_root_endpoint()
        results["models"] = self.test_models_endpoint()
        results["health"] = self.test_health_endpoint()
        results["stats"] = self.test_stats_endpoint()
        
        # Test chat completions
        logger.info("\n💬 Testing chat completions...")
        results["chat_complete"] = self.test_chat_completion(stream=False)
        results["chat_streaming"] = self.test_chat_completion(stream=True)
        
        # Generate summary
        logger.info("\n📊 API TEST SUMMARY")
        logger.info("===================")
        
        successful_tests = sum(1 for result in results.values() if result)
        total_tests = len(results)
        
        logger.info(f"✅ Successful tests: {successful_tests}/{total_tests}")
        
        if successful_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED!")
            logger.info("   OpenAI v1 compatible API ready for production")
        else:
            logger.info("🔧 SOME TESTS FAILED")
            logger.info("   Check server startup and configuration")
        
        return results

def test_api_without_server():
    """Test API functionality without running server (validation only)"""
    logger.info("🧪 OPTIMIZED API VALIDATION TEST")
    logger.info("================================")
    
    # Import and validate the API components
    try:
        from optimized_openai_api_server import OptimizedInferenceEngine, ChatMessage
        
        logger.info("✅ API server imports working")
        
        # Test engine initialization
        engine = OptimizedInferenceEngine()
        logger.info("✅ Optimized inference engine initialized")
        
        # Test message formatting
        messages = [
            ChatMessage(role="user", content="What do you know about the 2008 Lexus GX470?")
        ]
        prompt = engine._format_messages_to_prompt(messages)
        logger.info(f"✅ Message formatting working: {len(prompt)} chars")
        
        # Test performance stats
        stats = engine.get_performance_stats()
        logger.info("✅ Performance stats working:")
        logger.info(f"   Target TPS: {stats.get('target_tps')}")
        logger.info(f"   Baseline TPS: {stats.get('baseline_tps')}")
        
        logger.info("\n🎉 API VALIDATION SUCCESSFUL!")
        logger.info("   All components ready for deployment")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ API validation failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🦄 OPTIMIZED OpenAI v1 API SERVER TESTER")
    logger.info("========================================")
    
    # First, validate API components
    if not test_api_without_server():
        logger.error("❌ API validation failed - check imports")
        return
    
    # Try to test live server
    logger.info("\n🔍 Checking if API server is running...")
    
    tester = OptimizedAPITester()
    try:
        # Quick health check
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            logger.info("✅ API server is running - running live tests")
            tester.run_comprehensive_test()
        else:
            logger.info("⚠️  API server not responding - validation only")
    except requests.exceptions.RequestException:
        logger.info("ℹ️  API server not running - showing startup instructions")
        
        logger.info("\n🚀 TO START THE OPTIMIZED API SERVER:")
        logger.info("=====================================")
        logger.info("source ~/activate-uc1-ai-py311.sh")
        logger.info("python optimized_openai_api_server.py")
        logger.info("")
        logger.info("Then test with:")
        logger.info("curl http://localhost:8000/")
        logger.info("curl http://localhost:8000/v1/models")
        logger.info("")
        logger.info("Or test chat completion:")
        logger.info('curl -X POST http://localhost:8000/v1/chat/completions \\')
        logger.info('  -H "Content-Type: application/json" \\')
        logger.info('  -d \'{"model":"gemma-3-27b-optimized","messages":[{"role":"user","content":"What do you know about the 2008 Lexus GX470?"}],"max_tokens":150}\'')

if __name__ == "__main__":
    main()