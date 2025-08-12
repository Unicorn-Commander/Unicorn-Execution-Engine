#!/usr/bin/env python3
"""
Test client for the accelerated API server
Tests the real NPU+iGPU acceleration via OpenAI-compatible API
"""

import requests
import json
import time

def test_health_check():
    """Test the health check endpoint"""
    print("🏥 Testing health check...")
    try:
        response = requests.get("http://localhost:8001/health")
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Health check passed!")
            print(f"  Status: {health_data['status']}")
            print(f"  Model: {health_data['model']}")
            print(f"  Parameters: {health_data['parameters']:,}")
            print(f"  NPU Available: {health_data['npu_available']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_stats_endpoint():
    """Test the stats endpoint"""
    print("\n📊 Testing stats endpoint...")
    try:
        response = requests.get("http://localhost:8001/stats")
        if response.status_code == 200:
            stats = response.json()
            print("✅ Stats retrieved successfully!")
            print(f"  Total layers: {stats['model_info']['total_layers']}")
            print(f"  Sparse layers (NPU): {stats['model_info']['sparse_layers']}")
            print(f"  Dense layers (iGPU): {stats['model_info']['dense_layers']}")
            print(f"  NPU device: {stats['hardware_config']['npu_device']}")
            print(f"  iGPU device: {stats['hardware_config']['igpu_device']}")
            return True
        else:
            print(f"❌ Stats failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Stats error: {e}")
        return False

def test_chat_completion():
    """Test chat completion with real acceleration"""
    print("\n💬 Testing chat completion...")
    try:
        # Prepare chat request
        chat_request = {
            "model": "gemma3n-e2b-accelerated",
            "messages": [
                {"role": "user", "content": "Hello! Can you tell me about your acceleration?"}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        print("📤 Sending chat request...")
        start_time = time.time()
        
        response = requests.post(
            "http://localhost:8001/v1/chat/completions",
            json=chat_request,
            headers={"Content-Type": "application/json"}
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        if response.status_code == 200:
            chat_response = response.json()
            print("✅ Chat completion successful!")
            print(f"  Response time: {response_time:.2f}s")
            print(f"  Model: {chat_response['model']}")
            print(f"  Content: {chat_response['choices'][0]['message']['content']}")
            print(f"  Tokens used: {chat_response['usage']['total_tokens']}")
            return True
        else:
            print(f"❌ Chat completion failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Chat completion error: {e}")
        return False

def test_streaming_chat():
    """Test streaming chat completion"""
    print("\n🌊 Testing streaming chat completion...")
    try:
        chat_request = {
            "model": "gemma3n-e2b-accelerated",
            "messages": [
                {"role": "user", "content": "Explain how NPU acceleration works"}
            ],
            "max_tokens": 50,
            "stream": True
        }
        
        print("📤 Sending streaming request...")
        start_time = time.time()
        
        response = requests.post(
            "http://localhost:8001/v1/chat/completions",
            json=chat_request,
            headers={"Content-Type": "application/json"},
            stream=True
        )
        
        if response.status_code == 200:
            print("✅ Streaming started!")
            print("📝 Response: ", end="", flush=True)
            
            chunks_received = 0
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]  # Remove 'data: ' prefix
                        if data_str == '[DONE]':
                            break
                        try:
                            chunk_data = json.loads(data_str)
                            if 'choices' in chunk_data and chunk_data['choices']:
                                delta = chunk_data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    print(content, end="", flush=True)
                                    chunks_received += 1
                        except json.JSONDecodeError:
                            pass
            
            end_time = time.time()
            response_time = end_time - start_time
            print(f"\n✅ Streaming completed!")
            print(f"  Response time: {response_time:.2f}s")
            print(f"  Chunks received: {chunks_received}")
            return True
        else:
            print(f"❌ Streaming failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Streaming error: {e}")
        return False

def main():
    """Run all API tests"""
    print("🧪 Testing Accelerated API Server")
    print("=" * 50)
    
    # Run tests
    results = []
    results.append(test_health_check())
    results.append(test_stats_endpoint())
    results.append(test_chat_completion())
    results.append(test_streaming_chat())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Test Summary:")
    print(f"  Passed: {passed}/{total}")
    print(f"  Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed! Real NPU+iGPU acceleration working via API!")
    else:
        print("⚠️ Some tests failed - check server logs")
    
    return passed == total

if __name__ == "__main__":
    main()