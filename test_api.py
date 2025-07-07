#!/usr/bin/env python3
"""
Test script for Gemma 3n E2B OpenAI-compatible API
"""

import requests
import json
import time

API_BASE = "http://localhost:8000"

def test_api_endpoints():
    """Test all API endpoints"""
    print("🧪 Testing Gemma 3n E2B API Server")
    print("=" * 50)
    
    # Test root endpoint
    print("1. Testing root endpoint...")
    try:
        response = requests.get(f"{API_BASE}/")
        print(f"✅ Root: {response.status_code}")
        print(f"   Message: {response.json().get('message', 'N/A')}")
    except Exception as e:
        print(f"❌ Root failed: {e}")
    
    # Test health check
    print("\n2. Testing health check...")
    try:
        response = requests.get(f"{API_BASE}/health")
        health = response.json()
        print(f"✅ Health: {response.status_code}")
        print(f"   Status: {health.get('status', 'N/A')}")
        print(f"   Model loaded: {health.get('model_loaded', False)}")
        print(f"   Requests served: {health.get('performance_stats', {}).get('requests_served', 0)}")
    except Exception as e:
        print(f"❌ Health failed: {e}")
    
    # Test models list
    print("\n3. Testing models list...")
    try:
        response = requests.get(f"{API_BASE}/v1/models")
        models = response.json()
        print(f"✅ Models: {response.status_code}")
        print(f"   Available models: {len(models.get('data', []))}")
        for model in models.get('data', []):
            print(f"   - {model.get('id', 'Unknown')}")
    except Exception as e:
        print(f"❌ Models failed: {e}")
    
    # Test chat completion
    print("\n4. Testing chat completion...")
    try:
        chat_request = {
            "model": "qwen2.5-7b",
            "messages": [
                {"role": "user", "content": "Hello! Tell me about AI on edge devices."}
            ],
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        start_time = time.time()
        response = requests.post(f"{API_BASE}/v1/chat/completions", json=chat_request)
        end_time = time.time()
        
        completion = response.json()
        print(f"✅ Chat completion: {response.status_code}")
        print(f"   Response time: {(end_time - start_time)*1000:.1f}ms")
        
        if 'choices' in completion and completion['choices']:
            content = completion['choices'][0]['message']['content']
            print(f"   Generated: {content[:100]}{'...' if len(content) > 100 else ''}")
        
        if 'performance' in completion:
            perf = completion['performance']
            print(f"   TPS: {perf.get('tps', 'N/A'):.1f}")
            print(f"   TTFT: {perf.get('ttft_ms', 'N/A'):.1f}ms")
            print(f"   Memory: {perf.get('memory_mb', 'N/A'):.1f}MB")
            
    except Exception as e:
        print(f"❌ Chat completion failed: {e}")
    
    # Test completion
    print("\n5. Testing text completion...")
    try:
        completion_request = {
            "model": "qwen2.5-7b",
            "prompt": "The future of AI will be",
            "max_tokens": 30,
            "temperature": 0.8
        }
        
        start_time = time.time()
        response = requests.post(f"{API_BASE}/v1/completions", json=completion_request)
        end_time = time.time()
        
        completion = response.json()
        print(f"✅ Text completion: {response.status_code}")
        print(f"   Response time: {(end_time - start_time)*1000:.1f}ms")
        
        if 'choices' in completion and completion['choices']:
            text = completion['choices'][0]['text']
            print(f"   Generated: {text}")
        
        if 'performance' in completion:
            perf = completion['performance']
            print(f"   TPS: {perf.get('tps', 'N/A'):.1f}")
            print(f"   TTFT: {perf.get('ttft_ms', 'N/A'):.1f}ms")
            
    except Exception as e:
        print(f"❌ Text completion failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 API testing completed!")

def test_streaming():
    """Test streaming responses"""
    print("\n🌊 Testing Streaming Response...")
    
    try:
        chat_request = {
            "model": "qwen2.5-7b",
            "messages": [
                {"role": "user", "content": "Explain hybrid NPU+iGPU execution"}
            ],
            "max_tokens": 40,
            "stream": True
        }
        
        response = requests.post(f"{API_BASE}/v1/chat/completions", json=chat_request, stream=True)
        
        print("✅ Streaming response:")
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: ') and not line.endswith('[DONE]'):
                    try:
                        data = json.loads(line[6:])  # Remove 'data: ' prefix
                        if 'choices' in data and data['choices']:
                            content = data['choices'][0].get('delta', {}).get('content', '')
                            if content:
                                print(content, end='', flush=True)
                    except json.JSONDecodeError:
                        pass
        print("\n")
        
    except Exception as e:
        print(f"❌ Streaming failed: {e}")

if __name__ == "__main__":
    print("Waiting for server to start...")
    time.sleep(30)
    
    test_api_endpoints()
    test_streaming()
    
    print("\n💡 Usage Examples:")
    print("   curl -X POST http://localhost:8000/v1/chat/completions \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{\"model\":\"gemma-3n-e2b\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}]}'")
    print()
    print("   # Or use any OpenAI-compatible client/GUI with:")
    print("   # Base URL: http://localhost:8000")
    print("   # Model: gemma-3n-e2b")
    print("   # API Key: (not required)")