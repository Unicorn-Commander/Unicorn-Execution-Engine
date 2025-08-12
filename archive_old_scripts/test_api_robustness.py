#!/usr/bin/env python3
"""
Test script to validate API robustness and error handling
"""

import requests
import json
import time
import sys

def test_api_endpoint(url, description):
    """Test an API endpoint with error handling"""
    print(f"Testing: {description}")
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            print(f"  ✅ {description} - OK")
            return True
        else:
            print(f"  ❌ {description} - Status: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ {description} - Error: {e}")
        return False

def test_chat_completion(base_url):
    """Test chat completion endpoint"""
    print("Testing: Chat completion")
    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "gemma-3n-e4b-it",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                print(f"  ✅ Chat completion - OK")
                return True
            else:
                print(f"  ❌ Chat completion - Invalid response format")
                return False
        else:
            print(f"  ❌ Chat completion - Status: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print(f"  ❌ Chat completion - Error: {e}")
        return False

def test_streaming(base_url):
    """Test streaming endpoint"""
    print("Testing: Streaming completion")
    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "gemma-3n-e4b-it",
                "messages": [{"role": "user", "content": "Count to 3"}],
                "max_tokens": 10,
                "stream": True
            },
            timeout=30,
            stream=True
        )
        
        if response.status_code == 200:
            chunks = 0
            for line in response.iter_lines():
                if line:
                    chunks += 1
                    if chunks >= 3:  # Test first few chunks
                        break
            
            if chunks > 0:
                print(f"  ✅ Streaming - OK ({chunks} chunks)")
                return True
            else:
                print(f"  ❌ Streaming - No chunks received")
                return False
        else:
            print(f"  ❌ Streaming - Status: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Streaming - Error: {e}")
        return False

def main():
    """Main test function"""
    base_url = "http://localhost:8000"
    
    print("🧪 API Robustness Test")
    print("=" * 40)
    
    # Test basic endpoints
    tests = [
        (f"{base_url}/health", "Health check"),
        (f"{base_url}/v1/models", "Models endpoint"),
        (f"{base_url}/v1/metrics", "Metrics endpoint"),
    ]
    
    passed = 0
    total = len(tests) + 2  # +2 for chat completion and streaming
    
    for url, description in tests:
        if test_api_endpoint(url, description):
            passed += 1
    
    # Test chat completion
    if test_chat_completion(base_url):
        passed += 1
    
    # Test streaming
    if test_streaming(base_url):
        passed += 1
    
    print("=" * 40)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())