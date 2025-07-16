#!/usr/bin/env python3
"""
Test client for the persistent model server
"""

import requests
import json
import time
import sys

def test_inference(prompt, max_tokens=50):
    """Test inference via the model server"""
    url = "http://localhost:8011/generate"
    
    print(f"🦄 Testing inference...")
    print(f"📝 Prompt: '{prompt}'")
    print(f"🎯 Max tokens: {max_tokens}")
    print()
    
    # Make request
    try:
        response = requests.post(url, json={
            'prompt': prompt,
            'max_tokens': max_tokens
        }, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"📊 Generation time: {result['generation_time']:.2f}s")
            print(f"⚡ Tokens per second: {result['tokens_per_second']:.1f}")
            print(f"🎯 Completion: '{result['completion']}'")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.json())
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Is it running?")
        print("   Start with: python3 persistent_model_server.py")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    # Test prompts
    prompts = [
        "Magic Unicorn Unconventional Technology & Stuff is a groundbreaking Applied AI company that",
        "The future of artificial intelligence is",
        "In the world of technology,",
    ]
    
    # Check server health first
    try:
        health = requests.get("http://localhost:8011/health")
        if health.status_code == 200:
            print("✅ Server is healthy")
            print()
        else:
            print("⚠️ Server may not be ready")
    except:
        print("❌ Server not running!")
        print("Start with: python3 persistent_model_server.py")
        return
    
    # Test each prompt
    for prompt in prompts:
        test_inference(prompt, max_tokens=20)
        print("\n" + "="*60 + "\n")
        time.sleep(1)  # Brief pause between requests

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Custom prompt from command line
        prompt = " ".join(sys.argv[1:])
        test_inference(prompt)
    else:
        main()