#!/usr/bin/env python3
"""
Test client for persistent model server
"""

import requests
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERVER_URL = "http://localhost:8011"

def wait_for_server(timeout=30):
    """Wait for server to be ready"""
    logger.info("⏳ Waiting for server to be ready...")
    start = time.time()
    
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{SERVER_URL}/health", timeout=1)
            if r.status_code == 200:
                data = r.json()
                if data.get('model_loaded'):
                    logger.info("✅ Server is ready!")
                    return True
        except:
            pass
        time.sleep(1)
    
    logger.error("❌ Server timeout")
    return False

def test_magic_unicorn():
    """Test Magic Unicorn prompt"""
    prompt = "Magic Unicorn Unconventional Technology & Stuff is a groundbreaking Applied AI company that"
    
    logger.info("🦄 Testing Magic Unicorn prompt...")
    logger.info(f"📝 Prompt: '{prompt}'")
    
    try:
        r = requests.post(
            f"{SERVER_URL}/generate",
            json={
                "prompt": prompt,
                "max_tokens": 50,
                "temperature": 0.7
            },
            timeout=60
        )
        
        if r.status_code == 200:
            data = r.json()
            logger.info(f"\n✅ Response received!")
            logger.info(f"   Completion: {data['completion']}")
            logger.info(f"   Tokens: {data['tokens_generated']}")
            logger.info(f"   Time: {data['generation_time']:.1f}s")
            logger.info(f"   TPS: {data['tokens_per_second']:.1f}")
            
            return data['tokens_per_second']
        else:
            logger.error(f"❌ Error: {r.status_code}")
            logger.error(r.text)
            return 0
            
    except Exception as e:
        logger.error(f"❌ Request failed: {e}")
        return 0

def benchmark_sustained():
    """Benchmark sustained performance"""
    logger.info("\n📊 Benchmarking sustained performance...")
    
    prompts = [
        "The future of artificial intelligence is",
        "In a world where technology",
        "Once upon a time in Silicon Valley",
        "The key to successful AI development is",
        "Magic Unicorn's next product will"
    ]
    
    total_tps = 0
    
    for i, prompt in enumerate(prompts):
        logger.info(f"\n🔄 Test {i+1}/5: '{prompt[:30]}...'")
        
        try:
            r = requests.post(
                f"{SERVER_URL}/generate",
                json={
                    "prompt": prompt,
                    "max_tokens": 20,
                    "temperature": 0.7
                },
                timeout=30
            )
            
            if r.status_code == 200:
                data = r.json()
                tps = data['tokens_per_second']
                total_tps += tps
                logger.info(f"   ✅ {tps:.1f} TPS")
            else:
                logger.error(f"   ❌ Failed")
                
        except Exception as e:
            logger.error(f"   ❌ Error: {e}")
        
        time.sleep(0.5)  # Brief pause
    
    avg_tps = total_tps / 5
    logger.info(f"\n📊 Average sustained TPS: {avg_tps:.1f}")
    
    if avg_tps >= 81:
        logger.info("🎉 TARGET ACHIEVED! 81+ TPS!")
    elif avg_tps >= 50:
        logger.info("✅ Good performance!")
    else:
        logger.info("⚠️ Performance below target")

def main():
    """Run tests"""
    if not wait_for_server():
        logger.error("Server not available. Start with:")
        logger.error("python3 persistent_server_working.py")
        return
    
    # Get server info
    try:
        r = requests.get(f"{SERVER_URL}/info")
        if r.status_code == 200:
            logger.info(f"\n📋 Server info: {r.json()}")
    except:
        pass
    
    # Test Magic Unicorn
    tps = test_magic_unicorn()
    
    # Benchmark sustained performance
    benchmark_sustained()

if __name__ == "__main__":
    main()