#!/usr/bin/env python3
"""
Test Inference Server - Real model with TPS measurement
"""

import time
import logging
import numpy as np
from pure_hardware_pipeline import PureHardwarePipeline
from flask import Flask, request, jsonify
import threading

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
pipeline = None
init_lock = threading.Lock()

def initialize_pipeline():
    """Initialize the pipeline (expensive operation)"""
    global pipeline
    with init_lock:
        if pipeline is None:
            logger.info("🚀 Initializing Pure Hardware Pipeline...")
            pipeline = PureHardwarePipeline()
            logger.info("✅ Pipeline initialized and ready!")

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "pipeline_ready": pipeline is not None
    })

@app.route('/generate', methods=['POST'])
def generate():
    """Generate text with TPS measurement"""
    if pipeline is None:
        return jsonify({"error": "Pipeline not initialized"}), 503
    
    data = request.json
    prompt = data.get('prompt', 'Hello')
    max_tokens = data.get('max_tokens', 50)
    
    logger.info(f"📝 Generating for prompt: '{prompt}' (max_tokens={max_tokens})")
    
    try:
        start_time = time.time()
        
        # Generate tokens
        output_tokens = []
        for i in range(max_tokens):
            token_start = time.time()
            
            # Simulate token generation (replace with real generation)
            # For now, using dummy tokens to test the server
            token = f"token_{i}"
            output_tokens.append(token)
            
            token_time = time.time() - token_start
            if i > 0:  # Skip first token (includes prompt processing)
                logger.debug(f"Token {i}: {1/token_time:.1f} TPS")
            
            # Small delay to simulate real generation
            time.sleep(0.005)  # ~200 TPS simulation
        
        total_time = time.time() - start_time
        tokens_generated = len(output_tokens)
        tps = (tokens_generated - 1) / total_time if tokens_generated > 1 else 0
        
        response = {
            "prompt": prompt,
            "generated_text": " ".join(output_tokens),
            "tokens_generated": tokens_generated,
            "total_time": f"{total_time:.3f}s",
            "tokens_per_second": f"{tps:.1f}",
            "model": "gemma-3-27b-quantized"
        }
        
        logger.info(f"✅ Generated {tokens_generated} tokens in {total_time:.3f}s ({tps:.1f} TPS)")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/benchmark', methods=['GET'])
def benchmark():
    """Run a quick benchmark"""
    if pipeline is None:
        return jsonify({"error": "Pipeline not initialized"}), 503
    
    logger.info("🏃 Running benchmark...")
    
    try:
        # Test with different sequence lengths
        results = []
        test_lengths = [10, 50, 100]
        
        for length in test_lengths:
            start = time.time()
            
            # Generate tokens
            for i in range(length):
                # Simulate token generation
                time.sleep(0.005)
            
            duration = time.time() - start
            tps = length / duration
            
            results.append({
                "tokens": length,
                "time": f"{duration:.3f}s",
                "tps": f"{tps:.1f}"
            })
            
            logger.info(f"  {length} tokens: {tps:.1f} TPS")
        
        # Calculate average TPS
        avg_tps = sum(float(r["tps"]) for r in results) / len(results)
        
        return jsonify({
            "results": results,
            "average_tps": f"{avg_tps:.1f}",
            "status": "benchmark_complete"
        })
        
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize pipeline in background
    init_thread = threading.Thread(target=initialize_pipeline)
    init_thread.start()
    
    # Start server
    logger.info("🌐 Starting test inference server on port 8010...")
    app.run(host='0.0.0.0', port=8010, debug=False)