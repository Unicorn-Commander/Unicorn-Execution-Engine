#!/usr/bin/env python3
"""
Persistent Model Server - Keeps model loaded in GPU memory for multiple inferences
NO SIMULATIONS - Real model, real GPU, real inference
"""

import logging
import time
import subprocess
import threading
from flask import Flask, request, jsonify
from pure_hardware_pipeline_gpu_fixed import PureHardwarePipelineGPUFixed

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model instance
model_pipeline = None
model_lock = threading.Lock()

def clear_cache():
    """Clear system file cache"""
    try:
        subprocess.run(['sudo', 'sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches'], 
                      check=True, capture_output=True)
        logger.info("✅ System cache cleared")
    except Exception as e:
        logger.warning(f"Failed to clear cache: {e}")

def initialize_model():
    """Initialize and load model to GPU once"""
    global model_pipeline
    
    logger.info("🚀 Initializing persistent model server...")
    logger.info("🦄 Loading 27B Gemma model to GPU memory...")
    
    # Initialize pipeline
    model_pipeline = PureHardwarePipelineGPUFixed()
    model_path = "quantized_models/gemma-3-27b-it-layer-by-layer"
    
    # Load model
    start_time = time.time()
    if not model_pipeline.initialize(model_path):
        logger.error("❌ Failed to initialize model")
        return False
    
    load_time = time.time() - start_time
    logger.info(f"✅ Model loaded in {load_time:.1f}s")
    logger.info(f"   Layers in GPU: {len(model_pipeline.layer_weights_gpu)}")
    
    # Clear cache after loading
    logger.info("🧹 Clearing file cache after model load...")
    clear_cache()
    
    # Check GPU memory
    try:
        result = subprocess.run(['radeontop', '-d', '-', '-l', '1'], 
                              capture_output=True, text=True, timeout=2)
        for line in result.stdout.split('\n'):
            if 'vram' in line or 'gtt' in line:
                logger.info(f"   GPU Memory: {line.strip()}")
    except:
        pass
    
    logger.info("🎯 Model server ready for inference!")
    return True

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_pipeline is not None
    })

@app.route('/generate', methods=['POST'])
def generate():
    """Generate text from prompt"""
    global model_pipeline
    
    if model_pipeline is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 503
    
    data = request.json
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 50)
    
    if not prompt:
        return jsonify({
            'error': 'No prompt provided'
        }), 400
    
    logger.info(f"📝 Generating for prompt: '{prompt[:50]}...'")
    
    # Thread-safe generation
    with model_lock:
        start_time = time.time()
        
        try:
            # Monitor GPU during generation
            gpu_monitor = threading.Thread(target=monitor_gpu_async)
            gpu_monitor.start()
            
            # Generate tokens
            result = model_pipeline.generate_tokens(prompt, max_tokens=max_tokens)
            
            generation_time = time.time() - start_time
            tps = max_tokens / generation_time
            
            logger.info(f"✅ Generated {max_tokens} tokens in {generation_time:.2f}s ({tps:.1f} TPS)")
            
            return jsonify({
                'prompt': prompt,
                'completion': result,
                'tokens_generated': max_tokens,
                'generation_time': generation_time,
                'tokens_per_second': tps
            })
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return jsonify({
                'error': str(e)
            }), 500

def monitor_gpu_async():
    """Monitor GPU usage during inference"""
    try:
        # Sample GPU usage multiple times
        for _ in range(5):
            result = subprocess.run(['radeontop', '-d', '-', '-l', '1'], 
                                  capture_output=True, text=True, timeout=1)
            for line in result.stdout.split('\n'):
                if 'gpu' in line and '%' in line:
                    # Extract GPU percentage
                    import re
                    match = re.search(r'gpu\s+(\d+\.\d+)%', line)
                    if match:
                        gpu_usage = float(match.group(1))
                        if gpu_usage > 10:  # Significant usage
                            logger.info(f"🔥 GPU Usage: {gpu_usage}%")
            time.sleep(0.2)
    except:
        pass

def main():
    """Start the persistent model server"""
    logger.info("🦄 Magic Unicorn Model Server Starting...")
    
    # Initialize model
    if not initialize_model():
        logger.error("Failed to initialize model")
        return
    
    # Start Flask server
    logger.info("🚀 Starting inference server on port 8011...")
    app.run(host='0.0.0.0', port=8011, threaded=True)

if __name__ == "__main__":
    main()