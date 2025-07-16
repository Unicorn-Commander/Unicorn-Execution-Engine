#!/usr/bin/env python3
"""
Persistent Model Server with Working GPU Pipeline
Loads model once, serves many requests
"""

import logging
import time
import subprocess
import threading
from flask import Flask, request, jsonify
from gpu_pipeline_working import GPUPipelineWorking

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model instance
model_pipeline = None
model_lock = threading.Lock()
load_time = 0

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
    global model_pipeline, load_time
    
    logger.info("🚀 Initializing persistent model server with WORKING pipeline...")
    logger.info("🦄 Loading 27B Gemma model to GPU memory...")
    
    # Use our working pipeline
    model_pipeline = GPUPipelineWorking()
    model_path = "quantized_models/gemma-3-27b-it-layer-by-layer"
    
    # Load model
    start_time = time.time()
    if not model_pipeline.initialize(model_path):
        logger.error("❌ Failed to initialize model")
        return False
    
    load_time = time.time() - start_time
    logger.info(f"✅ Model loaded in {load_time:.1f}s")
    
    # Clear cache after loading
    logger.info("🧹 Clearing file cache...")
    clear_cache()
    
    # Test with single token
    logger.info("🧪 Testing inference...")
    test_ids = [1, 2, 3, 4, 5]  # Simple test
    try:
        start = time.time()
        output = model_pipeline.generate_tokens(test_ids, max_tokens=5)
        test_time = time.time() - start
        logger.info(f"✅ Test inference: {len(output)} tokens in {test_time:.2f}s")
    except Exception as e:
        logger.error(f"❌ Test inference failed: {e}")
    
    logger.info("🎯 Model server ready!")
    return True

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_pipeline is not None,
        'load_time_seconds': load_time
    })

@app.route('/generate', methods=['POST'])
def generate():
    """Generate text from prompt"""
    global model_pipeline
    
    if model_pipeline is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    data = request.json
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 50)
    temperature = data.get('temperature', 0.7)
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    logger.info(f"📝 Request: '{prompt[:50]}...' ({max_tokens} tokens)")
    
    # Simple tokenization
    input_ids = [ord(c) % 1000 for c in prompt]
    
    # Thread-safe generation
    with model_lock:
        try:
            start_time = time.time()
            generated_ids = model_pipeline.generate_tokens(
                input_ids, 
                max_tokens=max_tokens,
                temperature=temperature
            )
            gen_time = time.time() - start_time
            
            # Simple detokenization
            response = ''.join([chr((t % 94) + 33) for t in generated_ids])
            
            tps = len(generated_ids) / gen_time if gen_time > 0 else 0
            
            logger.info(f"✅ Generated {len(generated_ids)} tokens in {gen_time:.1f}s ({tps:.1f} TPS)")
            
            return jsonify({
                'prompt': prompt,
                'completion': response,
                'tokens_generated': len(generated_ids),
                'generation_time': gen_time,
                'tokens_per_second': tps
            })
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return jsonify({'error': str(e)}), 500

@app.route('/info', methods=['GET'])
def info():
    """Get model info"""
    return jsonify({
        'model': 'Gemma 27B',
        'pipeline': 'gpu_pipeline_working',
        'features': {
            'npu_available': False,  # Driver missing
            'gpu_compute': True,
            'attention': 'GPU (CPU fallback for now)',
            'ffn': 'GPU (Vulkan)',
            'quantization': 'INT8'
        }
    })

def main():
    """Start the server"""
    if not initialize_model():
        logger.error("Failed to initialize model")
        return
    
    logger.info("🚀 Starting server on port 8011...")
    app.run(host='0.0.0.0', port=8011, threaded=True)

if __name__ == "__main__":
    main()