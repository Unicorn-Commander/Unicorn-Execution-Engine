#!/usr/bin/env python3
"""
Run Real Inference Server - Load actual 27B model and serve it
"""

import time
import logging
import json
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from pure_hardware_pipeline import PureHardwarePipeline
import threading
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global state
pipeline = None
model_loaded = False
loading_status = "initializing"

class InferenceHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Get memory usage
            vram_mb, gtt_mb = check_gpu_memory()
            
            response = {
                "status": "ready" if model_loaded else loading_status,
                "model": "gemma-3-27b-quantized (layer-by-layer)",
                "memory": {
                    "vram_mb": vram_mb,
                    "vram_gb": f"{vram_mb/1024:.1f}",
                    "gtt_mb": gtt_mb,
                    "gtt_gb": f"{gtt_mb/1024:.1f}",
                    "total_gb": f"{(vram_mb + gtt_mb)/1024:.1f}"
                },
                "endpoints": {
                    "status": "GET /",
                    "generate": "POST /generate"
                }
            }
            
            self.wfile.write(json.dumps(response, indent=2).encode())
            
        else:
            self.send_error(404)
    
    def do_POST(self):
        if self.path == '/generate':
            if not model_loaded:
                self.send_error(503, f"Model still {loading_status}")
                return
                
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                prompt = data.get('prompt', 'Hello')
                max_tokens = data.get('max_tokens', 50)
                
                self.generate_text(prompt, max_tokens)
            except Exception as e:
                self.send_error(400, str(e))
        else:
            self.send_error(404)
    
    def generate_text(self, prompt, max_tokens):
        """Generate text using the pipeline"""
        logger.info(f"📝 Generating for: '{prompt}' (max_tokens={max_tokens})")
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        start_time = time.time()
        
        try:
            # Use the real pipeline
            output = pipeline.generate(prompt, max_tokens=max_tokens)
            
            generation_time = time.time() - start_time
            tps = max_tokens / generation_time if generation_time > 0 else 0
            
            response = {
                "prompt": prompt,
                "generated": output,
                "tokens": max_tokens,
                "time": f"{generation_time:.3f}s",
                "tps": f"{tps:.1f}"
            }
            
            logger.info(f"✅ Generated {max_tokens} tokens at {tps:.1f} TPS")
            
            self.wfile.write(json.dumps(response, indent=2).encode())
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            error_response = {"error": str(e)}
            self.wfile.write(json.dumps(error_response).encode())
    
    def log_message(self, format, *args):
        """Override to reduce logging"""
        if '/favicon.ico' not in args[0]:
            logger.info(f"{self.client_address[0]} - {args[0]}")

def check_gpu_memory():
    """Check current GPU memory usage"""
    try:
        result = subprocess.run(['radeontop', '-d', '-', '-l', '1'], 
                              capture_output=True, text=True, timeout=1)
        if result.stdout:
            import re
            vram_match = re.search(r'vram\s+(\d+\.\d+)%\s+(\d+\.\d+)mb', result.stdout)
            gtt_match = re.search(r'gtt\s+(\d+\.\d+)%\s+(\d+\.\d+)mb', result.stdout)
            
            if vram_match and gtt_match:
                return float(vram_match.group(2)), float(gtt_match.group(2))
    except:
        pass
    return 0, 0

def monitor_loading():
    """Monitor memory during loading"""
    global loading_status
    
    while loading_status != "complete":
        vram_mb, gtt_mb = check_gpu_memory()
        if vram_mb > 0 or gtt_mb > 0:
            logger.info(f"📊 Loading... VRAM: {vram_mb:.0f}MB ({vram_mb/1024:.1f}GB), GTT: {gtt_mb:.0f}MB ({gtt_mb/1024:.1f}GB)")
        time.sleep(5)

def initialize_pipeline():
    """Initialize the pipeline with real model"""
    global pipeline, model_loaded, loading_status
    
    try:
        logger.info("🚀 Initializing Pure Hardware Pipeline...")
        loading_status = "initializing"
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_loading, daemon=True)
        monitor_thread.start()
        
        # Initialize pipeline
        pipeline = PureHardwarePipeline()
        
        loading_status = "complete"
        model_loaded = True
        
        # Final memory check
        time.sleep(2)
        vram_mb, gtt_mb = check_gpu_memory()
        
        logger.info("✅ Pipeline initialized!")
        logger.info(f"📊 Final memory usage:")
        logger.info(f"   VRAM: {vram_mb:.0f}MB ({vram_mb/1024:.1f}GB)")
        logger.info(f"   GTT: {gtt_mb:.0f}MB ({gtt_mb/1024:.1f}GB)")
        logger.info(f"   Total: {(vram_mb + gtt_mb)/1024:.1f}GB")
        
        if vram_mb < 1000 and gtt_mb < 1000:
            logger.warning("⚠️ Model might not be fully loaded to GPU memory!")
            logger.warning("Expected: ~16GB VRAM + ~10GB GTT")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize pipeline: {e}")
        loading_status = "failed"
        import traceback
        traceback.print_exc()

def run_server():
    """Run the HTTP server"""
    server_address = ('', 8010)
    httpd = HTTPServer(server_address, InferenceHandler)
    
    logger.info("🌐 Real inference server running on http://0.0.0.0:8010")
    logger.info("📡 Loading model...")
    
    httpd.serve_forever()

if __name__ == '__main__':
    # Initialize pipeline in background
    init_thread = threading.Thread(target=initialize_pipeline)
    init_thread.start()
    
    # Run server
    run_server()