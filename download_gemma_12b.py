#!/usr/bin/env python3
"""Download Gemma-3-12B-IT model for testing"""

import os
from huggingface_hub import snapshot_download
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_gemma_12b():
    """Download Gemma-3-12B-IT model"""
    model_id = "google/gemma-3-12b-it"
    local_dir = "./models/gemma-3-12b-it"
    
    logger.info(f"🚀 Downloading {model_id}...")
    logger.info(f"📁 Destination: {local_dir}")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)
        
        # Download the model
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=["*.md", "*.txt"]  # Skip docs
        )
        
        logger.info("✅ Download complete!")
        
        # Check the size
        total_size = 0
        for root, dirs, files in os.walk(local_dir):
            for f in files:
                if f.endswith('.safetensors'):
                    fp = os.path.join(root, f)
                    total_size += os.path.getsize(fp)
        
        logger.info(f"📊 Total model size: {total_size / 1024 / 1024 / 1024:.1f} GB")
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        logger.info("You may need to login with: huggingface-cli login")

if __name__ == "__main__":
    download_gemma_12b()