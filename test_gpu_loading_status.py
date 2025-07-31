#!/usr/bin/env python3
"""Quick test to check GPU loading status"""

import subprocess
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def monitor_gpu():
    """Monitor GPU memory during loading"""
    logger.info("Starting GPU memory monitoring...")
    
    # Get initial state
    result = subprocess.run(['radeontop', '-d', '-', '-l', '1'], 
                          capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if 'vram' in line.lower() and 'gtt' in line.lower():
            logger.info(f"Initial: {line.strip()}")
            break
    
    # Start the benchmark in background
    logger.info("Starting benchmark...")
    proc = subprocess.Popen(['python3', 'benchmark_final_performance.py'], 
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.STDOUT,
                           text=True)
    
    # Monitor for 60 seconds
    max_vram = 0
    max_gtt = 0
    for i in range(30):
        time.sleep(2)
        result = subprocess.run(['radeontop', '-d', '-', '-l', '1'], 
                              capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'vram' in line.lower() and 'gtt' in line.lower():
                # Extract VRAM and GTT values
                import re
                vram_match = re.search(r'vram\s+[\d.]+%\s+([\d.]+)mb', line)
                gtt_match = re.search(r'gtt\s+[\d.]+%\s+([\d.]+)mb', line)
                if vram_match and gtt_match:
                    vram_mb = float(vram_match.group(1))
                    gtt_mb = float(gtt_match.group(1))
                    max_vram = max(max_vram, vram_mb)
                    max_gtt = max(max_gtt, gtt_mb)
                    logger.info(f"T+{i*2}s: VRAM={vram_mb:.1f}MB, GTT={gtt_mb:.1f}MB")
                break
        
        # Check process output
        try:
            output = proc.stdout.readline()
            if output and ('error' in output.lower() or 'fail' in output.lower()):
                logger.error(f"Process error: {output.strip()}")
        except:
            pass
    
    # Kill the process
    proc.terminate()
    
    logger.info(f"\nSummary:")
    logger.info(f"Max VRAM: {max_vram:.1f}MB ({max_vram/1024:.2f}GB)")
    logger.info(f"Max GTT: {max_gtt:.1f}MB ({max_gtt/1024:.2f}GB)")
    logger.info(f"Total GPU memory used: {(max_vram + max_gtt)/1024:.2f}GB")
    
    if max_vram < 5000:  # Less than 5GB
        logger.error("❌ GPU loading NOT working - VRAM too low")
    else:
        logger.info("✅ GPU loading appears to be working!")

if __name__ == "__main__":
    monitor_gpu()