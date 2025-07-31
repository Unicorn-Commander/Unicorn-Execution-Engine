#!/usr/bin/env python3
"""Quick system monitor to check if model loading is working correctly."""

import subprocess
import psutil
import time

def get_gpu_stats():
    """Get GPU memory usage from radeontop."""
    try:
        result = subprocess.run(['radeontop', '-d', '-', '-l', '1'], 
                              capture_output=True, text=True, timeout=2)
        for line in result.stdout.split('\n'):
            if 'vram' in line.lower():
                # Extract VRAM usage
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'vram' in part.lower() and i + 1 < len(parts):
                        return parts[i + 1]
        return "Unknown"
    except:
        return "Error"

def check_processes():
    """Check if benchmark is running and its resource usage."""
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
        try:
            if 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.cmdline())
                if 'benchmark' in cmdline or 'pipeline' in cmdline:
                    return {
                        'pid': proc.info['pid'],
                        'cpu': proc.cpu_percent(interval=0.1),
                        'memory': proc.memory_percent(),
                        'ram_mb': proc.memory_info().rss / 1024 / 1024
                    }
        except:
            continue
    return None

def main():
    print("🔍 Monitoring system resources...")
    print("=" * 60)
    
    # Initial GPU check
    initial_vram = get_gpu_stats()
    print(f"Initial VRAM: {initial_vram}")
    
    # Monitor for 30 seconds
    start_time = time.time()
    high_cpu_count = 0
    high_ram_count = 0
    
    while time.time() - start_time < 30:
        proc_info = check_processes()
        if proc_info:
            current_vram = get_gpu_stats()
            
            print(f"\n⏱️  Time: {int(time.time() - start_time)}s")
            print(f"📊 Process: PID {proc_info['pid']}")
            print(f"   CPU: {proc_info['cpu']:.1f}%")
            print(f"   RAM: {proc_info['ram_mb']:.0f}MB ({proc_info['memory']:.1f}%)")
            print(f"   VRAM: {current_vram}")
            
            # Check for warning signs
            if proc_info['cpu'] > 90:
                high_cpu_count += 1
            if proc_info['ram_mb'] > 10000:  # >10GB RAM
                high_ram_count += 1
                
        time.sleep(2)
    
    print("\n" + "=" * 60)
    print("📋 ANALYSIS:")
    
    if high_cpu_count > 3:
        print("⚠️  HIGH CPU USAGE - Likely loading to CPU memory (BAD)")
    if high_ram_count > 3:
        print("⚠️  HIGH RAM USAGE - Model loading to system RAM instead of VRAM (BAD)")
    if initial_vram == get_gpu_stats():
        print("⚠️  NO VRAM CHANGE - GPU loading not working!")
    
    print("\n💡 RECOMMENDATION:")
    if high_cpu_count > 3 or high_ram_count > 3:
        print("   INTERRUPT! The model is loading to CPU/RAM, not GPU.")
        print("   This will take forever and won't use the GPU properly.")
    else:
        print("   Let it continue - seems to be working correctly.")

if __name__ == "__main__":
    main()