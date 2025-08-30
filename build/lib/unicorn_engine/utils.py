"""
Utility functions for hardware detection and benchmarking
"""

import os
import platform
import subprocess
from typing import List, Dict, Any
import psutil

def detect_hardware() -> Dict[str, Any]:
    """
    Detect available hardware accelerators
    
    Returns:
        Dictionary with hardware information
    """
    hardware = {
        "cpu": {
            "model": platform.processor() or "Unknown",
            "cores": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True),
            "frequency": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
        },
        "memory": {
            "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        },
        "accelerators": []
    }
    
    # Check for AMD NPU
    if os.path.exists("/dev/accel/accel0"):
        hardware["accelerators"].append({
            "type": "npu",
            "vendor": "amd",
            "model": "Phoenix NPU",
            "capabilities": "16 TOPS INT8",
            "device": "/dev/accel/accel0"
        })
    
    # Check for AMD GPU
    if os.path.exists("/dev/dri/card0"):
        try:
            # Try to get GPU info
            result = subprocess.run(
                ["lspci", "-nn"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if "AMD" in result.stdout and ("Radeon" in result.stdout or "GPU" in result.stdout):
                hardware["accelerators"].append({
                    "type": "gpu",
                    "vendor": "amd",
                    "model": "AMD Radeon",
                    "capabilities": "Vulkan 1.3, ROCm",
                    "device": "/dev/dri/card0"
                })
        except:
            pass
    
    # Check for NVIDIA GPU
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            gpu_name = result.stdout.strip()
            hardware["accelerators"].append({
                "type": "gpu",
                "vendor": "nvidia",
                "model": gpu_name,
                "capabilities": "CUDA, TensorRT",
                "device": "/dev/nvidia0"
            })
    except:
        pass
    
    return hardware

def get_available_backends() -> List[str]:
    """
    Get list of available acceleration backends
    
    Returns:
        List of backend names
    """
    backends = []
    
    # Always have CPU
    backends.append("cpu")
    
    # Check for NPU
    if os.path.exists("/dev/accel/accel0"):
        backends.append("npu")
    
    # Check for GPU
    if os.path.exists("/dev/dri/card0"):
        backends.append("amd_gpu")
        backends.append("vulkan")
    
    # Check for NVIDIA
    if os.path.exists("/dev/nvidia0"):
        backends.append("nvidia_gpu")
        backends.append("vulkan")
    
    # Check for Vulkan support
    try:
        result = subprocess.run(
            ["vulkaninfo", "--summary"],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            if "vulkan" not in backends:
                backends.append("vulkan")
    except:
        pass
    
    return backends

def benchmark_hardware(backend: str = "auto") -> Dict[str, float]:
    """
    Benchmark hardware performance
    
    Args:
        backend: Backend to benchmark (auto, npu, gpu, cpu)
        
    Returns:
        Performance metrics
    """
    from .backends import AutoAccelerator, NPUAccelerator, AMDGPUAccelerator, CPUAccelerator
    
    if backend == "auto":
        accelerator = AutoAccelerator()
    elif backend == "npu":
        accelerator = NPUAccelerator()
    elif backend == "gpu":
        accelerator = AMDGPUAccelerator()
    elif backend == "cpu":
        accelerator = CPUAccelerator()
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    return accelerator.benchmark()

def estimate_performance(
    model_size: str,
    audio_duration_seconds: float,
    backend: str = "auto"
) -> Dict[str, Any]:
    """
    Estimate performance for a given workload
    
    Args:
        model_size: Model size (base, small, medium, large, large-v2, large-v3)
        audio_duration_seconds: Duration of audio in seconds
        backend: Backend to use
        
    Returns:
        Performance estimates
    """
    # Performance multipliers for different model sizes
    size_multipliers = {
        "base": 1.0,
        "small": 1.5,
        "medium": 2.5,
        "large": 4.0,
        "large-v2": 4.2,
        "large-v3": 4.5,
    }
    
    multiplier = size_multipliers.get(model_size, 4.0)
    
    # Get backend performance
    perf = benchmark_hardware(backend)
    
    # Calculate estimates
    processing_time = (audio_duration_seconds / perf["speedup"]) * multiplier
    power_consumption = perf["power_watts"] * (processing_time / 3600)  # kWh
    
    return {
        "backend": backend,
        "model_size": model_size,
        "audio_duration_s": audio_duration_seconds,
        "processing_time_s": round(processing_time, 2),
        "real_time_factor": round(processing_time / audio_duration_seconds, 4),
        "speedup": round(audio_duration_seconds / processing_time, 1),
        "power_kwh": round(power_consumption, 6),
        "tokens_per_second": perf.get("tokens_per_sec", 0),
    }

def get_optimal_batch_size(backend: str = "auto", memory_gb: Optional[float] = None) -> int:
    """
    Get optimal batch size for given backend and memory
    
    Args:
        backend: Backend to use
        memory_gb: Available memory in GB (auto-detect if None)
        
    Returns:
        Optimal batch size
    """
    if memory_gb is None:
        memory_gb = psutil.virtual_memory().available / (1024**3)
    
    # Backend-specific batch size recommendations
    batch_sizes = {
        "npu": min(32, int(memory_gb * 4)),  # NPU can handle large batches
        "amd_gpu": min(16, int(memory_gb * 2)),
        "nvidia_gpu": min(24, int(memory_gb * 3)),
        "vulkan": min(8, int(memory_gb * 1.5)),
        "cpu": min(4, int(memory_gb * 0.5)),
    }
    
    return batch_sizes.get(backend, 1)

__all__ = [
    "detect_hardware",
    "get_available_backends",
    "benchmark_hardware",
    "estimate_performance",
    "get_optimal_batch_size",
]