"""
Unicorn Execution Engine
Hardware-accelerated AI inference runtime

Copyright (c) 2025 Magic Unicorn Unconventional Technology & Stuff Inc.
"""

__version__ = "1.0.0"
__author__ = "Magic Unicorn Inc."
__email__ = "hello@magicunicorn.tech"

import os
import warnings
from typing import Optional, Union, Dict, Any

# Import backend accelerators
from .backends import (
    AutoAccelerator,
    NPUAccelerator,
    AMDGPUAccelerator,
    VulkanAccelerator,
    CPUAccelerator,
)

# Import model classes
from .models import (
    NPUWhisperX,
    WhisperModel,
    LLMModel,
    VisionModel,
)

# Import utilities
from .utils import (
    detect_hardware,
    get_available_backends,
    benchmark_hardware,
)

# Check available hardware
AVAILABLE_BACKENDS = get_available_backends()

def load_model(
    model_name: str,
    backend: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Load a model with automatic backend selection
    
    Args:
        model_name: Hugging Face model name or local path
        backend: Force specific backend (npu, gpu, vulkan, cpu)
        device: Device index for multi-device systems
        **kwargs: Additional model configuration
    
    Returns:
        Loaded model ready for inference
    
    Example:
        >>> model = load_model("magicunicorn/whisper-large-v3-amd-npu-int8")
        >>> result = model.transcribe("audio.wav")
    """
    
    if backend is None:
        # Auto-detect best backend
        accelerator = AutoAccelerator()
    else:
        # Use specified backend
        backend_map = {
            "npu": NPUAccelerator,
            "amd_gpu": AMDGPUAccelerator,
            "vulkan": VulkanAccelerator,
            "cpu": CPUAccelerator,
        }
        
        if backend not in backend_map:
            raise ValueError(f"Unknown backend: {backend}. Choose from {list(backend_map.keys())}")
        
        accelerator = backend_map[backend]()
    
    # Load model with selected accelerator
    return accelerator.from_pretrained(model_name, device=device, **kwargs)

# Convenience functions
def transcribe(
    audio_path: str,
    model: str = "magicunicorn/whisper-large-v3-amd-npu-int8",
    **kwargs
) -> Dict[str, Any]:
    """
    Transcribe audio using best available backend
    
    Args:
        audio_path: Path to audio file
        model: Model to use for transcription
        **kwargs: Additional transcription options
    
    Returns:
        Transcription results with text, segments, and metadata
    """
    whisper_model = load_model(model)
    return whisper_model.transcribe(audio_path, **kwargs)

def generate(
    prompt: str,
    model: str = "magicunicorn/granite-3b-amd-igpu-q4",
    **kwargs
) -> str:
    """
    Generate text using best available backend
    
    Args:
        prompt: Input prompt
        model: Model to use for generation
        **kwargs: Generation parameters
    
    Returns:
        Generated text
    """
    llm_model = load_model(model)
    return llm_model.generate(prompt, **kwargs)

# Print hardware info on import
if len(AVAILABLE_BACKENDS) > 0:
    print(f"🦄 Unicorn Engine v{__version__} initialized")
    print(f"   Available backends: {', '.join(AVAILABLE_BACKENDS)}")
else:
    warnings.warn(
        "No accelerated backends detected. Falling back to CPU. "
        "Install appropriate drivers for better performance."
    )

__all__ = [
    # Version info
    "__version__",
    
    # Main functions
    "load_model",
    "transcribe",
    "generate",
    
    # Accelerators
    "AutoAccelerator",
    "NPUAccelerator",
    "AMDGPUAccelerator",
    "VulkanAccelerator",
    "CPUAccelerator",
    
    # Models
    "NPUWhisperX",
    "WhisperModel",
    "LLMModel",
    "VisionModel",
    
    # Utilities
    "detect_hardware",
    "get_available_backends",
    "benchmark_hardware",
    "AVAILABLE_BACKENDS",
]