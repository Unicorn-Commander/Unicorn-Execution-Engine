"""
Intel iGPU Execution Module for Unicorn Execution Engine
=========================================================

This module provides optimized AI inference on Intel integrated GPUs
using OpenVINO runtime acceleration.

Target Hardware: Intel Iris Xe, Intel Arc iGPU, Intel UHD Graphics
"""

import onnxruntime as ort
import numpy as np
from typing import Dict, Any, Optional, List
import logging
import os
import subprocess

logger = logging.getLogger(__name__)


class IntelIGPUExecutor:
    """
    Execution provider for Intel integrated GPUs using OpenVINO.
    
    Features:
    - Automatic hardware detection
    - FP16 precision optimization
    - Shared memory architecture
    - Power-efficient inference
    """
    
    def __init__(self, cache_dir: str = "./openvino_cache"):
        """
        Initialize Intel iGPU executor.
        
        Args:
            cache_dir: Directory for OpenVINO model cache
        """
        self.cache_dir = cache_dir
        self.device_info = self._detect_intel_gpu()
        self.providers = self._configure_providers()
        
    def _detect_intel_gpu(self) -> Dict[str, Any]:
        """
        Detect Intel GPU capabilities.
        
        Returns:
            Dictionary with GPU information
        """
        info = {
            "available": False,
            "device_name": "Unknown",
            "execution_units": 0,
            "memory_mb": 0
        }
        
        try:
            # Check for Intel GPU using clinfo or similar
            result = subprocess.run(
                ["lspci", "-nn"], 
                capture_output=True, 
                text=True,
                timeout=5
            )
            
            if "Intel" in result.stdout and ("VGA" in result.stdout or "Display" in result.stdout):
                info["available"] = True
                
                # Parse GPU model
                for line in result.stdout.split('\n'):
                    if "Intel" in line and ("VGA" in line or "Display" in line):
                        # Extract model name
                        parts = line.split(': ')
                        if len(parts) > 1:
                            info["device_name"] = parts[1].split('[')[0].strip()
                        
                        # Detect execution units based on known models
                        if "Iris Xe" in line:
                            info["execution_units"] = 96  # Common Iris Xe config
                        elif "UHD" in line:
                            info["execution_units"] = 32  # Common UHD config
                        elif "Arc" in line:
                            info["execution_units"] = 128  # Arc iGPU estimate
                        break
                
                logger.info(f"Detected Intel GPU: {info['device_name']}")
                
        except Exception as e:
            logger.warning(f"Could not detect Intel GPU: {e}")
            
        return info
    
    def _configure_providers(self) -> List:
        """
        Configure ONNX Runtime providers for Intel iGPU.
        
        Returns:
            List of execution providers
        """
        providers = []
        
        if self.device_info["available"]:
            # Primary: OpenVINO for Intel GPU
            providers.append(('OpenVINOExecutionProvider', {
                'device_type': 'GPU',
                'precision': 'FP16',
                'cache_dir': self.cache_dir,
                'enable_dynamic_shapes': True,
                'num_of_threads': 4
            }))
            logger.info("Configured OpenVINO GPU provider")
        
        # Fallback: CPU
        providers.append('CPUExecutionProvider')
        
        return providers
    
    def create_session(self, model_path: str) -> ort.InferenceSession:
        """
        Create optimized inference session for Intel iGPU.
        
        Args:
            model_path: Path to ONNX model
            
        Returns:
            Configured InferenceSession
        """
        # Create cache directory if needed
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Session options for optimization
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True
        
        # Create session with Intel iGPU providers
        session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=self.providers
        )
        
        # Log actual provider being used
        actual_provider = session.get_providers()[0]
        logger.info(f"Using provider: {actual_provider}")
        
        return session
    
    def optimize_input(self, input_data: np.ndarray) -> np.ndarray:
        """
        Optimize input data for Intel iGPU.
        
        Args:
            input_data: Input numpy array
            
        Returns:
            Optimized input array
        """
        # Ensure contiguous memory layout
        if not input_data.flags['C_CONTIGUOUS']:
            input_data = np.ascontiguousarray(input_data)
        
        # Convert to FP16 if beneficial
        if input_data.dtype == np.float32 and self.device_info["available"]:
            # Keep as FP32 - OpenVINO handles conversion
            pass
        
        return input_data
    
    def run_inference(self, session: ort.InferenceSession, 
                     inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """
        Run optimized inference on Intel iGPU.
        
        Args:
            session: ONNX Runtime session
            inputs: Dictionary of input tensors
            
        Returns:
            List of output tensors
        """
        # Optimize inputs
        optimized_inputs = {
            name: self.optimize_input(tensor) 
            for name, tensor in inputs.items()
        }
        
        # Run inference
        outputs = session.run(None, optimized_inputs)
        
        return outputs
    
    def get_performance_hints(self) -> Dict[str, Any]:
        """
        Get performance optimization hints for current hardware.
        
        Returns:
            Dictionary of optimization hints
        """
        hints = {
            "batch_size": 1,  # iGPUs work best with small batches
            "precision": "FP16",
            "memory_pattern": "shared",  # Uses system RAM
            "power_mode": "balanced"
        }
        
        if self.device_info["execution_units"] >= 96:
            # High-end iGPU (Iris Xe)
            hints["batch_size"] = 4
            hints["power_mode"] = "performance"
        elif self.device_info["execution_units"] >= 32:
            # Mid-range iGPU (UHD)
            hints["batch_size"] = 2
            
        return hints


# Example usage for Kokoro TTS
class KokoroIntelOptimized:
    """
    Kokoro TTS optimized for Intel iGPU execution.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize Kokoro with Intel iGPU optimization.
        
        Args:
            model_path: Path to kokoro-v0_19.onnx
        """
        self.executor = IntelIGPUExecutor()
        self.session = self.executor.create_session(model_path)
        
    def synthesize(self, tokens: np.ndarray, style: np.ndarray, 
                  speed: float = 1.0) -> np.ndarray:
        """
        Synthesize speech using Intel iGPU acceleration.
        
        Args:
            tokens: Phoneme token IDs
            style: Voice embedding (256-dim)
            speed: Speech rate multiplier
            
        Returns:
            Audio waveform
        """
        inputs = {
            'tokens': tokens.astype(np.int64),
            'style': style.astype(np.float32),
            'speed': np.array([speed], dtype=np.float32)
        }
        
        outputs = self.executor.run_inference(self.session, inputs)
        return outputs[0]


if __name__ == "__main__":
    # Test Intel iGPU detection
    executor = IntelIGPUExecutor()
    print(f"Intel iGPU Available: {executor.device_info['available']}")
    print(f"Device: {executor.device_info['device_name']}")
    print(f"Execution Units: {executor.device_info['execution_units']}")
    print(f"Performance Hints: {executor.get_performance_hints()}")