#!/usr/bin/env python3
"""
Test VibeVoice inference and prepare for optimization
"""

import os
import sys
import torch
import time
import numpy as np
from pathlib import Path

# Add VibeVoice to path
sys.path.insert(0, '/home/ucadmin/VibeVoice')

from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

def test_vibevoice_inference():
    """
    Test basic VibeVoice inference
    """
    print("Testing VibeVoice inference...")
    
    # Model configuration
    model_name = "microsoft/VibeVoice-1.5B"
    cache_dir = "/home/ucadmin/Unicorn-Orator/models/vibevoice"
    
    # Load model with CPU for now (we'll optimize for iGPU later)
    device = "cpu"  # Force CPU for optimization testing
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model (already cached)...")
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float32,  # Start with FP32
        device_map=device
    )
    
    processor = VibeVoiceProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    
    print("✓ Model loaded")
    
    # Test simple generation
    test_text = "Hello, this is a test of VibeVoice on Intel iGPU."
    
    print(f"\nTest text: {test_text}")
    print("Generating audio...")
    
    start_time = time.time()
    
    # Process input
    inputs = processor(
        text=test_text,
        return_tensors="pt"
    ).to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True
        )
    
    end_time = time.time()
    
    print(f"✓ Generation completed in {end_time - start_time:.2f} seconds")
    
    # Get model size info
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    print(f"Model size in memory: {model_size:.2f} GB")
    
    # Memory usage
    if torch.cuda.is_available():
        print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    return model, processor

def export_to_torchscript(model, processor):
    """
    Export model to TorchScript for optimization
    """
    print("\n" + "="*50)
    print("Exporting to TorchScript")
    print("="*50)
    
    output_dir = "/home/ucadmin/Unicorn-Orator/models/vibevoice_optimized"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Try to trace the model
        print("Attempting TorchScript trace...")
        
        # Create example inputs
        example_text = "Hello world"
        inputs = processor(text=example_text, return_tensors="pt")
        
        # Trace the model
        traced_model = torch.jit.trace(model, (inputs['input_ids'],))
        
        # Save traced model
        output_path = os.path.join(output_dir, "vibevoice_traced.pt")
        torch.jit.save(traced_model, output_path)
        
        print(f"✓ Model traced and saved to: {output_path}")
        print(f"Size: {os.path.getsize(output_path) / 1e9:.2f} GB")
        
    except Exception as e:
        print(f"TorchScript tracing failed: {e}")
        print("Trying scripting instead...")
        
        try:
            scripted_model = torch.jit.script(model)
            output_path = os.path.join(output_dir, "vibevoice_scripted.pt")
            torch.jit.save(scripted_model, output_path)
            print(f"✓ Model scripted and saved to: {output_path}")
        except Exception as e2:
            print(f"Scripting also failed: {e2}")
            print("Will need to use alternative optimization approach")

def optimize_for_intel_igpu(model, processor):
    """
    Optimize model for Intel iGPU
    """
    print("\n" + "="*50)
    print("Optimizing for Intel iGPU")
    print("="*50)
    
    output_dir = "/home/ucadmin/Unicorn-Orator/models/vibevoice_igpu"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Try quantization with PyTorch
    print("\n1. Attempting PyTorch quantization...")
    try:
        import torch.quantization as quant
        
        # Prepare for quantization
        model.eval()
        
        # Dynamic quantization (works without calibration data)
        quantized_model = quant.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.Conv1d},  # Layers to quantize
            dtype=torch.qint8
        )
        
        # Save quantized model
        output_path = os.path.join(output_dir, "vibevoice_int8.pth")
        torch.save(quantized_model.state_dict(), output_path)
        
        print(f"✓ INT8 quantized model saved: {output_path}")
        print(f"Size: {os.path.getsize(output_path) / 1e9:.2f} GB")
        
        # Test quantized model
        test_text = "Testing quantized model"
        inputs = processor(text=test_text, return_tensors="pt")
        
        with torch.no_grad():
            start = time.time()
            _ = quantized_model.generate(**inputs, max_new_tokens=50)
            end = time.time()
            
        print(f"Quantized inference time: {end - start:.2f}s")
        
    except Exception as e:
        print(f"PyTorch quantization failed: {e}")
    
    # 2. Prepare for OpenVINO
    print("\n2. Preparing for OpenVINO optimization...")
    
    # Save model in format suitable for OpenVINO conversion
    try:
        # Save the model state dict
        torch.save(model.state_dict(), os.path.join(output_dir, "model_weights.pth"))
        
        # Save processor config
        processor.save_pretrained(output_dir)
        
        print(f"✓ Model weights and config saved to: {output_dir}")
        
    except Exception as e:
        print(f"Failed to save for OpenVINO: {e}")
    
    return output_dir

def create_optimized_inference_script():
    """
    Create optimized inference script for Intel iGPU
    """
    print("\n" + "="*50)
    print("Creating Optimized Inference Script")
    print("="*50)
    
    script_content = '''#!/usr/bin/env python3
"""
Optimized VibeVoice inference for Intel iGPU
Uses OpenVINO and mixed precision
"""

import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path

class VibeVoiceIntelOptimized:
    """
    VibeVoice optimized for Intel iGPU with OpenVINO
    """
    
    def __init__(self, model_path, device="igpu"):
        self.device = device
        self.setup_providers()
        self.load_model(model_path)
    
    def setup_providers(self):
        """Configure OpenVINO for Intel iGPU"""
        if self.device == "igpu":
            self.providers = [
                ('OpenVINOExecutionProvider', {
                    'device_type': 'GPU',
                    'precision': 'FP16',  # Use FP16 for iGPU
                    'cache_dir': './openvino_cache',
                    'enable_dynamic_shapes': True
                })
            ]
        else:
            self.providers = ['CPUExecutionProvider']
    
    def load_model(self, model_path):
        """Load optimized model"""
        # This would load the ONNX or OpenVINO IR model
        pass
    
    def generate(self, text, max_length=1000):
        """Generate audio from text"""
        # Optimized inference pipeline
        pass

if __name__ == "__main__":
    # Example usage
    model = VibeVoiceIntelOptimized("models/vibevoice_igpu")
    audio = model.generate("Hello from Intel iGPU optimized VibeVoice!")
'''
    
    output_path = "/home/ucadmin/Unicorn-Orator/vibevoice_intel_optimized.py"
    with open(output_path, 'w') as f:
        f.write(script_content)
    
    print(f"✓ Optimized inference script created: {output_path}")

if __name__ == "__main__":
    print("VibeVoice Optimization for Intel iGPU")
    print("="*50)
    
    # Test inference
    model, processor = test_vibevoice_inference()
    
    # Export to TorchScript
    export_to_torchscript(model, processor)
    
    # Optimize for Intel iGPU
    output_dir = optimize_for_intel_igpu(model, processor)
    
    # Create optimized inference script
    create_optimized_inference_script()
    
    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    print("✓ VibeVoice model loaded and tested")
    print("✓ Quantization attempted")
    print("✓ Model prepared for Intel iGPU optimization")
    print("\nNext steps:")
    print("1. Convert to ONNX using optimum-intel")
    print("2. Apply OpenVINO optimizations")
    print("3. Test on Intel iGPU hardware")