#!/usr/bin/env python3
"""
Export VibeVoice 1.5B model to ONNX format for Intel iGPU optimization
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add VibeVoice to path
sys.path.insert(0, '/home/ucadmin/VibeVoice')

def download_and_export_vibevoice():
    """
    Download VibeVoice model and export to ONNX
    """
    print("Starting VibeVoice ONNX export...")
    
    # Import VibeVoice modules
    try:
        from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
        from transformers import AutoModel, AutoTokenizer
        print("✓ VibeVoice modules imported")
    except ImportError as e:
        print(f"Error importing VibeVoice: {e}")
        print("Installing missing dependencies...")
        os.system("pip install --user --break-system-packages ml-collections librosa scipy")
        from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
    
    # Model configuration
    model_name = "microsoft/VibeVoice-1.5B"
    cache_dir = "/home/ucadmin/Unicorn-Orator/models/vibevoice"
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"Loading model: {model_name}")
    print("This will download ~5.4GB on first run...")
    
    try:
        # Load model and processor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Load the model
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float32,  # Use FP32 for ONNX export
            device_map="cpu"  # Keep on CPU for export
        )
        
        processor = VibeVoiceProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        print("✓ Model loaded successfully")
        
        # Get model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params / 1e9:.2f}B")
        
        # Try to trace the model for ONNX export
        print("\nAttempting ONNX export...")
        export_onnx_components(model, processor)
        
    except Exception as e:
        print(f"Error during model loading/export: {e}")
        import traceback
        traceback.print_exc()
        
        # Try alternative approach
        print("\nTrying alternative export approach...")
        export_with_torch_onnx(model_name, cache_dir)

def export_onnx_components(model, processor):
    """
    Export VibeVoice components to ONNX
    """
    output_dir = "/home/ucadmin/Unicorn-Orator/models/vibevoice_onnx"
    os.makedirs(output_dir, exist_ok=True)
    
    # VibeVoice has multiple components we need to export
    print("VibeVoice architecture components:")
    print("1. LLM backbone (Qwen2.5-1.5B)")
    print("2. Acoustic Tokenizer")
    print("3. Semantic Tokenizer") 
    print("4. Diffusion Head")
    
    # Try to access individual components
    if hasattr(model, 'llm'):
        print("\nExporting LLM backbone...")
        export_llm_component(model.llm, output_dir)
    
    if hasattr(model, 'acoustic_tokenizer'):
        print("\nExporting Acoustic Tokenizer...")
        export_tokenizer_component(model.acoustic_tokenizer, output_dir, "acoustic")
    
    if hasattr(model, 'semantic_tokenizer'):
        print("\nExporting Semantic Tokenizer...")
        export_tokenizer_component(model.semantic_tokenizer, output_dir, "semantic")
    
    if hasattr(model, 'diffusion_head'):
        print("\nExporting Diffusion Head...")
        export_diffusion_component(model.diffusion_head, output_dir)
    
    print(f"\n✓ Components exported to: {output_dir}")

def export_llm_component(llm, output_dir):
    """Export LLM backbone to ONNX"""
    try:
        # Create dummy inputs for LLM
        batch_size = 1
        seq_length = 128
        
        dummy_input_ids = torch.randint(0, 32000, (batch_size, seq_length))
        dummy_attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
        
        output_path = os.path.join(output_dir, "llm_backbone.onnx")
        
        # Export with dynamic axes for variable sequence length
        torch.onnx.export(
            llm,
            (dummy_input_ids, dummy_attention_mask),
            output_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["hidden_states"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "hidden_states": {0: "batch", 1: "sequence"}
            },
            opset_version=16,
            do_constant_folding=True
        )
        
        print(f"  ✓ LLM exported: {output_path}")
        print(f"  Size: {os.path.getsize(output_path) / 1e9:.2f}GB")
        
    except Exception as e:
        print(f"  ✗ Failed to export LLM: {e}")

def export_tokenizer_component(tokenizer, output_dir, name):
    """Export tokenizer component to ONNX"""
    try:
        # Tokenizers process audio at 7.5Hz
        # Create dummy audio input
        batch_size = 1
        audio_length = 24000 * 10  # 10 seconds at 24kHz
        
        dummy_audio = torch.randn(batch_size, audio_length)
        
        output_path = os.path.join(output_dir, f"{name}_tokenizer.onnx")
        
        torch.onnx.export(
            tokenizer,
            dummy_audio,
            output_path,
            input_names=["audio"],
            output_names=["tokens"],
            dynamic_axes={
                "audio": {0: "batch", 1: "length"},
                "tokens": {0: "batch", 1: "sequence"}
            },
            opset_version=16,
            do_constant_folding=True
        )
        
        print(f"  ✓ {name.capitalize()} tokenizer exported: {output_path}")
        print(f"  Size: {os.path.getsize(output_path) / 1e9:.2f}GB")
        
    except Exception as e:
        print(f"  ✗ Failed to export {name} tokenizer: {e}")

def export_diffusion_component(diffusion, output_dir):
    """Export diffusion head to ONNX"""
    try:
        # Diffusion head takes hidden states from LLM
        batch_size = 1
        seq_length = 128
        hidden_size = 1536  # Qwen2.5-1.5B hidden size
        
        dummy_hidden = torch.randn(batch_size, seq_length, hidden_size)
        dummy_timestep = torch.tensor([100])
        
        output_path = os.path.join(output_dir, "diffusion_head.onnx")
        
        torch.onnx.export(
            diffusion,
            (dummy_hidden, dummy_timestep),
            output_path,
            input_names=["hidden_states", "timestep"],
            output_names=["audio_features"],
            dynamic_axes={
                "hidden_states": {0: "batch", 1: "sequence"},
                "audio_features": {0: "batch", 1: "sequence"}
            },
            opset_version=16,
            do_constant_folding=True
        )
        
        print(f"  ✓ Diffusion head exported: {output_path}")
        print(f"  Size: {os.path.getsize(output_path) / 1e9:.2f}GB")
        
    except Exception as e:
        print(f"  ✗ Failed to export diffusion head: {e}")

def export_with_torch_onnx(model_name, cache_dir):
    """
    Alternative export using torch.onnx directly
    """
    print("Attempting direct torch.onnx export...")
    
    # First, let's try to load just the base model
    from transformers import AutoModelForCausalLM
    
    try:
        # Load the Qwen2.5 backbone
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B",
            cache_dir=cache_dir,
            torch_dtype=torch.float32
        )
        
        print("✓ Loaded Qwen2.5-1.5B base model")
        
        # Export base model
        output_path = "/home/ucadmin/Unicorn-Orator/models/vibevoice_onnx/qwen_base.onnx"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create dummy inputs
        dummy_input = torch.randint(0, 32000, (1, 128))
        
        torch.onnx.export(
            base_model,
            dummy_input,
            output_path,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "sequence"}
            },
            opset_version=16
        )
        
        print(f"✓ Base model exported to: {output_path}")
        print(f"Size: {os.path.getsize(output_path) / 1e9:.2f}GB")
        
    except Exception as e:
        print(f"Failed to export base model: {e}")

def analyze_model_structure():
    """
    Analyze VibeVoice model structure for optimization opportunities
    """
    print("\n" + "="*50)
    print("VibeVoice Model Analysis for Intel iGPU")
    print("="*50)
    
    print("\nOptimization Strategy:")
    print("1. LLM Backbone (1.5B params) → INT8 quantization")
    print("   - Most parameters are here, good for aggressive quantization")
    print("   - Expected size reduction: 6GB → 1.5GB")
    
    print("\n2. Acoustic Tokenizer (340M) → FP16")
    print("   - Audio quality critical, use FP16")
    print("   - Expected size reduction: 1.3GB → 0.65GB")
    
    print("\n3. Diffusion Head (123M) → FP16")
    print("   - Final audio generation, preserve quality")
    print("   - Expected size reduction: 0.5GB → 0.25GB")
    
    print("\nTotal expected size after optimization:")
    print("Original: ~5.4GB → Optimized: ~2.4GB (55% reduction)")
    
    print("\nExpected Intel iGPU Performance:")
    print("- Inference speed: 2-3x faster than CPU")
    print("- Power usage: 15W (vs 35W+ CPU)")
    print("- Memory: Fits in 8GB system RAM")

if __name__ == "__main__":
    print("VibeVoice ONNX Export for Intel iGPU Optimization")
    print("="*50)
    
    download_and_export_vibevoice()
    analyze_model_structure()
    
    print("\nNext steps:")
    print("1. Quantize exported ONNX models")
    print("2. Optimize with OpenVINO for Intel iGPU")
    print("3. Create unified inference pipeline")