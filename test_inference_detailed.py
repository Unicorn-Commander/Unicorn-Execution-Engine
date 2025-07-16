#!/usr/bin/env python3
"""Test the inference pipeline with very detailed debugging"""

import logging
import traceback
import numpy as np
from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed

# Set logging to DEBUG level
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Suppress some verbose loggers
logging.getLogger('real_vulkan_matrix_compute').setLevel(logging.INFO)

def test_inference_detailed():
    """Test inference with very detailed debugging"""
    
    print("🚀 Testing Inference Pipeline (Detailed Debug)")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = PureHardwarePipelineFixed()
    
    try:
        if pipeline.initialize('quantized_models/gemma-3-27b-it-layer-by-layer'):
            print("\n✅ Pipeline initialized!")
            
            # Set the logger to debug
            pipeline_logger = logging.getLogger('pure_hardware_pipeline_fixed')
            pipeline_logger.setLevel(logging.DEBUG)
            
            # Test with simple input
            prompt = "Hello"
            input_ids = [ord(c) for c in prompt]  # Simple ASCII encoding
            
            print(f"\n📝 Input: '{prompt}'")
            print(f"   Token IDs: {input_ids}")
            
            # Manually step through the first part of generate_tokens
            embed_weight = pipeline.get_weight_from_gpu('shared_language_model.model.embed_tokens.weight')
            if embed_weight is not None:
                print(f"\n✅ Embedding weight shape: {embed_weight.shape}")
                
                # Get initial hidden states
                hidden_states = embed_weight[input_ids]
                print(f"✅ Hidden states shape after embedding lookup: {hidden_states.shape}")
                
                # Add batch dimension
                if hidden_states.ndim == 2:
                    hidden_states = hidden_states[np.newaxis, :]
                    print(f"✅ Hidden states shape after adding batch dim: {hidden_states.shape}")
                
                # Test forward through first layer
                layer_idx = 0
                if layer_idx in pipeline.layer_weights_gpu:
                    print(f"\n🔄 Testing layer {layer_idx} forward pass...")
                    print(f"   Input shape: {hidden_states.shape}")
                    
                    try:
                        output, kv_cache = pipeline.forward_layer(layer_idx, hidden_states, kv_cache=None)
                        print(f"✅ Layer output shape: {output.shape}")
                        
                        # Try the full generation
                        print("\n🔄 Testing full generation...")
                        generated_ids = pipeline.generate_tokens(input_ids, max_tokens=5)
                        print(f"✅ Generated token IDs: {generated_ids}")
                        
                    except Exception as e:
                        print(f"\n❌ Error in forward pass: {e}")
                        traceback.print_exc()
                        
                        # Add more specific debugging
                        print("\n🔍 Checking layer weights...")
                        layer_weights = pipeline.layer_weights_gpu[layer_idx]
                        for key in list(layer_weights.keys())[:5]:
                            print(f"   {key}: {layer_weights[key]}")
                else:
                    print(f"\n⚠️ Layer {layer_idx} not in GPU")
            else:
                print("\n❌ Could not get embedding weights")
            
            pipeline.cleanup()
        else:
            print("\n❌ Failed to initialize pipeline")
            
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    test_inference_detailed()