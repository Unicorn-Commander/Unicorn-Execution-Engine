#!/usr/bin/env python3
"""Test the inference pipeline with detailed debugging"""

import logging
import traceback
import numpy as np
from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed

logging.basicConfig(level=logging.INFO)

def test_inference_debug():
    """Test inference with detailed debugging"""
    
    print("🚀 Testing Inference Pipeline (Debug Mode)")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = PureHardwarePipelineFixed()
    
    try:
        if pipeline.initialize('quantized_models/gemma-3-27b-it-layer-by-layer'):
            print("\n✅ Pipeline initialized!")
            
            # Test with simple input
            prompt = "Hello"
            input_ids = [ord(c) for c in prompt]  # Simple ASCII encoding
            
            print(f"\n📝 Input: '{prompt}'")
            print(f"   Token IDs: {input_ids}")
            
            # Check embedding weights
            embed_weight = pipeline.get_weight_from_gpu('shared_language_model.model.embed_tokens.weight')
            if embed_weight is not None:
                print(f"\n✅ Embedding weight shape: {embed_weight.shape}")
                print(f"   Dtype: {embed_weight.dtype}")
                
                # Get initial hidden states
                try:
                    hidden_states = embed_weight[input_ids]
                    print(f"\n✅ Initial hidden states shape: {hidden_states.shape}")
                    
                    # Test forward pass through one layer
                    layer_idx = 0
                    if layer_idx in pipeline.layer_weights_gpu:
                        print(f"\n🔄 Testing layer {layer_idx} forward pass...")
                        
                        # Add batch dimension
                        if hidden_states.ndim == 2:
                            hidden_states = hidden_states[np.newaxis, :]  # Add batch dim
                        
                        output, kv_cache = pipeline.forward_layer(layer_idx, hidden_states, kv_cache=None)
                        print(f"✅ Layer output shape: {output.shape}")
                        
                    else:
                        print(f"\n⚠️ Layer {layer_idx} not in GPU")
                        
                except Exception as e:
                    print(f"\n❌ Error in forward pass: {e}")
                    traceback.print_exc()
            else:
                print("\n❌ Could not get embedding weights")
            
            pipeline.cleanup()
        else:
            print("\n❌ Failed to initialize pipeline")
            
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    test_inference_debug()