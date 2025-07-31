#!/usr/bin/env python3
"""Test the inference pipeline directly"""

import logging
import traceback
from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed

logging.basicConfig(level=logging.INFO)

def test_inference():
    """Test inference with the fixed pipeline"""
    
    print("🚀 Testing Inference Pipeline")
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
            
            # Generate tokens
            print("\n🔄 Generating tokens...")
            generated_ids = pipeline.generate_tokens(
                input_ids=input_ids,
                max_tokens=10,
                temperature=0.7,
                top_p=0.9
            )
            
            # Convert back to text
            generated_text = "".join([chr(min(max(c, 32), 126)) for c in generated_ids])  # Keep in printable range
            
            print(f"\n✅ Generated Token IDs: {generated_ids}")
            print(f"✅ Generated Text: '{generated_text}'")
            
            pipeline.cleanup()
        else:
            print("\n❌ Failed to initialize pipeline")
            
    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    test_inference()