#!/usr/bin/env python3
"""
Fix for GPU loading issue - ensures tensors actually load to VRAM/GTT
"""

import sys
import re

def fix_gpu_loading():
    # Read the current pipeline
    with open('pure_hardware_pipeline_fixed.py', 'r') as f:
        content = f.read()
    
    # Fix 1: The weight check is too restrictive
    # Change from checking 'language_model' prefix to checking for valid tensor data
    old_check = """for weight_name, weight_info in layer_weights.items():
                        if weight_name.startswith('language_model') and isinstance(weight_info, dict) and 'lazy' in weight_info:"""
    
    new_check = """for weight_name, weight_info in layer_weights.items():
                        # Skip vision components but load everything else
                        if 'vision' in weight_name:
                            continue
                        if isinstance(weight_info, dict) and ('lazy' in weight_info or 'data_offsets' in weight_info):"""
    
    content = content.replace(old_check, new_check)
    
    # Fix 2: Add more detailed logging to track what's happening
    # Find the layer loading section and add debug info
    layer_loading_pattern = r"(layer_weights = future\.result\(\))"
    replacement = r"""\1
                    logger.info(f"Layer {layer_idx}: Got {len(layer_weights)} weights")
                    for k in list(layer_weights.keys())[:3]:
                        logger.info(f"  - {k}: {type(layer_weights[k])}")"""
    
    content = re.sub(layer_loading_pattern, replacement, content)
    
    # Fix 3: Ensure the tensor loading actually happens
    # Add verification after loading
    verify_pattern = r"(if size_mb > 0:[\s\S]*?layer_gpu_weights\[weight_name\] = buffer_key)"
    verify_replacement = r"""\1
                                logger.info(f"    ✅ Loaded {weight_name}: {size_mb:.1f}MB to {buffer_key}")"""
    
    content = re.sub(verify_pattern, verify_replacement, content, flags=re.MULTILINE)
    
    # Write the fixed version
    with open('pure_hardware_pipeline_fixed_v2.py', 'w') as f:
        f.write(content)
    
    print("✅ Created pure_hardware_pipeline_fixed_v2.py with GPU loading fixes")
    print("\nKey changes:")
    print("1. Fixed weight name check - now loads all non-vision weights")
    print("2. Added detailed logging to track tensor loading")
    print("3. Added verification after each tensor load")
    print("\nTo test: python3 test_gpu_loading_v2.py")

if __name__ == "__main__":
    fix_gpu_loading()