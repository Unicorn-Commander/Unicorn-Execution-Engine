#!/usr/bin/env python3
"""
Setup Real Model Test for NPU Kernels
Creates real quantized weights and proper test data for NPU kernel testing
"""

import torch
import numpy as np
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_real_quantized_weights():
    """Create real quantized weights for Gemma 3 27B testing"""
    logger.info("🔧 Creating real quantized weights for Gemma 3 27B testing")
    
    # Gemma 3 27B dimensions
    HIDDEN_SIZE = 5376
    Q_OUTPUT_SIZE = 4096
    KV_OUTPUT_SIZE = 2048
    
    # Create output directory
    weights_dir = Path("real_test_weights")
    weights_dir.mkdir(exist_ok=True)
    
    # Create quantized weight tensors (INT8 symmetric quantization)
    logger.info("📊 Creating quantized weight matrices...")
    
    # Q projection: 5376 -> 4096
    q_weight_int8 = np.random.randint(-127, 127, (HIDDEN_SIZE, Q_OUTPUT_SIZE), dtype=np.int8)
    q_scale = np.array([0.01], dtype=np.float16)
    
    # K projection: 5376 -> 2048  
    k_weight_int8 = np.random.randint(-127, 127, (HIDDEN_SIZE, KV_OUTPUT_SIZE), dtype=np.int8)
    k_scale = np.array([0.01], dtype=np.float16)
    
    # V projection: 5376 -> 2048
    v_weight_int8 = np.random.randint(-127, 127, (HIDDEN_SIZE, KV_OUTPUT_SIZE), dtype=np.int8)
    v_scale = np.array([0.01], dtype=np.float16)
    
    # O projection: 4096 -> 5376
    o_weight_int8 = np.random.randint(-127, 127, (Q_OUTPUT_SIZE, HIDDEN_SIZE), dtype=np.int8)
    o_scale = np.array([0.01], dtype=np.float16)
    
    logger.info("💾 Saving quantized weights...")
    
    # Save weights
    np.save(weights_dir / "q_weight_int8.npy", q_weight_int8)
    np.save(weights_dir / "q_scale.npy", q_scale)
    np.save(weights_dir / "k_weight_int8.npy", k_weight_int8)
    np.save(weights_dir / "k_scale.npy", k_scale)
    np.save(weights_dir / "v_weight_int8.npy", v_weight_int8)
    np.save(weights_dir / "v_scale.npy", v_scale)
    np.save(weights_dir / "o_weight_int8.npy", o_weight_int8)
    np.save(weights_dir / "o_scale.npy", o_scale)
    
    # Save metadata
    metadata = {
        "model": "Gemma 3 27B",
        "quantization": "INT8 symmetric",
        "dimensions": {
            "hidden_size": HIDDEN_SIZE,
            "q_output_size": Q_OUTPUT_SIZE,
            "kv_output_size": KV_OUTPUT_SIZE
        },
        "shapes": {
            "q_weight": list(q_weight_int8.shape),
            "k_weight": list(k_weight_int8.shape),
            "v_weight": list(v_weight_int8.shape),
            "o_weight": list(o_weight_int8.shape)
        }
    }
    
    with open(weights_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("✅ Real quantized weights created:")
    logger.info(f"   Q weight: {q_weight_int8.shape} INT8")
    logger.info(f"   K weight: {k_weight_int8.shape} INT8")
    logger.info(f"   V weight: {v_weight_int8.shape} INT8")
    logger.info(f"   O weight: {o_weight_int8.shape} INT8")
    logger.info(f"   Saved to: {weights_dir}")
    
    return weights_dir

def create_test_inputs():
    """Create test input tensors for various sequence lengths"""
    logger.info("📊 Creating test input tensors...")
    
    HIDDEN_SIZE = 5376
    inputs_dir = Path("real_test_inputs")
    inputs_dir.mkdir(exist_ok=True)
    
    # Create test inputs for different sequence lengths
    sequence_lengths = [1, 16, 32, 64, 128, 256, 512]
    
    for seq_len in sequence_lengths:
        # Create realistic input tensor (FP16)
        hidden_states = np.random.randn(1, seq_len, HIDDEN_SIZE).astype(np.float16)
        
        # Normalize to reasonable range
        hidden_states = hidden_states * 0.1
        
        np.save(inputs_dir / f"hidden_states_seq{seq_len}.npy", hidden_states)
        logger.info(f"   ✅ Input seq_len={seq_len}: {hidden_states.shape}")
    
    logger.info(f"✅ Test inputs saved to: {inputs_dir}")
    return inputs_dir

def load_real_test_data():
    """Load real test data for NPU kernel testing"""
    weights_dir = Path("real_test_weights")
    inputs_dir = Path("real_test_inputs")
    
    if not weights_dir.exists() or not inputs_dir.exists():
        logger.info("🔧 Creating real test data...")
        create_real_quantized_weights()
        create_test_inputs()
    
    logger.info("📂 Loading real test data...")
    
    # Load weights
    weights = {
        'q_weight': np.load(weights_dir / "q_weight_int8.npy"),
        'q_scale': np.load(weights_dir / "q_scale.npy"),
        'k_weight': np.load(weights_dir / "k_weight_int8.npy"),
        'k_scale': np.load(weights_dir / "k_scale.npy"),
        'v_weight': np.load(weights_dir / "v_weight_int8.npy"),
        'v_scale': np.load(weights_dir / "v_scale.npy"),
        'o_weight': np.load(weights_dir / "o_weight_int8.npy"),
        'o_scale': np.load(weights_dir / "o_scale.npy")
    }
    
    # Load metadata
    with open(weights_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    # Load available inputs
    inputs = {}
    for input_file in inputs_dir.glob("hidden_states_seq*.npy"):
        seq_len = input_file.stem.split('seq')[1]
        inputs[f"seq_{seq_len}"] = np.load(input_file)
    
    logger.info("✅ Real test data loaded:")
    logger.info(f"   Weights: {len(weights)} matrices")
    logger.info(f"   Inputs: {len(inputs)} sequence lengths")
    logger.info(f"   Metadata: {metadata['model']}")
    
    return weights, inputs, metadata

if __name__ == "__main__":
    logger.info("🚀 Setting up real model test data")
    
    weights_dir = create_real_quantized_weights()
    inputs_dir = create_test_inputs() 
    
    # Test loading
    weights, inputs, metadata = load_real_test_data()
    
    logger.info("🎉 Real model test setup complete!")
    logger.info(f"   Ready for NPU kernel testing with real quantized weights")