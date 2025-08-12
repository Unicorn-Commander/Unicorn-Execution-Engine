#!/usr/bin/env python3
"""
Debug pipeline timeout issue
Quick diagnostic test to identify bottleneck
"""

import torch
import time
import logging
import sys
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('debug_timeout.log')
    ]
)
logger = logging.getLogger(__name__)

def test_initialization():
    """Test just the initialization phase"""
    logger.info("🔍 DIAGNOSTIC: Testing initialization only")
    
    try:
        from complete_npu_igpu_inference_pipeline import CompleteNPUIGPUInferencePipeline
        
        logger.info("✅ Import successful")
        
        # Test initialization
        start = time.time()
        pipeline = CompleteNPUIGPUInferencePipeline(use_fp16=True)
        init_time = time.time() - start
        logger.info(f"✅ Pipeline object created in {init_time:.2f}s")
        
        # Test hardware initialization
        start = time.time()
        success = pipeline.initialize_hardware()
        hw_init_time = time.time() - start
        logger.info(f"✅ Hardware initialization: {success} in {hw_init_time:.2f}s")
        
        return pipeline if success else None
        
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        logger.error(traceback.format_exc())
        return None

def test_single_layer_load(pipeline):
    """Test loading just one layer"""
    logger.info("🔍 DIAGNOSTIC: Testing single layer loading")
    
    try:
        # Test layer 0 loading
        start = time.time()
        layer_weights = pipeline.layer_loader(0)
        load_time = time.time() - start
        logger.info(f"✅ Layer 0 loaded in {load_time:.2f}s with {len(layer_weights)} tensors")
        
        # Test layer 1 loading
        start = time.time()
        layer_weights = pipeline.layer_loader(1)
        load_time = time.time() - start
        logger.info(f"✅ Layer 1 loaded in {load_time:.2f}s with {len(layer_weights)} tensors")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Layer loading failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_embedding_lookup(pipeline):
    """Test embedding lookup"""
    logger.info("🔍 DIAGNOSTIC: Testing embedding lookup")
    
    try:
        # Get embedding weights
        embed_weight_key = 'language_model.model.embed_tokens.weight'
        if embed_weight_key not in pipeline.shared_weights:
            logger.error(f"❌ Embedding key not found. Available: {list(pipeline.shared_weights.keys())}")
            return False
        
        embed_weight_info = pipeline.shared_weights[embed_weight_key]
        embed_weight = pipeline._ensure_float_tensor(embed_weight_info)
        
        # Test embedding lookup
        input_ids = torch.tensor([[1, 450, 3437]], dtype=torch.long)
        start = time.time()
        hidden_states = torch.nn.functional.embedding(input_ids, embed_weight)
        embed_time = time.time() - start
        logger.info(f"✅ Embedding lookup completed in {embed_time:.2f}s, shape: {hidden_states.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Embedding lookup failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_single_layer_compute(pipeline):
    """Test computing just one transformer layer"""
    logger.info("🔍 DIAGNOSTIC: Testing single layer computation")
    
    try:
        # Create sample input
        input_ids = torch.tensor([[1, 450, 3437]], dtype=torch.long)
        
        # Get embedding
        embed_weight_key = 'language_model.model.embed_tokens.weight'
        embed_weight_info = pipeline.shared_weights[embed_weight_key]
        embed_weight = pipeline._ensure_float_tensor(embed_weight_info)
        hidden_states = torch.nn.functional.embedding(input_ids, embed_weight)
        
        logger.info(f"Input hidden_states shape: {hidden_states.shape}")
        
        # Load layer 0 weights
        start = time.time()
        layer_weights = pipeline.layer_loader(0)
        load_time = time.time() - start
        logger.info(f"Layer 0 loaded in {load_time:.2f}s")
        
        # Compute layer 0
        start = time.time()
        output = pipeline.compute_transformer_layer(hidden_states, layer_weights)
        compute_time = time.time() - start
        logger.info(f"✅ Layer 0 computed in {compute_time:.2f}s, output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Layer computation failed: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Run diagnostic tests"""
    logger.info("🚀 Starting pipeline timeout diagnostic")
    
    # Test 1: Initialization
    pipeline = test_initialization()
    if not pipeline:
        logger.error("❌ Initialization failed, stopping diagnostics")
        return
    
    # Test 2: Layer loading
    if not test_single_layer_load(pipeline):
        logger.error("❌ Layer loading failed, stopping diagnostics")
        return
    
    # Test 3: Embedding lookup
    if not test_embedding_lookup(pipeline):
        logger.error("❌ Embedding lookup failed, stopping diagnostics")
        return
    
    # Test 4: Single layer computation
    if not test_single_layer_compute(pipeline):
        logger.error("❌ Layer computation failed, stopping diagnostics")
        return
    
    logger.info("🎉 All diagnostic tests passed!")
    logger.info("The issue is likely in the full generation loop (28 layers × N tokens)")

if __name__ == "__main__":
    main()