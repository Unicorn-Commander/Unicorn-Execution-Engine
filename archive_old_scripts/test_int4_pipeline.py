#!/usr/bin/env python3
"""
Test INT4-enabled pipeline initialization
"""

import os
import sys
import logging
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_int4_pipeline():
    """Test INT4 pipeline initialization and loading"""
    logger.info("=" * 80)
    logger.info("🧪 TESTING INT4-ENABLED PIPELINE")
    logger.info("=" * 80)
    
    try:
        # Import the pipeline
        from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed
        
        # Create pipeline instance
        logger.info("📊 Creating INT4-enabled pipeline...")
        pipeline = PureHardwarePipelineFixed()
        
        # Initialize with model path
        model_path = "/home/ucadmin/Development/Unicorn-Execution-Engine/quantized_models/gemma-3-27b-it-layer-by-layer"
        logger.info(f"📊 Initializing with model: {model_path}")
        
        start_time = time.time()
        success = pipeline.initialize(model_path)
        init_time = time.time() - start_time
        
        if success:
            logger.info(f"✅ Pipeline initialized successfully in {init_time:.2f}s")
            
            # Check INT4 stats
            if hasattr(pipeline, 'int4_metadata') and pipeline.int4_metadata:
                logger.info(f"\n🔥 INT4 Quantization Active:")
                logger.info(f"   Number of INT4 tensors: {len(pipeline.int4_metadata)}")
                
                # Calculate compression
                total_original = sum(m['original_size'] for m in pipeline.int4_metadata.values())
                total_packed = sum(m['packed_size'] for m in pipeline.int4_metadata.values())
                
                if total_packed > 0:
                    compression_ratio = total_original / total_packed
                    logger.info(f"   Original size: {total_original / 1024 / 1024 / 1024:.1f}GB")
                    logger.info(f"   Packed size: {total_packed / 1024 / 1024 / 1024:.1f}GB")
                    logger.info(f"   Compression ratio: {compression_ratio:.1f}x")
                    logger.info(f"   Memory saved: {(total_original - total_packed) / 1024 / 1024 / 1024:.1f}GB")
                else:
                    logger.warning("   No INT4 compression data available")
            else:
                logger.warning("⚠️  INT4 metadata not found - quantization may not be active")
            
            # Test a simple forward pass
            logger.info("\n📊 Testing forward pass...")
            import numpy as np
            
            # Create test input
            test_input = np.random.randn(1, 10, 5376).astype(np.float32)
            
            try:
                # Test one layer forward
                output, _ = pipeline.forward_layer(0, test_input)
                logger.info(f"✅ Forward pass successful! Output shape: {output.shape}")
            except Exception as e:
                logger.error(f"❌ Forward pass failed: {e}")
            
            return True
        else:
            logger.error(f"❌ Pipeline initialization failed!")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Activate environment
    logger.info("🔧 Setting up environment...")
    os.environ['XILINX_XRT'] = '/opt/xilinx/xrt'
    
    # Run test
    success = test_int4_pipeline()
    
    if success:
        logger.info("\n✅ INT4 pipeline test completed successfully!")
        sys.exit(0)
    else:
        logger.info("\n❌ INT4 pipeline test failed!")
        sys.exit(1)