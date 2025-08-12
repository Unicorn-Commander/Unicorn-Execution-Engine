#!/usr/bin/env python3
"""
Basic INT4 integration test
"""

import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_int4_packing():
    """Test INT4 packing functionality"""
    logger.info("=" * 80)
    logger.info("🧪 TESTING INT4 PACKING")
    logger.info("=" * 80)
    
    try:
        from integrate_int4_quantization import INT4Integration
        
        # Create test tensor
        test_tensor = np.random.randn(1024, 1024).astype(np.float32)
        logger.info(f"Test tensor shape: {test_tensor.shape}")
        logger.info(f"Original size: {test_tensor.nbytes / 1024 / 1024:.2f} MB")
        
        # Pack to INT4
        packed_data, scale, zero_point = INT4Integration.pack_int4_weights(test_tensor)
        
        logger.info(f"Packed size: {packed_data.nbytes / 1024 / 1024:.2f} MB")
        logger.info(f"Compression ratio: {test_tensor.nbytes / packed_data.nbytes:.1f}x")
        logger.info(f"Scale: {scale:.6f}, Zero point: {zero_point}")
        
        # Verify packing worked
        expected_packed_size = (test_tensor.size + 1) // 2  # 2 values per byte
        actual_packed_size = packed_data.size
        
        logger.info(f"Expected packed size: {expected_packed_size} bytes")
        logger.info(f"Actual packed size: {actual_packed_size} bytes")
        
        if actual_packed_size == expected_packed_size:
            logger.info("✅ INT4 packing test PASSED!")
            return True
        else:
            logger.error("❌ INT4 packing test FAILED - size mismatch")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_int4_packing()
    exit(0 if success else 1)