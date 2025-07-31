#!/usr/bin/env python3
"""
LZMA Fallback Module
Provides fallback for missing _lzma module in Python environment
"""

import sys
import logging

logger = logging.getLogger(__name__)

class LZMAFallback:
    """Fallback implementation when _lzma is not available"""
    
    def __init__(self):
        logger.warning("⚠️  Using LZMA fallback - compression/decompression disabled")
    
    def compress(self, data, **kwargs):
        """Fallback compress - returns data uncompressed"""
        logger.warning("⚠️  LZMA compression not available, returning uncompressed data")
        return data
    
    def decompress(self, data, **kwargs):
        """Fallback decompress - returns data as-is"""
        logger.warning("⚠️  LZMA decompression not available, returning data as-is")
        return data

# Try to import real lzma, fall back if not available
try:
    import lzma
    import _lzma
    logger.info("✅ Real LZMA module available")
except ImportError as e:
    logger.warning(f"⚠️  LZMA module not available: {e}")
    
    # Create fallback module
    class FallbackLZMA:
        LZ4F_VERSION = 100
        
        def compress(self, data, format=None, check=None, preset=None, filters=None):
            return data
            
        def decompress(self, data, format=None, memlimit=None, filters=None):
            return data
            
        def open(self, filename, mode="rb", **kwargs):
            # For file operations, just use regular file I/O
            if 'b' in mode:
                return open(filename, mode.replace('x', 'w'))
            else:
                return open(filename, mode.replace('x', 'w'), encoding='utf-8')
    
    # Create fallback _lzma module
    class Fallback_LZMA:
        LZMA_OK = 0
        LZMA_STREAM_END = 1
        
        def LZMACompressor(self, **kwargs):
            class MockCompressor:
                def compress(self, data):
                    return data
                def flush(self):
                    return b''
            return MockCompressor()
            
        def LZMADecompressor(self, **kwargs):
            class MockDecompressor:
                def decompress(self, data):
                    return data
                eof = False
            return MockDecompressor()
    
    # Inject fallback modules
    lzma = FallbackLZMA()
    _lzma = Fallback_LZMA()
    
    # Add to sys.modules to make them importable
    sys.modules['lzma'] = lzma
    sys.modules['_lzma'] = _lzma
    
    logger.info("✅ LZMA fallback modules installed")

def ensure_lzma_available():
    """Ensure lzma modules are available (with fallback if needed)"""
    
    try:
        import lzma
        import _lzma
        return True
    except ImportError:
        # Import this module to trigger fallback installation
        import lzma_fallback
        return False

if __name__ == "__main__":
    # Test the fallback
    ensure_lzma_available()
    
    import lzma
    import _lzma
    
    print("✅ LZMA modules available (possibly fallback)")
    
    # Test basic functionality
    test_data = b"Hello, Magic Unicorn!"
    compressed = lzma.compress(test_data)
    decompressed = lzma.decompress(compressed)
    
    print(f"Original: {test_data}")
    print(f"Compressed: {compressed}")
    print(f"Decompressed: {decompressed}")
    print("✅ LZMA fallback test complete")