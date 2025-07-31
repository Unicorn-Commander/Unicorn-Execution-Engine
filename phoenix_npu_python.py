#!/usr/bin/env python3.13
"""
Phoenix NPU kernel using MLIR-AIE Python API
"""

import sys
import os
sys.path.append('npu_env/lib/python3.13/site-packages/mlir_aie/python')

import numpy as np
from aie.mlir.ir import *
from aie.mlir.passmanager import *
from aie.dialects.aie import *
from aie.dialects.scf import *
from aie.dialects.arith import *

def create_phoenix_npu_design():
    """Create a simple NPU design for Phoenix"""
    with Context() as ctx, Location.unknown():
        # Register necessary dialects
        ctx.load_all_available_dialects()
        
        # Create module
        module = Module.create()
        
        with InsertionPoint(module.body):
            # Create device for NPU1 (Phoenix)
            @device(AIEDevice.npu1)
            def npu_device():
                # Define tiles
                tile_0_0 = tile(0, 0)  # Shim tile
                tile_0_1 = tile(0, 1)  # Memory tile
                tile_0_2 = tile(0, 2)  # Core tile
                
                # Define a simple core that increments values
                @core(tile_0_2)
                def simple_core():
                    # Just a marker for now
                    pass
        
        return module

if __name__ == "__main__":
    print("🦄 Creating Phoenix NPU design...")
    
    try:
        module = create_phoenix_npu_design()
        print("✅ Design created successfully!")
        
        # Print the MLIR
        print("\n📄 Generated MLIR:")
        print(module)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()