//
// Simple NPU Kernel for Gemma3n Sparse Attention
// Phoenix NPU with 5 columns
//

module {
  aie.device(npu1) {
    
    // Define tiles
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    
    // Define memory buffers
    %buf0 = aie.buffer(%tile_0_2) : memref<1024xf32>
    %buf1 = aie.buffer(%tile_1_2) : memref<1024xf32>
    
    // Define locks
    %lock0 = aie.lock(%tile_0_2, 0) 
    %lock1 = aie.lock(%tile_1_2, 0)
    
    // Core computation
    aie.core(%tile_0_2) {
      aie.use_lock(%lock0, "Acquire", 0)
      
      // Simple attention computation placeholder
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1024 = arith.constant 1024 : index
      
      scf.for %i = %c0 to %c1024 step %c1 {
        %val = memref.load %buf0[%i] : memref<1024xf32>
        %result = arith.mulf %val, %val : f32
        memref.store %result, %buf0[%i] : memref<1024xf32>
      }
      
      aie.use_lock(%lock0, "Release", 1)
      aie.end
    }
    
    aie.core(%tile_1_2) {
      aie.use_lock(%lock1, "Acquire", 0)
      
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1024 = arith.constant 1024 : index
      
      scf.for %i = %c0 to %c1024 step %c1 {
        %val = memref.load %buf1[%i] : memref<1024xf32>
        %doubled = arith.addf %val, %val : f32
        memref.store %doubled, %buf1[%i] : memref<1024xf32>
      }
      
      aie.use_lock(%lock1, "Release", 1)
      aie.end
    }
  }
}