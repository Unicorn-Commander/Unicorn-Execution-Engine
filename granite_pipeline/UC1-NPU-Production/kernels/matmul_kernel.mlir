// MLIR-AIE Matrix Multiplication Kernel for AMD NPU
// Optimized for UC1-Embedding-NPU

module @matmul_npu {
  aie.device(npu) {
    // Define tiles for computation
    %tile_0_0 = aie.tile(0, 0)  // Memory tile
    %tile_0_1 = aie.tile(0, 1)  // Compute tile
    %tile_0_2 = aie.tile(0, 2)  // Compute tile
    
    // Define memory buffers
    %buf_a = aie.buffer(%tile_0_0) { sym_name = "buf_a" } : memref<256x256xf32>
    %buf_b = aie.buffer(%tile_0_0) { sym_name = "buf_b" } : memref<256x256xf32>
    %buf_c = aie.buffer(%tile_0_0) { sym_name = "buf_c" } : memref<256x256xf32>
    
    // Matrix multiplication kernel
    %core_0_1 = aie.core(%tile_0_1) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c256 = arith.constant 256 : index
      
      // Initialize output to zero
      scf.for %i = %c0 to %c256 step %c1 {
        scf.for %j = %c0 to %c256 step %c1 {
          %zero = arith.constant 0.0 : f32
          memref.store %zero, %buf_c[%i, %j] : memref<256x256xf32>
        }
      }
      
      // Matrix multiplication: C = A * B
      scf.for %i = %c0 to %c256 step %c1 {
        scf.for %j = %c0 to %c256 step %c1 {
          scf.for %k = %c0 to %c256 step %c1 {
            %a_val = memref.load %buf_a[%i, %k] : memref<256x256xf32>
            %b_val = memref.load %buf_b[%k, %j] : memref<256x256xf32>
            %c_val = memref.load %buf_c[%i, %j] : memref<256x256xf32>
            
            %mul = arith.mulf %a_val, %b_val : f32
            %sum = arith.addf %c_val, %mul : f32
            
            memref.store %sum, %buf_c[%i, %j] : memref<256x256xf32>
          }
        }
      }
      
      aie.end
    }
    
    // Data movement configuration
    %mem_0_0 = aie.mem(%tile_0_0) {
      %dma0 = aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
        aie.use_lock(%lock_a, "Acquire", 1)
        aie.dma_bd(%buf_a : memref<256x256xf32>, 0, 65536)
        aie.use_lock(%lock_a, "Release", 0)
        aie.next_bd ^bd1
      ^bd1:
        aie.use_lock(%lock_b, "Acquire", 1)
        aie.dma_bd(%buf_b : memref<256x256xf32>, 0, 65536)
        aie.use_lock(%lock_b, "Release", 0)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    
    // Lock definitions for synchronization
    %lock_a = aie.lock(%tile_0_0, 0) { sym_name = "lock_a" }
    %lock_b = aie.lock(%tile_0_0, 1) { sym_name = "lock_b" }
    %lock_c = aie.lock(%tile_0_0, 2) { sym_name = "lock_c" }
  }
}