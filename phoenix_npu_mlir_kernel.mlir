// Phoenix NPU Attention Kernel in MLIR-AIE
// Target: AMD XDNA1 with 5 columns (Phoenix APU)
// Author: Magic Unicorn Project

module @gemma3_attention {
  // Define the AIE device with 5 columns
  aie.device(npu1_5col) {
    // Memory tile configuration (row 0)
    %tile_0_0 = aie.tile(0, 0)
    %tile_1_0 = aie.tile(1, 0)
    %tile_2_0 = aie.tile(2, 0)
    %tile_3_0 = aie.tile(3, 0)
    %tile_4_0 = aie.tile(4, 0)
    
    // Compute tiles (rows 1-3)
    %tile_0_1 = aie.tile(0, 1)
    %tile_1_1 = aie.tile(1, 1)
    %tile_2_1 = aie.tile(2, 1)
    %tile_3_1 = aie.tile(3, 1)
    %tile_4_1 = aie.tile(4, 1)
    
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_2 = aie.tile(3, 2)
    %tile_4_2 = aie.tile(4, 2)
    
    %tile_0_3 = aie.tile(0, 3)
    %tile_1_3 = aie.tile(1, 3)
    %tile_2_3 = aie.tile(2, 3)
    %tile_3_3 = aie.tile(3, 3)
    %tile_4_3 = aie.tile(4, 3)
    
    // Buffer declarations
    %buffer_q = aie.buffer(%tile_0_0) : memref<256x128xi8>
    %buffer_k = aie.buffer(%tile_1_0) : memref<256x128xi8>
    %buffer_v = aie.buffer(%tile_2_0) : memref<256x128xi8>
    %buffer_out = aie.buffer(%tile_3_0) : memref<256x128xi8>
    
    // Locks for synchronization
    %lock_q = aie.lock(%tile_0_0, 0)
    %lock_k = aie.lock(%tile_1_0, 0)
    %lock_v = aie.lock(%tile_2_0, 0)
    %lock_out = aie.lock(%tile_3_0, 0)
    
    // Define attention kernel for Gemma 3 4B
    // 20 attention heads, each tile handles 1 head
    aie.core(%tile_0_1) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c128 = arith.constant 128 : index  // head_dim
      %c256 = arith.constant 256 : index  // seq_len
      
      // Process attention head 0
      aie.use_lock(%lock_q, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_k, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_v, AcquireGreaterEqual, 1)
      
      // Allocate local score buffer
      %scores = memref.alloca() : memref<256x256xi32>
      
      // INT8 attention computation
      scf.for %i = %c0 to %c256 step %c1 {
        scf.for %j = %c0 to %c256 step %c1 {
          // QK^T computation with INT8
          %c0_i32 = arith.constant 0 : i32
          %score = scf.for %k = %c0 to %c128 step %c1 iter_args(%sum = %c0_i32) -> i32 {
            %q_val = memref.load %buffer_q[%i, %k] : memref<256x128xi8>
            %k_val = memref.load %buffer_k[%j, %k] : memref<256x128xi8>
            %q_ext = arith.extsi %q_val : i8 to i32
            %k_ext = arith.extsi %k_val : i8 to i32
            %prod = arith.muli %q_ext, %k_ext : i32
            %new_sum = arith.addi %sum, %prod : i32
            scf.yield %new_sum : i32
          }
          
          // Store score (will be softmaxed)
          memref.store %score, %scores[%i, %j] : memref<256x256xi32>
        }
      }
      
      // Release locks
      aie.use_lock(%lock_out, Release, 1)
      aie.end
    }
    
    // Configure remaining cores for other attention heads
    // Core 1 processes head 1
    aie.core(%tile_1_1) {
      // Similar logic for head 1
      aie.end
    }
    
    // Data movement configuration
    %mem_0_0 = aie.mem(%tile_0_0) {
      %dma_0 = aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
        aie.use_lock(%lock_q, AcquireGreaterEqual, 0)
        aie.dma_bd(%buffer_q : memref<256x128xi8>, 0, 32768)
        aie.use_lock(%lock_q, Release, 1)
        aie.next_bd ^bd0
      ^end:
        aie.end
    }
  }
}