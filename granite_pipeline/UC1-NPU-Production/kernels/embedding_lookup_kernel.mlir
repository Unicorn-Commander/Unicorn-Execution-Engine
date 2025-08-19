// MLIR-AIE Embedding Lookup Kernel for NPU
// Optimized for UC1-EMB format

module @embedding_lookup_npu {
  aie.device(npu) {
    // Define tiles
    %tile_0_0 = aie.tile(0, 0)  // Memory tile for embedding table
    %tile_0_1 = aie.tile(0, 1)  // Compute tile for lookup
    %tile_0_2 = aie.tile(0, 2)  // Compute tile for processing
    
    // Embedding table buffer (50K vocab x 768 dim)
    %embed_table = aie.buffer(%tile_0_0) { sym_name = "embed_table" } : memref<50000x768xf32>
    
    // Input tokens buffer (batch_size x seq_length)
    %input_tokens = aie.buffer(%tile_0_0) { sym_name = "input_tokens" } : memref<4096xi32>
    
    // Output embeddings buffer
    %output_embeds = aie.buffer(%tile_0_0) { sym_name = "output_embeds" } : memref<4096x768xf32>
    
    // Embedding lookup kernel
    %core_0_1 = aie.core(%tile_0_1) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4096 = arith.constant 4096 : index
      %c768 = arith.constant 768 : index
      
      // Process each token
      scf.for %i = %c0 to %c4096 step %c1 {
        // Load token ID
        %token_id = memref.load %input_tokens[%i] : memref<4096xi32>
        %token_idx = arith.index_cast %token_id : i32 to index
        
        // Lookup embedding vector
        scf.for %j = %c0 to %c768 step %c1 {
          %embed_val = memref.load %embed_table[%token_idx, %j] : memref<50000x768xf32>
          memref.store %embed_val, %output_embeds[%i, %j] : memref<4096x768xf32>
        }
      }
      
      aie.end
    }
    
    // Layer normalization kernel
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4096 = arith.constant 4096 : index
      %c768 = arith.constant 768 : index
      %eps = arith.constant 1.0e-5 : f32
      
      // Normalize each embedding
      scf.for %i = %c0 to %c4096 step %c1 {
        // Calculate mean
        %sum_init = arith.constant 0.0 : f32
        %sum = scf.for %j = %c0 to %c768 step %c1 iter_args(%s = %sum_init) -> f32 {
          %val = memref.load %output_embeds[%i, %j] : memref<4096x768xf32>
          %new_sum = arith.addf %s, %val : f32
          scf.yield %new_sum : f32
        }
        
        %f768 = arith.constant 768.0 : f32
        %mean = arith.divf %sum, %f768 : f32
        
        // Calculate variance
        %var_init = arith.constant 0.0 : f32
        %variance = scf.for %j = %c0 to %c768 step %c1 iter_args(%v = %var_init) -> f32 {
          %val = memref.load %output_embeds[%i, %j] : memref<4096x768xf32>
          %diff = arith.subf %val, %mean : f32
          %diff_sq = arith.mulf %diff, %diff : f32
          %new_var = arith.addf %v, %diff_sq : f32
          scf.yield %new_var : f32
        }
        
        %var_norm = arith.divf %variance, %f768 : f32
        %var_eps = arith.addf %var_norm, %eps : f32
        %std_dev = math.sqrt %var_eps : f32
        
        // Normalize
        scf.for %j = %c0 to %c768 step %c1 {
          %val = memref.load %output_embeds[%i, %j] : memref<4096x768xf32>
          %centered = arith.subf %val, %mean : f32
          %normalized = arith.divf %centered, %std_dev : f32
          memref.store %normalized, %output_embeds[%i, %j] : memref<4096x768xf32>
        }
      }
      
      aie.end
    }
    
    // DMA configuration for data movement
    %mem_0_0 = aie.mem(%tile_0_0) {
      %dma0 = aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
        aie.use_lock(%lock_input, "Acquire", 1)
        aie.dma_bd(%input_tokens : memref<4096xi32>, 0, 4096)
        aie.use_lock(%lock_input, "Release", 0)
        aie.next_bd ^bd1
      ^bd1:
        aie.use_lock(%lock_output, "Acquire", 0)
        aie.dma_bd(%output_embeds : memref<4096x768xf32>, 0, 3145728)
        aie.use_lock(%lock_output, "Release", 1)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    
    // Synchronization locks
    %lock_input = aie.lock(%tile_0_0, 0) { sym_name = "lock_input" }
    %lock_output = aie.lock(%tile_0_0, 1) { sym_name = "lock_output" }
    %lock_table = aie.lock(%tile_0_0, 2) { sym_name = "lock_table" }
  }
}