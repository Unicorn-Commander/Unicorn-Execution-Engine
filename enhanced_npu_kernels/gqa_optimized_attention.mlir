
// Grouped Query Attention - Optimized for Gemma's 32Q/16KV configuration
// Target: Maximum efficiency for asymmetric head counts
// Features: Head broadcasting, memory coalescing, vectorization

module {
  func.func @gqa_optimized_attention(
    %query: memref<32x256x168xf16>,      // 32 query heads
    %key: memref<16x256x168xf16>,        // 16 key heads  
    %value: memref<16x256x168xf16>,      // 16 value heads
    %output: memref<32x256x168xf16>
  ) {
    
    // GQA: Each KV head serves 2 Q heads (32/16 = 2)
    %heads_per_kv = arith.constant 2 : index
    
    // Process KV heads in groups of 8 (for 16-way Q processing)
    scf.for %kv_group = (0) to (16) step (8) {
      
      // Load 8 KV heads into NPU SRAM
      %k_group = memref.subview %key[%kv_group, 0, 0] [8, 256, 168] [1, 1, 1]
      %v_group = memref.subview %value[%kv_group, 0, 0] [8, 256, 168] [1, 1, 1]
      
      // Process corresponding 16 Q heads (8 KV * 2 Q/KV = 16 Q)
      %q_start = arith.muli %kv_group, %heads_per_kv : index
      %q_group = memref.subview %query[%q_start, 0, 0] [16, 256, 168] [1, 1, 1]
      
      // Optimized GQA computation with broadcasting
      call @gqa_compute_16way(%q_group, %k_group, %v_group, %output, %q_start)
        : (memref<16x256x168xf16>, memref<8x256x168xf16>, memref<8x256x168xf16>, 
           memref<32x256x168xf16>, index) -> ()
    }
    
    return
  }
  
  func.func @gqa_compute_16way(
    %queries: memref<16x256x168xf16>,    // 16 Q heads
    %keys: memref<8x256x168xf16>,        // 8 K heads
    %values: memref<8x256x168xf16>,      // 8 V heads  
    %output: memref<32x256x168xf16>,
    %q_offset: index
  ) -> () {
    
    // Process each K/V head with its 2 corresponding Q heads
    scf.for %kv_idx = (0) to (8) step (1) {
      
      // Get the 2 Q heads for this KV head
      %q1_idx = arith.muli %kv_idx, arith.constant 2 : index
      %q2_idx = arith.addi %q1_idx, arith.constant 1 : index
      
      // Vectorized attention for both Q heads simultaneously
      scf.parallel (%seq_i) = (0) to (256) step (1) {
        
        // Load Q vectors for both heads
        %q1_vec = memref.subview %queries[%q1_idx, %seq_i, 0] [1, 1, 168] [1, 1, 1]
        %q2_vec = memref.subview %queries[%q2_idx, %seq_i, 0] [1, 1, 168] [1, 1, 1]
        
        // Compute attention scores for both Q heads with shared K
        %scores1 = memref.alloc() : memref<1x256xf16>
        %scores2 = memref.alloc() : memref<1x256xf16>
        
        scf.for %seq_j = (0) to (256) step (1) {
          
          // Shared K vector for both computations
          %k_vec = memref.subview %keys[%kv_idx, %seq_j, 0] [1, 1, 168] [1, 1, 1]
          
          // Vectorized dot products
          %score1 = call @vectorized_dot_product(%q1_vec, %k_vec) 
            : (memref<1x1x168xf16>, memref<1x1x168xf16>) -> f16
          %score2 = call @vectorized_dot_product(%q2_vec, %k_vec)
            : (memref<1x1x168xf16>, memref<1x1x168xf16>) -> f16
          
          memref.store %score1, %scores1[0, %seq_j] : memref<1x256xf16>
          memref.store %score2, %scores2[0, %seq_j] : memref<1x256xf16>
        }
        
        // Softmax for both attention distributions
        %probs1 = call @vectorized_softmax(%scores1) : (memref<1x256xf16>) -> memref<1x256xf16>
        %probs2 = call @vectorized_softmax(%scores2) : (memref<1x256xf16>) -> memref<1x256xf16>
        
        // Compute outputs using shared V
        %out1 = call @vectorized_attention_output(%probs1, %values, %kv_idx)
          : (memref<1x256xf16>, memref<8x256x168xf16>, index) -> memref<1x168xf16>
        %out2 = call @vectorized_attention_output(%probs2, %values, %kv_idx)
          : (memref<1x256xf16>, memref<8x256x168xf16>, index) -> memref<1x168xf16>
        
        // Store results
        %global_q1 = arith.addi %q_offset, %q1_idx : index
        %global_q2 = arith.addi %q_offset, %q2_idx : index
        
        call @store_attention_result(%out1, %output, %global_q1, %seq_i)
          : (memref<1x168xf16>, memref<32x256x168xf16>, index, index) -> ()
        call @store_attention_result(%out2, %output, %global_q2, %seq_i)  
          : (memref<1x168xf16>, memref<32x256x168xf16>, index, index) -> ()
      }
    }
    
    return
  }
}
