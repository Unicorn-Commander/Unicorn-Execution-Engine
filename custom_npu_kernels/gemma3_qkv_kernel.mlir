// Custom NPU Kernel for Gemma 3 27B Q/K/V Projections
// Target: NPU Phoenix 16 TOPS, 16 tiles, 2GB SRAM
// Architecture: 5376 → Q:4096, K/V:2048
// Quantization: INT8 symmetric with BF16 scales

module {
  // NPU Phoenix AIE2 architecture targeting
  aie.device(xcve2802) {
    
    // Memory layout for Gemma 3 27B attention
    // Input: [batch, seq_len, 5376] FP16
    // Q Weight: [5376, 4096] INT8
    // K Weight: [5376, 2048] INT8  
    // V Weight: [5376, 2048] INT8
    // Scales: BF16
    
    // Define memory buffers optimized for NPU Phoenix
    %buf_input = aie.buffer(%tile_0_0) { sym_name = "input_buffer" } : memref<1x64x5376xf16>
    %buf_q_weight = aie.buffer(%tile_0_1) { sym_name = "q_weight_buffer" } : memref<5376x4096xi8>
    %buf_k_weight = aie.buffer(%tile_0_2) { sym_name = "k_weight_buffer" } : memref<5376x2048xi8>
    %buf_v_weight = aie.buffer(%tile_0_3) { sym_name = "v_weight_buffer" } : memref<5376x2048xi8>
    %buf_q_scale = aie.buffer(%tile_1_0) { sym_name = "q_scale_buffer" } : memref<1xbf16>
    %buf_k_scale = aie.buffer(%tile_1_1) { sym_name = "k_scale_buffer" } : memref<1xbf16>
    %buf_v_scale = aie.buffer(%tile_1_2) { sym_name = "v_scale_buffer" } : memref<1xbf16>
    
    // Output buffers
    %buf_q_out = aie.buffer(%tile_2_0) { sym_name = "q_output_buffer" } : memref<1x64x4096xf16>
    %buf_k_out = aie.buffer(%tile_2_1) { sym_name = "k_output_buffer" } : memref<1x64x2048xf16>
    %buf_v_out = aie.buffer(%tile_2_2) { sym_name = "v_output_buffer" } : memref<1x64x2048xf16>
    
    // Define compute tiles for parallel execution
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_1_0 = aie.tile(1, 0)
    %tile_1_1 = aie.tile(1, 1)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_0 = aie.tile(2, 0)
    %tile_2_1 = aie.tile(2, 1)
    %tile_2_2 = aie.tile(2, 2)
    
    // Core compute functions for each projection
    // Q Projection: [1, 64, 5376] @ [5376, 4096] -> [1, 64, 4096]
    %core_q = aie.core(%tile_2_0) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c5376 = arith.constant 5376 : index
      %c4096 = arith.constant 4096 : index
      
      // Load scale factor
      %q_scale = memref.load %buf_q_scale[%c0] : memref<1xbf16>
      %q_scale_f16 = arith.extf %q_scale : bf16 to f16
      
      // Optimized matrix multiplication with INT8 dequantization
      scf.for %i = %c0 to %c1 step %c1 {
        scf.for %j = %c0 to %c64 step %c1 {
          scf.for %k = %c0 to %c4096 step %c1 {
            %sum = arith.constant 0.0 : f16
            %result = scf.for %l = %c0 to %c5376 step %c1 iter_args(%acc = %sum) -> (f16) {
              // Load input value
              %input_val = memref.load %buf_input[%i, %j, %l] : memref<1x64x5376xf16>
              
              // Load quantized weight and dequantize
              %weight_int8 = memref.load %buf_q_weight[%l, %k] : memref<5376x4096xi8>
              %weight_f16 = arith.sitofp %weight_int8 : i8 to f16
              %weight_dequant = arith.mulf %weight_f16, %q_scale_f16 : f16
              
              // Multiply and accumulate
              %prod = arith.mulf %input_val, %weight_dequant : f16
              %new_acc = arith.addf %acc, %prod : f16
              scf.yield %new_acc : f16
            }
            memref.store %result, %buf_q_out[%i, %j, %k] : memref<1x64x4096xf16>
          }
        }
      }
      aie.end
    }
    
    // K Projection: [1, 64, 5376] @ [5376, 2048] -> [1, 64, 2048]
    %core_k = aie.core(%tile_2_1) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c5376 = arith.constant 5376 : index
      %c2048 = arith.constant 2048 : index
      
      // Load scale factor
      %k_scale = memref.load %buf_k_scale[%c0] : memref<1xbf16>
      %k_scale_f16 = arith.extf %k_scale : bf16 to f16
      
      // Matrix multiplication for K projection
      scf.for %i = %c0 to %c1 step %c1 {
        scf.for %j = %c0 to %c64 step %c1 {
          scf.for %k = %c0 to %c2048 step %c1 {
            %sum = arith.constant 0.0 : f16
            %result = scf.for %l = %c0 to %c5376 step %c1 iter_args(%acc = %sum) -> (f16) {
              %input_val = memref.load %buf_input[%i, %j, %l] : memref<1x64x5376xf16>
              %weight_int8 = memref.load %buf_k_weight[%l, %k] : memref<5376x2048xi8>
              %weight_f16 = arith.sitofp %weight_int8 : i8 to f16
              %weight_dequant = arith.mulf %weight_f16, %k_scale_f16 : f16
              %prod = arith.mulf %input_val, %weight_dequant : f16
              %new_acc = arith.addf %acc, %prod : f16
              scf.yield %new_acc : f16
            }
            memref.store %result, %buf_k_out[%i, %j, %k] : memref<1x64x2048xf16>
          }
        }
      }
      aie.end
    }
    
    // V Projection: [1, 64, 5376] @ [5376, 2048] -> [1, 64, 2048]
    %core_v = aie.core(%tile_2_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c5376 = arith.constant 5376 : index
      %c2048 = arith.constant 2048 : index
      
      // Load scale factor
      %v_scale = memref.load %buf_v_scale[%c0] : memref<1xbf16>
      %v_scale_f16 = arith.extf %v_scale : bf16 to f16
      
      // Matrix multiplication for V projection
      scf.for %i = %c0 to %c1 step %c1 {
        scf.for %j = %c0 to %c64 step %c1 {
          scf.for %k = %c0 to %c2048 step %c1 {
            %sum = arith.constant 0.0 : f16
            %result = scf.for %l = %c0 to %c5376 step %c1 iter_args(%acc = %sum) -> (f16) {
              %input_val = memref.load %buf_input[%i, %j, %l] : memref<1x64x5376xf16>
              %weight_int8 = memref.load %buf_v_weight[%l, %k] : memref<5376x2048xi8>
              %weight_f16 = arith.sitofp %weight_int8 : i8 to f16
              %weight_dequant = arith.mulf %weight_f16, %v_scale_f16 : f16
              %prod = arith.mulf %input_val, %weight_dequant : f16
              %new_acc = arith.addf %acc, %prod : f16
              scf.yield %new_acc : f16
            }
            memref.store %result, %buf_v_out[%i, %j, %k] : memref<1x64x2048xf16>
          }
        }
      }
      aie.end
    }
  }
}