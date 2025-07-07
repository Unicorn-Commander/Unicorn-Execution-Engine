//
// Working NPU Kernel for Gemma3n Integration
//

module {
  aie.device(npu1) {
    
    %tile_0_2 = aie.tile(0, 2)
    
    %buf0 = aie.buffer(%tile_0_2) {sym_name = "input_buffer"} : memref<1024xf32>
    %buf1 = aie.buffer(%tile_0_2) {sym_name = "output_buffer"} : memref<1024xf32>
    
    %lock0 = aie.lock(%tile_0_2, 0) {init = 1 : i32}
    %lock1 = aie.lock(%tile_0_2, 1) {init = 0 : i32}
    
    aie.core(%tile_0_2) {
      aie.use_lock(%lock0, Acquire, 0)
      aie.use_lock(%lock1, Acquire, 1)
      
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1024 = arith.constant 1024 : index
      %scale = arith.constant 0.125 : f32
      
      // Simple attention-like computation
      scf.for %i = %c0 to %c1024 step %c1 {
        %val = memref.load %buf0[%i] : memref<1024xf32>
        %scaled = arith.mulf %val, %scale : f32
        memref.store %scaled, %buf1[%i] : memref<1024xf32>
      }
      
      aie.use_lock(%lock0, Release, 1)
      aie.use_lock(%lock1, Release, 0)
      aie.end
    }
  }
}