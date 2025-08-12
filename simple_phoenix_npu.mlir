// Simple Phoenix NPU kernel test
// Target: AMD XDNA1 with 5 columns

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    
    %buffer = aie.buffer(%tile_0_1) : memref<1024xi32>
    %lock = aie.lock(%tile_0_1, 0)
    
    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1024 = arith.constant 1024 : index
      
      aie.use_lock(%lock, AcquireGreaterEqual, 1)
      
      scf.for %i = %c0 to %c1024 step %c1 {
        %val = arith.index_cast %i : index to i32
        memref.store %val, %buffer[%i] : memref<1024xi32>
      }
      
      aie.use_lock(%lock, Release, 1)
      aie.end
    }
  }
}