module {
  aie.device(npu1) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %buffer_0_2 = aie.buffer(%tile_0_2) : memref<1024xf32> 
    %buffer_1_2 = aie.buffer(%tile_1_2) : memref<1024xf32> 
    %lock_0_2 = aie.lock(%tile_0_2, 0)
    %lock_1_2 = aie.lock(%tile_1_2, 0)
    %core_0_2 = aie.core(%tile_0_2) {
      aie.use_lock(%lock_0_2, Acquire, 0)
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1024 = arith.constant 1024 : index
      scf.for %arg0 = %c0 to %c1024 step %c1 {
        %0 = memref.load %buffer_0_2[%arg0] : memref<1024xf32>
        %1 = arith.mulf %0, %0 : f32
        memref.store %1, %buffer_0_2[%arg0] : memref<1024xf32>
      }
      aie.use_lock(%lock_0_2, Release, 1)
      aie.end
    }
    %core_1_2 = aie.core(%tile_1_2) {
      aie.use_lock(%lock_1_2, Acquire, 0)
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1024 = arith.constant 1024 : index
      scf.for %arg0 = %c0 to %c1024 step %c1 {
        %0 = memref.load %buffer_1_2[%arg0] : memref<1024xf32>
        %1 = arith.addf %0, %0 : f32
        memref.store %1, %buffer_1_2[%arg0] : memref<1024xf32>
      }
      aie.use_lock(%lock_1_2, Release, 1)
      aie.end
    }
  }
}

