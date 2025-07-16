module {
  func.func @test_gemma3_kernel(%arg0: memref<1x64x5376xf32>, %arg1: memref<5376x4096xi8>, %arg2: f32, %arg3: memref<1x64x4096xf32>) {
    // Simple matrix multiplication kernel
    affine.for %i = 0 to 1 {
      affine.for %j = 0 to 64 {
        affine.for %k = 0 to 4096 {
          %sum = affine.for %l = 0 to 5376 iter_args(%iter = arith.constant 0.0 : f32) -> f32 {
            %a = affine.load %arg0[%i, %j, %l] : memref<1x64x5376xf32>
            %b_int = affine.load %arg1[%l, %k] : memref<5376x4096xi8>
            %b = arith.sitofp %b_int : i8 to f32
            %b_scaled = arith.mulf %b, %arg2 : f32
            %prod = arith.mulf %a, %b_scaled : f32
            %acc = arith.addf %iter, %prod : f32
            affine.yield %acc : f32
          }
          affine.store %sum, %arg3[%i, %j, %k] : memref<1x64x4096xf32>
        }
      }
    }
    return
  }
}
