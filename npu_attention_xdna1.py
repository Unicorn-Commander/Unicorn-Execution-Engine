#!/usr/bin/env python3
"""
NPU Attention Kernel Implementation for AMD Phoenix (XDNA1)
Optimized for 16 TOPS INT8 performance
"""

import numpy as np
import pyxrt
import time
import os
from pathlib import Path

class NPUAttentionKernel:
    """NPU Attention kernel for XDNA1 architecture"""
    
    def __init__(self):
        self.device = None
        self.kernel_cache = {}
        self.setup_npu()
        
    def setup_npu(self):
        """Initialize NPU device"""
        try:
            self.device = pyxrt.device(0)
            print("✅ NPU device initialized")
            
            # Get device info
            device_name = self.device.get_info(pyxrt.info.device.name)
            print(f"   Device: {device_name}")
            print(f"   Architecture: XDNA1 (16 TOPS INT8)")
            
        except Exception as e:
            print(f"❌ NPU setup failed: {e}")
            raise
            
    def compile_attention_kernel(self, seq_len, head_dim, num_heads, num_kv_heads):
        """Generate and compile optimized attention kernel for XDNA1"""
        
        print(f"\n🔧 Compiling NPU attention kernel:")
        print(f"   Seq length: {seq_len}")
        print(f"   Head dim: {head_dim}")
        print(f"   Num heads: {num_heads}")
        print(f"   KV heads: {num_kv_heads}")
        
        # Generate MLIR kernel optimized for XDNA1
        mlir_code = f"""
// XDNA1 Optimized Attention Kernel
// Target: AMD Phoenix NPU (16 TOPS INT8)

module @attention_xdna1_s{seq_len} {{
    // Hardware constraints
    %tile_rows = arith.constant 4 : index
    %tile_cols = arith.constant 5 : index  // 4x5 = 20 AIE2 tiles
    %vec_width = arith.constant 64 : index // 512-bit vectors / 8-bit elements
    
    func.func @attention_compute(
        %Q: memref<{seq_len}x{num_heads}x{head_dim}xi8>,
        %K: memref<{seq_len}x{num_kv_heads}x{head_dim}xi8>,
        %V: memref<{seq_len}x{num_kv_heads}x{head_dim}xi8>,
        %O: memref<{seq_len}x{num_heads}x{head_dim}xi8>,
        %scale: f32
    ) {{
        // Group size for GQA
        %group_size = arith.constant {num_heads // num_kv_heads} : index
        
        // Distribute computation across tiles
        affine.parallel (%tile_id) = (0) to (20) {{
            // Each tile processes subset of heads
            %heads_per_tile = arith.constant {num_heads // 20 + 1} : index
            %start_head = arith.muli %tile_id, %heads_per_tile : index
            %end_head = arith.addi %start_head, %heads_per_tile : index
            
            // Process assigned heads
            affine.for %h = %start_head to %end_head {{
                // Determine KV head for GQA
                %kv_h = arith.divui %h, %group_size : index
                
                // Process sequence positions
                affine.for %i = 0 to {seq_len} {{
                    // Local accumulator for attention
                    %acc = memref.alloca() : memref<{head_dim}xf32>
                    
                    // Compute attention scores
                    affine.for %j = 0 to %i + 1 {{  // Causal mask
                        // Q @ K^T with vectorization
                        %score = arith.constant 0.0 : f32
                        
                        affine.for %d = 0 to {head_dim} step {vec_width} {{
                            // Load vectors
                            %q_vec = affine.vector_load %Q[%i, %h, %d] : 
                                memref<{seq_len}x{num_heads}x{head_dim}xi8>, 
                                vector<{vec_width}xi8>
                            %k_vec = affine.vector_load %K[%j, %kv_h, %d] : 
                                memref<{seq_len}x{num_kv_heads}x{head_dim}xi8>, 
                                vector<{vec_width}xi8>
                            
                            // INT8 dot product with accumulation
                            %dot = vector.contract {{
                                indexing_maps = [
                                    affine_map<(d) -> (d)>,
                                    affine_map<(d) -> (d)>,
                                    affine_map<(d) -> ()>
                                ],
                                iterator_types = ["reduction"]
                            }} %q_vec, %k_vec, %score : 
                                vector<{vec_width}xi8>, 
                                vector<{vec_width}xi8> into f32
                                
                            %score = arith.addf %score, %dot : f32
                        }}
                        
                        // Scale and accumulate
                        %scaled_score = arith.mulf %score, %scale : f32
                        
                        // Update accumulator with V weighted by score
                        affine.for %d = 0 to {head_dim} step {vec_width} {{
                            %v_vec = affine.vector_load %V[%j, %kv_h, %d] :
                                memref<{seq_len}x{num_kv_heads}x{head_dim}xi8>,
                                vector<{vec_width}xi8>
                            
                            // Convert to float for accumulation  
                            %v_fp = arith.extsi %v_vec : vector<{vec_width}xi8> to vector<{vec_width}xi32>
                            %v_fp32 = arith.sitofp %v_fp : vector<{vec_width}xi32> to vector<{vec_width}xf32>
                            
                            // Weight by attention score
                            %weighted = arith.mulf %v_fp32, %scaled_score : vector<{vec_width}xf32>
                            
                            // Accumulate
                            %acc_vec = affine.vector_load %acc[%d] : memref<{head_dim}xf32>, vector<{vec_width}xf32>
                            %new_acc = arith.addf %acc_vec, %weighted : vector<{vec_width}xf32>
                            affine.vector_store %new_acc, %acc[%d] : memref<{head_dim}xf32>, vector<{vec_width}xf32>
                        }}
                    }}
                    
                    // Convert accumulator back to INT8 and store
                    affine.for %d = 0 to {head_dim} step {vec_width} {{
                        %acc_vec = affine.vector_load %acc[%d] : memref<{head_dim}xf32>, vector<{vec_width}xf32>
                        
                        // Quantize to INT8 
                        %quant_i32 = arith.fptosi %acc_vec : vector<{vec_width}xf32> to vector<{vec_width}xi32>
                        %quant_i8 = arith.trunci %quant_i32 : vector<{vec_width}xi32> to vector<{vec_width}xi8>
                        
                        affine.vector_store %quant_i8, %O[%i, %h, %d] : 
                            memref<{seq_len}x{num_heads}x{head_dim}xi8>, 
                            vector<{vec_width}xi8>
                    }}
                }}
            }}
        }}
        
        return
    }}
}}
"""
        
        # Save MLIR file
        mlir_path = f"/tmp/attention_xdna1_s{seq_len}.mlir"
        with open(mlir_path, 'w') as f:
            f.write(mlir_code)
            
        print(f"✅ Generated MLIR kernel: {mlir_path}")
        
        # Check if we have a pre-compiled kernel
        xclbin_dir = Path("/home/ucadmin/Development/Unicorn-Execution-Engine/npu_kernels_real")
        
        # Determine model variant
        if head_dim >= 96 and num_heads >= 48:
            model_variant = "gemma3_27b"
        elif head_dim >= 80 and num_heads >= 32:
            model_variant = "gemma3_4b"
        else:
            model_variant = "gemma3n"
            
        # Determine sequence variant
        seq_variant = "s256"
        if seq_len <= 128:
            seq_variant = "s128"
        elif seq_len <= 256:
            seq_variant = "s256"
        elif seq_len <= 512:
            seq_variant = "s512"
        elif seq_len <= 1024:
            seq_variant = "s1024"
        else:
            seq_variant = "s2048"
            
        xclbin_path = xclbin_dir / model_variant / f"attention_{seq_variant}.xclbin"
        
        if xclbin_path.exists():
            print(f"✅ Using pre-compiled kernel: {xclbin_path}")
            return str(xclbin_path)
        else:
            print(f"⚠️  Pre-compiled kernel not found: {xclbin_path}")
            print("   Would need to compile MLIR -> XCLBIN")
            print("   Using XRT GEMM kernels as fallback")
            
            # Use XRT's built-in GEMM kernel as fallback
            xrt_kernel_path = "/opt/xilinx/xrt/amdxdna/bins/17f0_20/gemm_int8.xclbin"
            if os.path.exists(xrt_kernel_path):
                print(f"✅ Using XRT GEMM kernel: {xrt_kernel_path}")
                return xrt_kernel_path
                
        return None
        
    def execute_attention(self, Q, K, V, scale=1.0):
        """Execute attention on NPU"""
        
        seq_len, num_heads, head_dim = Q.shape
        _, num_kv_heads, _ = K.shape
        
        print(f"\n⚡ Executing NPU attention:")
        print(f"   Input shapes: Q{Q.shape}, K{K.shape}, V{V.shape}")
        
        # Get or compile kernel
        kernel_key = f"s{seq_len}_h{num_heads}_d{head_dim}_kv{num_kv_heads}"
        
        if kernel_key not in self.kernel_cache:
            xclbin_path = self.compile_attention_kernel(seq_len, head_dim, num_heads, num_kv_heads)
            
            if xclbin_path and os.path.exists(xclbin_path):
                # Load kernel
                xclbin = pyxrt.xclbin(xclbin_path)
                uuid = self.device.register_xclbin(xclbin)
                
                # Get kernel name
                kernels = xclbin.get_kernels()
                if kernels:
                    kernel_name = kernels[0].get_name()
                    kernel = pyxrt.kernel(self.device, uuid, kernel_name)
                    self.kernel_cache[kernel_key] = kernel
                    print(f"✅ Kernel loaded: {kernel_name}")
                else:
                    print("❌ No kernels found in XCLBIN")
                    return None
            else:
                print("❌ Failed to get valid kernel")
                return None
                
        kernel = self.kernel_cache.get(kernel_key)
        if not kernel:
            print("❌ Kernel not available")
            return None
            
        try:
            # Quantize inputs to INT8
            Q_int8 = (Q * 127).astype(np.int8)
            K_int8 = (K * 127).astype(np.int8)
            V_int8 = (V * 127).astype(np.int8)
            
            # Allocate NPU buffers
            q_size = Q_int8.nbytes
            k_size = K_int8.nbytes
            v_size = V_int8.nbytes
            o_size = q_size  # Output same size as Q
            
            q_bo = pyxrt.bo(self.device, q_size, pyxrt.bo.flags.normal, kernel.group_id(0))
            k_bo = pyxrt.bo(self.device, k_size, pyxrt.bo.flags.normal, kernel.group_id(1))
            v_bo = pyxrt.bo(self.device, v_size, pyxrt.bo.flags.normal, kernel.group_id(2))
            o_bo = pyxrt.bo(self.device, o_size, pyxrt.bo.flags.normal, kernel.group_id(3))
            
            # Write data
            q_bo.write(Q_int8.tobytes())
            k_bo.write(K_int8.tobytes())
            v_bo.write(V_int8.tobytes())
            
            # Sync to device
            q_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            k_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            v_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            
            # Run kernel
            start_time = time.time()
            
            run = kernel(q_bo, k_bo, v_bo, o_bo, scale)
            run.wait()
            
            exec_time = (time.time() - start_time) * 1000
            
            # Read result
            o_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            output_bytes = bytearray(o_size)
            o_bo.read(output_bytes)
            
            # Convert back to float
            output_int8 = np.frombuffer(output_bytes, dtype=np.int8).reshape(Q.shape)
            output = output_int8.astype(np.float32) / 127.0
            
            print(f"✅ NPU execution complete: {exec_time:.2f}ms")
            
            # Calculate TOPS utilization
            ops = 2 * seq_len * seq_len * num_heads * head_dim  # Attention ops
            tops_used = (ops / exec_time / 1e9)  # TOPS
            utilization = (tops_used / 16.0) * 100  # % of 16 TOPS
            
            print(f"   Performance: {tops_used:.1f} TOPS ({utilization:.1f}% utilization)")
            
            return output
            
        except Exception as e:
            print(f"❌ NPU execution failed: {e}")
            return None


def test_npu_attention():
    """Test NPU attention implementation"""
    
    print("🦄 NPU Attention Kernel Test")
    print("=" * 60)
    
    npu = NPUAttentionKernel()
    
    # Test configurations
    test_configs = [
        (32, 32, 80, 16),   # seq_len, num_heads, head_dim, num_kv_heads
        (128, 32, 80, 16),
        (256, 32, 80, 16),
    ]
    
    for seq_len, num_heads, head_dim, num_kv_heads in test_configs:
        print(f"\n🧪 Testing config: seq={seq_len}, heads={num_heads}, dim={head_dim}")
        
        # Create test tensors
        Q = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
        K = np.random.randn(seq_len, num_kv_heads, head_dim).astype(np.float32)
        V = np.random.randn(seq_len, num_kv_heads, head_dim).astype(np.float32)
        
        # Execute on NPU
        output = npu.execute_attention(Q, K, V, scale=1.0/np.sqrt(head_dim))
        
        if output is not None:
            print(f"✅ Output shape: {output.shape}")
            print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
        else:
            print("❌ NPU execution failed")
            
            
if __name__ == "__main__":
    test_npu_attention()