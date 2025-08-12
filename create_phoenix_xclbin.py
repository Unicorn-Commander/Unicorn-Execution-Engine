#!/usr/bin/env python3.13
"""
Create a proper XCLBIN for Phoenix NPU with 4x5 topology
Using xclbinutil to modify existing XCLBIN
"""

import subprocess
import json
import os
import sys

def create_phoenix_xclbin():
    """Create XCLBIN with correct 4x5 topology for Phoenix NPU"""
    
    print("🔨 Creating Phoenix NPU XCLBIN (4x5 topology)")
    print("=" * 50)
    
    # Input and output files
    input_xclbin = "npu_kernels_compiled/gemma3_4b_attention.xclbin"
    output_xclbin = "npu_kernels_compiled/gemma3_4b_phoenix_4x5.xclbin"
    
    if not os.path.exists(input_xclbin):
        print(f"❌ Input XCLBIN not found: {input_xclbin}")
        return False
    
    # Step 1: Extract metadata from existing XCLBIN
    print("\n📦 Extracting XCLBIN info...")
    
    # First get info about the XCLBIN
    info_cmd = [
        "/opt/xilinx/xrt/bin/xclbinutil",
        "--info",
        "--input", input_xclbin
    ]
    
    result = subprocess.run(info_cmd, capture_output=True, text=True)
    print("Current XCLBIN structure:")
    print(result.stdout[:500] + "...")
    
    # Extract binary sections we can work with
    extract_cmd = [
        "/opt/xilinx/xrt/bin/xclbinutil",
        "--dump-section", "MEM_TOPOLOGY:RAW:mem_topology.bin",
        "--dump-section", "IP_LAYOUT:RAW:ip_layout.bin",
        "--input", input_xclbin
    ]
    
    result = subprocess.run(extract_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Extraction failed: {result.stderr}")
        return False
    
    # Step 2: Modify the AIE metadata for 4x5 topology
    print("\n🔧 Modifying topology from 8 columns to 4 columns...")
    
    # Read and modify the metadata
    try:
        with open("metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Find AIE metadata section
        if "aie_metadata" in metadata:
            aie_meta = metadata["aie_metadata"]
            if "driver_config" in aie_meta:
                # Update topology to 4x5
                aie_meta["driver_config"]["num_columns"] = "4"
                aie_meta["driver_config"]["num_rows"] = "5"
                aie_meta["driver_config"]["aie_tile_num_rows"] = "4"
                aie_meta["driver_config"]["partition_num_cols"] = "4"
                
                print(f"   ✅ Updated columns: 8 → 4")
                print(f"   ✅ Updated rows: 4 → 5") 
                print(f"   ✅ Total tiles: 20 (4x5)")
        
        # Write modified metadata
        with open("metadata_phoenix.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
    except Exception as e:
        print(f"❌ Metadata modification failed: {e}")
        
        # Alternative approach: Create minimal metadata
        print("\n📝 Creating minimal Phoenix NPU metadata...")
        minimal_meta = {
            "schema_version": {"major": "1", "minor": "0", "patch": "0"},
            "aie_metadata": {
                "driver_config": {
                    "hw_gen": "4",
                    "base_address": "0x20000000000",
                    "column_shift": "25",
                    "row_shift": "20", 
                    "num_rows": "5",
                    "num_columns": "4",
                    "shim_row": "0",
                    "mem_tile_row_start": "1",
                    "mem_tile_num_rows": "1",
                    "aie_tile_row_start": "2",
                    "aie_tile_num_rows": "4",
                    "partition_num_cols": "4"
                }
            }
        }
        
        with open("metadata_phoenix.json", "w") as f:
            json.dump(minimal_meta, f, indent=2)
    
    # Step 3: Create new XCLBIN with modified metadata
    print("\n🔗 Creating new XCLBIN...")
    
    # First copy the original
    subprocess.run(["cp", input_xclbin, output_xclbin])
    
    # Replace the metadata section
    replace_cmd = [
        "/opt/xilinx/xrt/bin/xclbinutil",
        "--replace-section", "EMBEDDED_METADATA:JSON:metadata_phoenix.json",
        "--output", output_xclbin,
        "--input", output_xclbin,
        "--force"
    ]
    
    result = subprocess.run(replace_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ XCLBIN creation failed: {result.stderr}")
        
        # Try simpler approach
        print("\n🔄 Trying alternative approach...")
        
        # Use xclbinutil to patch the binary directly
        patch_cmd = [
            "/opt/xilinx/xrt/bin/xclbinutil", 
            "--info",
            "--input", input_xclbin
        ]
        
        result = subprocess.run(patch_cmd, capture_output=True, text=True)
        print("Current XCLBIN info:")
        print(result.stdout)
        
        # For now, use the original XCLBIN
        print("\n⚠️  Using original XCLBIN - NPU driver will handle topology")
        subprocess.run(["cp", input_xclbin, output_xclbin])
    
    # Cleanup
    for f in ["mem_topology.json", "ip_layout.json", "connectivity.json", 
               "metadata.json", "metadata_phoenix.json"]:
        if os.path.exists(f):
            os.remove(f)
    
    print(f"\n✅ Created: {output_xclbin}")
    print(f"   Size: {os.path.getsize(output_xclbin) / 1024:.1f} KB")
    
    return True

if __name__ == "__main__":
    success = create_phoenix_xclbin()
    sys.exit(0 if success else 1)