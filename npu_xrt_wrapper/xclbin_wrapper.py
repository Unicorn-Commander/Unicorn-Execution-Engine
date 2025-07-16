#!/usr/bin/env python3
"""
XCLBIN Wrapper - Creates XCLBIN format containers for NPU kernels
Based on Xilinx XCLBIN format specification for XRT runtime
"""

import os
import sys
import struct
import json
import uuid
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# XCLBIN Magic and Version
XCLBIN_MAGIC = b"xclbin2"
XCLBIN_VERSION_MAJOR = 2
XCLBIN_VERSION_MINOR = 0
XCLBIN_VERSION_PATCH = 0

# Section types based on XCLBIN spec
class SectionType:
    AXLF = 0
    BITSTREAM = 1
    CLEARING_BITSTREAM = 2
    EMBEDDED_METADATA = 3
    FIRMWARE = 4
    DEBUG_DATA = 5
    SCHED_FIRMWARE = 6
    MEM_TOPOLOGY = 7
    CONNECTIVITY = 8
    IP_LAYOUT = 9
    DEBUG_IP_LAYOUT = 10
    DESIGN_CHECK_POINT = 11
    CLOCK_FREQ_TOPOLOGY = 12
    MCS = 13
    BMC = 14
    BUILD_METADATA = 15
    KEYVALUE_METADATA = 16
    USER_METADATA = 17
    DNA_CERTIFICATE = 18
    PDI = 19
    BITSTREAM_PARTIAL_PDI = 20
    PARTITION_METADATA = 21
    EMULATION_DATA = 22
    SYSTEM_METADATA = 23
    SOFT_KERNEL = 24
    ASK_FLASH = 25
    AIE_METADATA = 26
    ASK_GROUP_TOPOLOGY = 27
    ASK_GROUP_CONNECTIVITY = 28
    SMARTNIC = 29
    AIE_PARTITION = 30
    AIE_RESOURCES = 31
    OVERLAY = 32
    VENDER_METADATA = 33

class XCLBINWrapper:
    """Creates XCLBIN containers for NPU kernels"""
    
    def __init__(self):
        self.sections = []
        self.metadata = {}
        
    def add_kernel_binary(self, kernel_path: Path, kernel_name: str) -> None:
        """Add NPU kernel binary to XCLBIN"""
        
        with open(kernel_path, 'rb') as f:
            kernel_data = f.read()
        
        # Parse our NPU kernel header
        if len(kernel_data) >= 16:
            magic, instruction_count, _, _ = struct.unpack('<IIII', kernel_data[:16])
            if magic == 0x4e505541:  # "NPUA"
                logger.info(f"NPU kernel: {instruction_count} instructions, {len(kernel_data)} bytes")
        
        # Create AIE_PARTITION section for NPU kernel
        section = {
            'type': SectionType.AIE_PARTITION,
            'name': kernel_name,
            'data': kernel_data,
            'offset': 0,  # Will be calculated later
            'size': len(kernel_data)
        }
        self.sections.append(section)
        
        # Add metadata
        self.metadata[kernel_name] = {
            'type': 'npu_kernel',
            'size': len(kernel_data),
            'instruction_count': instruction_count if 'instruction_count' in locals() else 0
        }
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to XCLBIN"""
        self.metadata[key] = value
    
    def create_mem_topology(self) -> bytes:
        """Create memory topology section"""
        # Memory topology header
        mem_count = 2  # DDR and NPU SRAM
        data = struct.pack('<I', mem_count)
        
        # DDR bank
        data += struct.pack('<B', 1)  # enabled
        data += struct.pack('<B', 0)  # type (DDR)
        data += struct.pack('<H', 0)  # reserved
        data += struct.pack('<Q', 0x100000000)  # size (4GB)
        data += struct.pack('<Q', 0x0)  # base address
        data += b'DDR[0]\x00' * 5  # tag (padded to 32 bytes)
        
        # NPU SRAM bank
        data += struct.pack('<B', 1)  # enabled
        data += struct.pack('<B', 3)  # type (SRAM)
        data += struct.pack('<H', 0)  # reserved
        data += struct.pack('<Q', 0x80000000)  # size (2GB)
        data += struct.pack('<Q', 0x200000000)  # base address
        data += b'NPU_SRAM\x00' * 4  # tag (padded to 32 bytes)
        
        return data
    
    def create_ip_layout(self, kernel_names: List[str]) -> bytes:
        """Create IP layout section"""
        ip_count = len(kernel_names)
        data = struct.pack('<I', ip_count)
        
        for i, name in enumerate(kernel_names):
            # IP type: 1 = kernel
            data += struct.pack('<I', 1)
            # Properties
            data += struct.pack('<I', 0)
            # Base address
            data += struct.pack('<Q', 0x1000 * i)
            # IP name (64 bytes)
            ip_name = name.encode('utf-8')[:63] + b'\x00'
            data += ip_name.ljust(64, b'\x00')
        
        return data
    
    def create_clock_freq_topology(self) -> bytes:
        """Create clock frequency topology"""
        # Number of clocks
        clock_count = 2
        data = struct.pack('<I', clock_count)
        
        # NPU clock (1GHz)
        data += struct.pack('<H', 1000)  # freq MHz
        data += struct.pack('<H', 0)     # reserved
        data += b'npu_clk\x00'.ljust(32, b'\x00')  # name
        
        # System clock (500MHz)
        data += struct.pack('<H', 500)   # freq MHz
        data += struct.pack('<H', 0)     # reserved
        data += b'sys_clk\x00'.ljust(32, b'\x00')  # name
        
        return data
    
    def create_xclbin(self, output_path: Path) -> bool:
        """Create XCLBIN file"""
        
        logger.info(f"Creating XCLBIN: {output_path}")
        
        # Create header
        header = bytearray(256)  # XCLBIN header is 256 bytes
        
        # Magic
        header[0:7] = XCLBIN_MAGIC
        
        # Version
        struct.pack_into('<HHI', header, 8, 
                        XCLBIN_VERSION_MAJOR,
                        XCLBIN_VERSION_MINOR,
                        XCLBIN_VERSION_PATCH)
        
        # UUID
        xclbin_uuid = uuid.uuid4()
        header[16:32] = xclbin_uuid.bytes
        
        # Platform VBNV (Vendor Board Name Version)
        platform_vbnv = b"xilinx_amd_phoenix_npu\x00"
        header[32:96] = platform_vbnv.ljust(64, b'\x00')
        
        # Feature ROM UUID
        feature_rom_uuid = uuid.uuid4()
        header[96:112] = feature_rom_uuid.bytes
        
        # Target device
        target_device = b"Phoenix\x00"
        header[112:128] = target_device.ljust(16, b'\x00')
        
        # Build metadata
        kernel_names = [s['name'] for s in self.sections if s['type'] == SectionType.AIE_PARTITION]
        
        # Add standard sections
        sections_data = []
        
        # Memory topology
        mem_topo_data = self.create_mem_topology()
        sections_data.append({
            'type': SectionType.MEM_TOPOLOGY,
            'data': mem_topo_data
        })
        
        # IP layout
        ip_layout_data = self.create_ip_layout(kernel_names)
        sections_data.append({
            'type': SectionType.IP_LAYOUT,
            'data': ip_layout_data
        })
        
        # Clock frequencies
        clock_data = self.create_clock_freq_topology()
        sections_data.append({
            'type': SectionType.CLOCK_FREQ_TOPOLOGY,
            'data': clock_data
        })
        
        # Build metadata
        build_metadata = {
            'build_date': datetime.now().isoformat(),
            'xrt_version': '2.20.0',
            'tool': 'unicorn_xclbin_wrapper',
            'kernels': kernel_names,
            'metadata': self.metadata
        }
        build_metadata_json = json.dumps(build_metadata, indent=2).encode('utf-8')
        sections_data.append({
            'type': SectionType.BUILD_METADATA,
            'data': build_metadata_json
        })
        
        # Add kernel sections
        for section in self.sections:
            sections_data.append(section)
        
        # Calculate offsets
        current_offset = 256  # After header
        current_offset += len(sections_data) * 64  # Section headers are 64 bytes each
        
        section_headers = bytearray()
        all_section_data = bytearray()
        
        for section in sections_data:
            # Section header (64 bytes)
            section_header = bytearray(64)
            struct.pack_into('<I', section_header, 0, section['type'])
            struct.pack_into('<Q', section_header, 8, current_offset)
            struct.pack_into('<Q', section_header, 16, len(section['data']))
            
            # Section name
            if 'name' in section:
                name_bytes = section['name'].encode('utf-8')[:31] + b'\x00'
                section_header[24:24+len(name_bytes)] = name_bytes
            
            section_headers.extend(section_header)
            all_section_data.extend(section['data'])
            current_offset += len(section['data'])
        
        # Update header with counts and size
        struct.pack_into('<I', header, 128, len(sections_data))  # num sections
        struct.pack_into('<Q', header, 136, current_offset)      # total size
        
        # Write XCLBIN file
        try:
            with open(output_path, 'wb') as f:
                f.write(header)
                f.write(section_headers)
                f.write(all_section_data)
            
            logger.info(f"✅ XCLBIN created: {output_path} ({current_offset} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write XCLBIN: {e}")
            return False

def create_npu_xclbin():
    """Create XCLBIN for NPU kernels"""
    
    logger.info("🔨 Creating XCLBIN for NPU kernels")
    
    wrapper = XCLBINWrapper()
    
    # First, let's compile some kernels
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from npu_mlir_kernel_compiler import NPUMLIRCompiler
    
    compiler = NPUMLIRCompiler()
    kernel_dir = Path("npu_kernels")
    kernel_dir.mkdir(exist_ok=True)
    
    # Compile standard kernel sizes
    kernel_configs = [
        (256, 5376, 32, 168),   # Gemma 27B config
        (512, 5376, 32, 168),
        (1024, 5376, 32, 168),
        (2048, 5376, 32, 168),
    ]
    
    kernel_files = []
    
    for seq_len, hidden_size, num_heads, head_dim in kernel_configs:
        logger.info(f"Compiling kernel for seq_len={seq_len}")
        kernel_data = compiler.compile_flash_attention(seq_len, hidden_size, num_heads, head_dim)
        
        # Save kernel
        kernel_path = kernel_dir / f"attention_{seq_len}_int8.bin"
        with open(kernel_path, 'wb') as f:
            f.write(kernel_data)
        kernel_files.append(kernel_path)
        logger.info(f"Saved kernel: {kernel_path} ({len(kernel_data)} bytes)")
    
    logger.info(f"Compiled {len(kernel_files)} kernel files")
    
    # Add each kernel
    for kernel_file in kernel_files:
        kernel_name = kernel_file.stem  # Remove .bin extension
        logger.info(f"Adding kernel: {kernel_name}")
        wrapper.add_kernel_binary(kernel_file, kernel_name)
    
    # Add platform metadata
    wrapper.add_metadata('platform', 'AMD Phoenix NPU')
    wrapper.add_metadata('device', 'Ryzen AI')
    wrapper.add_metadata('compute_units', 16)
    wrapper.add_metadata('memory_banks', ['DDR', 'NPU_SRAM'])
    
    # Create XCLBIN
    output_path = kernel_dir / "npu_attention_kernels.xclbin"
    if wrapper.create_xclbin(output_path):
        logger.info(f"✅ Success! XCLBIN created at: {output_path}")
        
        # Verify the file
        if output_path.exists():
            size = output_path.stat().st_size
            logger.info(f"XCLBIN size: {size} bytes")
            
            # Read and verify header
            with open(output_path, 'rb') as f:
                magic = f.read(7)
                if magic == XCLBIN_MAGIC:
                    logger.info("✅ XCLBIN header verified")
                else:
                    logger.error(f"Invalid magic: {magic}")
    else:
        logger.error("Failed to create XCLBIN")

if __name__ == "__main__":
    create_npu_xclbin()