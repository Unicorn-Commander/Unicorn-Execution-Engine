#!/usr/bin/env python3
"""
Gemma 3n E4B HMA Memory Bridge
Heterogeneous Memory Architecture bridge for elastic parameter management across NPU+iGPU+CPU
"""

import os
import sys
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import threading
import mmap
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryDevice(Enum):
    """HMA memory devices"""
    NPU_PHOENIX = "npu_phoenix"
    RADEON_780M = "radeon_780m"
    SYSTEM_CPU = "system_cpu"
    SHARED_HMA = "shared_hma"

class MemoryPriority(Enum):
    """Memory allocation priority"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKGROUND = "background"

@dataclass
class MemoryRegion:
    """Represents a memory region in HMA space"""
    region_id: str
    device: MemoryDevice
    start_address: int
    size: int
    allocated_size: int
    data_type: str
    priority: MemoryPriority
    elastic_enabled: bool
    last_accessed: float
    access_count: int
    dirty: bool

@dataclass
class ElasticMemoryBlock:
    """Elastic parameter memory block"""
    block_id: str
    parameter_id: str
    layer_id: int
    parameter_type: str
    base_size: int
    elastic_size: int
    current_size: int
    device_location: MemoryDevice
    activation_state: str
    memory_region: MemoryRegion
    compression_ratio: float
    access_pattern: str

class Gemma3nE4BHMAMemoryBridge:
    """HMA memory bridge for Gemma 3n E4B elastic parameter management"""
    
    def __init__(self, model_path: str = "./models/gemma-3n-e4b-it"):
        self.model_path = Path(model_path)
        
        # HMA unified memory architecture (96GB DDR5-5600)
        self.hma_config = {
            "total_memory": 96 * 1024**3,      # 96GB DDR5 unified memory
            "memory_bandwidth": 89.6 * 1024**3,  # 89.6 GB/s DDR5-5600
            "page_size": 4096,                 # 4KB pages
            "cache_line_size": 64,             # 64B cache lines
            "numa_domains": 1,                 # Single NUMA domain
            "coherency_protocol": "MESI",      # Cache coherency
            "dma_channels": 8,                 # DMA transfer channels
            "zero_copy_enabled": True,         # Zero-copy transfers
            "compression_enabled": True,       # Memory compression
            "prefetch_enabled": True          # Hardware prefetching
        }
        
        # Device-specific memory configuration in HMA space
        self.device_config = {
            "npu_phoenix": {
                "sram_size": 2 * 1024**3,      # 2GB dedicated SRAM
                "hma_allocation": 20 * 1024**3,  # 20GB HMA allocation
                "access_latency": 10,           # 10ns SRAM access
                "bandwidth": 1024 * 1024**3,    # 1TB/s SRAM bandwidth
                "preferred_data": ["attention", "embedding", "elastic_attention"],
                "coherency_domain": "npu",
                "dma_channels": 4
            },
            "radeon_780m": {
                "local_memory": 0,              # No dedicated VRAM (HMA only)
                "hma_allocation": 32 * 1024**3,  # 32GB HMA allocation  
                "access_latency": 100,          # 100ns HMA access
                "bandwidth": 89.6 * 1024**3,    # Shared DDR5 bandwidth
                "preferred_data": ["ffn", "projection", "elastic_ffn"],
                "coherency_domain": "igpu",
                "dma_channels": 2,
                "compute_units": 12
            },
            "system_cpu": {
                "cache_l3": 32 * 1024**2,      # 32MB L3 cache
                "hma_allocation": 44 * 1024**3,  # 44GB HMA allocation
                "access_latency": 80,           # 80ns DDR5 access
                "bandwidth": 89.6 * 1024**3,    # Full DDR5 bandwidth
                "preferred_data": ["orchestration", "inactive_elastic", "compressed"],
                "coherency_domain": "cpu",
                "numa_node": 0
            }
        }
        
        # Elastic parameter configuration for Gemma 3n E4B
        self.elastic_config = {
            "base_parameters": 2 * 1024**3,    # 2B base parameters
            "elastic_parameters": 2 * 1024**3,  # 2B elastic parameters
            "total_parameters": 4 * 1024**3,   # 4B total parameters
            "activation_granularity": 64 * 1024**2,  # 64M parameter blocks
            "compression_ratio": 0.6,          # 60% compression for inactive
            "prefetch_distance": 128 * 1024**2,  # 128M prefetch distance
            "memory_pooling": True,             # Memory pool management
            "lazy_loading": True,               # Lazy load elastic params
            "swapping_enabled": True            # Parameter swapping
        }
        
        # Memory regions registry
        self.memory_regions = {}
        self.elastic_blocks = {}
        self.free_regions = {device: [] for device in MemoryDevice}
        self.allocation_map = {}
        
        # Performance monitoring
        self.performance_metrics = {
            "allocation_time": [],
            "deallocation_time": [],
            "transfer_time": [],
            "bandwidth_utilization": [],
            "cache_hit_rate": [],
            "compression_ratio": [],
            "memory_fragmentation": []
        }
        
        # Synchronization
        self.memory_lock = threading.RLock()
        self.allocation_stats = {
            "total_allocated": 0,
            "peak_allocation": 0,
            "allocation_count": 0,
            "deallocation_count": 0,
            "active_regions": 0
        }
        
        # Initialize HMA memory bridge
        self.initialize_hma_bridge()
        
    def initialize_hma_bridge(self):
        """Initialize HMA memory bridge with unified memory layout"""
        logger.info("🔧 Initializing HMA memory bridge...")
        
        # Calculate device memory allocations
        total_allocated = sum(config["hma_allocation"] for config in self.device_config.values())
        logger.info(f"   📊 Total HMA allocation: {total_allocated / 1024**3:.1f}GB / {self.hma_config['total_memory'] / 1024**3:.1f}GB")
        
        # Initialize memory regions for each device
        current_address = 0x100000000  # Start at 4GB offset
        
        for device_name, config in self.device_config.items():
            device = MemoryDevice(device_name)
            allocation_size = config["hma_allocation"]
            
            # Create primary memory region for device
            region_id = f"{device_name}_primary"
            region = MemoryRegion(
                region_id=region_id,
                device=device,
                start_address=current_address,
                size=allocation_size,
                allocated_size=0,
                data_type="mixed",
                priority=MemoryPriority.HIGH,
                elastic_enabled=True,
                last_accessed=time.time(),
                access_count=0,
                dirty=False
            )
            
            self.memory_regions[region_id] = region
            self.free_regions[device].append(region)
            
            logger.info(f"   ✅ {device_name}: {allocation_size / 1024**3:.1f}GB @ 0x{current_address:016x}")
            current_address += allocation_size
        
        # Initialize elastic parameter blocks
        self.initialize_elastic_blocks()
        
        logger.info(f"   ✅ HMA bridge initialized with {len(self.memory_regions)} regions")
        logger.info(f"   🔧 Zero-copy transfers: {self.hma_config['zero_copy_enabled']}")
        logger.info(f"   💾 Memory compression: {self.hma_config['compression_enabled']}")
    
    def initialize_elastic_blocks(self):
        """Initialize elastic parameter memory blocks"""
        logger.info("🔧 Initializing elastic parameter blocks...")
        
        # Gemma 3n E4B architecture
        num_layers = 24
        hidden_size = 3072
        intermediate_size = 8192
        
        block_count = 0
        
        for layer_idx in range(num_layers):
            # Attention elastic blocks
            attention_blocks = [
                ("q_proj_elastic", hidden_size * hidden_size // 2),
                ("k_proj_elastic", hidden_size * hidden_size // 2),
                ("v_proj_elastic", hidden_size * hidden_size // 2),
                ("o_proj_elastic", hidden_size * hidden_size // 2)
            ]
            
            for param_name, param_size in attention_blocks:
                block_id = f"layer_{layer_idx}_{param_name}"
                
                # Attention parameters prefer NPU
                device_location = MemoryDevice.NPU_PHOENIX
                memory_region = self.find_best_region(device_location, param_size)
                
                elastic_block = ElasticMemoryBlock(
                    block_id=block_id,
                    parameter_id=f"{layer_idx}_{param_name}",
                    layer_id=layer_idx,
                    parameter_type="attention",
                    base_size=param_size,
                    elastic_size=param_size,
                    current_size=0,  # Initially inactive
                    device_location=device_location,
                    activation_state="inactive",
                    memory_region=memory_region,
                    compression_ratio=0.8,  # 80% compression when inactive
                    access_pattern="sequential"
                )
                
                self.elastic_blocks[block_id] = elastic_block
                block_count += 1
            
            # FFN elastic blocks
            ffn_blocks = [
                ("gate_proj_elastic", hidden_size * intermediate_size // 2),
                ("up_proj_elastic", hidden_size * intermediate_size // 2),
                ("down_proj_elastic", intermediate_size * hidden_size // 2)
            ]
            
            for param_name, param_size in ffn_blocks:
                block_id = f"layer_{layer_idx}_{param_name}"
                
                # FFN parameters prefer iGPU
                device_location = MemoryDevice.RADEON_780M
                memory_region = self.find_best_region(device_location, param_size)
                
                elastic_block = ElasticMemoryBlock(
                    block_id=block_id,
                    parameter_id=f"{layer_idx}_{param_name}",
                    layer_id=layer_idx,
                    parameter_type="ffn",
                    base_size=param_size,
                    elastic_size=param_size,
                    current_size=0,  # Initially inactive
                    device_location=device_location,
                    activation_state="inactive",
                    memory_region=memory_region,
                    compression_ratio=0.6,  # 60% compression when inactive
                    access_pattern="random"
                )
                
                self.elastic_blocks[block_id] = elastic_block
                block_count += 1
        
        total_elastic_memory = sum(block.base_size for block in self.elastic_blocks.values())
        logger.info(f"   ✅ Initialized {block_count} elastic blocks")
        logger.info(f"   📊 Total elastic memory: {total_elastic_memory / 1024**3:.1f}GB")
    
    def find_best_region(self, preferred_device: MemoryDevice, size: int) -> MemoryRegion:
        """Find best memory region for allocation"""
        
        # First try preferred device
        for region in self.free_regions[preferred_device]:
            if region.size - region.allocated_size >= size:
                return region
        
        # Fallback to any available region
        for device, regions in self.free_regions.items():
            for region in regions:
                if region.size - region.allocated_size >= size:
                    return region
        
        # Return the largest available region
        all_regions = [region for regions in self.free_regions.values() for region in regions]
        return max(all_regions, key=lambda r: r.size - r.allocated_size)
    
    def allocate_elastic_memory(self, block_id: str, activation_state: str = "active") -> bool:
        """Allocate memory for elastic parameter block"""
        
        if block_id not in self.elastic_blocks:
            logger.error(f"❌ Elastic block {block_id} not found")
            return False
        
        block = self.elastic_blocks[block_id]
        
        with self.memory_lock:
            start_time = time.time()
            
            try:
                # Calculate required size based on activation state
                if activation_state == "active":
                    required_size = block.elastic_size
                elif activation_state == "compressed":
                    required_size = int(block.elastic_size * block.compression_ratio)
                else:
                    required_size = 0
                
                if required_size == 0:
                    block.current_size = 0
                    block.activation_state = "inactive"
                    logger.info(f"   ✅ Deactivated elastic block {block_id}")
                    return True
                
                # Check if already allocated
                if block.current_size >= required_size:
                    logger.info(f"   ✅ Block {block_id} already allocated ({block.current_size} bytes)")
                    return True
                
                # Find suitable memory region
                region = self.find_best_region(block.device_location, required_size)
                
                if region.size - region.allocated_size < required_size:
                    # Try compression or swapping
                    if self.try_make_space(block.device_location, required_size):
                        region = self.find_best_region(block.device_location, required_size)
                    else:
                        logger.warning(f"⚠️ Insufficient memory for block {block_id}")
                        return False
                
                # Allocate memory
                allocation_address = region.start_address + region.allocated_size
                region.allocated_size += required_size
                
                # Update block
                block.current_size = required_size
                block.activation_state = activation_state
                block.memory_region = region
                
                # Update statistics
                self.allocation_stats["total_allocated"] += required_size
                self.allocation_stats["allocation_count"] += 1
                self.allocation_stats["active_regions"] += 1
                
                if self.allocation_stats["total_allocated"] > self.allocation_stats["peak_allocation"]:
                    self.allocation_stats["peak_allocation"] = self.allocation_stats["total_allocated"]
                
                # Record performance metrics
                allocation_time = time.time() - start_time
                self.performance_metrics["allocation_time"].append(allocation_time)
                
                logger.info(f"   ✅ Allocated {required_size / 1024**2:.1f}MB for {block_id} on {block.device_location.value}")
                
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to allocate memory for {block_id}: {e}")
                return False
    
    def deallocate_elastic_memory(self, block_id: str) -> bool:
        """Deallocate memory for elastic parameter block"""
        
        if block_id not in self.elastic_blocks:
            logger.error(f"❌ Elastic block {block_id} not found")
            return False
        
        block = self.elastic_blocks[block_id]
        
        with self.memory_lock:
            start_time = time.time()
            
            try:
                if block.current_size == 0:
                    logger.info(f"   ✅ Block {block_id} already deallocated")
                    return True
                
                # Deallocate memory
                region = block.memory_region
                region.allocated_size -= block.current_size
                
                # Update statistics
                self.allocation_stats["total_allocated"] -= block.current_size
                self.allocation_stats["deallocation_count"] += 1
                self.allocation_stats["active_regions"] -= 1
                
                # Reset block
                deallocated_size = block.current_size
                block.current_size = 0
                block.activation_state = "inactive"
                
                # Record performance metrics
                deallocation_time = time.time() - start_time
                self.performance_metrics["deallocation_time"].append(deallocation_time)
                
                logger.info(f"   ✅ Deallocated {deallocated_size / 1024**2:.1f}MB for {block_id}")
                
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to deallocate memory for {block_id}: {e}")
                return False
    
    def try_make_space(self, device: MemoryDevice, required_size: int) -> bool:
        """Try to make space by compressing or swapping"""
        
        logger.info(f"🔧 Trying to make {required_size / 1024**2:.1f}MB space on {device.value}")
        
        # Find candidates for compression/swapping
        candidates = []
        for block in self.elastic_blocks.values():
            if (block.device_location == device and 
                block.activation_state == "active" and 
                block.current_size > 0):
                
                # Calculate space saving potential
                if block.activation_state == "active":
                    compressed_size = int(block.current_size * block.compression_ratio)
                    space_saved = block.current_size - compressed_size
                    candidates.append((block, space_saved, "compress"))
        
        # Sort by space saving potential
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        total_space_freed = 0
        
        for block, space_saved, action in candidates:
            if total_space_freed >= required_size:
                break
            
            if action == "compress":
                if self.compress_elastic_block(block.block_id):
                    total_space_freed += space_saved
                    logger.info(f"   ✅ Compressed {block.block_id}, freed {space_saved / 1024**2:.1f}MB")
        
        return total_space_freed >= required_size
    
    def compress_elastic_block(self, block_id: str) -> bool:
        """Compress an elastic parameter block"""
        
        if block_id not in self.elastic_blocks:
            return False
        
        block = self.elastic_blocks[block_id]
        
        if block.activation_state != "active":
            return False
        
        # Simulate compression
        original_size = block.current_size
        compressed_size = int(original_size * block.compression_ratio)
        space_freed = original_size - compressed_size
        
        # Update memory region
        block.memory_region.allocated_size -= space_freed
        
        # Update block
        block.current_size = compressed_size
        block.activation_state = "compressed"
        
        # Update statistics
        self.allocation_stats["total_allocated"] -= space_freed
        
        return True
    
    def transfer_elastic_block(self, block_id: str, target_device: MemoryDevice) -> bool:
        """Transfer elastic block between devices using zero-copy HMA"""
        
        if block_id not in self.elastic_blocks:
            logger.error(f"❌ Elastic block {block_id} not found")
            return False
        
        block = self.elastic_blocks[block_id]
        
        if block.device_location == target_device:
            logger.info(f"   ✅ Block {block_id} already on {target_device.value}")
            return True
        
        with self.memory_lock:
            start_time = time.time()
            
            try:
                # Find target region
                target_region = self.find_best_region(target_device, block.current_size)
                
                if target_region.size - target_region.allocated_size < block.current_size:
                    if not self.try_make_space(target_device, block.current_size):
                        logger.warning(f"⚠️ Insufficient space on {target_device.value} for {block_id}")
                        return False
                    target_region = self.find_best_region(target_device, block.current_size)
                
                # Simulate zero-copy transfer (HMA allows direct memory access)
                if self.hma_config["zero_copy_enabled"]:
                    # Zero-copy: just update pointers and coherency
                    transfer_time = block.current_size / (self.hma_config["memory_bandwidth"] * 0.8)  # 80% efficiency
                    time.sleep(transfer_time * 0.001)  # Simulate coherency overhead
                else:
                    # Traditional copy
                    transfer_time = block.current_size / self.hma_config["memory_bandwidth"]
                    time.sleep(transfer_time)
                
                # Update source region
                block.memory_region.allocated_size -= block.current_size
                
                # Update target region
                target_region.allocated_size += block.current_size
                
                # Update block
                old_device = block.device_location
                block.device_location = target_device
                block.memory_region = target_region
                
                # Record performance metrics
                actual_transfer_time = time.time() - start_time
                self.performance_metrics["transfer_time"].append(actual_transfer_time)
                
                # Calculate bandwidth utilization
                bandwidth_used = block.current_size / actual_transfer_time
                bandwidth_utilization = bandwidth_used / self.hma_config["memory_bandwidth"]
                self.performance_metrics["bandwidth_utilization"].append(bandwidth_utilization)
                
                logger.info(f"   ✅ Transferred {block_id} from {old_device.value} to {target_device.value}")
                logger.info(f"   📊 Transfer time: {actual_transfer_time:.3f}s, Bandwidth: {bandwidth_used / 1024**3:.1f} GB/s")
                
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to transfer {block_id}: {e}")
                return False
    
    def activate_elastic_parameters(self, layer_ids: List[int], parameter_types: List[str]) -> Dict[str, bool]:
        """Activate elastic parameters for specified layers and types"""
        
        logger.info(f"🚀 Activating elastic parameters for layers {layer_ids}, types {parameter_types}")
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for layer_id in layer_ids:
                for param_type in parameter_types:
                    # Find matching elastic blocks
                    matching_blocks = [
                        block_id for block_id, block in self.elastic_blocks.items()
                        if block.layer_id == layer_id and param_type in block.parameter_type
                    ]
                    
                    for block_id in matching_blocks:
                        future = executor.submit(self.allocate_elastic_memory, block_id, "active")
                        futures.append((block_id, future))
            
            # Collect results
            for block_id, future in futures:
                try:
                    success = future.result(timeout=30)
                    results[block_id] = success
                except Exception as e:
                    logger.error(f"❌ Failed to activate {block_id}: {e}")
                    results[block_id] = False
        
        successful_activations = sum(1 for success in results.values() if success)
        logger.info(f"   ✅ Activated {successful_activations}/{len(results)} elastic parameters")
        
        return results
    
    def deactivate_elastic_parameters(self, layer_ids: List[int], parameter_types: List[str]) -> Dict[str, bool]:
        """Deactivate elastic parameters for specified layers and types"""
        
        logger.info(f"🛑 Deactivating elastic parameters for layers {layer_ids}, types {parameter_types}")
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for layer_id in layer_ids:
                for param_type in parameter_types:
                    # Find matching elastic blocks
                    matching_blocks = [
                        block_id for block_id, block in self.elastic_blocks.items()
                        if block.layer_id == layer_id and param_type in block.parameter_type
                    ]
                    
                    for block_id in matching_blocks:
                        future = executor.submit(self.deallocate_elastic_memory, block_id)
                        futures.append((block_id, future))
            
            # Collect results
            for block_id, future in futures:
                try:
                    success = future.result(timeout=30)
                    results[block_id] = success
                except Exception as e:
                    logger.error(f"❌ Failed to deactivate {block_id}: {e}")
                    results[block_id] = False
        
        successful_deactivations = sum(1 for success in results.values() if success)
        logger.info(f"   ✅ Deactivated {successful_deactivations}/{len(results)} elastic parameters")
        
        return results
    
    def optimize_memory_layout(self) -> Dict[str, Any]:
        """Optimize memory layout for better performance"""
        
        logger.info("🔧 Optimizing HMA memory layout...")
        
        optimization_results = {
            "blocks_moved": 0,
            "blocks_compressed": 0,
            "fragmentation_reduced": 0.0,
            "bandwidth_improved": 0.0
        }
        
        # Analyze current layout
        device_utilization = {}
        for device in MemoryDevice:
            total_size = sum(
                region.size for region in self.memory_regions.values() 
                if region.device == device
            )
            allocated_size = sum(
                region.allocated_size for region in self.memory_regions.values() 
                if region.device == device
            )
            
            utilization = allocated_size / total_size if total_size > 0 else 0
            device_utilization[device] = utilization
        
        # Find optimization opportunities
        overloaded_devices = [
            device for device, util in device_utilization.items() 
            if util > 0.9
        ]
        
        underutilized_devices = [
            device for device, util in device_utilization.items() 
            if util < 0.5
        ]
        
        # Move blocks from overloaded to underutilized devices
        for overloaded_device in overloaded_devices:
            for underutilized_device in underutilized_devices:
                # Find candidate blocks to move
                candidate_blocks = [
                    block for block in self.elastic_blocks.values()
                    if (block.device_location == overloaded_device and 
                        block.activation_state == "compressed")
                ]
                
                # Move blocks
                for block in candidate_blocks[:3]:  # Move up to 3 blocks
                    if self.transfer_elastic_block(block.block_id, underutilized_device):
                        optimization_results["blocks_moved"] += 1
        
        # Compress inactive blocks
        inactive_blocks = [
            block for block in self.elastic_blocks.values()
            if block.activation_state == "active" and block.access_pattern == "sequential"
        ]
        
        for block in inactive_blocks[:5]:  # Compress up to 5 blocks
            if self.compress_elastic_block(block.block_id):
                optimization_results["blocks_compressed"] += 1
        
        logger.info(f"   ✅ Optimization complete:")
        logger.info(f"   📊 Blocks moved: {optimization_results['blocks_moved']}")
        logger.info(f"   📊 Blocks compressed: {optimization_results['blocks_compressed']}")
        
        return optimization_results
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory status"""
        
        device_status = {}
        
        for device in MemoryDevice:
            if device == MemoryDevice.SHARED_HMA:
                continue
                
            regions = [r for r in self.memory_regions.values() if r.device == device]
            total_size = sum(r.size for r in regions)
            allocated_size = sum(r.allocated_size for r in regions)
            free_size = total_size - allocated_size
            
            active_blocks = [
                b for b in self.elastic_blocks.values() 
                if b.device_location == device and b.current_size > 0
            ]
            
            device_status[device.value] = {
                "total_size": total_size,
                "allocated_size": allocated_size,
                "free_size": free_size,
                "utilization": allocated_size / total_size if total_size > 0 else 0,
                "active_blocks": len(active_blocks),
                "regions": len(regions)
            }
        
        overall_status = {
            "devices": device_status,
            "allocation_stats": self.allocation_stats.copy(),
            "performance_metrics": {
                "avg_allocation_time": np.mean(self.performance_metrics["allocation_time"]) if self.performance_metrics["allocation_time"] else 0,
                "avg_transfer_time": np.mean(self.performance_metrics["transfer_time"]) if self.performance_metrics["transfer_time"] else 0,
                "avg_bandwidth_utilization": np.mean(self.performance_metrics["bandwidth_utilization"]) if self.performance_metrics["bandwidth_utilization"] else 0
            },
            "elastic_blocks": {
                "total": len(self.elastic_blocks),
                "active": len([b for b in self.elastic_blocks.values() if b.activation_state == "active"]),
                "compressed": len([b for b in self.elastic_blocks.values() if b.activation_state == "compressed"]),
                "inactive": len([b for b in self.elastic_blocks.values() if b.activation_state == "inactive"])
            }
        }
        
        return overall_status
    
    def save_hma_state(self, output_path: str):
        """Save HMA memory bridge state"""
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Saving HMA memory bridge state to {output_dir}")
        
        # Save memory status
        status_file = output_dir / "memory_status.json"
        with open(status_file, 'w') as f:
            import json
            
            def convert_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, MemoryDevice):
                    return obj.value
                elif isinstance(obj, MemoryPriority):
                    return obj.value
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(item) for item in obj]
                else:
                    return obj
            
            status_data = convert_types(self.get_memory_status())
            json.dump(status_data, f, indent=2)
        
        # Save configuration
        config_file = output_dir / "hma_config.json"
        with open(config_file, 'w') as f:
            import json
            config_data = {
                "hma_config": self.hma_config,
                "device_config": self.device_config,
                "elastic_config": self.elastic_config,
                "timestamp": time.time()
            }
            json.dump(config_data, f, indent=2)
        
        logger.info("✅ HMA memory bridge state saved successfully!")
        
        return output_dir

def main():
    """Main function for testing HMA memory bridge"""
    
    logger.info("🦄 Gemma 3n E4B HMA Memory Bridge")
    logger.info("=" * 60)
    
    # Initialize HMA memory bridge
    hma_bridge = Gemma3nE4BHMAMemoryBridge()
    
    # Test elastic parameter activation
    logger.info("🚀 Testing elastic parameter activation...")
    
    # Activate attention parameters for first 4 layers
    activation_results = hma_bridge.activate_elastic_parameters([0, 1, 2, 3], ["attention"])
    
    successful_activations = sum(1 for success in activation_results.values() if success)
    logger.info(f"   ✅ Activated {successful_activations}/{len(activation_results)} attention parameters")
    
    # Activate FFN parameters for layers 5-8
    ffn_activation_results = hma_bridge.activate_elastic_parameters([5, 6, 7, 8], ["ffn"])
    
    successful_ffn = sum(1 for success in ffn_activation_results.values() if success)
    logger.info(f"   ✅ Activated {successful_ffn}/{len(ffn_activation_results)} FFN parameters")
    
    # Test memory transfer
    logger.info("🔄 Testing memory transfers...")
    
    # Transfer some blocks between devices
    transfer_count = 0
    for block_id, block in list(hma_bridge.elastic_blocks.items())[:3]:
        if block.activation_state == "active":
            target_device = MemoryDevice.RADEON_780M if block.device_location == MemoryDevice.NPU_PHOENIX else MemoryDevice.NPU_PHOENIX
            if hma_bridge.transfer_elastic_block(block_id, target_device):
                transfer_count += 1
    
    logger.info(f"   ✅ Transferred {transfer_count} blocks between devices")
    
    # Test memory optimization
    logger.info("🔧 Testing memory optimization...")
    
    optimization_results = hma_bridge.optimize_memory_layout()
    logger.info(f"   ✅ Optimization: {optimization_results['blocks_moved']} moved, {optimization_results['blocks_compressed']} compressed")
    
    # Get final memory status
    memory_status = hma_bridge.get_memory_status()
    
    logger.info("📊 Final Memory Status:")
    for device_name, status in memory_status["devices"].items():
        logger.info(f"   {device_name}: {status['utilization']:.1%} utilized, {status['active_blocks']} active blocks")
    
    logger.info(f"   Total allocation: {memory_status['allocation_stats']['total_allocated'] / 1024**3:.1f}GB")
    logger.info(f"   Peak allocation: {memory_status['allocation_stats']['peak_allocation'] / 1024**3:.1f}GB")
    
    # Test deactivation
    logger.info("🛑 Testing elastic parameter deactivation...")
    
    deactivation_results = hma_bridge.deactivate_elastic_parameters([0, 1], ["attention"])
    successful_deactivations = sum(1 for success in deactivation_results.values() if success)
    logger.info(f"   ✅ Deactivated {successful_deactivations}/{len(deactivation_results)} parameters")
    
    # Save HMA state
    output_path = "./hma_memory_states/gemma-3n-e4b-test"
    hma_bridge.save_hma_state(output_path)
    
    # Performance summary
    logger.info("=" * 60)
    logger.info("🎯 HMA MEMORY BRIDGE COMPLETE!")
    logger.info(f"📁 Output: {output_path}")
    logger.info(f"💾 Total HMA memory: {hma_bridge.hma_config['total_memory'] / 1024**3:.1f}GB")
    logger.info(f"🔧 Zero-copy transfers: {hma_bridge.hma_config['zero_copy_enabled']}")
    logger.info(f"📊 Elastic blocks: {len(hma_bridge.elastic_blocks)}")
    logger.info(f"⚡ Memory bandwidth: {hma_bridge.hma_config['memory_bandwidth'] / 1024**3:.1f} GB/s")
    logger.info("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())