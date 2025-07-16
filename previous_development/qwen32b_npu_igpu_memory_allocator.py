#!/usr/bin/env python3
"""
Qwen 2.5 32B NPU+iGPU Memory Allocation Strategy
HMA-optimized zero-copy memory management for AMD hardware
"""

import os
import sys
import time
import torch
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeviceType(Enum):
    NPU_PHOENIX = "npu_phoenix"
    RADEON_780M = "radeon_780m"
    SYSTEM_MEMORY = "system_memory"

@dataclass
class MemoryPool:
    """Memory pool configuration for each device"""
    device_type: DeviceType
    total_memory: int  # bytes
    allocated_memory: int = 0
    free_memory: int = 0
    buffers: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.buffers is None:
            self.buffers = {}
        self.free_memory = self.total_memory - self.allocated_memory

@dataclass
class LayerAllocation:
    """Layer allocation configuration"""
    layer_id: int
    layer_type: str  # attention, ffn, embedding, output
    device: DeviceType
    memory_required: int
    precision: str
    optimization_hints: Dict = None

class Qwen32BMemoryAllocator:
    """Advanced memory allocator for Qwen 2.5 32B across NPU+iGPU+System"""
    
    def __init__(self):
        self.hardware_config = self.detect_hardware_configuration()
        self.memory_pools = self.initialize_memory_pools()
        self.allocation_strategy = self.create_allocation_strategy()
        self.hma_bridges = self.setup_hma_bridges()
        
    def detect_hardware_configuration(self) -> Dict:
        """Detect and configure AMD hardware"""
        
        logger.info("🔍 Detecting AMD hardware configuration...")
        
        config = {
            "npu_phoenix": {
                "detected": False,
                "memory": 2 * 1024**3,  # 2GB SRAM
                "tops": 16,
                "turbo_mode": False
            },
            "radeon_780m": {
                "detected": False,
                "memory": 16 * 1024**3,  # 16GB DDR5 allocation
                "compute_units": 12,
                "vulkan_support": False
            },
            "system_memory": {
                "total": 96 * 1024**3,  # 96GB DDR5
                "available": 80 * 1024**3,  # 80GB available
                "bandwidth": 89.6 * 1024**3  # GB/s
            }
        }
        
        # Check NPU Phoenix
        try:
            if os.path.exists("/dev/accel/accel0") or os.path.exists("/sys/class/accel"):
                config["npu_phoenix"]["detected"] = True
                logger.info("   ✅ NPU Phoenix detected")
                
                # Try to enable turbo mode
                try:
                    import subprocess
                    result = subprocess.run(["xrt-smi", "examine"], capture_output=True, text=True)
                    if "phoenix" in result.stdout.lower():
                        config["npu_phoenix"]["turbo_mode"] = True
                        logger.info("   🚀 NPU turbo mode available")
                except:
                    pass
        except:
            logger.warning("   ⚠️ NPU Phoenix not detected")
        
        # Check Radeon 780M
        try:
            import subprocess
            result = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True)
            if "radv phoenix" in result.stdout.lower():
                config["radeon_780m"]["detected"] = True
                config["radeon_780m"]["vulkan_support"] = True
                logger.info("   ✅ Radeon 780M with Vulkan detected")
        except:
            logger.warning("   ⚠️ Radeon 780M Vulkan not detected")
        
        return config
    
    def initialize_memory_pools(self) -> Dict[DeviceType, MemoryPool]:
        """Initialize memory pools for each device"""
        
        pools = {}
        
        # NPU Phoenix pool (2GB SRAM)
        pools[DeviceType.NPU_PHOENIX] = MemoryPool(
            device_type=DeviceType.NPU_PHOENIX,
            total_memory=self.hardware_config["npu_phoenix"]["memory"]
        )
        
        # Radeon 780M pool (16GB DDR5)
        pools[DeviceType.RADEON_780M] = MemoryPool(
            device_type=DeviceType.RADEON_780M,
            total_memory=self.hardware_config["radeon_780m"]["memory"]
        )
        
        # System memory pool (80GB DDR5)
        pools[DeviceType.SYSTEM_MEMORY] = MemoryPool(
            device_type=DeviceType.SYSTEM_MEMORY,
            total_memory=self.hardware_config["system_memory"]["available"]
        )
        
        logger.info("💾 Memory pools initialized:")
        for device, pool in pools.items():
            logger.info(f"   {device.value}: {pool.total_memory / 1024**3:.1f}GB")
        
        return pools
    
    def create_allocation_strategy(self) -> Dict:
        """Create optimal allocation strategy for Qwen 2.5 32B"""
        
        # Qwen 2.5 32B architecture parameters
        model_config = {
            "num_layers": 64,
            "hidden_size": 5120,
            "num_attention_heads": 40,
            "intermediate_size": 27392,
            "vocab_size": 152064
        }
        
        # Calculate layer memory requirements
        attention_memory_per_layer = self.calculate_attention_memory(model_config)
        ffn_memory_per_layer = self.calculate_ffn_memory(model_config)
        embedding_memory = self.calculate_embedding_memory(model_config)
        
        logger.info(f"📊 Memory per attention layer: {attention_memory_per_layer / 1024**2:.1f}MB")
        logger.info(f"📊 Memory per FFN layer: {ffn_memory_per_layer / 1024**2:.1f}MB")
        logger.info(f"📊 Embedding memory: {embedding_memory / 1024**2:.1f}MB")
        
        # Optimal allocation strategy
        strategy = {
            "npu_allocation": {
                "layers": self.allocate_npu_layers(model_config, attention_memory_per_layer),
                "components": ["attention", "layer_norm_1"],
                "precision": "INT8",
                "optimization": "attention_fusion"
            },
            "igpu_allocation": {
                "layers": self.allocate_igpu_layers(model_config, ffn_memory_per_layer),
                "components": ["ffn", "layer_norm_2"],
                "precision": "INT4",
                "optimization": "ffn_fusion"
            },
            "system_allocation": {
                "components": ["embedding", "output", "remaining_layers"],
                "precision": "FP16",
                "optimization": "memory_efficient"
            }
        }
        
        return strategy
    
    def calculate_attention_memory(self, config: Dict) -> int:
        """Calculate memory required for attention layer"""
        
        hidden_size = config["hidden_size"]
        
        # Q, K, V, O projection matrices
        qkv_memory = 4 * hidden_size * hidden_size * 1  # INT8 = 1 byte
        
        # Attention cache and intermediate tensors
        cache_memory = 2 * hidden_size * 32768 * 2  # KV cache for max context, FP16
        
        return qkv_memory + cache_memory
    
    def calculate_ffn_memory(self, config: Dict) -> int:
        """Calculate memory required for FFN layer"""
        
        hidden_size = config["hidden_size"]
        intermediate_size = config["intermediate_size"]
        
        # Gate, Up, Down projections
        ffn_memory = (
            hidden_size * intermediate_size +    # Gate
            hidden_size * intermediate_size +    # Up  
            intermediate_size * hidden_size      # Down
        ) * 0.5  # INT4 = 0.5 bytes
        
        return int(ffn_memory)
    
    def calculate_embedding_memory(self, config: Dict) -> int:
        """Calculate memory required for embedding layer"""
        
        vocab_size = config["vocab_size"]
        hidden_size = config["hidden_size"]
        
        # Embedding + output projection
        embedding_memory = vocab_size * hidden_size * 2  # FP16 = 2 bytes
        
        return embedding_memory
    
    def allocate_npu_layers(self, config: Dict, memory_per_layer: int) -> List[int]:
        """Determine which layers can fit on NPU Phoenix"""
        
        npu_memory = self.memory_pools[DeviceType.NPU_PHOENIX].total_memory
        max_layers = min(config["num_layers"], npu_memory // memory_per_layer)
        
        # Allocate first N attention layers to NPU
        allocated_layers = list(range(min(max_layers, config["num_layers"] // 3)))
        
        logger.info(f"🔧 NPU allocation: {len(allocated_layers)} attention layers")
        return allocated_layers
    
    def allocate_igpu_layers(self, config: Dict, memory_per_layer: int) -> List[int]:
        """Determine which layers can fit on Radeon 780M"""
        
        igpu_memory = self.memory_pools[DeviceType.RADEON_780M].total_memory
        max_layers = min(config["num_layers"], igpu_memory // memory_per_layer)
        
        # Allocate middle layers to iGPU
        start_layer = config["num_layers"] // 3
        allocated_layers = list(range(start_layer, start_layer + min(max_layers, config["num_layers"] // 2)))
        
        logger.info(f"🔧 iGPU allocation: {len(allocated_layers)} FFN layers")
        return allocated_layers
    
    def setup_hma_bridges(self) -> Dict:
        """Setup HMA (Heterogeneous Memory Architecture) bridges"""
        
        logger.info("🌉 Setting up HMA zero-copy bridges...")
        
        bridges = {
            "npu_to_igpu": {
                "enabled": True,
                "bandwidth": 40 * 1024**3,  # 40GB/s theoretical
                "latency": 1.5e-6,  # 1.5μs
                "buffer_size": 256 * 1024**2  # 256MB buffer
            },
            "igpu_to_system": {
                "enabled": True,
                "bandwidth": 89.6 * 1024**3,  # DDR5 bandwidth
                "latency": 0.5e-6,  # 0.5μs
                "buffer_size": 512 * 1024**2  # 512MB buffer
            },
            "npu_to_system": {
                "enabled": True,
                "bandwidth": 25 * 1024**3,  # 25GB/s
                "latency": 2.0e-6,  # 2.0μs
                "buffer_size": 128 * 1024**2  # 128MB buffer
            }
        }
        
        return bridges
    
    def allocate_layer(self, layer_id: int, layer_type: str, memory_required: int) -> LayerAllocation:
        """Allocate specific layer to optimal device"""
        
        # Allocation logic based on layer type and memory requirements
        if layer_type == "attention":
            device = DeviceType.NPU_PHOENIX
            precision = "INT8"
        elif layer_type == "ffn":
            device = DeviceType.RADEON_780M
            precision = "INT4"
        else:  # embedding, output, etc.
            device = DeviceType.SYSTEM_MEMORY
            precision = "FP16"
        
        # Check if device has enough memory
        pool = self.memory_pools[device]
        if pool.free_memory < memory_required:
            # Fallback to system memory
            device = DeviceType.SYSTEM_MEMORY
            precision = "FP16"
            pool = self.memory_pools[device]
        
        # Allocate memory
        if pool.free_memory >= memory_required:
            pool.allocated_memory += memory_required
            pool.free_memory -= memory_required
            
            allocation = LayerAllocation(
                layer_id=layer_id,
                layer_type=layer_type,
                device=device,
                memory_required=memory_required,
                precision=precision,
                optimization_hints={"device_optimized": True}
            )
            
            logger.debug(f"   ✅ Layer {layer_id} ({layer_type}) → {device.value} ({precision})")
            return allocation
        else:
            logger.error(f"   ❌ Insufficient memory for layer {layer_id}")
            return None
    
    def create_memory_map(self) -> Dict:
        """Create complete memory map for Qwen 2.5 32B"""
        
        logger.info("🗺️ Creating Qwen 32B memory map...")
        
        memory_map = {
            "layers": [],
            "embeddings": None,
            "total_memory_used": 0,
            "device_utilization": {}
        }
        
        # Allocate layers based on strategy
        strategy = self.allocation_strategy
        
        # NPU layers (attention)
        for layer_id in strategy["npu_allocation"]["layers"]:
            memory_req = self.calculate_attention_memory({"hidden_size": 5120})
            allocation = self.allocate_layer(layer_id, "attention", memory_req)
            if allocation:
                memory_map["layers"].append(allocation)
        
        # iGPU layers (FFN)
        for layer_id in strategy["igpu_allocation"]["layers"]:
            memory_req = self.calculate_ffn_memory({"hidden_size": 5120, "intermediate_size": 27392})
            allocation = self.allocate_layer(layer_id, "ffn", memory_req)
            if allocation:
                memory_map["layers"].append(allocation)
        
        # Remaining layers to system memory
        allocated_ids = {alloc.layer_id for alloc in memory_map["layers"]}
        for layer_id in range(64):  # 64 layers total
            if layer_id not in allocated_ids:
                # Allocate both attention and FFN to system memory
                attn_memory = self.calculate_attention_memory({"hidden_size": 5120})
                ffn_memory = self.calculate_ffn_memory({"hidden_size": 5120, "intermediate_size": 27392})
                
                attn_alloc = self.allocate_layer(layer_id, "attention", attn_memory)
                ffn_alloc = self.allocate_layer(layer_id, "ffn", ffn_memory)
                
                if attn_alloc:
                    memory_map["layers"].append(attn_alloc)
                if ffn_alloc:
                    memory_map["layers"].append(ffn_alloc)
        
        # Embeddings
        embedding_memory = self.calculate_embedding_memory({"vocab_size": 152064, "hidden_size": 5120})
        embedding_alloc = self.allocate_layer(-1, "embedding", embedding_memory)
        memory_map["embeddings"] = embedding_alloc
        
        # Calculate utilization
        for device, pool in self.memory_pools.items():
            utilization = (pool.allocated_memory / pool.total_memory) * 100
            memory_map["device_utilization"][device.value] = {
                "allocated": pool.allocated_memory,
                "total": pool.total_memory,
                "utilization_percent": utilization
            }
        
        # Summary
        total_allocated = sum(pool.allocated_memory for pool in self.memory_pools.values())
        memory_map["total_memory_used"] = total_allocated
        
        logger.info("📊 Memory allocation summary:")
        for device, util in memory_map["device_utilization"].items():
            logger.info(f"   {device}: {util['allocated']/1024**3:.1f}GB / {util['total']/1024**3:.1f}GB ({util['utilization_percent']:.1f}%)")
        
        return memory_map
    
    def optimize_memory_layout(self, memory_map: Dict) -> Dict:
        """Optimize memory layout for performance"""
        
        logger.info("⚡ Optimizing memory layout for performance...")
        
        optimizations = {
            "layer_reordering": self.optimize_layer_order(memory_map),
            "buffer_pooling": self.create_buffer_pools(),
            "prefetch_strategy": self.create_prefetch_strategy(memory_map),
            "cache_optimization": self.optimize_cache_usage()
        }
        
        return optimizations
    
    def optimize_layer_order(self, memory_map: Dict) -> List[int]:
        """Optimize layer execution order to minimize memory transfers"""
        
        # Group layers by device to minimize transfers
        npu_layers = []
        igpu_layers = []
        system_layers = []
        
        for allocation in memory_map["layers"]:
            if allocation.device == DeviceType.NPU_PHOENIX:
                npu_layers.append(allocation.layer_id)
            elif allocation.device == DeviceType.RADEON_780M:
                igpu_layers.append(allocation.layer_id)
            else:
                system_layers.append(allocation.layer_id)
        
        # Optimal execution order: NPU → iGPU → System
        optimized_order = sorted(npu_layers) + sorted(igpu_layers) + sorted(system_layers)
        
        logger.info(f"   🔄 Optimized layer order: {len(optimized_order)} layers")
        return optimized_order
    
    def create_buffer_pools(self) -> Dict:
        """Create buffer pools for efficient memory management"""
        
        buffer_pools = {}
        
        for device, pool in self.memory_pools.items():
            # Reserve 10% of memory for dynamic buffers
            buffer_size = int(pool.total_memory * 0.1)
            
            buffer_pools[device.value] = {
                "size": buffer_size,
                "buffers": [],
                "free_buffers": [],
                "allocation_strategy": "round_robin"
            }
        
        return buffer_pools
    
    def create_prefetch_strategy(self, memory_map: Dict) -> Dict:
        """Create prefetch strategy to hide memory latency"""
        
        prefetch_strategy = {
            "enabled": True,
            "lookahead_layers": 2,  # Prefetch 2 layers ahead
            "prefetch_size": 128 * 1024**2,  # 128MB prefetch buffer
            "priority_layers": []  # High-priority layers for prefetching
        }
        
        return prefetch_strategy
    
    def optimize_cache_usage(self) -> Dict:
        """Optimize cache usage across devices"""
        
        cache_config = {
            "npu_cache": {
                "size": 256 * 1024**2,  # 256MB
                "policy": "lru",
                "line_size": 64
            },
            "igpu_cache": {
                "size": 512 * 1024**2,  # 512MB
                "policy": "lfu",
                "line_size": 128
            },
            "system_cache": {
                "size": 2 * 1024**3,  # 2GB
                "policy": "adaptive",
                "line_size": 256
            }
        }
        
        return cache_config

def main():
    """Test memory allocator"""
    
    logger.info("🦄 Qwen 2.5 32B NPU+iGPU Memory Allocator")
    logger.info("=" * 60)
    
    # Initialize allocator
    allocator = Qwen32BMemoryAllocator()
    
    # Create memory map
    memory_map = allocator.create_memory_map()
    
    # Optimize layout
    optimizations = allocator.optimize_memory_layout(memory_map)
    
    logger.info("=" * 60)
    logger.info("✅ Memory allocation strategy created!")
    logger.info(f"📊 Total layers allocated: {len(memory_map['layers'])}")
    logger.info(f"💾 Total memory used: {memory_map['total_memory_used'] / 1024**3:.1f}GB")
    logger.info("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())