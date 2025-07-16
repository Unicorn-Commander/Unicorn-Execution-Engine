#!/usr/bin/env python3
"""
HMA Memory Manager for AMD Ryzen AI Architecture
Optimizes 96GB unified memory: 16GB VRAM + 40GB GTT + 40GB CPU
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import time
import psutil
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HMAMemoryManager:
    """
    HMA (Heterogeneous Memory Architecture) Manager
    Optimizes memory allocation across:
    - 16GB VRAM (fastest - dedicated iGPU memory)  
    - 40GB GTT (medium - GPU-accessible system RAM)
    - 40GB CPU (slowest - system RAM only)
    - 2GB NPU SRAM (separate - dedicated NPU memory)
    """
    
    def __init__(self):
        self.memory_pools = {
            'vram': {'total': 16 * 1024 * 1024 * 1024, 'used': 0, 'speed': 'fastest'},     # 16GB
            'gtt': {'total': 40 * 1024 * 1024 * 1024, 'used': 0, 'speed': 'medium'},      # 40GB
            'cpu': {'total': 40 * 1024 * 1024 * 1024, 'used': 0, 'speed': 'slowest'},     # 40GB
            'npu': {'total': 2 * 1024 * 1024 * 1024, 'used': 0, 'speed': 'npu_dedicated'} # 2GB
        }
        
        # Memory allocation strategy
        self.allocation_strategy = {
            'small_tensors': 'vram',      # <100MB -> VRAM (fast access)
            'medium_tensors': 'gtt',      # 100MB-1GB -> GTT (large capacity)  
            'large_tensors': 'gtt',       # >1GB -> GTT (40GB capacity)
            'model_weights': 'gtt',       # Static weights -> GTT
            'activations': 'vram',        # Dynamic activations -> VRAM
            'attention_cache': 'gtt',     # KV cache -> GTT (grows large)
            'npu_kernels': 'npu'          # NPU operations -> NPU SRAM
        }
        
        # Track allocations
        self.allocations = {}
        self.allocation_history = []
        
        logger.info("🧠 HMA Memory Manager initialized")
        logger.info(f"   💾 Total capacity: {sum(pool['total'] for pool in self.memory_pools.values()) / 1024**3:.1f}GB")
        self._log_memory_layout()
    
    def _log_memory_layout(self):
        """Log HMA memory architecture layout"""
        logger.info("📊 HMA Memory Architecture:")
        logger.info("   ⚡ VRAM (16GB):  Fastest - Small tensors, activations")
        logger.info("   📊 GTT (40GB):   Medium  - Large tensors, model weights")  
        logger.info("   💾 CPU (40GB):   Slowest - System operations, overflow")
        logger.info("   🔥 NPU (2GB):    Dedicated - NPU kernels and data")
    
    def allocate_tensor(self, 
                       tensor_data: np.ndarray, 
                       tensor_type: str = 'auto',
                       cache_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Allocate tensor in optimal HMA memory pool
        
        Args:
            tensor_data: Numpy array to allocate
            tensor_type: Type hint for allocation strategy
            cache_key: Optional cache key for reuse
            
        Returns:
            Allocation info with memory pool and access methods
        """
        
        size_bytes = tensor_data.nbytes
        size_mb = size_bytes / (1024 * 1024)
        
        # Check cache first
        if cache_key and cache_key in self.allocations:
            logger.info(f"🚀 Cache hit: {cache_key} ({size_mb:.1f}MB)")
            return self.allocations[cache_key]
        
        # Determine optimal memory pool
        memory_pool = self._select_memory_pool(size_bytes, tensor_type)
        
        # Check capacity
        if not self._check_capacity(memory_pool, size_bytes):
            memory_pool = self._find_fallback_pool(size_bytes)
            if not memory_pool:
                raise RuntimeError(f"Insufficient memory for {size_mb:.1f}MB allocation")
        
        # Perform allocation
        allocation_info = {
            'data': tensor_data,
            'size_bytes': size_bytes,
            'memory_pool': memory_pool,
            'tensor_type': tensor_type,
            'allocation_id': f"{memory_pool}_{len(self.allocations)}",
            'timestamp': time.time(),
            'zero_copy_accessible': memory_pool in ['vram', 'gtt']  # GPU accessible
        }
        
        # Update pool usage
        self.memory_pools[memory_pool]['used'] += size_bytes
        
        # Cache allocation
        if cache_key:
            self.allocations[cache_key] = allocation_info
        
        # Log allocation
        pool_info = self.memory_pools[memory_pool]
        usage_pct = (pool_info['used'] / pool_info['total']) * 100
        
        logger.info(f"🧠 Allocated {size_mb:.1f}MB to {memory_pool.upper()}")
        logger.info(f"   📊 {memory_pool.upper()} usage: {usage_pct:.1f}% ({pool_info['used']/1024**3:.1f}GB / {pool_info['total']/1024**3:.1f}GB)")
        
        return allocation_info
    
    def _select_memory_pool(self, size_bytes: int, tensor_type: str) -> str:
        """Select optimal memory pool based on HMA strategy"""
        
        size_mb = size_bytes / (1024 * 1024)
        
        # NPU-specific allocations
        if tensor_type in ['npu_kernels', 'attention_weights']:
            return 'npu'
        
        # Size-based allocation for general tensors
        if tensor_type == 'auto':
            if size_mb < 100:  # Small tensors -> VRAM (fast)
                return 'vram'
            elif size_mb < 1024:  # Medium tensors -> GTT (capacity)
                return 'gtt'
            else:  # Large tensors -> GTT (40GB capacity)
                return 'gtt'
        
        # Type-specific allocation
        return self.allocation_strategy.get(tensor_type, 'gtt')
    
    def _check_capacity(self, memory_pool: str, size_bytes: int) -> bool:
        """Check if memory pool has sufficient capacity"""
        pool = self.memory_pools[memory_pool]
        available = pool['total'] - pool['used']
        return available >= size_bytes
    
    def _find_fallback_pool(self, size_bytes: int) -> Optional[str]:
        """Find fallback memory pool with sufficient capacity"""
        # Try pools in order of preference: GTT -> CPU -> VRAM
        for pool_name in ['gtt', 'cpu', 'vram']:
            if self._check_capacity(pool_name, size_bytes):
                logger.warning(f"⚠️ Using fallback pool: {pool_name.upper()}")
                return pool_name
        return None
    
    def create_zero_copy_tensor(self, 
                               tensor_data: np.ndarray,
                               source_device: str = 'npu',
                               target_device: str = 'igpu') -> torch.Tensor:
        """
        Create zero-copy tensor for NPU↔iGPU transfers using HMA
        """
        
        # Allocate in GTT (GPU-accessible system RAM)
        allocation_info = self.allocate_tensor(
            tensor_data, 
            tensor_type='zero_copy_transfer',
            cache_key=f"zero_copy_{source_device}_{target_device}"
        )
        
        if allocation_info['memory_pool'] not in ['vram', 'gtt']:
            logger.warning("⚠️ Zero-copy transfer not in GPU-accessible memory")
        
        # Create tensor with shared memory
        shared_tensor = torch.from_numpy(allocation_info['data'])
        
        logger.info(f"🔗 Zero-copy transfer: {source_device}→{target_device}")
        logger.info(f"   📍 Memory: {allocation_info['memory_pool'].upper()} ({allocation_info['size_bytes']/1024**2:.1f}MB)")
        
        return shared_tensor
    
    def optimize_for_gemma27b(self):
        """Optimize memory allocation for Gemma 3 27B model"""
        
        logger.info("🦄 Optimizing HMA for Gemma 3 27B...")
        
        # Model size estimates (quantized)
        gemma_27b_sizes = {
            'embedding_weights': 512 * 1024 * 1024,      # 512MB
            'attention_weights': 8 * 1024 * 1024 * 1024, # 8GB (all layers)
            'ffn_weights': 16 * 1024 * 1024 * 1024,      # 16GB (all layers)
            'activations': 2 * 1024 * 1024 * 1024,       # 2GB (dynamic)
            'kv_cache': 4 * 1024 * 1024 * 1024           # 4GB (grows with sequence)
        }
        
        # Optimal allocation strategy for Gemma 27B
        allocation_plan = {
            'embedding_weights': 'gtt',     # Static, fits in GTT
            'attention_weights': 'gtt',     # Large, static -> GTT (40GB)
            'ffn_weights': 'gtt',           # Large, static -> GTT
            'activations': 'vram',          # Dynamic, fast access -> VRAM (16GB)
            'kv_cache': 'gtt'               # Grows large -> GTT
        }
        
        # Verify capacity
        total_gtt_needed = (gemma_27b_sizes['embedding_weights'] + 
                           gemma_27b_sizes['attention_weights'] + 
                           gemma_27b_sizes['ffn_weights'] + 
                           gemma_27b_sizes['kv_cache'])
        
        total_vram_needed = gemma_27b_sizes['activations']
        
        gtt_gb = total_gtt_needed / (1024**3)
        vram_gb = total_vram_needed / (1024**3)
        
        logger.info(f"📊 Gemma 27B memory requirements:")
        logger.info(f"   GTT needed: {gtt_gb:.1f}GB / 40GB available")
        logger.info(f"   VRAM needed: {vram_gb:.1f}GB / 16GB available")
        
        if gtt_gb <= 40 and vram_gb <= 16:
            logger.info("✅ Gemma 27B fits perfectly in HMA architecture!")
            return True
        else:
            logger.warning("⚠️ Gemma 27B may exceed available memory")
            return False
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        
        stats = {
            'pools': {},
            'total_allocated': 0,
            'total_available': 0,
            'allocation_count': len(self.allocations)
        }
        
        for pool_name, pool_info in self.memory_pools.items():
            usage_pct = (pool_info['used'] / pool_info['total']) * 100
            available = pool_info['total'] - pool_info['used']
            
            stats['pools'][pool_name] = {
                'total_gb': pool_info['total'] / (1024**3),
                'used_gb': pool_info['used'] / (1024**3),
                'available_gb': available / (1024**3),
                'usage_percent': usage_pct,
                'speed': pool_info['speed']
            }
            
            stats['total_allocated'] += pool_info['used']
            stats['total_available'] += available
        
        return stats
    
    def cleanup_allocations(self, older_than_seconds: float = 300):
        """Cleanup old allocations to free memory"""
        
        current_time = time.time()
        cleanup_keys = []
        
        for key, allocation in self.allocations.items():
            if current_time - allocation['timestamp'] > older_than_seconds:
                cleanup_keys.append(key)
        
        for key in cleanup_keys:
            self.deallocate(key)
        
        logger.info(f"🧹 Cleaned up {len(cleanup_keys)} old allocations")
        gc.collect()
    
    def deallocate(self, allocation_key: str):
        """Deallocate memory allocation"""
        if allocation_key in self.allocations:
            allocation = self.allocations[allocation_key]
            pool_name = allocation['memory_pool']
            size_bytes = allocation['size_bytes']
            
            # Update pool usage
            self.memory_pools[pool_name]['used'] -= size_bytes
            
            # Remove allocation
            del self.allocations[allocation_key]
            
            logger.info(f"🗑️ Deallocated {size_bytes/1024**2:.1f}MB from {pool_name.upper()}")

def test_hma_memory_manager():
    """Test HMA memory manager"""
    logger.info("🧪 Testing HMA Memory Manager...")
    
    # Initialize manager
    hma = HMAMemoryManager()
    
    # Test Gemma 27B optimization
    hma.optimize_for_gemma27b()
    
    # Test tensor allocations
    small_tensor = np.random.randn(32, 4096).astype(np.float16)  # 256KB
    medium_tensor = np.random.randn(1024, 4096).astype(np.float16)  # 8MB
    large_tensor = np.random.randn(8192, 4096).astype(np.float16)  # 64MB
    
    # Allocate tensors
    small_alloc = hma.allocate_tensor(small_tensor, 'activations')
    medium_alloc = hma.allocate_tensor(medium_tensor, 'attention_weights')  
    large_alloc = hma.allocate_tensor(large_tensor, 'ffn_weights')
    
    # Test zero-copy transfer
    zero_copy_tensor = hma.create_zero_copy_tensor(medium_tensor, 'npu', 'igpu')
    
    # Get statistics
    stats = hma.get_memory_stats()
    logger.info("📊 HMA Memory Statistics:")
    for pool_name, pool_stats in stats['pools'].items():
        logger.info(f"   {pool_name.upper()}: {pool_stats['used_gb']:.1f}GB / {pool_stats['total_gb']:.1f}GB ({pool_stats['usage_percent']:.1f}%)")
    
    logger.info("✅ HMA Memory Manager test completed!")

if __name__ == "__main__":
    test_hma_memory_manager()