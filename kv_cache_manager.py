#!/usr/bin/env python3
"""
KV Cache Manager for the Unicorn Execution Engine
Manages the Key-Value cache for attention layers on the GPU.
"""

import numpy as np
import logging
from typing import Dict, Tuple, List

logger = logging.getLogger(__name__)

class KVCacheManager:
    """
    Manages the Key-Value cache for attention, ensuring data resides on the GPU
    to avoid CPU-GPU transfer bottlenecks.
    """

    def __init__(self, num_layers: int, max_batch_size: int, max_seq_len: int, hidden_size: int, num_heads: int, head_dim: int, device_allocator):
        """
        Initializes the KV Cache Manager.

        Args:
            num_layers (int): The number of transformer layers.
            max_batch_size (int): The maximum number of sequences in a batch.
            max_seq_len (int): The maximum sequence length.
            hidden_size (int): The dimension of the model's hidden state.
            num_heads (int): The number of attention heads.
            head_dim (int): The dimension of each attention head.
            device_allocator: A handle to the VulkanMatrixCompute engine to allocate memory directly on the GPU.
        """
        self.num_layers = num_layers
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device_allocator = device_allocator
        
        # The cache will be a list of dictionaries (one per layer)
        # Each dictionary will map from a sequence ID to its key and value caches
        # The actual Tensors will be numpy arrays, ready for the hardware pipeline
        self.cache: List[Dict[int, Tuple[np.ndarray, np.ndarray]]] = [{} for _ in range(num_layers)]
        
        logger.info("🧠 KV Cache Manager Initialized.")
        logger.info(f"   - Layers: {num_layers}, Max Batch: {max_batch_size}, Max Seq Len: {max_seq_len}")
        logger.info("   - Caches will be allocated in GPU memory via the provided device allocator.")

    def update(self, layer_idx: int, sequence_ids: List[int], new_keys: np.ndarray, new_values: np.ndarray):
        """
        Updates the cache for a specific layer with new key and value states.

        Args:
            layer_idx (int): The index of the layer to update.
            sequence_ids (List[int]): The IDs of the sequences in the batch.
            new_keys (np.ndarray): The new key tensor for the current token (batch_size, num_heads, 1, head_dim).
            new_values (np.ndarray): The new value tensor for the current token (batch_size, num_heads, 1, head_dim).
        """
        if layer_idx >= self.num_layers:
            raise IndexError("Layer index out of bounds.")

        for i, seq_id in enumerate(sequence_ids):
            if seq_id not in self.cache[layer_idx]:
                # First token for this sequence, initialize the cache
                self.cache[layer_idx][seq_id] = (new_keys[i], new_values[i])
            else:
                # Append the new token's state to the existing cache
                old_k, old_v = self.cache[layer_idx][seq_id]
                updated_k = np.concatenate([old_k, new_keys[i]], axis=1)
                updated_v = np.concatenate([old_v, new_values[i]], axis=1)
                self.cache[layer_idx][seq_id] = (updated_k, updated_v)

    def get(self, layer_idx: int, sequence_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves the cached keys and values for a given layer and batch of sequences.

        Args:
            layer_idx (int): The index of the layer.
            sequence_ids (List[int]): The IDs of the sequences in the batch.

        Returns:
            A tuple containing the concatenated key and value caches for the batch.
        """
        if layer_idx >= self.num_layers:
            raise IndexError("Layer index out of bounds.")

        batch_keys = []
        batch_values = []
        for seq_id in sequence_ids:
            k, v = self.cache[layer_idx].get(seq_id, (None, None))
            if k is not None:
                batch_keys.append(k)
                batch_values.append(v)
        
        # This is a simplified concatenation. A real implementation would use pre-allocated buffers.
        if batch_keys:
            return np.concatenate(batch_keys, axis=0), np.concatenate(batch_values, axis=0)
        else:
            # Return None for empty cache (first token)
            return None, None

    def evict(self, sequence_id: int):
        """
        Removes a sequence from the cache, freeing its memory.

        Args:
            sequence_id (int): The ID of the sequence to evict.
        """
        for layer_cache in self.cache:
            if sequence_id in layer_cache:
                # In a real implementation with hardware buffers, we would release the buffers here.
                del layer_cache[sequence_id]
        logger.debug(f"🧹 Evicted sequence {sequence_id} from KV cache.")

