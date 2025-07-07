# NPU Development for Computer Vision, LLM Inference, and Embeddings

## Overview

This guide extends our NPU development knowledge from speech processing to other AI workloads including computer vision, large language model (LLM) inference, and embedding generation. These patterns leverage the same foundational NPU infrastructure while addressing domain-specific optimization requirements.

## Computer Vision on NPU

### Supported Vision Operations

#### 1. Convolution Operations
**NPU Advantages**: High parallelism, efficient memory access patterns

```python
# npu_vision_kernels.py - Computer vision NPU kernels
import numpy as np
from typing import Tuple, Optional

class NPUConvolutionEngine:
    """NPU-optimized convolution operations for computer vision"""
    
    def __init__(self):
        self.npu_accelerator = XRTNPUAccelerator()
        self.memory_manager = NPUMemoryManager()
    
    def conv2d_npu(self, 
                   input_tensor: np.ndarray,      # [N, C, H, W]
                   weight_tensor: np.ndarray,     # [Out_C, In_C, K_H, K_W]
                   stride: Tuple[int, int] = (1, 1),
                   padding: Tuple[int, int] = (0, 0)) -> np.ndarray:
        """NPU-accelerated 2D convolution"""
        
        # Optimize tensor layout for NPU
        input_npu = self._optimize_conv_input(input_tensor)
        weight_npu = self._optimize_conv_weights(weight_tensor)
        
        # Calculate output dimensions
        N, C_in, H_in, W_in = input_tensor.shape
        C_out, _, K_H, K_W = weight_tensor.shape
        
        H_out = (H_in + 2 * padding[0] - K_H) // stride[0] + 1
        W_out = (W_in + 2 * padding[1] - K_W) // stride[1] + 1
        
        try:
            # NPU convolution using optimized matrix multiplication
            # Reshape convolution as matrix multiplication (im2col approach)
            input_matrix = self._im2col_npu(input_npu, K_H, K_W, stride, padding)
            weight_matrix = weight_npu.reshape(C_out, -1)
            
            # Perform matrix multiplication on NPU
            output_matrix = self.npu_accelerator.matrix_multiply(
                weight_matrix.astype(np.float16),
                input_matrix.astype(np.float16)
            )
            
            # Reshape to output tensor
            output_tensor = output_matrix.reshape(N, C_out, H_out, W_out)
            return output_tensor.astype(np.float32)
            
        except Exception as e:
            # Fallback to CPU implementation
            return self._conv2d_cpu_fallback(input_tensor, weight_tensor, stride, padding)
    
    def _optimize_conv_input(self, tensor: np.ndarray) -> np.ndarray:
        """Optimize input tensor layout for NPU convolution"""
        # Convert NCHW to NPU-preferred layout
        if tensor.flags['C_CONTIGUOUS']:
            return tensor
        else:
            return np.ascontiguousarray(tensor)
    
    def _optimize_conv_weights(self, weights: np.ndarray) -> np.ndarray:
        """Optimize weight tensor layout for NPU"""
        # Ensure weights are in optimal format for matrix multiplication
        return np.ascontiguousarray(weights.astype(np.float16))
    
    def _im2col_npu(self, input_tensor, kernel_h, kernel_w, stride, padding):
        """NPU-optimized im2col transformation"""
        # Implement efficient im2col for NPU matrix multiplication
        N, C, H, W = input_tensor.shape
        
        # Pad input if necessary
        if padding[0] > 0 or padding[1] > 0:
            input_tensor = np.pad(input_tensor, 
                                ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
                                mode='constant')
        
        # Calculate output dimensions
        out_h = (H + 2 * padding[0] - kernel_h) // stride[0] + 1
        out_w = (W + 2 * padding[1] - kernel_w) // stride[1] + 1
        
        # Create column matrix
        col_matrix = np.zeros((C * kernel_h * kernel_w, N * out_h * out_w), dtype=np.float16)
        
        for y in range(out_h):
            for x in range(out_w):
                for n in range(N):
                    patch = input_tensor[n, :, 
                                       y*stride[0]:y*stride[0]+kernel_h, 
                                       x*stride[1]:x*stride[1]+kernel_w]
                    col_idx = n * out_h * out_w + y * out_w + x
                    col_matrix[:, col_idx] = patch.flatten()
        
        return col_matrix
```

#### 2. Object Detection Pipeline
```python
class NPUObjectDetector:
    """Complete object detection pipeline optimized for NPU"""
    
    def __init__(self, model_path: str):
        self.conv_engine = NPUConvolutionEngine()
        self.npu_session = self._create_npu_session(model_path)
        self.post_processor = NPUPostProcessor()
    
    def detect_objects(self, image: np.ndarray) -> list:
        """Detect objects in image using NPU acceleration"""
        
        # Preprocessing on NPU
        preprocessed = self._preprocess_image_npu(image)
        
        # Feature extraction using NPU convolutions
        features = self._extract_features_npu(preprocessed)
        
        # Detection head inference
        detections = self._run_detection_head(features)
        
        # Post-processing
        final_detections = self.post_processor.process(detections)
        
        return final_detections
    
    def _preprocess_image_npu(self, image: np.ndarray) -> np.ndarray:
        """NPU-accelerated image preprocessing"""
        
        # Resize using NPU matrix operations
        resized = self._resize_npu(image, target_size=(640, 640))
        
        # Normalization using NPU
        normalized = self._normalize_npu(resized)
        
        return normalized
    
    def _extract_features_npu(self, image: np.ndarray) -> dict:
        """Extract multi-scale features using NPU"""
        
        features = {}
        
        # Backbone feature extraction
        x = image
        for i, layer_config in enumerate(self.backbone_config):
            if layer_config['type'] == 'conv':
                x = self.conv_engine.conv2d_npu(
                    x, 
                    layer_config['weights'],
                    stride=layer_config['stride'],
                    padding=layer_config['padding']
                )
            elif layer_config['type'] == 'pool':
                x = self._pool_npu(x, layer_config['kernel_size'])
            
            # Store feature maps at different scales
            if i in self.feature_levels:
                features[f'level_{i}'] = x
        
        return features
```

## LLM Inference on NPU

### NPU-Optimized LLM Components

#### 1. Attention Mechanism
```python
class NPUAttentionEngine:
    """NPU-optimized attention mechanism for transformer models"""
    
    def __init__(self, hidden_size: int, num_heads: int):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.npu_matmul = NPUMatrixMultiplier()
        
    def multi_head_attention_npu(self, 
                                query: np.ndarray,
                                key: np.ndarray, 
                                value: np.ndarray,
                                mask: Optional[np.ndarray] = None) -> np.ndarray:
        """NPU-accelerated multi-head attention"""
        
        batch_size, seq_len, hidden_size = query.shape
        
        # Linear projections using NPU matrix multiplication
        q_proj = self.npu_matmul.multiply(query, self.w_q)  # [B, L, H]
        k_proj = self.npu_matmul.multiply(key, self.w_k)    # [B, L, H]
        v_proj = self.npu_matmul.multiply(value, self.w_v)  # [B, L, H]
        
        # Reshape for multi-head attention
        q_heads = q_proj.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k_heads = k_proj.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v_heads = v_proj.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation: [B, H, L, D]
        q_heads = np.transpose(q_heads, (0, 2, 1, 3))
        k_heads = np.transpose(k_heads, (0, 2, 1, 3))
        v_heads = np.transpose(v_heads, (0, 2, 1, 3))
        
        # Attention scores using NPU batch matrix multiplication
        attention_scores = self._batch_matmul_npu(q_heads, k_heads.transpose(0, 1, 3, 2))
        attention_scores = attention_scores / np.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores + (mask * -1e9)
        
        # Softmax attention weights
        attention_weights = self._softmax_npu(attention_scores)
        
        # Apply attention to values
        attended_values = self._batch_matmul_npu(attention_weights, v_heads)
        
        # Concatenate heads
        attended_values = np.transpose(attended_values, (0, 2, 1, 3))
        output = attended_values.reshape(batch_size, seq_len, hidden_size)
        
        # Output projection
        final_output = self.npu_matmul.multiply(output, self.w_o)
        
        return final_output
```

## Embedding Generation on NPU

### NPU-Optimized Embedding Pipeline

#### 1. Dense Embedding Generation
```python
class NPUEmbeddingGenerator:
    """NPU-accelerated embedding generation for various modalities"""
    
    def __init__(self, model_path: str, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.encoder_session = self._load_encoder_model(model_path)
        self.pooling_strategy = 'mean'  # 'mean', 'cls', 'max'
        
    def generate_text_embeddings(self, texts: list) -> np.ndarray:
        """Generate embeddings for text inputs using NPU"""
        
        # Tokenize texts
        input_ids, attention_masks = self._tokenize_texts(texts)
        
        # Encode using NPU-accelerated transformer
        hidden_states = self._encode_texts_npu(input_ids, attention_masks)
        
        # Pool to fixed-size embeddings
        embeddings = self._pool_embeddings_npu(hidden_states, attention_masks)
        
        # Normalize embeddings
        normalized_embeddings = self._normalize_embeddings_npu(embeddings)
        
        return normalized_embeddings
    
    def generate_image_embeddings(self, images: np.ndarray) -> np.ndarray:
        """Generate embeddings for image inputs using NPU"""
        
        # Image preprocessing
        preprocessed = self._preprocess_images_npu(images)
        
        # Vision encoder (CNN or Vision Transformer)
        features = self._encode_images_npu(preprocessed)
        
        # Global pooling to embeddings
        embeddings = self._global_pool_npu(features)
        
        # Normalize
        normalized_embeddings = self._normalize_embeddings_npu(embeddings)
        
        return normalized_embeddings
```

## Cross-Domain Performance Optimizations

### Universal NPU Optimization Patterns

#### 1. Memory-Aware Batching
```python
class NPUBatchOptimizer:
    """Optimize batch sizes across different AI workloads"""
    
    def __init__(self):
        self.workload_profiles = {
            'vision_conv': {'memory_factor': 4.0, 'compute_factor': 2.0},
            'llm_attention': {'memory_factor': 8.0, 'compute_factor': 1.5},
            'embedding_encode': {'memory_factor': 2.0, 'compute_factor': 3.0}
        }
    
    def optimize_batch_size(self, workload_type: str, input_shape: tuple, available_memory: int) -> int:
        """Calculate optimal batch size for NPU workload"""
        
        profile = self.workload_profiles.get(workload_type, {'memory_factor': 2.0, 'compute_factor': 2.0})
        
        # Estimate memory per sample
        memory_per_sample = np.prod(input_shape) * profile['memory_factor'] * 2  # FP16
        
        # Calculate maximum batch size based on memory
        max_batch_memory = available_memory // memory_per_sample
        
        # Adjust for compute efficiency
        optimal_batch = min(max_batch_memory, 64)  # NPU sweet spot
        
        # Round to power of 2 for optimal NPU utilization
        optimal_batch = 2 ** int(np.log2(optimal_batch))
        
        return max(1, optimal_batch)
```

This comprehensive guide demonstrates how the NPU development patterns from speech processing can be effectively extended to computer vision, LLM inference, and embedding generation, providing a solid foundation for diverse AI workloads on AMD Ryzen AI NPU hardware.