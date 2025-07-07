# AMD Ryzen AI NPU Developer Guide

## Table of Contents
1. [Overview](#overview)
2. [Architecture Understanding](#architecture-understanding)
3. [Development Environment Setup](#development-environment-setup)
4. [NPU Programming Models](#npu-programming-models)
5. [Lessons Learned](#lessons-learned)
6. [Best Practices](#best-practices)
7. [Debugging and Profiling](#debugging-and-profiling)
8. [Performance Optimization](#performance-optimization)
9. [Common Pitfalls](#common-pitfalls)
10. [Example Workflows](#example-workflows)

## Overview

This guide provides comprehensive documentation for developing applications using AMD Ryzen AI NPU, based on real-world experience from implementing the first complete ONNX Whisper system with NPU acceleration.

### What This Guide Covers
- NPU architecture and capabilities
- Software stack integration
- Custom kernel development
- Performance optimization techniques
- Real-world deployment strategies

### Prerequisites
- AMD Ryzen AI processor (Phoenix, Hawk Point, or Strix)
- Ubuntu 25.04+ or equivalent with Linux kernel 6.10+
- Basic understanding of C++ and Python
- Familiarity with ONNX and machine learning concepts

## Architecture Understanding

### NPU Hardware Architecture

#### Phoenix NPU Specifications
- **Compute Units**: 6 NPU accelerator instances
- **Memory**: Shared system memory with dedicated NPU allocation
- **Precision**: Native BF16, FP16 support; FP32 via conversion
- **Bandwidth**: High-bandwidth memory interface
- **Power**: Low-power design optimized for continuous inference

#### NPU Memory Hierarchy
```
┌─────────────────┐
│   CPU Memory    │ ← Main system RAM
└─────────────────┘
         ↕ PCIe/Fabric
┌─────────────────┐
│  NPU L2 Cache   │ ← Shared across compute units
└─────────────────┘
         ↕
┌─────────────────┐
│  Compute Units  │ ← 6x parallel processors
│  (Local Cache)  │
└─────────────────┘
```

### Software Architecture Stack

```
┌─────────────────────────────────────┐
│        Application Layer            │ ← Your Python/C++ code
├─────────────────────────────────────┤
│     High-Level APIs                 │ ← ONNX Runtime, PyTorch
├─────────────────────────────────────┤
│     NPU Execution Providers        │ ← Custom XRT integration
├─────────────────────────────────────┤
│     MLIR-AIE / IRON Framework      │ ← Kernel compilation
├─────────────────────────────────────┤
│     XRT (Xilinx Runtime)           │ ← Device management
├─────────────────────────────────────┤
│     XDNA Kernel Driver             │ ← Hardware interface
├─────────────────────────────────────┤
│     Hardware (NPU)                 │ ← Physical NPU device
└─────────────────────────────────────┘
```

## Development Environment Setup

### Recommended Setup: Docker Container

For the most stable and reliable development environment, we strongly recommend using the provided Docker container. This method isolates the NPU toolchain from your host system, preventing library conflicts and ensuring reproducibility.

**Steps:**
1.  **Build the Image:**
    ```bash
    cd NPU-Development/
    docker build -t npu-dev-env .
    ```
2.  **Run the Container:**
    ```bash
    docker run -it --device=/dev/accel/accel0 --device=/dev/dri:/dev/dri -v $(pwd):/workspace npu-dev-env
    ```
3.  **Verify:** Inside the container, run `xrt-smi examine` and `./scripts/verify_npu_setup.sh` to confirm the NPU is accessible.

### Legacy Setup: Native Installation (Not Recommended)

**Warning:** The native installation script (`install_npu_stack.sh`) has known issues and is not the recommended path.

```bash
# Download and run our installation script
cd NPU-Development/scripts/
./install_npu_stack.sh

# Activate environment
source ~/npu-dev/setup_npu_env.sh
```

## Known Issues & Workarounds

### 1. Native Installation Script Failures

-   **Issue:** The `install_npu_stack.sh` script often fails during the `pip install` or `cmake` build steps.
-   **Workaround:** Use the Docker-based setup. The `Dockerfile` in the `NPU-Development` directory contains the corrected and validated installation steps.

### 2. XRT Environment Conflicts

-   **Issue:** Sourcing the XRT setup script (`/opt/xilinx/xrt/setup.sh`) can interfere with other system libraries.
-   **Workaround:** The Docker container provides an isolated environment, preventing these conflicts. If you must use a native installation, consider using a wrapper script to isolate the environment variables.


### Manual Environment Activation
```bash
# XRT environment
source /opt/xilinx/xrt/setup.sh

# MLIR-AIE environment
source ~/npu-dev/mlir-aie/ironenv/bin/activate
source ~/npu-dev/mlir-aie/utils/env_setup.sh

# Python environment
source ~/npu-dev/npu_dev_env/bin/activate
```

### Verification Commands
```bash
# Check NPU detection
lspci | grep -i "signal processing"
lsmod | grep amdxdna
ls /dev/accel/

# Check XRT functionality
xrt-smi examine

# Test Python imports
python -c "import mlir_aie; print('MLIR-AIE OK')"
python -c "import onnxruntime; print('ONNX Runtime OK')"
```

## NPU Programming Models

### 1. High-Level: ONNX Runtime Integration

**Best for**: Production applications, existing ONNX models

```python
import onnxruntime as ort
import numpy as np

# Create session with NPU provider (when available)
providers = [
    'VitisAIExecutionProvider',  # NPU provider
    'CPUExecutionProvider'       # Fallback
]

session = ort.InferenceSession('model.onnx', providers=providers)

# Run inference
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
result = session.run(None, {'input': input_data})
```

### 2. Mid-Level: XRT Direct Integration

**Best for**: Custom optimizations, direct NPU control

```python
from whisperx_npu_accelerator import XRTNPUAccelerator

class NPUProcessor:
    def __init__(self):
        self.accelerator = XRTNPUAccelerator()
        
    def process_matrix(self, matrix_a, matrix_b):
        """Direct NPU matrix multiplication"""
        try:
            # Convert to NPU-compatible format
            a_fp16 = matrix_a.astype(np.float16)
            b_fp16 = matrix_b.astype(np.float16)
            
            # NPU computation
            result = self.accelerator.matrix_multiply(a_fp16, b_fp16)
            return result
            
        except Exception as e:
            # Graceful CPU fallback
            return np.dot(matrix_a, matrix_b)
```

### 3. Low-Level: MLIR-AIE Kernels

**Best for**: Maximum performance, custom operations

```python
# Example AIE kernel compilation
from mlir_aie import compile_kernel

kernel_source = """
func.func @matrix_mult(%A: memref<64x64xf16>, %B: memref<64x64xf16>, 
                      %C: memref<64x64xf16>) {
  // Custom NPU kernel implementation
  affine.for %i = 0 to 64 {
    affine.for %j = 0 to 64 {
      affine.for %k = 0 to 64 {
        %a = affine.load %A[%i, %k] : memref<64x64xf16>
        %b = affine.load %B[%k, %j] : memref<64x64xf16>
        %c = affine.load %C[%i, %j] : memref<64x64xf16>
        %prod = arith.mulf %a, %b : f16
        %sum = arith.addf %c, %prod : f16
        affine.store %sum, %C[%i, %j] : memref<64x64xf16>
      }
    }
  }
  return
}
"""

# Compile for NPU
compiled_kernel = compile_kernel(kernel_source, target="aie")
```

## NPU Configuration Status

### Current Limitations ⚠️

#### Missing NPU Hardware Optimizations
Our current implementation lacks several key NPU optimizations:

```bash
# MISSING: NPU turbo mode and performance configuration
xrt-smi configure --device 0 --turbo on
xrt-smi configure --device 0 --power-profile performance
xrt-smi configure --device 0 --frequency max
```

**Impact**: We're likely running at 60-70% of potential NPU performance without these optimizations.

#### Incomplete Hybrid Execution
Unlike Ryzen AI v1.2/v1.4, we lack:
- OGA (ONNX Generator API) integration
- True NPU+iGPU+CPU load balancing
- Vulkan compute for iGPU acceleration

**Current Implementation**: Basic NPU→iGPU→CPU fallback only

### Future Enhancement Roadmap
See `NPU_OPTIMIZATION_GUIDE.md` for complete roadmap including:
- XRT-SMI turbo configuration
- Ryzen AI v1.4 feature adoption
- Vulkan + iGPU hybrid execution
- OGA integration for text generation

## Recent Fixes and Updates

### 🆕 **July 2025 - Critical XRT Environment Fix**

**Issue Resolved**: XRT environment setup was failing in production, causing NPU systems to fall back to dummy/demo mode.

**Root Cause**: The `whisperx_npu_accelerator.py` XRT environment parsing was incorrect, leading to:
- ⚠️ XRT environment setup failed warnings
- NPU detected but not properly initialized
- GUI falling back to demo mode instead of real NPU processing

**Solution Implemented**:
```python
# Fixed XRT environment setup in whisperx_npu_accelerator.py
def _setup_xrt_environment(self):
    """Set up XRT environment variables - FIXED VERSION"""
    xrt_setup = "/opt/xilinx/xrt/setup.sh"
    if os.path.exists(xrt_setup):
        try:
            # Source XRT environment and get all environment variables
            env_vars = subprocess.run(
                f"bash -c 'source {xrt_setup} && env'", 
                shell=True, capture_output=True, text=True
            )
            
            if env_vars.returncode == 0:
                xrt_vars_set = 0
                for line in env_vars.stdout.split('\n'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        # Set XRT-related environment variables
                        if any(x in key for x in ['XILINX_XRT', 'XRT', 'LD_LIBRARY_PATH', 'PATH', 'PYTHONPATH']):
                            os.environ[key] = value
                            xrt_vars_set += 1
                
                if xrt_vars_set > 0:
                    logger.info(f"✅ XRT environment configured ({xrt_vars_set} variables set)")
```

**Results**:
- ✅ XRT environment configured (11 variables set)
- ✅ NPU Phoenix detected and ready
- ✅ Real NPU processing instead of demo mode
- ✅ AdvancedNPUBackend fully functional

### 🆕 **GUI Backend Integration Fix**

**Issue**: GUI was using AdvancedNPUBackend for initialization but checking AlwaysListeningNPU for operation, causing fallback to demo mode.

**Fix**: Updated `start_always_listening()` method to properly handle both backends:
```python
if self.always_listening_system:
    # Legacy backend path
    success = self.always_listening_system.start_always_listening(...)
elif hasattr(self, 'advanced_backend') and self.advanced_backend:
    # Advanced NPU backend path - FIXED
    success = self._start_advanced_always_listening()
else:
    # Demo mode (only if no backends available)
    success = self.start_demo_mode()
```

### ✅ **Current Production Status**
- **NPU Detection**: ✅ AMD Phoenix NPU with firmware 1.5.5.391
- **XRT Integration**: ✅ Fully working with 11 environment variables
- **Backend Status**: ✅ AdvancedNPUBackend operational
- **Performance**: ✅ Real NPU acceleration active
- **Demo Mode**: ❌ No longer used in production

## Lessons Learned

### Critical Discoveries

#### 1. Memory Management is Key
**Problem**: NPU memory allocation is complex and underdocumented.

**Solution**: Implement temporary file-based memory management:
```python
def npu_safe_operation(data):
    """Safe NPU operation with memory management"""
    temp_file = None
    try:
        # Use temporary files for NPU memory
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name
            np.save(f, data)
        
        # NPU operation
        result = npu_compute(temp_file)
        return result
        
    except Exception:
        # Clean fallback
        return cpu_fallback(data)
    finally:
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)
```

#### 2. Precision Matters
**Problem**: NPU prefers FP16, but many models use FP32.

**Solution**: Automatic precision conversion with validation:
```python
def prepare_for_npu(tensor):
    """Convert tensor to NPU-optimal format"""
    if tensor.dtype == np.float32:
        # Convert to FP16 with range checking
        if np.max(np.abs(tensor)) > 65504:  # FP16 max
            # Use mixed precision
            return tensor.astype(np.float32)  # Keep FP32
        else:
            return tensor.astype(np.float16)  # Convert to FP16
    return tensor
```

#### 3. Concurrent NPU Usage is Possible
**Problem**: Initially thought NPU was single-threaded.

**Discovery**: NPU can handle multiple concurrent operations with proper resource management:
```python
class NPUResourceManager:
    def __init__(self, max_concurrent=3):
        self.semaphore = threading.Semaphore(max_concurrent)
        self.active_operations = {}
    
    def schedule_operation(self, operation_id, func, *args):
        """Schedule NPU operation with resource management"""
        with self.semaphore:
            return func(*args)
```

#### 4. Batch Processing Optimization
**Problem**: Single operations were inefficient.

**Solution**: Dynamic batching based on NPU capacity:
```python
def optimal_batch_size(operation_type, input_shape):
    """Determine optimal batch size for NPU operation"""
    base_memory = np.prod(input_shape) * 2  # FP16 = 2 bytes
    
    if operation_type == "matrix_mult":
        # NPU can handle larger matrices efficiently
        return min(64, 2048 * 2048 // base_memory)
    elif operation_type == "attention":
        # Attention is memory-intensive
        return min(16, 1024 * 1024 // base_memory)
    else:
        return 8  # Conservative default
```

### Integration Challenges

#### XRT Environment Setup
**Challenge**: XRT environment variables conflict with system libraries.

**Solution**: Isolated environment activation:
```bash
# Create XRT activation wrapper
function activate_xrt() {
    # Save current environment
    OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
    OLD_PATH=$PATH
    
    # Set XRT environment
    source /opt/xilinx/xrt/setup.sh
    
    # Custom cleanup function
    deactivate_xrt() {
        export LD_LIBRARY_PATH=$OLD_LD_LIBRARY_PATH
        export PATH=$OLD_PATH
        unset deactivate_xrt
    }
}
```

#### ONNX Model Compatibility
**Challenge**: Not all ONNX operators work with NPU.

**Solution**: Operator fallback mapping:
```python
SUPPORTED_NPU_OPS = {
    'MatMul', 'Gemm', 'Conv', 'Add', 'Mul', 'Relu', 'Sigmoid'
}

def create_hybrid_session(model_path):
    """Create ONNX session with NPU/CPU operator splitting"""
    # Analyze model for NPU-compatible operations
    model = onnx.load(model_path)
    
    npu_ops = []
    cpu_ops = []
    
    for node in model.graph.node:
        if node.op_type in SUPPORTED_NPU_OPS:
            npu_ops.append(node.name)
        else:
            cpu_ops.append(node.name)
    
    # Configure providers with operator assignment
    providers = [
        ('VitisAIExecutionProvider', {'ops': npu_ops}),
        ('CPUExecutionProvider', {'ops': cpu_ops})
    ]
    
    return ort.InferenceSession(model_path, providers=providers)
```

## Best Practices

### 1. Development Workflow

```python
class NPUDevelopmentWorkflow:
    """Recommended development pattern for NPU applications"""
    
    def __init__(self):
        self.npu_available = self.check_npu_availability()
        self.performance_metrics = {}
    
    def develop_feature(self, feature_name):
        """Standard NPU feature development process"""
        
        # 1. Implement CPU version first
        cpu_impl = self.implement_cpu_version(feature_name)
        cpu_result = self.test_cpu_implementation(cpu_impl)
        
        # 2. Profile CPU performance
        cpu_time = self.benchmark_implementation(cpu_impl)
        
        # 3. Implement NPU version
        if self.npu_available:
            npu_impl = self.implement_npu_version(feature_name)
            npu_result = self.test_npu_implementation(npu_impl)
            
            # 4. Validate numerical accuracy
            if not self.validate_accuracy(cpu_result, npu_result):
                return cpu_impl  # Fall back to CPU
            
            # 5. Benchmark performance
            npu_time = self.benchmark_implementation(npu_impl)
            
            # 6. Choose best implementation
            if npu_time < cpu_time * 0.8:  # 20% improvement threshold
                return npu_impl
        
        return cpu_impl
```

### 2. Error Handling Patterns

```python
def robust_npu_operation(operation_func, *args, **kwargs):
    """Robust NPU operation with comprehensive error handling"""
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Attempt NPU operation
            return operation_func(*args, **kwargs)
            
        except NPUMemoryError:
            # Memory issue - reduce batch size
            if 'batch_size' in kwargs:
                kwargs['batch_size'] = max(1, kwargs['batch_size'] // 2)
                retry_count += 1
                continue
            else:
                break
                
        except NPUTimeoutError:
            # Timeout - retry with longer timeout
            if 'timeout' in kwargs:
                kwargs['timeout'] *= 2
                retry_count += 1
                continue
            else:
                break
                
        except Exception as e:
            # Unknown error - log and fallback
            log_error(f"NPU operation failed: {e}")
            break
    
    # Fallback to CPU implementation
    return cpu_fallback_operation(*args, **kwargs)
```

### 3. Performance Monitoring

```python
class NPUPerformanceMonitor:
    """Monitor NPU performance and resource utilization"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = None
    
    def start_operation(self, operation_name):
        """Start monitoring an NPU operation"""
        self.start_time = time.perf_counter()
        self.operation_name = operation_name
    
    def end_operation(self):
        """End monitoring and record metrics"""
        if self.start_time:
            duration = time.perf_counter() - self.start_time
            self.metrics[self.operation_name].append(duration)
            
            # Log performance
            avg_time = np.mean(self.metrics[self.operation_name])
            logging.info(f"{self.operation_name}: {duration:.3f}s (avg: {avg_time:.3f}s)")
    
    def get_performance_report(self):
        """Generate performance report"""
        report = {}
        for operation, times in self.metrics.items():
            report[operation] = {
                'count': len(times),
                'total_time': sum(times),
                'average_time': np.mean(times),
                'min_time': min(times),
                'max_time': max(times),
                'std_time': np.std(times)
            }
        return report
```

## Debugging and Profiling

### NPU Debugging Tools

#### 1. XRT Debugging
```bash
# Enable XRT debugging
export XRT_DEBUG=1
export XRT_VERBOSE=1

# Monitor NPU usage
xrt-smi examine -v

# Check NPU device status
xrt-smi dump
```

#### 2. MLIR-AIE Debugging
```bash
# Enable MLIR-AIE debugging
export MLIR_ENABLE_DUMP=1
export AIE_DEBUG=1

# Compile with debug info
mlir-opt --debug-only=aie-dialect input.mlir
```

#### 3. Custom NPU Profiler
```python
class NPUProfiler:
    """Custom NPU operation profiler"""
    
    def __init__(self):
        self.operation_stats = {}
        self.memory_usage = []
    
    def profile_operation(self, func, *args, **kwargs):
        """Profile NPU operation with detailed metrics"""
        
        # Pre-operation state
        memory_before = self.get_npu_memory_usage()
        start_time = time.perf_counter()
        
        try:
            # Execute operation
            result = func(*args, **kwargs)
            
            # Post-operation metrics
            end_time = time.perf_counter()
            memory_after = self.get_npu_memory_usage()
            
            # Record statistics
            op_name = func.__name__
            self.operation_stats[op_name] = {
                'execution_time': end_time - start_time,
                'memory_used': memory_after - memory_before,
                'success': True
            }
            
            return result
            
        except Exception as e:
            # Record failure
            self.operation_stats[func.__name__] = {
                'execution_time': time.perf_counter() - start_time,
                'error': str(e),
                'success': False
            }
            raise
    
    def get_npu_memory_usage(self):
        """Get current NPU memory usage"""
        try:
            # Parse xrt-smi output for memory usage
            result = subprocess.run(['xrt-smi', 'examine'], 
                                  capture_output=True, text=True)
            # Parse memory information from output
            # Implementation depends on XRT version
            return 0  # Placeholder
        except:
            return 0
```

## Performance Optimization

### Optimization Strategies

#### 1. Memory Layout Optimization
```python
def optimize_memory_layout(tensor, operation_type):
    """Optimize tensor memory layout for NPU operations"""
    
    if operation_type == "matrix_multiply":
        # NPU prefers row-major layout
        if tensor.flags['F_CONTIGUOUS']:  # Column-major
            tensor = np.ascontiguousarray(tensor)
    
    elif operation_type == "convolution":
        # NPU prefers NHWC layout for convolutions
        if tensor.ndim == 4 and tensor.shape[1] < tensor.shape[3]:
            # Convert NCHW to NHWC
            tensor = np.transpose(tensor, (0, 2, 3, 1))
    
    return tensor
```

#### 2. Quantization for NPU
```python
def quantize_for_npu(model_path, output_path):
    """Quantize ONNX model for optimal NPU performance"""
    
    from onnxruntime.quantization import quantize_dynamic, QuantType
    
    # Dynamic quantization to INT8 for NPU
    quantize_dynamic(
        model_input=model_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
        extra_options={
            'WeightSymmetric': True,
            'ActivationSymmetric': False,
            'EnableSubgraph': True,
            'OptimizeLevel': 99
        }
    )
```

#### 3. Pipeline Parallelism
```python
class NPUPipeline:
    """Implement pipeline parallelism for NPU operations"""
    
    def __init__(self, stages):
        self.stages = stages
        self.queues = [Queue(maxsize=2) for _ in range(len(stages) + 1)]
        self.workers = []
    
    def start_pipeline(self):
        """Start pipeline workers"""
        for i, stage in enumerate(self.stages):
            worker = Thread(
                target=self._worker,
                args=(stage, self.queues[i], self.queues[i + 1])
            )
            worker.start()
            self.workers.append(worker)
    
    def _worker(self, stage_func, input_queue, output_queue):
        """Pipeline worker function"""
        while True:
            try:
                data = input_queue.get(timeout=1)
                if data is None:  # Shutdown signal
                    break
                
                result = stage_func(data)
                output_queue.put(result)
                
            except Empty:
                continue
    
    def process(self, input_data):
        """Process data through pipeline"""
        self.queues[0].put(input_data)
        return self.queues[-1].get()
```

## Common Pitfalls

### 1. NPU Memory Leaks
**Problem**: NPU memory not properly released.
**Solution**: Always use context managers or explicit cleanup.

```python
class NPUContext:
    """Context manager for NPU operations"""
    
    def __enter__(self):
        self.npu_handle = initialize_npu()
        return self.npu_handle
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'npu_handle'):
            cleanup_npu(self.npu_handle)
```

### 2. Incorrect Tensor Shapes
**Problem**: NPU operations fail with incompatible tensor shapes.
**Solution**: Shape validation and automatic reshaping.

```python
def validate_npu_shapes(tensor_a, tensor_b, operation):
    """Validate and fix tensor shapes for NPU operations"""
    
    if operation == "matrix_multiply":
        if tensor_a.shape[-1] != tensor_b.shape[-2]:
            raise ValueError(f"Incompatible shapes for matrix multiply: "
                           f"{tensor_a.shape} x {tensor_b.shape}")
        
        # Ensure minimum size for NPU efficiency
        if min(tensor_a.shape) < 32:
            # Pad to minimum efficient size
            tensor_a = pad_to_size(tensor_a, min_size=32)
            tensor_b = pad_to_size(tensor_b, min_size=32)
    
    return tensor_a, tensor_b
```

### 3. Environment Conflicts
**Problem**: XRT and system libraries conflict.
**Solution**: Containerized development environment.

```dockerfile
# NPU Development Container
FROM ubuntu:25.04

# Install NPU development stack
COPY NPU-Development/scripts/install_npu_stack.sh /tmp/
RUN /tmp/install_npu_stack.sh

# Set up isolated environment
ENV NPU_DEV_ROOT=/opt/npu-dev
WORKDIR $NPU_DEV_ROOT

# Default activation
CMD ["bash", "-c", "source setup_npu_env.sh && bash"]
```

## Example Workflows

### Complete NPU Application Template

```python
#!/usr/bin/env python3
"""
NPU Application Template
Demonstrates best practices for NPU application development
"""

import logging
import numpy as np
from pathlib import Path
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NPUApplication:
    """Template for NPU-accelerated applications"""
    
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.npu_available = self._check_npu_availability()
        self.session = None
        self.performance_monitor = NPUPerformanceMonitor()
    
    def _check_npu_availability(self):
        """Check if NPU is available and working"""
        try:
            # Check for NPU device
            if not Path('/dev/accel/accel0').exists():
                return False
            
            # Test XRT functionality
            result = subprocess.run(['xrt-smi', 'examine'], 
                                  capture_output=True, check=True)
            return 'NPU' in result.stdout.decode()
            
        except Exception as e:
            logger.warning(f"NPU not available: {e}")
            return False
    
    def initialize(self):
        """Initialize NPU session"""
        try:
            if self.npu_available:
                # NPU providers
                providers = [
                    'VitisAIExecutionProvider',
                    'CPUExecutionProvider'
                ]
            else:
                providers = ['CPUExecutionProvider']
            
            self.session = ort.InferenceSession(
                str(self.model_path), 
                providers=providers
            )
            
            logger.info(f"Initialized with providers: {self.session.get_providers()}")
            
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            raise
    
    @contextmanager
    def npu_operation(self, operation_name):
        """Context manager for NPU operations with monitoring"""
        self.performance_monitor.start_operation(operation_name)
        try:
            yield
        finally:
            self.performance_monitor.end_operation()
    
    def process(self, input_data):
        """Main processing function"""
        with self.npu_operation("inference"):
            # Prepare input
            if isinstance(input_data, np.ndarray):
                input_dict = {"input": input_data}
            else:
                input_dict = input_data
            
            # Run inference
            try:
                outputs = self.session.run(None, input_dict)
                return outputs[0] if len(outputs) == 1 else outputs
                
            except Exception as e:
                logger.error(f"Inference failed: {e}")
                # Implement fallback logic here
                raise
    
    def get_performance_report(self):
        """Get performance statistics"""
        return self.performance_monitor.get_performance_report()

# Usage example
if __name__ == "__main__":
    app = NPUApplication("model.onnx")
    app.initialize()
    
    # Process data
    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    result = app.process(input_data)
    
    # Print performance report
    report = app.get_performance_report()
    print(f"Performance Report: {report}")
```

This comprehensive guide provides the foundation for NPU development based on real-world experience. The patterns and practices documented here will accelerate development for any NPU-based application.