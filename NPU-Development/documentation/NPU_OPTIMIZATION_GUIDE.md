# NPU Optimization and Hybrid Execution Guide

## Current NPU Configuration Status

### XRT-SMI Configuration (⚠️ Incomplete)

**Current Usage:**
```bash
# Basic detection and monitoring only
xrt-smi examine          # NPU device detection
xrt-smi dump            # Device status monitoring
```

**Missing Optimizations:**
```bash
# NOT YET IMPLEMENTED - Future optimizations needed
xrt-smi configure --device 0 --turbo on        # Enable turbo mode
xrt-smi configure --device 0 --power-profile performance
xrt-smi configure --device 0 --frequency max   # Maximum clock frequency
xrt-smi configure --device 0 --memory-bandwidth high
```

### Current NPU Performance Settings

**What We Have:**
- FP16 precision optimization
- Dynamic batch sizing (typically 64 for matrix operations)
- Concurrent session management (3 simultaneous operations)
- Memory-efficient temporary file allocation
- Graceful CPU fallback mechanisms

**What We're Missing:**
- Hardware turbo mode activation
- Memory frequency optimization
- Power profile configuration
- Thermal management settings

## Ryzen AI Version Analysis

### Current Implementation: Custom Phoenix NPU

**Hardware Detected:**
- AMD Phoenix NPU with firmware 1.5.5.391
- 6 NPU accelerator instances
- Linux kernel 6.14+ with native amdxdna driver

**Ryzen AI Integration Status:**
- ❌ **Not using Ryzen AI v1.2 or v1.4 directly**
- ✅ **Custom XRT + MLIR-AIE implementation**
- ❌ **No OGA (ONNX Generator API) integration**
- ⚠️ **Limited hybrid execution compared to official Ryzen AI**

### Ryzen AI v1.2 vs v1.4 Comparison

#### Ryzen AI v1.2 Features (Not Implemented)
```python
# v1.2 Hybrid Execution Pattern
def ryzen_ai_v12_hybrid():
    """
    v1.2 supported hybrid NPU+CPU execution
    - Automatic operator splitting
    - Dynamic load balancing
    - Fallback management
    """
    providers = [
        ('RyzenAIExecutionProvider', {
            'device_id': 0,
            'hybrid_mode': True,
            'cpu_fallback': True
        }),
        'CPUExecutionProvider'
    ]
```

#### Ryzen AI v1.4 Features (Target for Implementation)
```python
# v1.4 Advanced Hybrid Execution
def ryzen_ai_v14_hybrid():
    """
    v1.4 supports NPU+iGPU+CPU hybrid execution
    - OGA integration for text generation
    - Vulkan compute for iGPU
    - Advanced scheduling
    """
    providers = [
        ('RyzenAIExecutionProvider', {
            'device_id': 0,
            'enable_igpu': True,
            'vulkan_compute': True,
            'oga_integration': True
        }),
        'DMLExecutionProvider',    # DirectML for iGPU
        'CPUExecutionProvider'
    ]
```

## Current Hybrid Execution Implementation

### Existing Multi-Backend Architecture

**Current Implementation:**
```python
# Our current hybrid approach
class NPUResourceManager:
    """Current hybrid execution system"""
    
    def __init__(self):
        self.backends = {
            'npu': NPUBackend(),           # Primary XRT-based NPU
            'igpu': iGPUBackend(),         # CUDA/OpenCL fallback
            'cpu': CPUBackend()            # CPU baseline
        }
    
    def execute_operation(self, operation, data):
        """Current operation scheduling"""
        try:
            # Try NPU first
            return self.backends['npu'].execute(operation, data)
        except NPUError:
            # Fallback to iGPU
            try:
                return self.backends['igpu'].execute(operation, data)
            except GPUError:
                # Final CPU fallback
                return self.backends['cpu'].execute(operation, data)
```

**Current iGPU Support:**
```python
# Limited iGPU implementation
class GPUType(Enum):
    CUDA = "cuda"           # NVIDIA CUDA ✅
    OPENCL = "opencl"       # OpenCL (Intel/AMD) ⚠️ Partial
    METAL = "metal"         # Apple Metal ✅
    VULKAN = "vulkan"       # Vulkan compute ❌ Not implemented
    CPU_OPTIMIZED = "cpu"   # Optimized CPU ✅
```

## Future Roadmap: Advanced Hybrid Execution

### Phase 1: NPU Hardware Optimization

#### 1.1 XRT-SMI Configuration Enhancement
```bash
#!/bin/bash
# npu_turbo_optimization.sh - Planned implementation

# Enable turbo mode for maximum performance
xrt-smi configure --device 0 --turbo on

# Set performance power profile
xrt-smi configure --device 0 --power-profile performance

# Maximum frequency configuration
xrt-smi configure --device 0 --frequency max

# Optimize memory bandwidth
xrt-smi configure --device 0 --memory-bandwidth high

# Verify configuration
xrt-smi examine --verbose
```

#### 1.2 Advanced NPU Resource Management
```python
# Planned: Advanced NPU optimization
class AdvancedNPUOptimizer:
    """Enhanced NPU configuration and optimization"""
    
    def __init__(self):
        self.turbo_enabled = False
        self.performance_profile = 'balanced'
        self.thermal_monitoring = True
    
    def enable_turbo_mode(self):
        """Enable NPU turbo mode for maximum performance"""
        subprocess.run(['xrt-smi', 'configure', '--device', '0', '--turbo', 'on'])
        self.turbo_enabled = True
    
    def set_performance_profile(self, profile='performance'):
        """Set NPU power profile (balanced/performance/power-save)"""
        subprocess.run(['xrt-smi', 'configure', '--device', '0', 
                       '--power-profile', profile])
        self.performance_profile = profile
    
    def optimize_for_workload(self, workload_type):
        """Optimize NPU settings for specific workload types"""
        if workload_type == 'llm_inference':
            self.enable_turbo_mode()
            self.set_performance_profile('performance')
        elif workload_type == 'continuous_audio':
            self.set_performance_profile('balanced')  # Better thermal management
```

### Phase 2: Ryzen AI Integration

#### 2.1 OGA (ONNX Generator API) Integration
```python
# Planned: OGA integration for text generation
class OGANPUIntegration:
    """Integrate OGA with NPU acceleration"""
    
    def __init__(self):
        self.oga_session = None
        self.npu_accelerator = XRTNPUAccelerator()
    
    def create_oga_session(self, model_path):
        """Create OGA session with NPU backend"""
        import onnxruntime_genai as og
        
        # Custom OGA configuration for NPU
        config = og.GeneratorParams(model_path)
        config.set_option('device', 'npu')
        config.set_option('provider', 'RyzenAIExecutionProvider')
        
        self.oga_session = og.Generator(config)
    
    def generate_with_npu(self, prompt, max_tokens=100):
        """Generate text using OGA + NPU acceleration"""
        tokens = self.oga_session.encode(prompt)
        
        # Use NPU for matrix operations in generation
        for _ in range(max_tokens):
            # Custom NPU acceleration for attention and FFN
            next_token = self._npu_accelerated_step(tokens)
            tokens.append(next_token)
            
            if next_token == self.oga_session.eos_token:
                break
        
        return self.oga_session.decode(tokens)
```

#### 2.2 Ryzen AI v1.4 Feature Adoption
```python
# Planned: Adopt v1.4 hybrid patterns
class RyzenAIHybridEngine:
    """Implement Ryzen AI v1.4 style hybrid execution"""
    
    def __init__(self):
        self.execution_providers = [
            ('RyzenAIExecutionProvider', {
                'device_id': 0,
                'npu_config': {
                    'turbo_mode': True,
                    'precision': 'fp16',
                    'concurrent_streams': 3
                },
                'igpu_config': {
                    'vulkan_compute': True,
                    'memory_pool_size': '2GB'
                },
                'hybrid_scheduling': {
                    'load_balancing': 'dynamic',
                    'fallback_strategy': 'graceful'
                }
            }),
            'DMLExecutionProvider',    # DirectML for Windows iGPU
            'CPUExecutionProvider'
        ]
    
    def create_hybrid_session(self, model_path):
        """Create session with v1.4 style hybrid execution"""
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Enable advanced hybrid features
        session_options.add_session_config_entry('hybrid_execution', 'true')
        session_options.add_session_config_entry('dynamic_load_balancing', 'true')
        
        return ort.InferenceSession(
            model_path, 
            providers=self.execution_providers,
            sess_options=session_options
        )
```

### Phase 3: Vulkan + iGPU Integration

#### 3.1 Vulkan Compute Implementation
```python
# Planned: Vulkan compute for iGPU acceleration
class VulkanNPUBridge:
    """Bridge between NPU and Vulkan compute for hybrid execution"""
    
    def __init__(self):
        self.vulkan_device = None
        self.compute_pipeline = None
        self.npu_fallback = XRTNPUAccelerator()
    
    def initialize_vulkan(self):
        """Initialize Vulkan compute for iGPU operations"""
        import vulkan as vk
        
        # Create Vulkan instance and device
        self.vulkan_device = self._create_vulkan_device()
        
        # Create compute pipeline for matrix operations
        self.compute_pipeline = self._create_compute_pipeline()
    
    def hybrid_matrix_multiply(self, a, b):
        """Hybrid matrix multiplication using NPU + Vulkan"""
        
        # Determine optimal execution strategy
        if self._should_use_npu(a.shape, b.shape):
            try:
                return self.npu_fallback.matrix_multiply(a, b)
            except NPUError:
                pass  # Fall through to Vulkan
        
        # Use Vulkan compute for iGPU acceleration
        return self._vulkan_matrix_multiply(a, b)
    
    def _vulkan_matrix_multiply(self, a, b):
        """Matrix multiplication using Vulkan compute shaders"""
        # Upload matrices to GPU memory
        buffer_a = self._upload_to_gpu(a)
        buffer_b = self._upload_to_gpu(b)
        
        # Dispatch compute shader
        self._dispatch_compute(buffer_a, buffer_b)
        
        # Download result
        return self._download_from_gpu()
```

#### 3.2 Advanced Scheduling Algorithm
```python
# Planned: Intelligent hybrid scheduling
class IntelligentHybridScheduler:
    """Advanced scheduling for NPU + iGPU + CPU hybrid execution"""
    
    def __init__(self):
        self.performance_history = {}
        self.current_workload = None
        self.thermal_monitor = ThermalMonitor()
    
    def schedule_operation(self, operation, data_shape, priority='normal'):
        """Intelligently schedule operation across available compute units"""
        
        # Analyze operation characteristics
        op_profile = self._profile_operation(operation, data_shape)
        
        # Check thermal constraints
        thermal_state = self.thermal_monitor.get_state()
        
        # Make scheduling decision
        if thermal_state.npu_temp < 80 and op_profile.npu_efficiency > 0.8:
            return self._execute_on_npu(operation, data_shape)
        elif op_profile.parallel_efficiency > 0.6:
            return self._execute_on_vulkan_igpu(operation, data_shape)
        else:
            return self._execute_on_cpu(operation, data_shape)
    
    def _profile_operation(self, operation, shape):
        """Profile operation to determine best execution target"""
        return OperationProfile(
            npu_efficiency=self._estimate_npu_efficiency(operation, shape),
            parallel_efficiency=self._estimate_parallel_efficiency(operation, shape),
            memory_requirements=self._estimate_memory_usage(operation, shape)
        )
```

### Phase 4: Performance Optimization

#### 4.1 Thermal Management
```python
# Planned: Thermal-aware optimization
class ThermalAwareOptimizer:
    """Optimize performance while managing thermal constraints"""
    
    def __init__(self):
        self.thermal_thresholds = {
            'npu_max': 85,      # Celsius
            'cpu_max': 90,
            'warning': 80
        }
        self.cooling_strategies = {}
    
    def monitor_and_adjust(self):
        """Continuously monitor thermals and adjust performance"""
        temps = self._read_temperatures()
        
        if temps['npu'] > self.thermal_thresholds['warning']:
            self._reduce_npu_frequency()
            self._increase_igpu_usage()
        
        if temps['npu'] > self.thermal_thresholds['npu_max']:
            self._emergency_thermal_protection()
```

#### 4.2 Memory Pool Management
```python
# Planned: Unified memory management
class UnifiedMemoryManager:
    """Manage memory across NPU, iGPU, and CPU efficiently"""
    
    def __init__(self):
        self.memory_pools = {
            'npu': NPUMemoryPool(size='1GB'),
            'vulkan': VulkanMemoryPool(size='2GB'),
            'system': SystemMemoryPool(size='4GB')
        }
        self.allocation_strategy = 'adaptive'
    
    def allocate_unified_buffer(self, size, access_pattern='read_write'):
        """Allocate memory accessible by multiple compute units"""
        
        if access_pattern == 'npu_primary':
            # Allocate in NPU memory with CPU mapping
            return self._allocate_npu_mapped(size)
        elif access_pattern == 'streaming':
            # Use system memory with all units mapping
            return self._allocate_streaming_buffer(size)
        else:
            # Default to most flexible allocation
            return self._allocate_flexible_buffer(size)
```

## Implementation Priority

### Immediate (Next Sprint)
1. **XRT-SMI Turbo Configuration**: Enable hardware optimization
2. **Thermal Monitoring**: Basic temperature tracking
3. **Vulkan Detection**: Enumerate Vulkan compute capabilities

### Short-term (1-2 Months)
1. **OGA Integration**: Implement ONNX Generator API support
2. **Basic Vulkan Compute**: Matrix operations on iGPU
3. **Improved Hybrid Scheduling**: Dynamic load balancing

### Long-term (3-6 Months)
1. **Full Ryzen AI v1.4 Compatibility**: Complete feature parity
2. **Advanced Thermal Management**: Thermal-aware optimization
3. **Unified Memory Architecture**: Zero-copy between compute units

## Expected Performance Improvements

### With NPU Turbo Mode
- **15-25% performance increase** for sustained workloads
- **Reduced latency** for short burst operations
- **Better thermal efficiency** with dynamic frequency scaling

### With Vulkan iGPU Integration
- **3-5x speedup** over CPU for parallel operations
- **Reduced NPU thermal load** through load distribution
- **Better battery life** on mobile platforms

### With OGA Integration
- **Native text generation** support with NPU acceleration
- **Improved token generation speed** for LLM workloads
- **Better memory efficiency** for large language models

This roadmap positions us to achieve true hybrid execution comparable to or exceeding Ryzen AI v1.4 capabilities while maintaining our custom NPU optimization advantages.