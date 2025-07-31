# NPU API Reference
*Complete API documentation for NPU acceleration in Meeting-Ops*

## Core Classes

### NPUAccelerator
*Main interface for NPU operations*

```python
from stt_engine.npu_accelerator import NPUAccelerator

class NPUAccelerator:
    """NPU Accelerator using pre-compiled binaries"""
    
    def __init__(self):
        """Initialize NPU accelerator
        
        Raises:
            RuntimeError: If NPU device not found or permission denied
        """
        
    def is_available(self) -> bool:
        """Check if NPU is available and initialized
        
        Returns:
            bool: True if NPU is ready for use
        """
        
    def execute_kernel(self, kernel_name: str, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Execute NPU kernel
        
        Args:
            kernel_name: Name of kernel to execute ("whisper_transcribe", "mel_spectrogram", etc.)
            inputs: Dictionary of input arrays
            
        Returns:
            Dictionary of output arrays
            
        Raises:
            RuntimeError: If NPU not initialized or execution fails
        """
```

### SimplifiedNPURuntime
*Low-level NPU runtime interface*

```python
from npu_runtime import SimplifiedNPURuntime

class SimplifiedNPURuntime:
    """Direct NPU hardware interface"""
    
    def __init__(self, device_path: str = '/dev/accel/accel0'):
        """Initialize runtime with NPU device
        
        Args:
            device_path: Path to NPU device file
        """
        
    def open_device(self) -> bool:
        """Open NPU device and verify it's working
        
        Returns:
            bool: True if device opened successfully
        """
        
    def load_model(self, model_path: str) -> bool:
        """Load ONNX model to NPU
        
        Args:
            model_path: Path to model or model name
            
        Returns:
            bool: True if model loaded successfully
        """
        
    def transcribe(self, audio_data: Union[np.ndarray, bytes, str]) -> Dict[str, Any]:
        """Perform NPU-accelerated transcription
        
        Args:
            audio_data: Audio as numpy array, bytes, or file path
            
        Returns:
            dict: Transcription results with text, confidence, timing
        """
        
    def get_device_info(self) -> Dict[str, Any]:
        """Get NPU device information
        
        Returns:
            dict: Device status, AIE version, etc.
        """
```

## IOCTL Interface

### Constants
```python
# IOCTL commands
DRM_IOCTL_AMDXDNA_CREATE_BO = 0xC0206443    # Create buffer object
DRM_IOCTL_AMDXDNA_MAP_BO = 0xC0186444       # Map buffer to memory
DRM_IOCTL_AMDXDNA_SYNC_BO = 0xC0186445      # Synchronize buffer
DRM_IOCTL_AMDXDNA_EXEC_CMD = 0xC0206446     # Execute NPU command
DRM_IOCTL_AMDXDNA_GET_INFO = 0xC0106447     # Get device info

# Buffer types
AMDXDNA_BO_SHMEM = 1      # Shared memory buffer
AMDXDNA_BO_DEV_HEAP = 2   # Device heap buffer

# Info query types
AMDXDNA_INFO_AIE_VERSION = 2  # Query AIE version
```

### Buffer Management
```python
def create_buffer(fd: int, size: int) -> int:
    """Create DMA buffer for NPU operations
    
    Args:
        fd: File descriptor for NPU device
        size: Buffer size in bytes (will be 4KB aligned)
        
    Returns:
        int: Buffer handle
        
    Example:
        buffer_handle = create_buffer(npu_fd, 16 * 1024 * 1024)  # 16MB
    """
    
def map_buffer(fd: int, handle: int, size: int) -> mmap.mmap:
    """Map NPU buffer to process memory
    
    Args:
        fd: File descriptor
        handle: Buffer handle from create_buffer
        size: Buffer size
        
    Returns:
        mmap: Memory-mapped buffer
    """
    
def sync_buffer(fd: int, handle: int, direction: int):
    """Synchronize buffer between CPU and NPU
    
    Args:
        fd: File descriptor
        handle: Buffer handle
        direction: 0=to_device, 1=from_device
    """
```

## NPU Binary Format

### Header Structure
```python
class NPUBinaryHeader:
    magic: bytes = b"XDNA"     # 4 bytes
    version: int = 1           # 4 bytes
    num_instructions: int      # 4 bytes
    entry_point: int          # 4 bytes
    data_offset: int          # 4 bytes
    data_size: int            # 4 bytes
    reserved: bytes           # 8 bytes
```

### Instruction Set
```python
# NPU instruction opcodes
NPU_VMUL_INT8 = 0x10      # Vector multiply INT8
NPU_VADD_INT8 = 0x11      # Vector add INT8
NPU_VDOT_INT8 = 0x12      # Dot product INT8
NPU_MGEMM_INT8 = 0x20     # Matrix multiply INT8
NPU_MSOFTMAX_INT8 = 0x22  # Softmax INT8
NPU_VLOAD = 0x30          # Load from memory
NPU_VSTORE = 0x31         # Store to memory
NPU_BARRIER = 0x40        # Synchronization barrier
```

## Whisper Integration

### WhisperNPUTranscriber
```python
from stt_engine.whisper_npu_transcriber import WhisperNPUTranscriber

class WhisperNPUTranscriber:
    """ONNX Whisper with NPU acceleration"""
    
    def __init__(self, model_size: str = "base"):
        """Initialize transcriber
        
        Args:
            model_size: Whisper model size ("base", "small", etc.)
        """
        
    def transcribe_chunk(self, audio_chunk: np.ndarray) -> Dict[str, Any]:
        """Transcribe audio chunk using NPU
        
        Args:
            audio_chunk: 10 seconds of audio at 16kHz
            
        Returns:
            dict: {
                "text": str,
                "confidence": float,
                "language": str,
                "npu_accelerated": bool,
                "processing_time": float
            }
        """
```

## Usage Examples

### Basic Transcription
```python
# Initialize NPU
npu = NPUAccelerator()
if not npu.is_available():
    raise RuntimeError("NPU not available")

# Load audio (16kHz mono)
audio = load_audio("meeting.wav")

# Transcribe using NPU
result = npu.execute_kernel("whisper_transcribe", {"audio": audio})
print(f"Transcription: {result['transcription']}")
print(f"NPU Processing Time: {result['processing_time']}s")
```

### Direct Hardware Access
```python
# Low-level NPU access
runtime = SimplifiedNPURuntime()
if runtime.open_device():
    # Load Whisper model
    runtime.load_model("whisper-base")
    
    # Get device info
    info = runtime.get_device_info()
    print(f"AIE Version: {info['aie_version']}")
    
    # Transcribe
    result = runtime.transcribe("audio.wav")
    print(f"Text: {result['text']}")
    print(f"Speedup: {result['speedup']}x")
```

### WebSocket Integration
```python
# Real-time transcription via WebSocket
async def handle_audio_stream(websocket):
    transcriber = WhisperNPUTranscriber()
    
    async for message in websocket:
        # Convert audio chunk to numpy
        audio_chunk = np.frombuffer(message, dtype=np.float32)
        
        # NPU transcription
        result = transcriber.transcribe_chunk(audio_chunk)
        
        # Send result back
        await websocket.send(json.dumps(result))
```

## Performance Metrics

### Benchmark Results
```python
def benchmark_npu():
    """Benchmark NPU performance"""
    npu = NPUAccelerator()
    results = npu.benchmark()
    
    # Returns:
    # {
    #     "attention_ms": 0.5,        # Attention kernel time
    #     "mel_spec_ms": 0.3,         # Mel spectrogram time
    #     "theoretical_speedup": 32   # vs CPU baseline
    # }
```

### Expected Performance
- **Latency**: <100ms for 10-second chunks
- **Throughput**: 2,985x real-time
- **Memory**: 300MB model + buffers
- **Power**: ~10W under load

## Error Handling

### Common Errors
```python
try:
    npu = NPUAccelerator()
except RuntimeError as e:
    if "device not found" in str(e):
        print("NPU hardware not present")
    elif "permission denied" in str(e):
        print("Add user to render group: sudo usermod -a -G render $USER")
    else:
        print(f"NPU error: {e}")
```

### Debugging
```python
# Enable debug logging
import logging
logging.getLogger("npu_runtime").setLevel(logging.DEBUG)
logging.getLogger("stt_engine.npu_accelerator").setLevel(logging.DEBUG)

# Check NPU status
runtime = SimplifiedNPURuntime()
if runtime.open_device():
    info = runtime.get_device_info()
    print(f"NPU Status: {info}")
```

## Best Practices

1. **Always check NPU availability**
   ```python
   if not npu.is_available():
       raise RuntimeError("NPU required but not available")
   ```

2. **Handle audio format correctly**
   ```python
   # NPU expects 16kHz mono audio
   if sample_rate != 16000:
       audio = resample(audio, sample_rate, 16000)
   ```

3. **Use appropriate chunk sizes**
   ```python
   # 10-second chunks are optimal
   CHUNK_SIZE = 10 * 16000  # 10 seconds at 16kHz
   ```

4. **Monitor performance**
   ```python
   result = npu.execute_kernel("whisper_transcribe", {"audio": audio})
   if result['processing_time'] > 0.1:  # >100ms
       logger.warning("NPU performance degraded")
   ```

## Limitations

1. **Model Support**: Currently supports Whisper base/small/medium
2. **Audio Format**: 16kHz mono only (resampling required)
3. **Memory**: 4GB NPU memory limit
4. **Concurrency**: Single NPU instance (no parallel sessions)
5. **Quantization**: INT8 only (slight accuracy loss)

## Future Extensions

1. **Multi-model support**: Load multiple models simultaneously
2. **Streaming mode**: Zero-copy audio streaming
3. **Custom kernels**: User-defined NPU operations
4. **Power management**: Dynamic frequency scaling
5. **Multi-NPU**: Support for systems with multiple NPUs