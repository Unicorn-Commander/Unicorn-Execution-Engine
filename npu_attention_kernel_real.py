#\!/usr/bin/env python3
"""
Real NPU Attention Kernel - Direct AMD Phoenix NPU Hardware Acceleration
No simulations - real hardware or failure
"""

import numpy as np
import logging
import ctypes
import os
from typing import Dict, Tuple, List, Optional, Any

logger = logging.getLogger(__name__)

class NPUAttentionKernelReal:
    """Real NPU Attention Kernel with direct hardware acceleration"""

    def __init__(self, seq_length=256, d_model=5376, num_heads=32):
        self.seq_length = seq_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.initialized = False
        
        # NPU hardware handles
        self.npu_device = None
        self.npu_context = None
        self.xdna_driver = None
        
        # Kernel data
        self.kernel_binary = None
        self.kernel_config = None
        self.kernel_loaded = False
        
        logger.info("🧠 Real NPU Attention Kernel Initialized.")
        logger.info(f"   - Sequence Length: {seq_length}")
        logger.info(f"   - Model Dimension: {d_model}")
        logger.info(f"   - Number of Heads: {num_heads}")
        logger.info(f"   - Head Dimension: {self.head_dim}")

    def initialize(self) -> bool:
        """Initialize real NPU hardware"""
        logger.info("⚡ Initializing Real NPU Hardware...")
        
        try:
            # Check for NPU device
            if not self._detect_npu_device():
                logger.error("❌ AMD Phoenix NPU not detected")
                return False
                
            # Load NPU driver
            if not self._load_npu_driver():
                logger.error("❌ Failed to load NPU driver")
                return False
                
            # Initialize NPU context
            if not self._initialize_npu_context():
                logger.error("❌ Failed to initialize NPU context")
                return False
                
            # Load attention kernel
            if not self._load_attention_kernel():
                logger.error("❌ Failed to load attention kernel")
                return False
                
            self.initialized = True
            logger.info("✅ Real NPU Hardware initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ NPU initialization failed: {e}")
            return False

    def _detect_npu_device(self) -> bool:
        """Detect AMD Phoenix NPU device"""
        try:
            # Check for NPU accelerator device
            npu_devices = []
            
            # Method 1: Check /dev/accel devices
            accel_devices = "/dev/accel"
            if os.path.exists(accel_devices):
                for device in os.listdir(accel_devices):
                    accel_path = f"{accel_devices}/{device}"
                    if os.path.exists(accel_path):
                        npu_devices.append(accel_path)
                        logger.info(f"✅ Found NPU accelerator device: {accel_path}")
            
            # Method 2: Check /sys/class/accel devices
            sys_accel = "/sys/class/accel"
            if os.path.exists(sys_accel):
                for device in os.listdir(sys_accel):
                    device_path = f"{sys_accel}/{device}/device/vendor"
                    if os.path.exists(device_path):
                        with open(device_path, 'r') as f:
                            vendor = f.read().strip()
                        if vendor == "0x1022":  # AMD vendor ID
                            # Check device ID for Phoenix NPU
                            device_id_path = f"{sys_accel}/{device}/device/device"
                            if os.path.exists(device_id_path):
                                with open(device_id_path, 'r') as f:
                                    device_id = f.read().strip()
                                if device_id == "0x1502":  # Phoenix NPU device ID
                                    logger.info(f"✅ Found AMD Phoenix NPU: {device} (vendor: {vendor}, device: {device_id})")
                                    npu_devices.append(device)
            
            if not npu_devices:
                logger.warning("⚠️ No AMD Phoenix NPU devices found")
                return False
                
            logger.info(f"✅ AMD Phoenix NPU detection successful: {len(npu_devices)} devices")
            return True
            
        except Exception as e:
            logger.error(f"❌ NPU detection failed: {e}")
            return False

    def _load_npu_driver(self) -> bool:
        """Load NPU driver library"""
        try:
            # Try to load the XDNA driver
            driver_paths = [
                "/usr/local/xrt/lib/libxrt_driver_xdna.so",
                "/usr/lib/x86_64-linux-gnu/libxrt_driver_xdna.so",
                "/opt/xilinx/xrt/lib/libxrt_driver_xdna.so"
            ]
            
            for path in driver_paths:
                if os.path.exists(path):
                    try:
                        self.xdna_driver = ctypes.CDLL(path)
                        logger.info(f"✅ Loaded NPU driver: {path}")
                        return True
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to load {path}: {e}")
                        continue
            
            logger.error("❌ No NPU driver found")
            return False
            
        except Exception as e:
            logger.error(f"❌ NPU driver loading failed: {e}")
            return False

    def _initialize_npu_context(self) -> bool:
        """Initialize NPU execution context"""
        try:
            # Check NPU availability through multiple methods
            npu_interfaces = []
            
            # Method 1: Check /dev/accel device
            if os.path.exists("/dev/accel/accel0"):
                npu_interfaces.append("/dev/accel/accel0")
                logger.info("✅ NPU device interface: /dev/accel/accel0")
            
            # Method 2: Check AMDXDNA driver module
            if os.path.exists("/sys/module/amdxdna"):
                npu_interfaces.append("/sys/module/amdxdna")
                logger.info("✅ AMDXDNA driver module loaded")
            
            # Method 3: Check NPU PCI device
            npu_pci_path = "/sys/devices/pci0000:00/0000:00:08.2/0000:c7:00.1"
            if os.path.exists(npu_pci_path):
                npu_interfaces.append(npu_pci_path)
                logger.info(f"✅ NPU PCI device interface: {npu_pci_path}")
            
            # Method 4: Check for XRT runtime interfaces
            xrt_paths = [
                "/sys/kernel/tracing/events/amdxdna_trace",
                "/sys/bus/pci/drivers/amdxdna"
            ]
            
            for path in xrt_paths:
                if os.path.exists(path):
                    npu_interfaces.append(path)
                    logger.info(f"✅ XRT NPU interface: {path}")
            
            if not npu_interfaces:
                logger.error("❌ No NPU interfaces available")
                return False
            
            # Initialize context using the available interfaces
            self.npu_device = "/dev/accel/accel0"  # Primary NPU device
            self.npu_context = {
                "device": self.npu_device,
                "interfaces": npu_interfaces,
                "driver": "amdxdna",
                "initialized": True
            }
            logger.info(f"✅ NPU context initialized with {len(npu_interfaces)} interfaces")
            return True
            
        except Exception as e:
            logger.error(f"❌ NPU context initialization failed: {e}")
            return False

    def _load_attention_kernel(self) -> bool:
        """Load attention computation kernel"""
        try:
            logger.info("⚡ Loading Flash Attention kernel for NPU...")
            
            # Check environment for kernel path
            kernel_path = os.environ.get('NPU_KERNEL_PATH')
            
            if not kernel_path:
                # Try to find best kernel
                kernel_dir = "/home/ucadmin/Development/Unicorn-Execution-Engine/npu_kernels"
                
                # Look for exact match first
                kernel_path = f"{kernel_dir}/attention_{self.seq_length}_int8.bin"
                
                if not os.path.exists(kernel_path):
                    # Try flash attention
                    kernel_path = f"{kernel_dir}/gemma-3n-e4b-attention/flash_attention_kernel.bin"
                    
            if os.path.exists(kernel_path):
                # Load the kernel binary
                with open(kernel_path, 'rb') as f:
                    self.kernel_binary = f.read()
                    
                self.kernel_loaded = True
                logger.info(f"✅ Loaded NPU kernel: {os.path.basename(kernel_path)} ({len(self.kernel_binary)} bytes)")
                
                # Load config if available
                import json
                config_path = os.path.dirname(kernel_path) + "/kernel_configs.json"
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        self.kernel_config = json.load(f)
                        logger.info("✅ Loaded kernel configuration")
                
                return True
            else:
                logger.error(f"❌ NPU kernel not found at {kernel_path}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Attention kernel loading failed: {e}")
            return False

    def compute_flash_attention(self, hidden_states: np.ndarray, q_proj_weight: np.ndarray, 
                               k_proj_weight: np.ndarray, v_proj_weight: np.ndarray, 
                               o_proj_weight: np.ndarray, kv_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Flash Attention on real NPU hardware
        """
        if not self.initialized:
            raise RuntimeError("Real NPU Kernel not initialized")

        logger.info(f"🔥 Computing Flash Attention on REAL NPU Hardware: {hidden_states.shape}")
        
        # This is where real NPU execution would happen
        # Since we don't have the full NPU kernel compiled, we must fail gracefully
        
        # Check if we can actually execute on NPU
        if not self._can_execute_on_npu():
            raise RuntimeError("NPU execution not available - no compiled kernel")
            
        # Real NPU execution would go here
        # For now, we must fail since we don't have real implementation
        raise NotImplementedError("Real NPU kernel execution not yet implemented. Need compiled MLIR-AIE2 kernel.")

    def _can_execute_on_npu(self) -> bool:
        """Check if we can actually execute on NPU"""
        # Real check would verify:
        # - NPU device is available
        # - Kernel is compiled and loaded
        # - Memory is allocated
        # - Context is ready
        
        if not self.initialized:
            return False
            
        if not self.kernel_loaded or not self.kernel_binary:
            return False
            
        # Check if NPU device and context are ready
        if not self.npu_device or not self.npu_context:
            return False
            
        # For now, we have the kernel binary but need XRT runtime
        # to actually execute it. This requires the full XRT API.
        return self.kernel_loaded

    def cleanup(self):
        """Clean up NPU resources"""
        logger.info("🧹 Cleaning up Real NPU Hardware resources...")
        
        if self.npu_context:
            # Clean up NPU context
            self.npu_context = None
            
        if self.npu_device:
            # Clean up NPU device
            self.npu_device = None
            
        if self.xdna_driver:
            # Clean up driver
            self.xdna_driver = None
            
        self.initialized = False
        logger.info("✅ NPU cleanup complete")
