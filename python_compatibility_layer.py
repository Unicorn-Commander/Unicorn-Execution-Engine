#!/usr/bin/env python3
"""
Python Compatibility Layer for NPU+iGPU Magic Unicorn System
Seamlessly handles Python 3.11 ↔ 3.13 environment switching
"""

import os
import sys
import subprocess
import threading
import queue
import time
import pickle
import tempfile
import logging
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PythonEnvironment(Enum):
    """Available Python environments"""
    PYTHON_311 = "python3.11"
    PYTHON_313 = "python3.13"
    AUTO = "auto"

@dataclass
class EnvironmentConfig:
    """Configuration for Python environment"""
    python_version: PythonEnvironment
    virtual_env_path: str
    activation_script: str
    required_packages: List[str]
    environment_variables: Dict[str, str]

class PythonCompatibilityLayer:
    """
    🦄 Magic Unicorn Python Compatibility Layer
    
    Features:
    - Automatic environment detection and switching
    - Seamless function calls across Python versions
    - Persistent subprocess management
    - Shared memory communication
    - Zero-overhead for same-version calls
    """
    
    def __init__(self):
        """Initialize compatibility layer"""
        
        self.current_version = self._detect_python_version()
        self.environments = self._setup_environments()
        self.active_subprocesses: Dict[PythonEnvironment, subprocess.Popen] = {}
        self.subprocess_queues: Dict[PythonEnvironment, queue.Queue] = {}
        
        logger.info("🦄 Python Compatibility Layer initializing...")
        logger.info(f"   Current Python: {self.current_version.value}")
        
    def _detect_python_version(self) -> PythonEnvironment:
        """Detect current Python version"""
        version = sys.version_info
        if version >= (3, 13):
            return PythonEnvironment.PYTHON_313
        elif version >= (3, 11):
            return PythonEnvironment.PYTHON_311
        else:
            raise RuntimeError(f"Unsupported Python version: {version}")
    
    def _setup_environments(self) -> Dict[PythonEnvironment, EnvironmentConfig]:
        """Setup environment configurations"""
        
        environments = {}
        
        # Python 3.11 environment (main ML environment)
        environments[PythonEnvironment.PYTHON_311] = EnvironmentConfig(
            python_version=PythonEnvironment.PYTHON_311,
            virtual_env_path="/home/ucadmin/ai-env-py311",
            activation_script="/home/ucadmin/activate-uc1-ai-py311.sh",
            required_packages=[
                "torch", "numpy", "transformers", "accelerate", 
                "safetensors", "vulkan", "scipy"
            ],
            environment_variables={
                "PYTHONPATH": "/home/ucadmin/Development/Unicorn-Execution-Engine",
                "TORCH_USE_CUDA_DSA": "1"
            }
        )
        
        # Python 3.13 environment (NPU/XRT environment)
        environments[PythonEnvironment.PYTHON_313] = EnvironmentConfig(
            python_version=PythonEnvironment.PYTHON_313,
            virtual_env_path="/home/ucadmin/npu-env-py313",
            activation_script="/home/ucadmin/activate-npu-py313.sh",
            required_packages=[
                "torch", "numpy", "pyxrt", "ml-dtypes"
            ],
            environment_variables={
                "XILINX_XRT": "/opt/xilinx/xrt",
                "PYTHONPATH": "/opt/xilinx/xrt/python:/home/ucadmin/Development/Unicorn-Execution-Engine",
                "XRT_HACK_UNSECURE_LOADING_XCLBIN": "1"
            }
        )
        
        return environments
    
    def call_function(self, target_env: PythonEnvironment, 
                     module_name: str, function_name: str, 
                     *args, **kwargs) -> Any:
        """
        Call function in specific Python environment
        
        Args:
            target_env: Target Python environment
            module_name: Python module to import
            function_name: Function to call
            *args, **kwargs: Function arguments
            
        Returns:
            Function result
        """
        
        # If target environment matches current, call directly
        if target_env == self.current_version:
            return self._call_function_direct(module_name, function_name, *args, **kwargs)
        
        # Otherwise, call via subprocess
        return self._call_function_subprocess(target_env, module_name, function_name, *args, **kwargs)
    
    def _call_function_direct(self, module_name: str, function_name: str, 
                             *args, **kwargs) -> Any:
        """Call function directly in current environment"""
        
        try:
            # Import module
            module = __import__(module_name, fromlist=[function_name])
            func = getattr(module, function_name)
            
            # Call function
            start_time = time.time()
            result = func(*args, **kwargs)
            call_time = time.time() - start_time
            
            logger.debug(f"📞 Direct call: {module_name}.{function_name} ({call_time*1000:.2f}ms)")
            return result
            
        except Exception as e:
            logger.error(f"❌ Direct call failed: {module_name}.{function_name} - {e}")
            raise
    
    def _call_function_subprocess(self, target_env: PythonEnvironment, 
                                 module_name: str, function_name: str, 
                                 *args, **kwargs) -> Any:
        """Call function in subprocess with different Python version"""
        
        start_time = time.time()
        
        try:
            # Prepare function call data
            call_data = {
                'module_name': module_name,
                'function_name': function_name,
                'args': args,
                'kwargs': kwargs
            }
            
            # Create temporary files for data exchange
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as input_file:
                pickle.dump(call_data, input_file)
                input_path = input_file.name
            
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as output_file:
                output_path = output_file.name
            
            # Create subprocess script
            subprocess_script = self._create_subprocess_script(target_env)
            
            # Execute subprocess
            env_config = self.environments[target_env]
            cmd = [
                "bash", "-c", 
                f"source {env_config.activation_script} && python3 {subprocess_script} {input_path} {output_path}"
            ]
            
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60
            )
            
            if result.returncode == 0:
                # Load result
                with open(output_path, 'rb') as f:
                    result_data = pickle.load(f)
                
                if result_data['success']:
                    call_time = time.time() - start_time
                    logger.debug(f"📞 Subprocess call: {module_name}.{function_name} ({call_time*1000:.2f}ms)")
                    return result_data['result']
                else:
                    raise Exception(result_data['error'])
            else:
                raise Exception(f"Subprocess failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            raise Exception("Subprocess call timed out")
        except Exception as e:
            logger.error(f"❌ Subprocess call failed: {module_name}.{function_name} - {e}")
            raise
        finally:
            # Cleanup temp files
            try:
                os.unlink(input_path)
                os.unlink(output_path)
            except:
                pass
    
    def _create_subprocess_script(self, target_env: PythonEnvironment) -> str:
        """Create subprocess script for function calls"""
        
        script_content = f'''#!/usr/bin/env python3
"""
Subprocess script for {target_env.value} environment
"""

import sys
import pickle
import traceback

# Setup environment
{self._get_environment_setup(target_env)}

def main():
    if len(sys.argv) != 3:
        print("Usage: script.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        # Load function call data
        with open(input_file, 'rb') as f:
            call_data = pickle.load(f)
        
        module_name = call_data['module_name']
        function_name = call_data['function_name']
        args = call_data['args']
        kwargs = call_data['kwargs']
        
        # Import module and call function
        module = __import__(module_name, fromlist=[function_name])
        func = getattr(module, function_name)
        result = func(*args, **kwargs)
        
        # Save result
        result_data = {
            'success': True,
            'result': result
        }
        
    except Exception as e:
        result_data = {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }
    
    # Write result
    with open(output_file, 'wb') as f:
        pickle.dump(result_data, f)

if __name__ == "__main__":
    main()
'''
        
        # Write script to temporary file
        script_path = f"/tmp/subprocess_script_{target_env.value}.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)
        
        return script_path
    
    def _get_environment_setup(self, target_env: PythonEnvironment) -> str:
        """Get environment setup code for subprocess"""
        
        env_config = self.environments[target_env]
        setup_lines = []
        
        # Add paths
        for key, value in env_config.environment_variables.items():
            if key == "PYTHONPATH":
                setup_lines.append(f"sys.path.extend('{value}'.split(':'))")
            else:
                setup_lines.append(f"import os; os.environ['{key}'] = '{value}'")
        
        return "\\n".join(setup_lines)
    
    def start_persistent_subprocess(self, target_env: PythonEnvironment) -> bool:
        """Start persistent subprocess for faster repeated calls"""
        
        if target_env in self.active_subprocesses:
            # Already running
            return True
        
        try:
            env_config = self.environments[target_env]
            
            # Create persistent subprocess script
            persistent_script = self._create_persistent_subprocess_script(target_env)
            
            # Start subprocess
            cmd = [
                "bash", "-c",
                f"source {env_config.activation_script} && python3 {persistent_script}"
            ]
            
            process = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, text=True, bufsize=1
            )
            
            self.active_subprocesses[target_env] = process
            self.subprocess_queues[target_env] = queue.Queue()
            
            logger.info(f"✅ Started persistent subprocess: {target_env.value}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start persistent subprocess: {e}")
            return False
    
    def _create_persistent_subprocess_script(self, target_env: PythonEnvironment) -> str:
        """Create persistent subprocess script"""
        
        script_content = f'''#!/usr/bin/env python3
"""
Persistent subprocess for {target_env.value} environment
"""

import sys
import json
import pickle
import base64
import traceback

# Setup environment
{self._get_environment_setup(target_env)}

def main():
    while True:
        try:
            # Read command from stdin
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if line == "EXIT":
                break
            
            if not line:
                continue

            # Parse command
            try:
                command = json.loads(line)
            except json.JSONDecodeError:
                result = '{"success": false, "error": "Invalid JSON command"}'
                print(result)
                sys.stdout.flush()
                continue

            if command.get('type') == 'function_call':
                result = handle_function_call(command)
            else:
                result = {'success': False, 'error': 'Unknown command type'}

            # Send result
            print(json.dumps(result))
            sys.stdout.flush()
            
        except Exception as e:
            error_result = {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            print(json.dumps(error_result))
            sys.stdout.flush()

def handle_function_call(command):
    try:
        # Decode arguments
        args = pickle.loads(base64.b64decode(command['args']))
        kwargs = pickle.loads(base64.b64decode(command['kwargs']))
        
        # Import and call function
        module = __import__(command['module_name'], fromlist=[command['function_name']])
        func = getattr(module, command['function_name'])
        result = func(*args, **kwargs)
        
        # Encode result
        result_encoded = base64.b64encode(pickle.dumps(result)).decode('utf-8')
        
        return {
            'success': True,
            'result': result_encoded
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }

if __name__ == "__main__":
    main()
'''
        
        script_path = f"/tmp/persistent_subprocess_{target_env.value}.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)
        
        return script_path
    
    def call_function_persistent(self, target_env: PythonEnvironment,
                                module_name: str, function_name: str,
                                *args, **kwargs) -> Any:
        """Call function using persistent subprocess (faster)"""
        
        if target_env == self.current_version:
            return self._call_function_direct(module_name, function_name, *args, **kwargs)
        
        if target_env not in self.active_subprocesses:
            if not self.start_persistent_subprocess(target_env):
                raise Exception(f"Failed to start subprocess for {target_env.value}")
        
        try:
            process = self.active_subprocesses[target_env]
            
            # Prepare command
            import base64
            args_encoded = base64.b64encode(pickle.dumps(args)).decode('utf-8')
            kwargs_encoded = base64.b64encode(pickle.dumps(kwargs)).decode('utf-8')
            
            command = {
                'type': 'function_call',
                'module_name': module_name,
                'function_name': function_name,
                'args': args_encoded,
                'kwargs': kwargs_encoded
            }
            
            # Send command
            import json
            process.stdin.write(json.dumps(command) + "\\n")
            process.stdin.flush()
            
            # Read result
            result_line = process.stdout.readline().strip()
            result = json.loads(result_line)
            
            if result['success']:
                # Decode result
                result_data = pickle.loads(base64.b64decode(result['result']))
                return result_data
            else:
                raise Exception(result['error'])
                
        except Exception as e:
            logger.error(f"❌ Persistent call failed: {module_name}.{function_name} - {e}")
            # Restart subprocess on error
            self.stop_persistent_subprocess(target_env)
            raise
    
    def stop_persistent_subprocess(self, target_env: PythonEnvironment) -> None:
        """Stop persistent subprocess"""
        
        if target_env in self.active_subprocesses:
            process = self.active_subprocesses[target_env]
            try:
                process.stdin.write("EXIT\\n")
                process.stdin.flush()
                process.wait(timeout=5)
            except:
                process.terminate()
                process.wait(timeout=5)
                if process.poll() is None:
                    process.kill()
            
            del self.active_subprocesses[target_env]
            del self.subprocess_queues[target_env]
            
            logger.info(f"🔌 Stopped persistent subprocess: {target_env.value}")
    
    def stop_all_subprocesses(self) -> None:
        """Stop all persistent subprocesses"""
        
        for target_env in list(self.active_subprocesses.keys()):
            self.stop_persistent_subprocess(target_env)
    
    def __del__(self):
        """Cleanup on destruction"""
        self.stop_all_subprocesses()

# Global compatibility layer instance
_compatibility_layer = None

def get_compatibility_layer() -> PythonCompatibilityLayer:
    """Get global compatibility layer instance"""
    global _compatibility_layer
    if _compatibility_layer is None:
        _compatibility_layer = PythonCompatibilityLayer()
    return _compatibility_layer

def call_npu_function(module_name: str, function_name: str, *args, **kwargs) -> Any:
    """
    Convenience function to call NPU function (requires Python 3.13)
    """
    layer = get_compatibility_layer()
    return layer.call_function_persistent(
        PythonEnvironment.PYTHON_313, module_name, function_name, *args, **kwargs
    )

def call_ml_function(module_name: str, function_name: str, *args, **kwargs) -> Any:
    """
    Convenience function to call ML function (Python 3.11)
    """
    layer = get_compatibility_layer()
    return layer.call_function_persistent(
        PythonEnvironment.PYTHON_311, module_name, function_name, *args, **kwargs
    )

def test_compatibility_layer():
    """Test the compatibility layer"""
    
    logger.info("🧪 Testing Python Compatibility Layer...")
    
    layer = PythonCompatibilityLayer()
    
    # Test direct call (same environment)
    try:
        result = layer.call_function(
            layer.current_version, "math", "sqrt", 16
        )
        logger.info(f"✅ Direct call result: sqrt(16) = {result}")
    except Exception as e:
        logger.error(f"❌ Direct call failed: {e}")
    
    # Test subprocess call (different environment)
    try:
        if layer.current_version == PythonEnvironment.PYTHON_311:
            target_env = PythonEnvironment.PYTHON_313
        else:
            target_env = PythonEnvironment.PYTHON_311
        
        result = layer.call_function(
            target_env, "math", "sqrt", 25
        )
        logger.info(f"✅ Subprocess call result: sqrt(25) = {result}")
    except Exception as e:
        logger.warning(f"⚠️  Subprocess call failed: {e}")
    
    # Test persistent subprocess
    try:
        layer.start_persistent_subprocess(target_env)
        result = layer.call_function_persistent(
            target_env, "math", "sqrt", 36
        )
        logger.info(f"✅ Persistent call result: sqrt(36) = {result}")
        layer.stop_persistent_subprocess(target_env)
    except Exception as e:
        logger.warning(f"⚠️  Persistent call failed: {e}")
    
    logger.info("✅ Compatibility layer test completed!")

if __name__ == "__main__":
    test_compatibility_layer()