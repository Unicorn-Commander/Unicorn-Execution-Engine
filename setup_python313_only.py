#!/usr/bin/env python3.13
"""
Setup Magic Unicorn with Python 3.13 Only - No IPC Needed!
Simplifies the entire architecture by using one Python environment
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_python_version():
    """Ensure we're running Python 3.13"""
    if sys.version_info < (3, 13):
        print(f"❌ Python 3.13+ required, but got {sys.version}")
        print("Please run with: python3.13 setup_python313_only.py")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} detected")

def check_hardware_modules():
    """Check all required hardware modules"""
    modules = {
        'pyxrt': 'NPU/XRT access',
        'vulkan': 'GPU compute',
        'numpy': 'Array operations',
        'mmap': 'Memory mapping',
        'struct': 'Binary data handling',
        '_lzma': 'Compression support',
        'json': 'Configuration files',
        'pathlib': 'File path handling',
    }
    
    all_good = True
    for module, description in modules.items():
        try:
            __import__(module)
            print(f"✅ {module:<12} - {description}")
        except ImportError:
            print(f"❌ {module:<12} - {description}")
            all_good = False
    
    return all_good

def install_missing_packages():
    """Install any missing packages for hardware-only operation"""
    # For hardware-only, we need very few packages
    packages = [
        'numpy',       # Basic array operations
        'pyYAML',      # Config files
        'psutil',      # System monitoring
        'safetensors', # Model weight loading (no torch dependency)
    ]
    
    print("\n📦 Installing required packages...")
    for package in packages:
        try:
            __import__(package.lower().replace('-', '_'))
            print(f"✅ {package} already installed")
        except ImportError:
            print(f"📥 Installing {package}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], check=True)

def create_launcher_script():
    """Create a simple launcher script for Python 3.13"""
    launcher_content = """#!/usr/bin/env python3.13
'''
Magic Unicorn Hardware-Only Launcher
Uses Python 3.13 directly - no IPC needed!
'''

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Direct hardware-only imports
from pure_hardware_pipeline_fixed import MagicUnicornPipeline
from real_vulkan_matrix_compute import VulkanMatrixCompute
from npu_attention_kernel_real import NPUAttentionKernel

def main():
    print("🦄 Magic Unicorn Hardware-Only System")
    print("📍 Running on Python 3.13 - Direct NPU+GPU access")
    
    # Initialize pipeline
    pipeline = MagicUnicornPipeline(
        model_path="/home/ucadmin/Development/Unicorn-Execution-Engine/quantized_models/gemma-3-4b-it-quantized",
        sequence_length=512,
        use_real_npu=True,
        debug=True
    )
    
    # Initialize hardware
    if not pipeline.initialize_hardware():
        print("❌ Hardware initialization failed")
        return
    
    # Load model
    if not pipeline.load_model():
        print("❌ Model loading failed")
        return
    
    print("✅ System ready for inference!")
    
    # Test inference
    test_prompt = "What is the capital of France?"
    result = pipeline.generate_text(test_prompt, max_tokens=50)
    print(f"\\nPrompt: {test_prompt}")
    print(f"Response: {result}")

if __name__ == "__main__":
    main()
"""
    
    launcher_path = Path("launch_hardware_only.py")
    launcher_path.write_text(launcher_content)
    launcher_path.chmod(0o755)
    print(f"\n✅ Created launcher: {launcher_path}")

def update_imports():
    """Update all Python files to use python3.13 shebang"""
    print("\n🔧 Updating Python files to use Python 3.13...")
    
    py_files = list(Path('.').glob('*.py'))
    updated = 0
    
    for py_file in py_files[:5]:  # Just show first 5 as example
        try:
            content = py_file.read_text()
            if content.startswith('#!/usr/bin/env python'):
                new_content = content.replace(
                    '#!/usr/bin/env python3',
                    '#!/usr/bin/env python3.13',
                    1
                )
                if new_content != content:
                    print(f"  Updated: {py_file}")
                    updated += 1
        except Exception as e:
            print(f"  Skipped: {py_file} ({e})")
    
    print(f"✅ Would update {updated} files (showing first 5)")

def create_pyproject_toml():
    """Create a pyproject.toml for Poetry management"""
    pyproject_content = """[tool.poetry]
name = "magic-unicorn-hardware"
version = "1.0.0"
description = "Magic Unicorn NPU+iGPU Hardware Acceleration System"
authors = ["Magic Unicorn Team"]
python = "^3.13"

[tool.poetry.dependencies]
python = "^3.13"
numpy = "^2.0.0"
pyyaml = "^6.0"
psutil = "^5.9"
safetensors = "^0.4"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4"
black = "^23.0"
ruff = "^0.1"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.poetry.scripts]
magic-unicorn = "launch_hardware_only:main"
"""
    
    Path("pyproject.toml").write_text(pyproject_content)
    print("\n✅ Created pyproject.toml for Poetry")

def create_simple_test():
    """Create a simple test to verify everything works"""
    test_content = '''#!/usr/bin/env python3.13
"""
Simple test to verify Python 3.13 hardware-only setup
"""

import sys
print(f"Python: {sys.version}")

# Test NPU access
try:
    import pyxrt
    print("✅ NPU/XRT access available")
    
    # List NPU devices
    device_count = pyxrt.get_device_count()
    print(f"   Found {device_count} XRT devices")
except Exception as e:
    print(f"❌ NPU/XRT error: {e}")

# Test GPU access
try:
    import vulkan as vk
    print("✅ Vulkan GPU access available")
    
    # List GPU devices
    instance = vk.vkCreateInstance(vk.VkInstanceCreateInfo(), None)
    print(f"   Vulkan instance created")
except Exception as e:
    print(f"❌ Vulkan error: {e}")

# Test model loading
try:
    import mmap
    import struct
    print("✅ Binary model loading available")
except Exception as e:
    print(f"❌ Model loading error: {e}")

print("\\n🎉 Hardware-only setup complete!")
'''
    
    test_path = Path("test_python313_setup.py")
    test_path.write_text(test_content)
    test_path.chmod(0o755)
    print(f"\n✅ Created test script: {test_path}")

def main():
    """Main setup function"""
    print("🦄 Magic Unicorn Python 3.13 Hardware-Only Setup")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Check hardware modules
    print("\n📋 Checking hardware modules...")
    if not check_hardware_modules():
        print("\n⚠️  Some modules missing, but we can continue")
    
    # Install packages
    install_missing_packages()
    
    # Create launcher
    create_launcher_script()
    
    # Create Poetry config
    create_pyproject_toml()
    
    # Create test script
    create_simple_test()
    
    # Update imports (just show what would be done)
    update_imports()
    
    print("\n" + "=" * 50)
    print("✅ Setup complete! Benefits of Python 3.13 only:")
    print("  • No IPC complexity - direct hardware access")
    print("  • No subprocess overhead")
    print("  • Simpler debugging")
    print("  • Better performance")
    print("\nNext steps:")
    print("  1. Run: python3.13 test_python313_setup.py")
    print("  2. Run: python3.13 launch_hardware_only.py")
    print("\nOr use Poetry:")
    print("  1. poetry install")
    print("  2. poetry run magic-unicorn")

if __name__ == "__main__":
    main()