#!/bin/bash
# Install PyTorch with ROCm support for AMD GPUs

echo "Installing PyTorch with ROCm support..."

# Option 1: Install via pip (recommended for ROCm 6.x)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Alternative for different ROCm versions:
# ROCm 6.1: https://download.pytorch.org/whl/rocm6.1
# ROCm 6.0: https://download.pytorch.org/whl/rocm6.0
# ROCm 5.7: https://download.pytorch.org/whl/rocm5.7

echo "Testing PyTorch ROCm installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'ROCm available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
"