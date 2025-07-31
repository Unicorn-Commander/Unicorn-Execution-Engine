#!/usr/bin/env python3.13
import mlir_aie
from mlir_aie import runtime as aie_runtime
import numpy as np
import subprocess

# Test if we can use MLIR-AIE
print("✅ MLIR-AIE imported successfully")

# Check available tools
result = subprocess.run(['which', 'aie-opt'], capture_output=True, text=True)
if result.returncode == 0:
    print(f"✅ aie-opt found: {result.stdout.strip()}")
else:
    print("❌ aie-opt not found")

result = subprocess.run(['which', 'aie-translate'], capture_output=True, text=True)
if result.returncode == 0:
    print(f"✅ aie-translate found: {result.stdout.strip()}")
else:
    print("❌ aie-translate not found")