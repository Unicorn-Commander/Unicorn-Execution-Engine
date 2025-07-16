#!/bin/bash
# Activate MLIR-AIE2 ironenv environment
source /home/ucadmin/mlir-aie2/ironenv/bin/activate

# Add MLIR-AIE2 Python bindings to path  
export PYTHONPATH="/home/ucladmin/mlir-aie2/ironenv/lib/python3.12/site-packages:$PYTHONPATH"

echo "✅ MLIR-AIE2 ironenv activated"
echo "🔧 PYTHONPATH: $PYTHONPATH"