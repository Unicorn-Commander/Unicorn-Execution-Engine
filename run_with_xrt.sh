#!/bin/bash
export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH
export LD_PRELOAD="/opt/xilinx/xrt/lib/libxrt_core.so:/opt/xilinx/xrt/lib/libxrt++.so"

# Find any llama binary
LLAMA=$(find . -name "llama-cli" -type f -executable | head -1)
if [ -z "$LLAMA" ]; then
    LLAMA=$(find . -name "llama-simple" -type f -executable | head -1)
fi

if [ -n "$LLAMA" ]; then
    echo "🚀 Running $LLAMA with XRT libraries preloaded"
    exec "$LLAMA" "$@"
else
    echo "❌ No llama binary found"
fi
