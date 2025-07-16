#!/bin/bash
# Fix NPU driver loading by setting library paths

echo "🔧 Fixing NPU driver environment..."

# Find XRT core libraries
XRT_LIB_PATH="/usr/local/xrt/lib"
if [ -d "$XRT_LIB_PATH" ]; then
    export LD_LIBRARY_PATH="$XRT_LIB_PATH:$LD_LIBRARY_PATH"
    echo "✅ Added $XRT_LIB_PATH to LD_LIBRARY_PATH"
fi

# Also check /opt/xilinx/xrt/lib
ALT_XRT_PATH="/opt/xilinx/xrt/lib"
if [ -d "$ALT_XRT_PATH" ]; then
    export LD_LIBRARY_PATH="$ALT_XRT_PATH:$LD_LIBRARY_PATH"
    echo "✅ Added $ALT_XRT_PATH to LD_LIBRARY_PATH"
fi

# Verify NPU device exists
if [ -e "/dev/accel/accel0" ]; then
    echo "✅ NPU device found at /dev/accel/accel0"
else
    echo "❌ NPU device not found!"
fi

# Check if libraries can be loaded now
echo "🔍 Checking XRT driver dependencies..."
ldd /usr/local/xrt/lib/libxrt_driver_xdna.so 2>&1 | grep -E "(not found|=>)" | head -5

echo "🚀 NPU environment ready!"