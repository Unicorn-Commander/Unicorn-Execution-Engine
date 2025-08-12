#!/usr/bin/env python3
"""
Production-ready NPU-accelerated chat interface for llama.cpp
"""

import subprocess
import sys
import os
import argparse

def check_npu_available():
    """Check if NPU device is available"""
    return os.path.exists("/dev/accel/accel0")

def run_npu_chat(model_path, ctx_size=2048, npu_enabled=True):
    """Run interactive chat with NPU acceleration"""
    print("🦄 Unicorn NPU Chat Interface")
    print("=" * 50)
    
    # Check NPU availability
    if npu_enabled and check_npu_available():
        print("✅ NPU Hardware: DETECTED (Phoenix NPU)")
        print("🚀 Acceleration: ENABLED")
        npu_flag = "--npu-attention"
    else:
        print("⚠️  NPU Hardware: NOT AVAILABLE")
        print("🖥️  Fallback: CPU mode")
        npu_flag = ""
    
    print(f"📊 Model: {os.path.basename(model_path)}")
    print(f"💾 Context: {ctx_size} tokens")
    print("=" * 50)
    print("\n💬 Start chatting! (Type 'exit' to quit)\n")
    
    # Build command
    cmd = [
        "./llama.cpp/build/bin/llama-cli",
        "-m", model_path,
        "--interactive",
        "--interactive-first",
        "--ctx-size", str(ctx_size),
        "--n-predict", "-1",
        "--color",
        "--multiline-input",
        "--conversation",
        "--chatml"
    ]
    
    if npu_flag:
        cmd.append(npu_flag)
    
    # Run interactive chat
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n👋 Thanks for using Unicorn NPU Chat!")

def main():
    parser = argparse.ArgumentParser(description="NPU-accelerated LLM chat interface")
    parser.add_argument("model", help="Path to GGUF model file")
    parser.add_argument("--ctx-size", type=int, default=2048, help="Context size (default: 2048)")
    parser.add_argument("--no-npu", action="store_true", help="Disable NPU acceleration")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found: {args.model}")
        sys.exit(1)
    
    if args.benchmark:
        print("📊 Running NPU performance benchmark...")
        subprocess.run(["bash", "./llama.cpp/benchmark_npu_performance.sh"])
    else:
        run_npu_chat(args.model, args.ctx_size, not args.no_npu)

if __name__ == "__main__":
    main()