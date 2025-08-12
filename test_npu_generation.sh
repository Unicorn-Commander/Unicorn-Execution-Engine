#!/bin/bash
cd /home/ucadmin/Development/Unicorn-Execution-Engine
./llama.cpp/build/bin/llama-cli -m gemma-3n-E4B-it-Q8_0.gguf -p "The magic unicorn represents" -n 10 --npu-attention