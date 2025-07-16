# 🚀 Background Quantization Commands

## Option 1: Run Both Models in Parallel (Recommended)

### Terminal 1 - Gemma 3 4B (5-10 minutes)
```bash
cd /home/ucadmin/Development/github_repos/Unicorn-Execution-Engine
python run_gemma3_4b_quantization.py
```

### Terminal 2 - Gemma 3 27B (15-25 minutes)
```bash
cd /home/ucadmin/Development/github_repos/Unicorn-Execution-Engine
python run_gemma3_27b_quantization.py
```

## Option 2: Background Processing with nohup

### Start Both in Background
```bash
cd /home/ucadmin/Development/github_repos/Unicorn-Execution-Engine

# Start 4B model in background
nohup python run_gemma3_4b_quantization.py > gemma3_4b_quantization.log 2>&1 &
echo "4B PID: $!" > quantization_pids.txt

# Start 27B model in background  
nohup python run_gemma3_27b_quantization.py > gemma3_27b_quantization.log 2>&1 &
echo "27B PID: $!" >> quantization_pids.txt

echo "Both models started in background!"
echo "Check progress with: tail -f gemma3_4b_quantization.log"
echo "Check 27B progress with: tail -f gemma3_27b_quantization.log"
```

### Monitor Progress
```bash
# Check 4B progress
tail -f gemma3_4b_quantization.log

# Check 27B progress (in another terminal)
tail -f gemma3_27b_quantization.log

# Check if processes are running
ps aux | grep python | grep quantization
```

## Option 3: Using tmux/screen

### With tmux
```bash
# Start tmux session
tmux new-session -d -s quantization

# Create windows for each model
tmux new-window -t quantization -n "4b" "cd /home/ucadmin/Development/github_repos/Unicorn-Execution-Engine && python run_gemma3_4b_quantization.py"
tmux new-window -t quantization -n "27b" "cd /home/ucadmin/Development/github_repos/Unicorn-Execution-Engine && python run_gemma3_27b_quantization.py"

# Attach to session
tmux attach-session -t quantization

# Switch between windows: Ctrl+b then 1 or 2
# Detach: Ctrl+b then d
```

## Option 4: Claude Code with Long Timeout

If you want me to run them, I can use very long timeouts:

```bash
# 4B model (should complete in 10 minutes)
timeout 1200s python run_gemma3_4b_quantization.py

# 27B model (should complete in 25 minutes)  
timeout 1800s python run_gemma3_27b_quantization.py
```

## Expected Outputs

### Gemma 3 4B Success
```
🦄 SUCCESS! Gemma 3 4B-IT quantized and ready!
📁 Location: ./quantized_models/gemma-3-4b-it-multimodal
⏱️ Time: 8.5 minutes
🎮 Test with: python terminal_chat.py --model ./quantized_models/gemma-3-4b-it-multimodal
```

### Gemma 3 27B Success
```
🦄 MULTIMODAL SUCCESS! Gemma 3 27B ready!
📁 Location: ./quantized_models/gemma-3-27b-it-multimodal
⏱️ Time: 18.3 minutes
🧪 Tests: 3/3 passed
🎮 Test with: python terminal_chat.py --model ./quantized_models/gemma-3-27b-it-multimodal
```

## Check Status

```bash
# Check what's in quantized_models directory
ls -la quantized_models/

# Check if models are complete (look for model_info.json)
ls -la quantized_models/gemma-3-4b-it-multimodal/
ls -la quantized_models/gemma-3-27b-it-multimodal/

# Test completed models
python terminal_chat.py --model ./quantized_models/gemma-3-4b-it-multimodal
```

Choose whichever method works best for your setup!