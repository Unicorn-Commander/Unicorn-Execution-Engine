# 🤖 PROMPT FOR GEMINI CLI (or other AI)

## **Task: Eliminate Transpose Operations for Fast Model Loading**

### **Context**
The Unicorn Execution Engine is a high-performance inference system for a 27B parameter model. Currently, model loading takes 2+ minutes because every tensor needs to be transposed from storage format to compute format. This happens in CPU and is single-threaded per tensor.

### **Your Mission**
Create a one-time model conversion script that pre-transposes all weights, so loading can be fast (like Ollama, <30 seconds).

### **Current Problem**
In `pure_hardware_pipeline_fixed.py`, the `_load_tensor_to_gpu()` function transposes weights:
```python
# This is SLOW - happens for every tensor during loading
if "proj.weight" in name:
    tensor_data = tensor_data.T  # Transpose operation
```

### **Solution Approach**
1. Create `convert_model_format.py` that:
   - Loads the original model once
   - Transposes all necessary weights
   - Saves in new format (e.g., `gemma-27b-pretransposed/`)
   
2. Modify `pure_hardware_pipeline_fixed.py` to:
   - Check if pretransposed model exists
   - Skip transpose operations if using converted model

### **Files to Work With**
- Create: `/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/convert_model_format.py`
- Modify: `/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/pure_hardware_pipeline_fixed.py`
- Model location: `/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/quantized_models/gemma-3-27b-it-layer-by-layer/`

### **Success Criteria**
- One-time conversion script works (may take time, that's OK)
- Loading converted model takes <30 seconds
- No transpose operations during loading
- Model still works correctly for inference

### **Testing**
```bash
cd /home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/
source /home/ucadmin/activate-pure-hardware-env.sh

# Convert model (one time)
python convert_model_format.py

# Test loading speed
python test_quick_performance.py
```

### **Hints**
- Use safetensors library for reading/writing
- Keep same file structure (layer-by-layer)
- Preserve INT8 quantization
- Use multiple threads for conversion
- Add progress bar for conversion

---

## **Alternative Task: Implement Proper Tokenization**

If the transpose task is too complex, you could instead work on:

### **Context**
Currently using dummy tokenization: `[ord(c) % 1000 for c in prompt]`

### **Your Mission**
Implement real Gemma tokenizer for proper text encoding/decoding.

### **Requirements**
1. Load actual Gemma tokenizer files
2. Implement `encode(text) -> token_ids`
3. Implement `decode(token_ids) -> text`
4. Handle special tokens (<bos>, <eos>, etc.)

### **Test Prompt**
"Magic Unicorn Unconventional Technology & Stuff is a groundbreaking Applied AI company that"

Should tokenize to proper Gemma tokens and generate coherent continuation.

---

Choose whichever task you're most comfortable with!