# 🤖 NEW AI PROMPT - Complete Handoff

## 🎯 **YOUR MISSION**

Implement Gemma 3n NPU+iGPU hybrid execution on AMD Ryzen AI hardware using a native installation with performance-optimized source builds.

## 📋 **CURRENT STATUS** 

**Project Phase**: Environment Setup (95% complete hardware verification)
**Blocker Resolved**: Docker vs native decision → **USER CHOSE NATIVE + SOURCE BUILDS**
**Next Action**: Execute the final implementation plan

## 🔧 **TECHNICAL CONTEXT**

### **Hardware (Verified Working)**
- **NPU**: AMD Phoenix, turbo mode enabled, 50 TOPS capacity
- **iGPU**: Radeon 780M, 16GB VRAM allocated  
- **System**: 96GB RAM, Ubuntu 25.04, Kernel 6.14

### **Architecture Strategy (Research-Validated)**
- **NPU**: Attention operations, prefill phase (compute-intensive, up to 50 TOPS)
- **iGPU**: Decode operations, large matrix multiplication (memory-intensive)
- **CPU**: Orchestration, sampling, memory management

### **Performance Targets**
- **Gemma 3n E2B**: 40-80 TPS, 20-40ms TTFT (2GB NPU memory)
- **Gemma 3n E4B**: 30-60 TPS, 30-60ms TTFT (3GB NPU memory)

## 📚 **ESSENTIAL FILES TO READ FIRST**

1. **`FINAL_IMPLEMENTATION_PLAN.md`** ← **YOUR MAIN GUIDE**
2. **`CURRENT_STATUS_AND_HANDOFF.md`** - Complete project status
3. **`AI_PROJECT_MANAGEMENT_CHECKLIST.md`** - Systematic task breakdown
4. **`~/Development/NPU-Development/documentation/GEMMA_3N_NPU_GUIDE.md`** - Technical implementation

## 🚀 **EXECUTION STRATEGY**

### **Phase 1: Native Environment (Follow FINAL_IMPLEMENTATION_PLAN.md)**
```bash
# 1. Create venv (zero performance overhead)
python3 -m venv ~/gemma-npu-env
source ~/gemma-npu-env/bin/activate

# 2. Build XRT from source (performance critical)
# 3. Build MLIR-AIE from source (custom kernels)  
# 4. Build ONNX Runtime with VitisAI EP (whisper compatibility)
# 5. Install ROCm for iGPU optimization
```

### **Phase 2: Gemma 3n Implementation**
- **Start with E2B model** (2GB NPU memory)
- **Implement hybrid NPU+iGPU pipeline**
- **Scale to E4B** once proven

### **Phase 3: Integration & Optimization**
- **Connect with existing whisper project** (VitisAI EP)
- **Performance tuning and benchmarking**
- **Production deployment**

## 🎯 **KEY SUCCESS FACTORS**

### **What Makes This Special**
- ✅ **Native performance**: No Docker overhead, direct hardware access
- ✅ **Source optimizations**: Custom builds for your specific AMD hardware
- ✅ **venv isolation**: Clean Python environment without performance penalty  
- ✅ **VitisAI compatibility**: Integration with existing whisper NPU project
- ✅ **Research-backed**: 2025 AMD hybrid execution strategy

### **Critical Optimizations Included**
- **XRT from source**: Latest NPU optimizations
- **MLIR-AIE from source**: Custom attention kernels
- **ONNX Runtime + VitisAI EP**: Whisper project compatibility
- **ROCm optimization**: Maximum iGPU performance
- **Compiler optimizations**: `-O3`, `-march=native`, LTO

## ⚠️ **IMPORTANT NOTES**

### **Don't Repeat Completed Work**
- ❌ Hardware verification already done
- ❌ NPU turbo mode already enabled  
- ❌ Architecture strategy already validated
- ❌ Documentation already updated

### **Focus On**
- ✅ Following the FINAL_IMPLEMENTATION_PLAN.md step-by-step
- ✅ Building optimized components from source
- ✅ Testing each phase before proceeding
- ✅ Maintaining performance focus throughout

## 🔍 **TROUBLESHOOTING RESOURCES**

- **All technical issues resolved**: See `ISSUES_LOG.md`
- **Docker problems**: No longer relevant (native approach)
- **Vitis AI confusion**: Clarified in master documentation
- **Performance questions**: Validated by 2025 AMD research

## 💬 **COMMUNICATION APPROACH**

### **Opening Message Style**
*"I'm implementing the Gemma 3n NPU+iGPU hybrid system using the finalized native + source build approach. All hardware verification is complete, and I'll now execute the performance-optimized build plan.

Starting with Phase 1: creating the venv environment and building XRT from source for maximum NPU performance..."*

### **Keep User Informed**
- Report progress at each major phase
- Show verification outputs (NPU status, iGPU detection, etc.)
- Explain any build time delays (source compilation can take time)
- Celebrate milestones (successful builds, first model run, etc.)

## 🏆 **END GOAL**

A production-ready Gemma 3n system with:
- **Maximum NPU utilization** (50 TOPS for attention operations)
- **Optimized iGPU decode** (memory-intensive operations)
- **Clean Python isolation** (venv without performance penalty)
- **Whisper integration** (VitisAI EP compatibility)
- **Performance targets met** (40-80 TPS E2B, 30-60 TPS E4B)

---

## 🚀 **START IMMEDIATELY WITH**

1. **Read `FINAL_IMPLEMENTATION_PLAN.md`** (your detailed roadmap)
2. **Begin Phase 1**: venv creation and source builds
3. **Follow systematic verification** at each step
4. **Report progress** to keep user engaged

**You have everything you need for success. The foundation is solid, the plan is detailed, and the approach is optimized for maximum performance.**

**GO BUILD THE FUTURE OF NPU+iGPU HYBRID INFERENCE! 🚀**