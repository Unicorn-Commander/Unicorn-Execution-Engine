# Project Issues Log

## Current Hang-ups:

*   **XRT Build Process:** The `build.sh` script for XRT is not directly executable by the agent due to security restrictions. This prevents automated building of XRT from source.

## Resolved Issues:

*   **Conflicting Dockerfile Approaches:** Previously, there was confusion regarding the correct Dockerfile to use, leading to attempts to build from a root Dockerfile that referenced unnecessary Vitis AI components and a missing script. This has been clarified:
    *   The correct Dockerfile is located at `NPU-Development/Dockerfile`.
    *   The `install_npu_stack.sh` script has been added to the project.
    *   It has been confirmed that no Vitis AI downloads are needed for the NPU stack, only the XDNA driver, XRT runtime, and MLIR-AIE framework.

## Change in Project Plan:

The project plan has shifted from a Docker-based setup to a native installation with Python virtual environments and source builds for XRT, MLIR-AIE, and ONNX Runtime. This aims for maximum performance and clean isolation.

## Progress on New Plan (Phase 1 & 2):

*   **Base System Preparation:** System dependencies have been installed.
*   **Python Virtual Environment:** A virtual environment (`~/gemma-npu-env`) has been created and base tools upgraded.
*   **XRT from Source:** XRT has been cloned, compilers set, and dependencies installed. The next step is to build XRT, which is currently blocked.