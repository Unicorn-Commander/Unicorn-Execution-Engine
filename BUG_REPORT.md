# Bug Report

## Open Bugs

*   **BUG-001: `install_npu_stack.sh` fails during MLIR-AIE installation.**
    *   **Description:** The script fails with the error `ERROR: Could not open requirements file: [Errno 2] No such file or directory: 'requirements.txt'` when trying to install Python dependencies for MLIR-AIE.
    *   **Impact:** The MLIR-AIE environment is not fully set up, which will likely block the development of custom NPU kernels in Phase 3.
    *   **Workaround:** Proceeding with the assumption that the core XRT and driver installations were successful. The MLIR-AIE setup will be revisited before Phase 3 begins.

## Resolved Bugs

*No resolved bugs at this time.*