# NPU Backend for llama.cpp (Simplified - no XRT dependency)
cmake_minimum_required(VERSION 3.15)
project(ggml_npu CXX C)

# Add llama.cpp as a subdirectory to ensure ggml is built
add_subdirectory(../llama.cpp ${CMAKE_CURRENT_BINARY_DIR}/llama.cpp_build)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(GGML_NPU_BUILD_TESTS "Build NPU backend tests" ON)
option(GGML_NPU_VERBOSE "Enable verbose NPU logging" ON)

# Find dependencies
find_package(Threads REQUIRED)

# NPU backend library with Direct Runtime
add_library(ggml-npu STATIC
    npu_kernel_loader_simple.cpp
    npu_backend_real.cpp
    ggml_npu_backend.cpp
    npu_vulkan_bridge.cpp
    ggml_npu_attention.cpp
    npu_runtime_direct.cpp
)

# Include directories
target_include_directories(ggml-npu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/ggml/include # Correct path to GGML headers
)

# Compile definitions
target_compile_definitions(ggml-npu PRIVATE
    $<$<BOOL:${GGML_NPU_VERBOSE}>:NPU_VERBOSE=1>
    NPU_RUNTIME_LOAD=1  # Load XRT at runtime
)

# Add position independent code flag for shared library linking
set_target_properties(ggml-npu PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Link libraries
target_link_libraries(ggml-npu
    PUBLIC
        Threads::Threads
        ${CMAKE_DL_LIBS}
        stdc++fs  # For filesystem
        dl        # For dlopen
)

# Tests
if(GGML_NPU_BUILD_TESTS)
    # Test NPU backend
    add_executable(test-npu test_npu_backend.cpp)
    target_link_libraries(test-npu PRIVATE ggml-npu ggml m)
    
    # Benchmark NPU
    add_executable(benchmark-npu benchmark_npu.cpp)
    target_link_libraries(benchmark-npu PRIVATE ggml-npu ggml m)
    
    # Test real kernels
    add_executable(test-real-kernels test_real_kernels.cpp)
    target_link_libraries(test-real-kernels PRIVATE ggml-npu ggml m)
endif()

# Copy kernel files for testing
file(GLOB KERNEL_FILES 
    "${CMAKE_CURRENT_SOURCE_DIR}/../npu_kernels_gemma3_4b/*.xclbin"
)
foreach(kernel_file ${KERNEL_FILES})
    configure_file(${kernel_file} ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
endforeach()

# Status messages
message(STATUS "NPU Backend Configuration (Simplified):")
message(STATUS "  Runtime XRT loading: ON")
message(STATUS "  Verbose logging: ${GGML_NPU_VERBOSE}")
message(STATUS "  Build tests: ${GGML_NPU_BUILD_TESTS}")
message(STATUS "  Kernel directory: ${CMAKE_CURRENT_SOURCE_DIR}/../npu_kernels_gemma3_4b/")