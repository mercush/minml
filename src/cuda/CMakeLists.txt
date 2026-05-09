# cmake-js entry point for the minml CUDA addon.
#
# Build with: `npx cmake-js compile -d src/cuda` (or via `npm run build:cuda`).
# Output:    `build/Release/minml_cuda.node`.
#
# Requires:
#   * CUDA toolkit (nvcc on PATH, libcudart available)
#   * Node.js + cmake-js (via devDependencies)
#   * node-addon-api (resolved via require())

cmake_minimum_required(VERSION 3.18)
project(minml_cuda LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# cmake-js variables: CMAKE_JS_INC = node header dir, CMAKE_JS_LIB = node
# import lib (Windows), CMAKE_JS_SRC = optional delay-load file (Windows).
include_directories(${CMAKE_JS_INC})

# Resolve node-addon-api include. The package is header-only and lives
# at <repo>/node_modules/node-addon-api in a standard (non-workspace)
# install. Verify napi.h is present so we fail at configure rather than
# during the compile.
set(NODE_ADDON_API_DIR "${CMAKE_SOURCE_DIR}/../../node_modules/node-addon-api")
if(NOT EXISTS "${NODE_ADDON_API_DIR}/napi.h")
    message(FATAL_ERROR
        "node-addon-api headers not found at ${NODE_ADDON_API_DIR}.\n"
        "Run `npm install` from the repo root, or set NODE_ADDON_API_DIR explicitly.")
endif()
message(STATUS "node-addon-api: ${NODE_ADDON_API_DIR}")

add_library(minml_cuda SHARED
    addon.cc
    kernels.cu
    ${CMAKE_JS_SRC}
)

target_include_directories(minml_cuda PRIVATE
    ${NODE_ADDON_API_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(minml_cuda PROPERTIES
    PREFIX ""
    SUFFIX ".node"
)

target_link_libraries(minml_cuda
    ${CMAKE_JS_LIB}
    cudart
    cuda            # driver API for cuModuleLoadData / cuLaunchKernel
    nvrtc           # runtime CUDA compiler
)
