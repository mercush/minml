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

# Resolve node-addon-api include via Node, since the package is
# pure-headers and ships its include path through `require`.
execute_process(
    COMMAND node -p "require('node-addon-api').include_dir"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/../..
    OUTPUT_VARIABLE NODE_ADDON_API_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

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
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(minml_cuda
    ${CMAKE_JS_LIB}
    cudart
)
