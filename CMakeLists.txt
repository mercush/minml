cmake_minimum_required(VERSION 3.20)
project(minml CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(MINML_BUILD_CUDA       "Build the CUDA backend"          OFF)
option(MINML_BUILD_WEBGPU     "Build the WebGPU backend"        OFF)
option(MINML_BUILD_PYTHON     "Build the Python bindings"       OFF)
option(MINML_BUILD_TS         "Build the TypeScript/WASM bindings (requires emcmake)" OFF)
option(MINML_BUILD_EXAMPLES   "Build the C++ example"           ON)

# ---- Core sources ----------------------------------------------------------
set(MINML_SOURCES
  src/device.cpp
  src/array.cpp
  src/ops.cpp
  src/cpu_backend.cpp
  src/stubs.cpp
)
set(MINML_DEFS MINML_HAS_CPU)

if(MINML_BUILD_CUDA)
  enable_language(CUDA)
  list(APPEND MINML_SOURCES src/cuda_backend.cu)
  list(APPEND MINML_DEFS MINML_HAS_CUDA)
endif()

if(MINML_BUILD_WEBGPU)
  list(APPEND MINML_SOURCES src/webgpu_backend.cpp)
  list(APPEND MINML_DEFS MINML_HAS_WEBGPU)
endif()

add_library(minml ${MINML_SOURCES})
target_include_directories(minml PUBLIC include)
target_include_directories(minml PRIVATE src)
target_compile_definitions(minml PRIVATE ${MINML_DEFS})

# ---- WebGPU wiring ---------------------------------------------------------
# Native build: link against Dawn (assumed already built and findable).
# Emscripten build (4.0.10+): use the emdawnwebgpu port, which provides
# webgpu.h and the JS bindings to the browser's WebGPU API. The old
# -sUSE_WEBGPU=1 flag was deprecated in favor of this port.
if(MINML_BUILD_WEBGPU)
  if(EMSCRIPTEN)
    # The port provides headers at compile time AND JS bindings at link time,
    # so it must appear in both options lists.
    target_compile_options(minml PUBLIC "--use-port=emdawnwebgpu")
    target_link_options(minml PUBLIC
      "--use-port=emdawnwebgpu"
      "-sASYNCIFY=1"               # for the spin-wait in webgpu_backend.cpp
      "-sALLOW_MEMORY_GROWTH=1")
  else()
    # User must point CMAKE_PREFIX_PATH at a Dawn install.
    find_package(Dawn REQUIRED)
    target_link_libraries(minml PUBLIC dawn::webgpu_dawn)
  endif()
endif()

# ---- Examples --------------------------------------------------------------
if(MINML_BUILD_EXAMPLES)
  add_executable(example examples/example.cpp)
  target_link_libraries(example PRIVATE minml)
endif()

# ---- Python bindings -------------------------------------------------------
if(MINML_BUILD_PYTHON)
  find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
  # nanobind ships its CMake config inside the Python package; ask the
  # active interpreter where it is, then add it to the search path.
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
  list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(_minml bindings/python/bind.cpp)
  target_link_libraries(_minml PRIVATE minml)
endif()

# ---- TypeScript/WASM bindings ----------------------------------------------
# Configure with: emcmake cmake -DMINML_BUILD_TS=ON -DMINML_BUILD_WEBGPU=ON ..
if(MINML_BUILD_TS)
  if(NOT EMSCRIPTEN)
    message(FATAL_ERROR "MINML_BUILD_TS requires the Emscripten toolchain (emcmake).")
  endif()
  if(NOT MINML_BUILD_WEBGPU)
    message(FATAL_ERROR
      "MINML_BUILD_TS now requires MINML_BUILD_WEBGPU=ON: the embind layer "
      "calls webgpu_init_with_device() and includes <webgpu/webgpu.h>.")
  endif()
  add_executable(minml_js bindings/ts/bind.cpp)
  target_link_libraries(minml_js PRIVATE minml)
  set_target_properties(minml_js PROPERTIES SUFFIX ".js")
  target_link_options(minml_js PRIVATE
    "-lembind"
    "-sMODULARIZE=1"
    "-sEXPORT_NAME=createMinml"
    "-sENVIRONMENT=web"
    "-sEXPORT_ES6=1"
    "-sALLOW_MEMORY_GROWTH=1")
endif()
