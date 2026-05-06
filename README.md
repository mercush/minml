# minml

A minimal multi-backend tensor library, structured the way MLX is, kept
small enough to read in one sitting. Two ops: `add` and `dot`, on 1-D
float32 vectors, with a CPU / CUDA / WebGPU backend and Python /
TypeScript bindings.

## Architecture

```
                 ┌──────────────────────────────┐
                 │       User-facing API         │
                 │   array.h · ops.h · device.h  │
                 └──────────────┬───────────────┘
                                │
                  ┌─────────────▼──────────────┐
                  │   Array (lazy by default)   │
                  │   · evaluated buffer ─OR─   │
                  │   · primitive + inputs      │
                  └─────────────┬──────────────┘
                                │ eval()
                  ┌─────────────▼──────────────┐
                  │   Primitive (AddPrim …)     │
                  │   dispatches on device      │
                  └──┬──────────┬──────────┬───┘
                     │          │          │
              ┌──────▼──┐  ┌────▼───┐  ┌───▼──────┐
              │   CPU   │  │  CUDA  │  │  WebGPU  │
              │ memcpy  │  │ nvcc   │  │ WGSL +   │
              │ + loops │  │ kernel │  │ webgpu.h │
              └─────────┘  └────────┘  └──────────┘
```

The pieces:

* **`Array`** holds *either* a `Buffer` (evaluated) *or* a `Primitive` plus
  a list of input `Array`s (lazy). Calling `eval()`/`tolist()`/`item()`
  walks the input DAG post-order, allocates output buffers, and runs each
  primitive.
* **`Primitive`** (`AddPrim`, `DotPrim`) is the op-specific node. Its
  single job is to dispatch on the output device to a per-backend free
  function (`cpu_add`, `cuda_add`, `webgpu_add`).
* **Backends** each provide five functions: `allocate`, `copy_h2d`,
  `copy_d2h`, `<op>_add`, `<op>_dot`. They live in their own translation
  unit and have their own `Buffer` subclass with the right destructor
  (`delete[]`, `cudaFree`, `wgpuBufferRelease`).
* **Stubs** (`src/stubs.cpp`) supply throwing implementations for every
  backend the build did *not* enable, so disabled backends become clean
  runtime errors instead of link errors.

This is the same design as MLX: a thin `Array`, a `Primitive` graph, and
backends sliced behind a fixed set of free functions. To add a new op,
add a `Primitive` subclass and one function per backend. To add a new
backend, add a translation unit implementing the five functions and a
`Buffer` subclass.

## Repository layout

```
include/minml/         Public headers (Array, ops, Device, Buffer)
src/                   Core + per-backend implementations
  array.cpp            Lazy graph traversal
  ops.cpp              add, dot, primitive dispatch
  cpu_backend.cpp      Reference (always built)
  cuda_backend.cu      nvcc-compiled
  webgpu_backend.cpp   webgpu.h C API; Dawn or Emscripten provides it
  webgpu_shaders.h     WGSL kernels as raw string literals
  stubs.cpp            Throwing fallbacks for disabled backends
bindings/python/       nanobind module
bindings/ts/           Emscripten + embind module
examples/              C++, Python, browser
```

## Building

### CPU only (always works, no GPU required)

```bash
cmake -S . -B build
cmake --build build -j
./build/example
```

### WebGPU, native (Dawn)

Build Dawn separately, then point CMake at it:

```bash
cmake -S . -B build -DMINML_BUILD_WEBGPU=ON \
    -DCMAKE_PREFIX_PATH=/path/to/dawn-install
cmake --build build -j
```

You'll need to call `webgpu_init_with_device(...)` from your application
with a Dawn-acquired device.

### Python + CUDA

```bash
uv venv
source .venv/bin/activate
uv pip install nanobind
cmake -S . -B build -DMINML_BUILD_PYTHON=ON -DMINML_BUILD_CUDA=ON
cmake --build build -j
PYTHONPATH=build python examples/example.py
```

### TypeScript / browser (the WebGPU+WASM target)

```bash
source /path/to/emsdk/emsdk_env.sh
emcmake cmake -S . -B build \
    -DMINML_BUILD_WEBGPU=ON \
    -DMINML_BUILD_TS=ON
emmake cmake --build build -j
# Compile examples/example.ts -> examples/example.js
(cd examples && tsc)
# Serves: examples/example.html, examples/example.js,
#         build/minml_js.js, build/minml_js.wasm
python -m http.server 8000 --directory .
```

The browser entry point is written in TypeScript (`examples/example.ts`),
with hand-written ambient types for the embind module in
`examples/minml_js.d.ts`. embind doesn't emit `.d.ts`, so those types track
`bindings/ts/bind.cpp` by hand.

The example runs on the WebGPU backend: `await m.initWebGPU()` acquires an
adapter and device inside WASM via `wgpuInstanceRequestAdapter` /
`wgpuAdapterRequestDevice`, ASYNCIFY suspends the call while the underlying
`navigator.gpu` Promises resolve, and the resulting `WGPUDevice` is handed
to `webgpu_init_with_device()`. Readbacks (`tolist()`, `item()`) suspend on
`wgpuBufferMapAsync` and surface as Promises on the JS side.

Open <http://localhost:8000/examples/example.html> in a Chromium-family
browser.

## What's deliberately missing

This is meant to *demonstrate the shape*, not be a real ML lib. The most
obvious gaps:

* **Only float32, only 1-D.** Adding shape/strides/dtype is mechanical and
  would not change any of the architecture above; you'd thread them
  through `Array`, the WGSL shaders, and the kernels.
* **No fusion, no scheduling, no streams.** `eval()` is a recursive
  post-order walk. MLX's real evaluator schedules across streams and
  fuses elementwise ops; the hooks for that are at `Primitive::eval` but
  unused here.
* **WebGPU `dot` uses an `atomicCompareExchangeWeak` loop on a u32-cast
  f32**, which is the cheapest way to write a one-pass float reduction
  in WGSL. For real workloads you'd do a multi-pass reduction.
* **No staging-buffer pool for WebGPU readback.** Each `tolist()` round-
  trips a fresh `MapRead` buffer.
* **No error router for WebGPU.** Production code would install
  `wgpuDeviceSetUncapturedErrorCallback` and surface those into C++
  exceptions.
