# minml

A minimal multi-backend tensor library, structured the way MLX is, kept
small enough to read in one sitting. Add / mul / dot on float32 vectors,
gather and sampling on multi-dim tensors, with a CPU and WebGPU backend.
Pure TypeScript — runs in Node and in the browser.

The user-facing surface is mostly sync (graph builders); only `tolist`,
`item`, and `init_webgpu` are async — those are the only places that can
block on a GPU.

## Architecture

```
                 ┌──────────────────────────────┐
                 │      User-facing API          │
                 │   Array, ops, Device, DType   │
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
                  └──┬─────────────┬───────────┬┘
                     │             │           │
              ┌──────▼──┐    ┌─────▼────┐  ┌───▼──────┐
              │   CPU   │    │  CUDA    │  │  WebGPU  │
              │  loops  │    │ N-API    │  │ navigator│
              │         │    │ addon    │  │   .gpu   │
              └─────────┘    └──────────┘  └──────────┘
```

The pieces:

- **`Array`** holds *either* a `Buffer` (evaluated) *or* a `Primitive`
  plus a list of input `Array`s (lazy). `tolist()` / `item()` walk the
  input DAG iteratively in post-order, allocate output buffers, and run
  each primitive.
- **`Primitive`** (`AddPrim`, `DotPrim`, …) is the op-specific node. Its
  single job is to dispatch on the output device to the right backend.
- **Backends** each provide allocate / h2d / d2h / kernels.
  - **CPU** loops over `Float32Array` / `Int32Array` views over a shared
    `ArrayBuffer`.
  - **WebGPU** uses the browser's native `navigator.gpu` — no wgpu, no
    wasm shim. WGSL kernels are inlined as strings.
  - **CUDA** is an *optional* Node-only N-API addon (`src/cuda/`,
    compiled via cmake-js). Build it with `npm run build:cuda` on a host
    with the CUDA toolkit; CPU / WebGPU keep working without it. If the
    `.node` binary isn't present, calls touching `Device.Cuda` throw
    `MinmlError.backend_not_built("cuda")`.

## Install / build

```bash
npm install
npm run build         # tsc -> dist/
npm test              # vitest: 16 tests, all CPU
```

### Optional CUDA addon (Linux/Windows + NVIDIA GPU)

```bash
npm run build:cuda    # cmake-js -> build/Release/minml_cuda.node
```

Requires `nvcc` on `PATH` and a working CUDA toolkit. Won't run on macOS
(no CUDA support since 2019). Once built, `Device.Cuda` works for `add`,
`mul`, `dot`; gather, sampling, and `ones` are CPU-only (matches the
original).

## Usage

```ts
import {
  Array,
  Device,
  PRNGKey,
  add,
  dot,
  randint,
  init_webgpu,
  set_default_device,
} from "minml";

// CPU
const x = Array.from_f32_1d([1, 2, 3, 4], Device.Cpu);
const y = Array.from_f32_1d([10, 20, 30, 40], Device.Cpu);
console.log(await add(x, y).tolist());          // [11, 22, 33, 44]
console.log(await dot(x, y).item());            // 300

// WebGPU (browser only)
await init_webgpu();
set_default_device(Device.WebGpu);
const a = Array.from_f32_1d([1, 2, 3, 4], Device.WebGpu);
const b = Array.from_f32_1d([10, 20, 30, 40], Device.WebGpu);
console.log(await dot(a, b).item());            // 300

// Splittable PRNG (JAX-style)
const key = PRNGKey.from_seed(42);
const r = randint(key.k0, key.k1, 0, 10, [64], Device.Cpu);
```

## Browser demo

```bash
npm run build           # build dist/
npm run build:example   # build examples/example.js
npx http-server -p 8888 .
open http://localhost:8888/examples/example.html
```

Output (in a Chromium-family browser with WebGPU enabled):

```
add -> 11, 22, 33, 44
dot -> 300
dot(x+y, x+y) -> 3630
```

## Repository layout

```
package.json
tsconfig.json
vitest.config.ts
src/
  index.ts                    # public re-exports
  array.ts                    # lazy Array + iterative eval
  buffer.ts primitive.ts
  device.ts dtype.ts error.ts
  deviceDispatch.ts           # backend selector
  ops.ts                      # add, mul, dot, gather, ones, randint, distributions
  prng.ts threefry.ts         # JAX-style splittable PRNG
  transforms.ts               # slice_axis0, stack, vmap_apply
  cpu/{backend,kernels,random}.ts
  cuda/                       # optional N-API addon
    backend.ts                # try-loads ../../build/Release/minml_cuda.node
    kernels.{cu,h}            # CUDA kernels + extern "C" entry points
    addon.cc                  # node-addon-api bridge
    CMakeLists.txt            # cmake-js build
  webgpu/{backend,shaders}.ts
tests/
  threefry.test.ts            # u32 wrapping correctness
  cpu.test.ts                 # round-trip the CPU backend through the lazy graph
examples/
  example.html
  example.ts                  # browser WebGPU demo
  tsconfig.json
```

## What's deliberately missing

Same scope as the original Rust port; this is meant to demonstrate the
*shape*, not be a real ML lib.

- **Only float32 / int32**, multi-dim shapes but no broadcasting.
- **No fusion, scheduling, or streams.** `eval()` is iterative
  post-order; nothing rewrites the graph.
- **WebGPU `dot` uses an `atomicCompareExchangeWeak` loop on a
  u32-cast f32** — cheapest one-pass float reduction in WGSL. For real
  workloads do a multi-pass reduction.
- **No staging-buffer pool for WebGPU readback.** Each `tolist()`
  round-trips a fresh `MapRead` buffer.
- **vmap is CPU-only**, loop-based — `slice_axis0` and `stack` throw on
  non-CPU. A graph-transformation vmap (per-primitive batching rules)
  would lift this restriction; the `Primitive` shape is ready for it.
- **`Normal::sample`** is unimplemented (kept as a placeholder; matches
  the original).

## Notes

This is a TypeScript port of an earlier Rust implementation. The Rust
source lives in git history (commits before the TS port) for anyone who
wants to compare the two. Native CPU SIMD, multi-threaded eval, and CUDA
all left with the Rust code — bringing them back would mean dropping back
into a native FFI surface (Node N-API, Wasm, or PyO3).
