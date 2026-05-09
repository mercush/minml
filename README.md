# minml

A minimal multi-backend tensor library, structured the way MLX is, kept
small enough to read in one sitting. Two ops: `add` and `dot`, on 1-D
float32 vectors, with a CPU / CUDA / WebGPU backend and Python /
TypeScript bindings.

The core is Rust. The Python and TypeScript APIs are async — readbacks
return coroutines / Promises uniformly across backends. WebGPU readback
goes through wgpu's native `map_async` (no Asyncify, no spin-wait).

## Architecture

```
                 ┌──────────────────────────────┐
                 │       User-facing API         │
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
                  └──┬──────────┬──────────┬───┘
                     │          │          │
              ┌──────▼──┐  ┌────▼───┐  ┌───▼──────┐
              │   CPU   │  │  CUDA  │  │  WebGPU  │
              │  loops  │  │  FFI   │  │   wgpu   │
              │         │  │ shim   │  │   crate  │
              └─────────┘  └────────┘  └──────────┘
```

The pieces:

* **`Array`** holds *either* a `Buffer` (evaluated) *or* a `Primitive`
  plus a list of input `Array`s (lazy). Calling `tolist()` / `item()` /
  `eval()` walks the input DAG iteratively in post-order, allocates
  output buffers, and runs each primitive.
* **`Primitive`** (`AddPrim`, `DotPrim`, …) is the op-specific node. Its
  single job is to dispatch on the output device to the right backend.
* **Backends** each provide allocate / h2d / d2h / kernels.
  - **CPU** is pure Rust loops.
  - **CUDA** is an extern "C" shim in `crates/minml-core/cuda/kernels.cu`
    built by `build.rs` via `cc::Build::cuda(true)`. Rust calls it via
    FFI through opaque handles.
  - **WebGPU** uses the [`wgpu`](https://wgpu.rs) crate. Same code
    targets native (Vulkan/Metal/DX12) and `wasm32-unknown-unknown`
    (`navigator.gpu`).
* **Async seam.** The internal `Backend` trait is sync — even WebGPU
  kernel dispatch is sync (`queue.submit` returns immediately). Only
  *readback* (`Buffer::slice.map_async`) and *device acquisition*
  (`request_adapter`/`request_device`) are async. Those are the
  user-facing async surfaces: `Array::tolist`, `Array::item`,
  `Array::eval`, and `init_webgpu`.

## Repository layout

```
Cargo.toml                          # workspace
crates/
  minml-core/                       # core library (Rust)
    Cargo.toml
    build.rs                        # nvcc when feature=cuda
    cuda/kernels.{cu,h}             # extern "C" CUDA shim
    src/
      lib.rs
      array.rs                      # lazy Array + iterative eval
      buffer.rs primitive.rs ops.rs
      device.rs dtype.rs error.rs
      device_dispatch.rs            # backend selector
      prng.rs threefry.rs           # JAX-style splittable PRNG
      transforms.rs                 # slice_axis0, stack, vmap_apply
      cpu/{mod,kernels,random,backend}.rs
      webgpu/{mod,shaders}.rs
      cuda/mod.rs                   # extern "C" decls + Backend impl
  minml-py/                         # pyo3 + maturin
    Cargo.toml pyproject.toml
    src/lib.rs                      # async Python surface
  minml-wasm/                       # wasm-bindgen (browser TS frontend)
    Cargo.toml
    index.ts                        # ESM re-export, no top-level await
    src/lib.rs                      # async TS surface
  minml-node/                       # napi-rs (Node TS frontend)
    Cargo.toml package.json
    src/lib.rs                      # native .node addon (links libcuda
                                    # + libnvrtc when --features cuda)
examples/
  example.py                        # async, asyncio.run
  example.ts                        # browser WASM + WebGPU
  example.html                      # loads the wasm demo
  example.js                        # tsc output
  example_node.ts                   # Node + native (CPU/CUDA/WebGPU)
```

## Building

### Rust workspace (CPU + native WebGPU)

```bash
cargo build -p minml-core --features webgpu
cargo test  -p minml-core --features webgpu
```

The `webgpu` feature is on by default. To strip wgpu out, build with
`--no-default-features`. CUDA is gated behind `--features cuda`; the
shim compiles only on a host with `nvcc` on `PATH`.

### Python bindings

```bash
uv venv
source .venv/bin/activate
uv pip install maturin
cd crates/minml-py
maturin develop --release
python ../../examples/example.py
```

On a CUDA host (Linux/Windows with `nvcc` on `PATH`), build with the
CUDA backend linked in:

```bash
maturin develop --release --features cuda
```

The `cuda` feature forwards through to `minml-core`, which compiles
`crates/minml-core/cuda/kernels.cu` via `cc::Build::cuda(true)` and
links `cudart`. Then the Python example just flips one line:

```python
device = m.Device.CUDA   # was m.Device.CPU
```

Same async API on every backend — readbacks like `await m.add(x,
y).tolist()` work uniformly. WebGPU is also available via `await
m.init_webgpu()` + `Device.WebGPU`.

### TypeScript / browser (WebGPU + WASM)

Requires a Rustup-managed toolchain with the wasm target installed
(`rustup target add wasm32-unknown-unknown`). Non-Rustup installs —
notably Homebrew's `rust` formula — ship without the wasm sysroot, and
`wasm-pack` will fail with `wasm32-unknown-unknown target not found in
sysroot`.

```bash
cd crates/minml-wasm
wasm-pack build --target web --release
(cd ../../examples && tsc)
cd ../..
python -m http.server 8888
```

Open <http://localhost:8888/examples/example.html> in a Chromium-family
browser. `await init()` instantiates the wasm module; `await
initWebGPU()` acquires a device through wgpu (which calls
`navigator.gpu` under the hood — no Asyncify, no spin loop). Readbacks
suspend on `Buffer::slice.map_async` and surface as Promises on the JS
side.

### TypeScript / Node (CPU / CUDA / native WebGPU)

The browser path above can't reach the CUDA backend: WASM in any
browser/Node WASM runtime is sandboxed and has no FFI to host C
libraries (libcuda, libnvrtc). To run TypeScript against the CUDA
backend you build the `minml-node` crate instead — a regular native
addon (`.node`) via [napi-rs](https://napi.rs), the same family as the
pyo3-built Python extension. Building with `--features cuda` links
libcuda + libnvrtc just like Python does.

```bash
cd crates/minml-node
pnpm install            # or npm install / yarn

# CPU only (and native WebGPU via wgpu's Vulkan/Metal/DX12 backends):
pnpm build

# CUDA backend — requires libcuda + libnvrtc on the host (Linux/Windows
# with the CUDA toolkit installed and on LD_LIBRARY_PATH / PATH).
pnpm build:cuda

cd ../..
node --experimental-strip-types examples/example_node.ts
```

`pnpm build:cuda` runs `napi build --release --cargo-flags="--features
cuda"`, producing `crates/minml-node/minml-node.<triple>.node` plus a
generated `index.js` / `index.d.ts`. The example imports from that
generated entry point and uses `Device.CUDA` directly:

```ts
import { Device, array, add, dot, setDefaultDevice }
  from "../crates/minml-node/index.js";

setDefaultDevice(Device.CUDA);
const x = array([1, 2, 3, 4], Device.CUDA);
const y = array([10, 20, 30, 40], Device.CUDA);
console.log(await dot(x, y).item());        // -> 300
```

The async surface is identical to the wasm/python bindings: `tolist`,
`item`, `eval`, and `initWebGPU` are Promises driven by napi-rs's Tokio
runtime. CPU and CUDA resolve immediately inside the Rust future;
native WebGPU (if you opt into it via `await initWebGPU()`) suspends on
the real `map_async`. Note the Node binding intentionally omits `jit`
for now — graph builders compose fine on the CUDA backend without it,
and the trace-and-callback shape is awkward to bind through stable
napi v2. If you need fusion from TS, the wasm + WebGPU path (or the
Python binding) already exposes it.

| Frontend     | Backends reachable           | Notes                          |
| ------------ | ---------------------------- | ------------------------------ |
| `minml-py`   | CPU, CUDA, WebGPU            | pyo3 extension; `jit` exposed  |
| `minml-wasm` | CPU (in-browser), WebGPU     | sandboxed; no host-FFI         |
| `minml-node` | CPU, CUDA, native WebGPU     | napi-rs addon; CUDA via toolkit|

## What's deliberately missing

Same scope as the C++ original; this is meant to demonstrate the
*shape*, not be a real ML lib.

* **Only float32 / int32, only 1-D** for ops; gather/sample carry
  multi-dim. Adding shape/strides/more dtypes is mechanical.
* **No fusion, scheduling, or streams.** `eval()` is iterative
  post-order. The hooks for a real evaluator go on `Primitive::eval`
  but are unused here.
* **WebGPU `dot` uses an `atomicCompareExchangeWeak` loop on a
  u32-cast f32** — cheapest one-pass float reduction in WGSL. For real
  workloads do a multi-pass reduction.
* **No staging-buffer pool for WebGPU readback.** Each `tolist()`
  round-trips a fresh `MapRead` buffer.
* **CUDA scope = add / mul / dot only.** Random ops and gather stay
  CPU-only on every backend (matches the C++ original).
* **vmap is CPU-only**, loop-based — `slice_axis0` and `stack` throw
  on non-CPU. A graph-transformation vmap (per-primitive batching
  rules) would lift this restriction; the `Primitive::eval` shape is
  ready for it.
* **Wasm vmap pytree returns are not supported.** The Python binding
  walks `__dict__` to collect Array leaves from a class instance and
  rebuild a fresh one; the wasm binding would need wasm-bindgen
  `instanceof` support that 0.2 doesn't expose for `#[wasm_bindgen]`
  structs. JS callers can return a single `Array` or a JS array of
  `Array`; for class-shaped returns, destructure manually.
