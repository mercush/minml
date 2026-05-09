// Node.js + TypeScript example, native CUDA backend.
//
// Run me on a CUDA host with libcuda + libnvrtc available:
//   cd crates/minml-node
//   pnpm install
//   pnpm build:cuda          # produces minml-node.<triple>.node
//   cd ../..
//   node --experimental-strip-types examples/example_node.ts
//
// `minml-node` is a regular native addon (.node) — same family as the
// Python pyo3 extension, *not* WASM. It links libcuda + libnvrtc
// directly when built with `--features cuda`, so `Device.CUDA` works
// from TypeScript exactly like it does from Python.
//
// Swap `device` to `Device.CPU` if you don't have a CUDA box but still
// want to see the API working.

import {
  Device,
  array,
  add,
  mul,
  dot,
  setDefaultDevice,
} from "../crates/minml-node/index.js";

async function main(): Promise<void> {
  const device = Device.CUDA; // set to Device.CPU if you don't have CUDA
  setDefaultDevice(device);

  const x = array([1.0, 2.0, 3.0, 4.0], device);
  const y = array([10.0, 20.0, 30.0, 40.0], device);

  console.log("add ->", await add(x, y).tolist());
  console.log("dot ->", await dot(x, y).item());

  // Lazy graph: the inner add is only evaluated when item() forces it.
  console.log(
    "dot(x+y, x+y) ->",
    await dot(add(x, y), add(x, y)).item(),
  );

  // Same op composition the wasm/python examples use; runs on the GPU
  // through cuda-oxide + NVRTC when device === Device.CUDA.
  console.log(
    "(x*y) + (x*x) ->",
    await add(mul(x, y), mul(x, x)).tolist(),
  );
}

main().catch((err) => {
  console.error("error:", err);
  process.exit(1);
});
