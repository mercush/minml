// Browser entry point for the minml WebGPU demo.
//
// Imports the compiled TS library from ../dist (run `npm run build` in the
// repo root first). Readbacks (tolist, item) are Promises driven by
// GPUBuffer.mapAsync.

import {
  add,
  Array,
  Device,
  dot,
  init_webgpu,
  jit,
  mul,
  set_default_device,
} from "../dist/src/index.js";

const out = document.getElementById("out") as HTMLElement;
out.textContent = "";
const log = (s: string): void => {
  out.textContent += s + "\n";
};

try {
  if (!navigator.gpu) {
    throw new Error("WebGPU is not available in this browser");
  }

  await init_webgpu();
  set_default_device(Device.WebGpu);

  const x = Array.from_f32_1d([1, 2, 3, 4], Device.WebGpu);
  const y = Array.from_f32_1d([10, 20, 30, 40], Device.WebGpu);

  log("add -> " + (await add(x, y).tolist()).join(", "));
  log("dot -> " + (await dot(x, y).item()));
  log("dot(x+y, x+y) -> " + (await dot(add(x, y), add(x, y)).item()));

  // ---- jit: kernel fusion ----
  // Without jit, `mul(add(a, b), add(a, a))` runs as three WGSL dispatches
  // with two intermediate storage buffers. jit rewrites the lazy DAG so the
  // whole expression becomes one runtime-generated WGSL kernel and one
  // dispatch — no intermediates touch global memory.
  const fused = jit((a: Array, b: Array) => mul(add(a, b), add(a, a)));
  log("jit (x+y)*(x+x) -> " + (await fused(x, y).tolist()).join(", "));

  // jit also accepts pytree-style returns — any class instance whose
  // enumerable fields are Arrays. Same shape goes in, same shape comes
  // out, with each leaf separately fused.
  class Pair {
    constructor(
      public sum: Array,
      public diff: Array,
    ) {}
  }
  const both = jit((a: Array, b: Array) =>
    new Pair(
      add(mul(a, b), mul(a, a)),       // a*b + a*a
      add(mul(a, a), mul(a, a)),       // 2 * a*a
    ),
  );
  const pair = both(x, y);
  log("jit pair.sum  -> " + (await pair.sum.tolist()).join(", "));
  log("jit pair.diff -> " + (await pair.diff.tolist()).join(", "));
} catch (err) {
  log("error: " + err);
  console.error(err);
}
