// minml-wasm — thin re-export of the wasm-bindgen package.
//
// Unlike the old embind index.ts (which awaited createMinml() at module
// load), this file does NOT use top-level await. The user calls
// `await init()` once, then everything else works:
//
//     import init, * as m from "./pkg/minml_wasm.js";
//     await init();
//     await m.initWebGPU();
//     m.setDefaultDevice(m.Device.WebGPU);
//
// vmap is curried for ergonomics: `vmap(f, in_axes)(...args)` matches
// JAX's call shape. f returns either a single Array, or a JS array of
// Array (one per leaf). Pytree-shaped class returns are not supported in
// the wasm binding (the C++ Asyncify version did via JS-side instanceof
// — see plan / README for the regression).

import init, * as wasm from "./pkg/minml_wasm.js";

export { init };
export const Device = wasm.Device;
export const DType = wasm.DType;
export const PRNGKey = wasm.PRNGKey;
export const Dirichlet = wasm.Dirichlet;
export const Categorical = wasm.Categorical;
export const Normal = wasm.Normal;
// Tensor is the wasm-bindgen Array class. Re-exported under a notebook-
// friendly name (avoids shadowing JS's built-in Array).
export const Tensor = wasm.Array;

export const setDefaultDevice = wasm.setDefaultDevice;
export const defaultDevice = wasm.defaultDevice;
export const array = wasm.array;
export const arrayInt = wasm.arrayInt;
export const add = wasm.add;
export const mul = wasm.mul;
export const dot = wasm.dot;
export const ones = wasm.ones;
export const randint = wasm.randint;
export const gather = wasm.gather;
export const dirichletSample = wasm.dirichletSample;
export const categoricalSample = wasm.categoricalSample;
export const initWebGPU = wasm.initWebGPU;

export const vmap = (f: Function, in_axes: number[]) =>
  (...args: unknown[]) => wasm.vmapApply(f, in_axes, args);
