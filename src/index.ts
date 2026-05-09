// minml — minimal lazy tensor library: CPU + WebGPU.
//
// Lazy `Array` graph, `Primitive`-per-op, per-backend free functions.
// The user-facing surface is mostly sync; only `Array.tolist`, `Array.item`,
// and `init_webgpu` are async.

export { Array } from "./array.js";
export type { Buffer } from "./buffer.js";
export { Device, default_device, set_default_device } from "./device.js";
export { DType, dtype_bytes } from "./dtype.js";
export { MinmlError, type MinmlErrorKind } from "./error.js";
export {
  add,
  Categorical,
  categorical_sample,
  Dirichlet,
  dirichlet_sample,
  dot,
  gather,
  mul,
  Normal,
  ones,
  randint,
} from "./ops.js";
export type { Primitive } from "./primitive.js";
export { PRNGKey } from "./prng.js";
export {
  jit,
  slice_axis0,
  stack,
  vmap_apply,
  type VmapCallable,
} from "./transforms.js";
export { init as init_webgpu } from "./webgpu/backend.js";
