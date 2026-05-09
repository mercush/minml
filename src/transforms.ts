// Transforms: slice_axis0, stack, vmap_apply.
//
// All sync, CPU-only — same scope as the Rust original. Slicing forces
// eval; the per-iter callable builds lazy graphs which are stacked back
// into a CPU array.

import { Array } from "./array.js";
import * as cpu_backend from "./cpu/backend.js";
import { Device } from "./device.js";
import { DType, dtype_bytes } from "./dtype.js";
import { MinmlError } from "./error.js";

function product(shape: number[]): number {
  let p = 1;
  for (const d of shape) p *= d;
  return p;
}

function product_after_first(shape: number[]): number {
  let p = 1;
  for (let i = 1; i < shape.length; i++) p *= shape[i];
  return p;
}

export function slice_axis0(arr_in: Array): Array[] {
  if (arr_in.device() !== Device.Cpu) {
    throw MinmlError.other("slice_axis0: CPU only for now");
  }
  arr_in.eval();
  if (arr_in.shape().length === 0) {
    throw MinmlError.other("slice_axis0: cannot slice a scalar");
  }
  const big_n = arr_in.shape()[0];
  const sub_shape = arr_in.shape().slice(1);
  const per = product_after_first(arr_in.shape());

  const buf = arr_in.buffer()!;
  const out: Array[] = [];
  switch (arr_in.dtype()) {
    case DType.Float32: {
      const data = cpu_backend.f32_view(buf);
      for (let i = 0; i < big_n; i++) {
        const chunk = data.slice(i * per, (i + 1) * per);
        out.push(Array.from_f32_with_shape(chunk, sub_shape, arr_in.device()));
      }
      break;
    }
    case DType.Int32: {
      const data = cpu_backend.i32_view(buf);
      for (let i = 0; i < big_n; i++) {
        const chunk = data.slice(i * per, (i + 1) * per);
        out.push(Array.from_i32_with_shape(chunk, sub_shape, arr_in.device()));
      }
      break;
    }
  }
  return out;
}

export function stack(parts: Array[]): Array {
  if (parts.length === 0) throw MinmlError.other("stack: empty input");
  const base_shape = parts[0].shape().slice();
  const dev = parts[0].device();
  const dt = parts[0].dtype();
  for (const p of parts) {
    if (p.shape().length !== base_shape.length) throw MinmlError.other("stack: shape mismatch");
    for (let i = 0; i < base_shape.length; i++) {
      if (p.shape()[i] !== base_shape[i]) throw MinmlError.other("stack: shape mismatch");
    }
    if (p.device() !== dev) throw MinmlError.other("stack: device mismatch");
    if (p.dtype() !== dt) throw MinmlError.other("stack: dtype mismatch");
  }
  if (dev !== Device.Cpu) throw MinmlError.other("stack: CPU only for now");

  const out_shape = [parts.length, ...base_shape];
  const per = product(base_shape);
  const bytes_per = per * dtype_bytes(dt);
  const buf = new Uint8Array(bytes_per * parts.length);

  for (let i = 0; i < parts.length; i++) {
    parts[i].eval();
    const pbuf = parts[i].buffer()!;
    const dst = buf.subarray(i * bytes_per, (i + 1) * bytes_per);
    cpu_backend.copy_buffer_to_host(pbuf, dst);
  }

  switch (dt) {
    case DType.Float32:
      return Array.from_f32_with_shape(
        new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4),
        out_shape,
        dev,
      );
    case DType.Int32:
      return Array.from_i32_with_shape(
        new Int32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4),
        out_shape,
        dev,
      );
  }
}

// Per-iteration callable. Receives:
//   * iter_index: which batch element we're on.
//   * args: Array arguments with batched ones already sliced (shape ==
//     orig.shape[1:]); unbatched ones passed through unchanged.
// Returns one Array per leaf of the function's logical return value.
export type VmapCallable = (iter_index: number, args: Array[]) => Array[];

export function vmap_apply(
  big_n: number,
  args: Array[],
  in_axes: number[],
  f: VmapCallable,
): Array[] {
  if (args.length !== in_axes.length) {
    throw MinmlError.vmap("args/in_axes size mismatch");
  }
  // Pre-slice batched inputs once.
  const sliced: Array[][] = args.map(() => []);
  for (let i = 0; i < args.length; i++) {
    if (in_axes[i] < 0) continue;
    if (in_axes[i] !== 0) throw MinmlError.vmap("only axis 0 supported");
    if (args[i].shape().length === 0) {
      throw MinmlError.vmap("cannot batch over a scalar");
    }
    if (args[i].shape()[0] !== big_n) {
      throw MinmlError.vmap("batched dims disagree");
    }
    sliced[i] = slice_axis0(args[i]);
  }

  const all_leaves: Array[][] = [];
  for (let b = 0; b < big_n; b++) {
    const per_iter: Array[] = [];
    for (let i = 0; i < args.length; i++) {
      if (in_axes[i] >= 0) {
        per_iter.push(sliced[i][b]);
      } else {
        per_iter.push(args[i]);
      }
    }
    all_leaves.push(f(b, per_iter));
  }
  if (all_leaves.length === 0) throw MinmlError.vmap("N=0");
  const n_leaves = all_leaves[0].length;
  for (const v of all_leaves) {
    if (v.length !== n_leaves) {
      throw MinmlError.vmap("leaf count varies across iterations");
    }
  }
  const stacked: Array[] = [];
  for (let l = 0; l < n_leaves; l++) {
    const parts = all_leaves.map((v) => v[l]);
    stacked.push(stack(parts));
  }
  return stacked;
}
