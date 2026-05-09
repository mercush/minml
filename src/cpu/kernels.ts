import type { Array } from "../array.js";
import { DType } from "../dtype.js";
import { MinmlError } from "../error.js";
import { f32_view, i32_view } from "./backend.js";

function must_buffer(a: Array) {
  const b = a.buffer();
  if (b === null) throw MinmlError.other("kernel: input not evaluated");
  return b;
}

export function add(a: Array, b: Array, out: Array): void {
  const n = out.size();
  const pa = f32_view(must_buffer(a));
  const pb = f32_view(must_buffer(b));
  const po = f32_view(must_buffer(out));
  for (let i = 0; i < n; i++) {
    po[i] = pa[i] + pb[i];
  }
}

export function mul(a: Array, b: Array, out: Array): void {
  const n = out.size();
  const pa = f32_view(must_buffer(a));
  const pb = f32_view(must_buffer(b));
  const po = f32_view(must_buffer(out));
  for (let i = 0; i < n; i++) {
    po[i] = pa[i] * pb[i];
  }
}

export function dot(a: Array, b: Array, out: Array): void {
  const n = a.size();
  const pa = f32_view(must_buffer(a));
  const pb = f32_view(must_buffer(b));
  const po = f32_view(must_buffer(out));
  let sum = 0.0;
  for (let i = 0; i < n; i++) {
    sum += pa[i] * pb[i];
  }
  po[0] = sum;
}

export function ones(out: Array): void {
  const n = out.size();
  const buf = must_buffer(out);
  switch (out.dtype()) {
    case DType.Float32: {
      const p = f32_view(buf);
      for (let i = 0; i < n; i++) p[i] = 1.0;
      break;
    }
    case DType.Int32: {
      const p = i32_view(buf);
      for (let i = 0; i < n; i++) p[i] = 1;
      break;
    }
  }
}

export function gather(table: Array, indices: Array, out: Array): void {
  const big_n = table.shape()[0];
  let trail = 1;
  for (let d = 1; d < table.shape().length; d++) {
    trail *= table.shape()[d];
  }
  const m = indices.size();
  const t = f32_view(must_buffer(table));
  const idx = i32_view(must_buffer(indices));
  const o = f32_view(must_buffer(out));

  for (let i = 0; i < m; i++) {
    const k = idx[i];
    if (k < 0 || k >= big_n) {
      throw MinmlError.gather_oob();
    }
    const src = k * trail;
    const dst = i * trail;
    for (let j = 0; j < trail; j++) {
      o[dst + j] = t[src + j];
    }
  }
}
