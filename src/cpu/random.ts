// CPU random / sampling — direct port of Rust cpu/random.rs.
//
// All draws come from Threefry-2x32. Per-element indexing uses
// (op_tag, key, output_index, sub_iter); op_tag prevents different ops
// from producing identical bits when given the same key.

import type { Array } from "../array.js";
import { MinmlError } from "../error.js";
import { threefry_2x32, threefry_u32, u32_to_unit_f32 } from "../threefry.js";
import { f32_view, i32_view } from "./backend.js";

const TAG_RANDINT = 0x52414e44 >>> 0; // 'RAND'
const TAG_DIRICHLET = 0x44495243 >>> 0; // 'DIRC'
const TAG_CATEGORICAL_U = 0x43415455 >>> 0; // 'CATU'

function uniform_for(
  k0: number,
  k1: number,
  tag: number,
  i: number,
  sub: number,
): number {
  const [a] = threefry_2x32((k0 ^ tag) >>> 0, k1, i >>> 0, sub >>> 0);
  return u32_to_unit_f32(a);
}

// Marsaglia & Tsang gamma sampler for shape >= 1, with the boost trick for
// shape < 1.
function sample_gamma(
  k0: number,
  k1: number,
  shape: number,
  base_idx: number,
): number {
  if (shape < 1.0) {
    const g = sample_gamma(k0, k1, shape + 1.0, base_idx);
    let u = uniform_for(k0, k1, 0xdeadbeef >>> 0, base_idx, 99);
    if (u < 1e-30) u = 1e-30;
    return g * Math.pow(u, 1.0 / shape);
  }
  const d = shape - 1.0 / 3.0;
  const c = 1.0 / Math.sqrt(9.0 * d);
  let sub = 0;
  for (;;) {
    let u1 = uniform_for(k0, k1, 0xcafe0001 >>> 0, base_idx, sub);
    sub += 1;
    const u2 = uniform_for(k0, k1, 0xcafe0002 >>> 0, base_idx, sub);
    sub += 1;
    if (u1 < 1e-30) u1 = 1e-30;
    const x = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(6.2831853 * u2);
    const v = 1.0 + c * x;
    if (v <= 0.0) continue;
    const v3 = v * v * v;
    const u = uniform_for(k0, k1, 0xcafe0003 >>> 0, base_idx, sub);
    sub += 1;
    if (u < 1.0 - 0.0331 * x * x * x * x) {
      return d * v3;
    }
    if (Math.log(u) < 0.5 * x * x + d * (1.0 - v3 + Math.log(v3))) {
      return d * v3;
    }
  }
}

function must_buffer(a: Array) {
  const b = a.buffer();
  if (b === null) throw MinmlError.other("random: input not evaluated");
  return b;
}

export function randint(
  k0: number,
  k1: number,
  low: number,
  high: number,
  out: Array,
): void {
  if (high <= low) {
    throw MinmlError.other("randint: high <= low");
  }
  const span = (high - low) >>> 0;
  const n = out.size();
  const p = i32_view(must_buffer(out));
  for (let i = 0; i < n; i++) {
    const bits = threefry_u32((k0 ^ TAG_RANDINT) >>> 0, k1, i >>> 0);
    p[i] = low + ((bits >>> 0) % span);
  }
}

export function dirichlet_sample(
  k0: number,
  k1: number,
  batch_shape: number[],
  alpha: Array,
  out: Array,
): void {
  const big_k = alpha.shape()[0];
  let big_b = 1;
  for (const d of batch_shape) big_b *= d;
  const a = f32_view(must_buffer(alpha));
  const o = f32_view(must_buffer(out));
  for (let b = 0; b < big_b; b++) {
    const base = b * big_k;
    let sum = 0.0;
    for (let k = 0; k < big_k; k++) {
      const idx = (base + k) >>> 0;
      o[base + k] = sample_gamma(
        (k0 ^ TAG_DIRICHLET) >>> 0,
        k1,
        a[k],
        idx,
      );
      sum += o[base + k];
    }
    if (sum > 0.0) {
      for (let k = 0; k < big_k; k++) {
        o[base + k] /= sum;
      }
    }
  }
}

export function categorical_sample(
  k0: number,
  k1: number,
  batch_shape: number[],
  probs: Array,
  out: Array,
): void {
  const big_k = probs.shape()[0];
  let big_b = 1;
  for (const d of batch_shape) big_b *= d;
  if (big_b === 0) big_b = 1;

  const p = f32_view(must_buffer(probs));
  const cdf = new Float32Array(big_k);
  let total = 0.0;
  for (let k = 0; k < big_k; k++) {
    total += p[k];
    cdf[k] = total;
  }
  if (total <= 0.0) {
    throw MinmlError.other("categorical: probs sum to 0");
  }
  const o = i32_view(must_buffer(out));
  for (let b = 0; b < big_b; b++) {
    const u = uniform_for(k0, k1, TAG_CATEGORICAL_U, b >>> 0, 0) * total;
    let pick = big_k - 1;
    for (let k = 0; k < big_k; k++) {
      if (u < cdf[k]) {
        pick = k;
        break;
      }
    }
    o[b] = pick;
  }
}
