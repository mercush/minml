// Op entry points + per-primitive eval dispatchers. Each constructor builds
// a lazy Array; eval() walks the DAG and runs per-backend kernels.

import { Array } from "./array.js";
import * as cpu_kernels from "./cpu/kernels.js";
import * as cpu_random from "./cpu/random.js";
import * as cuda from "./cuda/backend.js";
import { Device } from "./device.js";
import { DType } from "./dtype.js";
import { MinmlError } from "./error.js";
import type { Primitive } from "./primitive.js";
import { PRNGKey } from "./prng.js";
import * as webgpu from "./webgpu/backend.js";

function shapes_equal(a: number[], b: number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

function check_same_shape(a: Array, b: Array): void {
  if (!shapes_equal(a.shape(), b.shape())) throw MinmlError.shape_mismatch();
  if (a.device() !== b.device()) throw MinmlError.device_mismatch();
}

// ---- add ----

export function add(a: Array, b: Array): Array {
  check_same_shape(a, b);
  return Array.lazy(a.shape().slice(), a.dtype(), a.device(), new AddPrim(), [a, b]);
}

class AddPrim implements Primitive {
  name(): string {
    return "add";
  }
  fusion_class(): "elementwise" {
    return "elementwise";
  }
  eval(inputs: Array[], out: Array): void {
    switch (out.device()) {
      case Device.Cpu:
        cpu_kernels.add(inputs[0], inputs[1], out);
        return;
      case Device.Cuda:
        cuda.add(inputs[0], inputs[1], out);
        return;
      case Device.WebGpu:
        webgpu.add(inputs[0], inputs[1], out);
        return;
    }
  }
}

// ---- mul ----

export function mul(a: Array, b: Array): Array {
  check_same_shape(a, b);
  return Array.lazy(a.shape().slice(), a.dtype(), a.device(), new MulPrim(), [a, b]);
}

class MulPrim implements Primitive {
  name(): string {
    return "mul";
  }
  fusion_class(): "elementwise" {
    return "elementwise";
  }
  eval(inputs: Array[], out: Array): void {
    switch (out.device()) {
      case Device.Cpu:
        cpu_kernels.mul(inputs[0], inputs[1], out);
        return;
      case Device.Cuda:
        cuda.mul(inputs[0], inputs[1], out);
        return;
      case Device.WebGpu:
        webgpu.mul(inputs[0], inputs[1], out);
        return;
    }
  }
}

// ---- dot ----

export function dot(a: Array, b: Array): Array {
  check_same_shape(a, b);
  if (a.shape().length !== 1) throw MinmlError.dot_requires_1d();
  return Array.lazy([1], a.dtype(), a.device(), new DotPrim(), [a, b]);
}

class DotPrim implements Primitive {
  name(): string {
    return "dot";
  }
  fusion_class(): "reduction" {
    return "reduction";
  }
  eval(inputs: Array[], out: Array): void {
    switch (out.device()) {
      case Device.Cpu:
        cpu_kernels.dot(inputs[0], inputs[1], out);
        return;
      case Device.Cuda:
        cuda.dot(inputs[0], inputs[1], out);
        return;
      case Device.WebGpu:
        webgpu.dot(inputs[0], inputs[1], out);
        return;
    }
  }
}

// ---- ones ----

export function ones(shape: number[], dtype: DType, device: Device): Array {
  return Array.lazy(shape, dtype, device, new OnesPrim(), []);
}

class OnesPrim implements Primitive {
  name(): string {
    return "ones";
  }
  eval(_inputs: Array[], out: Array): void {
    if (out.device() === Device.Cpu) {
      cpu_kernels.ones(out);
      return;
    }
    throw MinmlError.op_not_implemented("ones", out.device());
  }
}

// ---- randint ----

export function randint(
  k0: number,
  k1: number,
  low: number,
  high: number,
  shape: number[],
  device: Device,
): Array {
  return Array.lazy(shape, DType.Int32, device, new RandIntPrim(k0, k1, low, high), []);
}

class RandIntPrim implements Primitive {
  constructor(
    readonly k0: number,
    readonly k1: number,
    readonly low: number,
    readonly high: number,
  ) {}
  name(): string {
    return "randint";
  }
  eval(_inputs: Array[], out: Array): void {
    if (out.device() === Device.Cpu) {
      cpu_random.randint(this.k0, this.k1, this.low, this.high, out);
      return;
    }
    throw MinmlError.op_not_implemented("randint", out.device());
  }
}

// ---- gather ----

export function gather(table: Array, indices: Array): Array {
  if (indices.dtype() !== DType.Int32) throw MinmlError.gather_indices_not_int32();
  if (table.shape().length === 0) throw MinmlError.gather_table_rank();
  const out_shape = indices.shape().slice();
  for (let d = 1; d < table.shape().length; d++) {
    out_shape.push(table.shape()[d]);
  }
  return Array.lazy(out_shape, table.dtype(), table.device(), new GatherPrim(), [table, indices]);
}

class GatherPrim implements Primitive {
  name(): string {
    return "gather";
  }
  eval(inputs: Array[], out: Array): void {
    if (out.device() === Device.Cpu) {
      cpu_kernels.gather(inputs[0], inputs[1], out);
      return;
    }
    throw MinmlError.op_not_implemented("gather", out.device());
  }
}

// ---- distribution sample primitives ----

export function dirichlet_sample(
  k0: number,
  k1: number,
  alpha: Array,
  batch_shape: number[],
): Array {
  if (alpha.shape().length !== 1) throw MinmlError.dirichlet_alpha_not_1d();
  const out_shape = batch_shape.slice();
  out_shape.push(alpha.shape()[0]);
  return Array.lazy(
    out_shape,
    DType.Float32,
    alpha.device(),
    new DirichletSamplePrim(k0, k1, batch_shape),
    [alpha],
  );
}

class DirichletSamplePrim implements Primitive {
  constructor(
    readonly k0: number,
    readonly k1: number,
    readonly batch_shape: number[],
  ) {}
  name(): string {
    return "dirichlet_sample";
  }
  eval(inputs: Array[], out: Array): void {
    if (out.device() === Device.Cpu) {
      cpu_random.dirichlet_sample(this.k0, this.k1, this.batch_shape, inputs[0], out);
      return;
    }
    throw MinmlError.op_not_implemented("dirichlet_sample", out.device());
  }
}

export function categorical_sample(
  k0: number,
  k1: number,
  probs: Array,
  batch_shape: number[],
): Array {
  if (probs.shape().length !== 1) throw MinmlError.categorical_probs_not_1d();
  return Array.lazy(
    batch_shape.slice(),
    DType.Int32,
    probs.device(),
    new CategoricalSamplePrim(k0, k1, batch_shape),
    [probs],
  );
}

class CategoricalSamplePrim implements Primitive {
  constructor(
    readonly k0: number,
    readonly k1: number,
    readonly batch_shape: number[],
  ) {}
  name(): string {
    return "categorical_sample";
  }
  eval(inputs: Array[], out: Array): void {
    if (out.device() === Device.Cpu) {
      cpu_random.categorical_sample(this.k0, this.k1, this.batch_shape, inputs[0], out);
      return;
    }
    throw MinmlError.op_not_implemented("categorical_sample", out.device());
  }
}

// ---- Distribution wrappers ----

export class Dirichlet {
  constructor(readonly alpha: Array) {}
  sample(key: PRNGKey, batch_shape: number[]): Array {
    return dirichlet_sample(key.k0, key.k1, this.alpha, batch_shape);
  }
}

export class Categorical {
  constructor(readonly probs: Array) {}
  sample(key: PRNGKey, batch_shape: number[]): Array {
    return categorical_sample(key.k0, key.k1, this.probs, batch_shape);
  }
}

export class Normal {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  sample(_key: PRNGKey, _batch_shape: number[]): Array {
    throw MinmlError.other("Normal::sample: not implemented yet");
  }
}
