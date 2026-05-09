// Fused primitives. Two flavours: elementwise (out[i] = expr) and
// reduce-sum (out[0] = sum_i expr). Both delegate to the per-backend
// executor (webgpu/fused.ts or cuda/fused.ts) at eval time.

import type { Array } from "../array.js";
import * as cuda_fused from "../cuda/fused.js";
import { Device } from "../device.js";
import { MinmlError } from "../error.js";
import type { Primitive } from "../primitive.js";
import * as webgpu_fused from "../webgpu/fused.js";
import type { Expr, Plan } from "./expr.js";

export class FusedElemPrim implements Primitive {
  readonly plan: Plan;

  constructor(body: Expr, size: number) {
    this.plan = { kind: "elementwise", body, size };
  }

  name(): string {
    return "fused_elem";
  }

  // Treated as opaque by re-fusion (no nesting); jit only fuses the
  // original add/mul/dot ops.
  eval(inputs: Array[], out: Array): void {
    switch (out.device()) {
      case Device.WebGpu:
        webgpu_fused.run(this.plan, inputs, out);
        return;
      case Device.Cuda:
        cuda_fused.run(this.plan, inputs, out);
        return;
      default:
        throw MinmlError.op_not_implemented("fused_elem", out.device());
    }
  }
}

export class FusedReducePrim implements Primitive {
  readonly plan: Plan;

  constructor(body: Expr, size: number) {
    this.plan = { kind: "reduce_sum", body, size };
  }

  name(): string {
    return "fused_reduce";
  }

  eval(inputs: Array[], out: Array): void {
    switch (out.device()) {
      case Device.WebGpu:
        webgpu_fused.run(this.plan, inputs, out);
        return;
      case Device.Cuda:
        cuda_fused.run(this.plan, inputs, out);
        return;
      default:
        throw MinmlError.op_not_implemented("fused_reduce", out.device());
    }
  }
}
