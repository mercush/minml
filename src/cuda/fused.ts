// Fused-kernel emitter for CUDA. Generates CUDA C source from a `Plan`,
// hands it to the addon's NVRTC compile (cached by source string), and
// launches via the driver API.

import type { Array } from "../array.js";
import { emit, type Expr, type Plan } from "../fused/expr.js";
import { compile_kernel, launch_elem, launch_reduce } from "./backend.js";

function count_inputs(e: Expr): number {
  switch (e.kind) {
    case "input":
      return e.index + 1;
    case "add":
    case "mul":
      return Math.max(count_inputs(e.a), count_inputs(e.b));
  }
}

function param_list(n: number): string {
  const parts: string[] = [];
  for (let i = 0; i < n; i++) parts.push(`const float* in${i}`);
  parts.push("float* out");
  parts.push("size_t n");
  return parts.join(", ");
}

function gen_elementwise_cu(plan: Plan & { kind: "elementwise" }): string {
  const n = count_inputs(plan.body);
  const body = emit(plan.body, (i) => `in${i}[i]`);
  return `
extern "C" __global__ void fused(${param_list(n)}) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = ${body};
}
`;
}

function gen_reduce_cu(plan: Plan & { kind: "reduce_sum" }): string {
  const n = count_inputs(plan.body);
  const body = emit(plan.body, (i) => `in${i}[i]`);
  return `
extern "C" __global__ void fused(${param_list(n)}) {
    __shared__ float scratch[256];
    int tid = threadIdx.x;
    float local = 0.f;
    for (size_t i = tid; i < n; i += blockDim.x) {
        local += ${body};
    }
    scratch[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, scratch[0]);
}
`;
}

export function run(plan: Plan, inputs: Array[], out: Array): void {
  if (plan.kind === "elementwise") {
    const source = gen_elementwise_cu(plan);
    const kernel = compile_kernel(source);
    launch_elem(kernel, inputs, out, plan.size);
  } else {
    const source = gen_reduce_cu(plan);
    const kernel = compile_kernel(source);
    launch_reduce(kernel, inputs, out, plan.size);
  }
}
