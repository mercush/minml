// Fused-kernel emitter for WebGPU. Generates WGSL on the fly from a
// `Plan`, hands it to the source-keyed pipeline cache in backend.ts,
// and dispatches with a dynamic bind group.

import type { Array } from "../array.js";
import { emit, type Expr, type Plan } from "../fused/expr.js";
import {
  compute_pipeline,
  get_device_and_queue,
  WebGpuBuffer,
} from "./backend.js";

// Largest input.index in the expression, plus 1. All inputs to a fused
// chain are same-shape (enforced by add/mul/dot's check_same_shape), so
// we don't need per-input length info.
function count_inputs(e: Expr): number {
  switch (e.kind) {
    case "input":
      return e.index + 1;
    case "add":
    case "mul":
      return Math.max(count_inputs(e.a), count_inputs(e.b));
  }
}

function gen_elementwise_wgsl(plan: Plan & { kind: "elementwise" }): string {
  const n = count_inputs(plan.body);
  const bindings: string[] = [];
  for (let i = 0; i < n; i++) {
    bindings.push(
      `@group(0) @binding(${i}) var<storage, read>       in${i} : array<f32>;`,
    );
  }
  bindings.push(
    `@group(0) @binding(${n}) var<storage, read_write> out : array<f32>;`,
  );
  const body = emit(plan.body, (i) => `in${i}[i]`);
  return `${bindings.join("\n")}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i < arrayLength(&out)) {
    out[i] = ${body};
  }
}
`;
}

function gen_reduce_wgsl(plan: Plan & { kind: "reduce_sum" }): string {
  const n = count_inputs(plan.body);
  const bindings: string[] = [];
  for (let i = 0; i < n; i++) {
    bindings.push(
      `@group(0) @binding(${i}) var<storage, read>           in${i} : array<f32>;`,
    );
  }
  bindings.push(
    `@group(0) @binding(${n}) var<storage, read_write>     out : array<atomic<u32>>;`,
  );
  const body = emit(plan.body, (i) => `in${i}[i]`);
  return `${bindings.join("\n")}

var<workgroup> scratch : array<f32, 64>;

fn atomic_add_f32(idx : u32, v : f32) {
  loop {
    let old_u = atomicLoad(&out[idx]);
    let old_f = bitcast<f32>(old_u);
    let new_u = bitcast<u32>(old_f + v);
    let res   = atomicCompareExchangeWeak(&out[idx], old_u, new_u);
    if (res.exchanged) { break; }
  }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(local_invocation_id)  lid : vec3<u32>) {
  let i = gid.x;
  let n = arrayLength(&in0);
  var v : f32 = 0.0;
  if (i < n) { v = ${body}; }
  scratch[lid.x] = v;
  workgroupBarrier();

  var stride : u32 = 32u;
  loop {
    if (stride == 0u) { break; }
    if (lid.x < stride) {
      scratch[lid.x] = scratch[lid.x] + scratch[lid.x + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  if (lid.x == 0u) {
    atomic_add_f32(0u, scratch[0]);
  }
}
`;
}

export function run(plan: Plan, inputs: Array[], out: Array): void {
  const { device, queue } = get_device_and_queue();
  const wgsl =
    plan.kind === "elementwise"
      ? gen_elementwise_wgsl(plan)
      : gen_reduce_wgsl(plan);
  const label = plan.kind === "elementwise" ? "fused_elem" : "fused_reduce";
  const pipe = compute_pipeline(wgsl, label);

  const out_handle = (out.buffer() as WebGpuBuffer).handle;
  if (plan.kind === "reduce_sum") {
    queue.writeBuffer(
      out_handle,
      0,
      new Uint8Array(4) as Uint8Array<ArrayBuffer>,
    );
  }

  const entries: GPUBindGroupEntry[] = inputs.map((arr, i) => ({
    binding: i,
    resource: { buffer: (arr.buffer() as WebGpuBuffer).handle },
  }));
  entries.push({
    binding: inputs.length,
    resource: { buffer: out_handle },
  });

  const bg = device.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries,
  });

  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipe);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(Math.ceil(plan.size / 64), 1, 1);
  pass.end();
  queue.submit([enc.finish()]);
}
