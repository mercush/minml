// src/webgpu_shaders.h
//
// WGSL kernels. Embedded as raw string literals so the build doesn't need
// a runtime file. dot uses a workgroup-shared partial sum + atomic add to
// the output buffer (one workgroup per chunk). For a real lib you'd avoid
// the f32 atomic by doing a multi-pass reduction; this is the minimal
// version.
#pragma once

namespace minml {

inline constexpr const char* kAddWgsl = R"WGSL(
@group(0) @binding(0) var<storage, read>        a   : array<f32>;
@group(0) @binding(1) var<storage, read>        b   : array<f32>;
@group(0) @binding(2) var<storage, read_write>  out : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i < arrayLength(&out)) {
    out[i] = a[i] + b[i];
  }
}
)WGSL";

// Dot product. Each workgroup reduces its slice into scratch[0], then thread 0
// of the workgroup adds it into out[0] using a CAS loop on the bit pattern
// (WGSL has no f32 atomics; we cast to u32 and CAS). Plenty fast for a demo.
inline constexpr const char* kDotWgsl = R"WGSL(
@group(0) @binding(0) var<storage, read>                 a   : array<f32>;
@group(0) @binding(1) var<storage, read>                 b   : array<f32>;
@group(0) @binding(2) var<storage, read_write>           out : array<atomic<u32>>;

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
  let n = arrayLength(&a);
  var v : f32 = 0.0;
  if (i < n) { v = a[i] * b[i]; }
  scratch[lid.x] = v;
  workgroupBarrier();

  // Tree reduction within the workgroup.
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
)WGSL";

}  // namespace minml
