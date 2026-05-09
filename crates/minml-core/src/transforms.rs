// Transforms: slice_axis0, stack, vmap_apply, jit.
//
// slice/stack/vmap stay sync + CPU-only (same scope as transforms.cpp:23).
// `jit` is a graph-rewrite pass: it traces the user's callable, walks the
// resulting lazy DAG, and folds chains of elementwise add/mul into a single
// FusedElementwise primitive. On WebGPU and CUDA that primitive emits one
// kernel (one launch, one round-trip through global memory) instead of one
// kernel per op + an intermediate buffer per result.

use crate::array::Array;
use crate::cpu::backend as cpu_backend;
use crate::device::Device;
use crate::dtype::{dtype_bytes, DType};
use crate::error::{MinmlError, Result};
use crate::primitive::Primitive;
use std::sync::Arc;

fn product_after_first(shape: &[usize]) -> usize {
    shape.iter().skip(1).product()
}

pub fn slice_axis0(arr_in: &Array) -> Result<Vec<Array>> {
    if arr_in.device() != Device::Cpu {
        return Err(MinmlError::Other("slice_axis0: CPU only for now".into()));
    }
    arr_in.eval()?;
    if arr_in.shape().is_empty() {
        return Err(MinmlError::Other("slice_axis0: cannot slice a scalar".into()));
    }
    let big_n = arr_in.shape()[0];
    let sub_shape: Vec<usize> = arr_in.shape()[1..].to_vec();
    let per = product_after_first(arr_in.shape());

    let buf = arr_in.buffer().expect("evaluated");
    let mut out: Vec<Array> = Vec::with_capacity(big_n);
    match arr_in.dtype() {
        DType::Float32 => {
            cpu_backend::with_f32(&*buf, |data| -> Result<()> {
                for i in 0..big_n {
                    let chunk = data[i * per..(i + 1) * per].to_vec();
                    out.push(Array::from_f32_with_shape(
                        chunk,
                        sub_shape.clone(),
                        arr_in.device(),
                    )?);
                }
                Ok(())
            })?;
        }
        DType::Int32 => {
            cpu_backend::with_i32(&*buf, |data| -> Result<()> {
                for i in 0..big_n {
                    let chunk = data[i * per..(i + 1) * per].to_vec();
                    out.push(Array::from_i32_with_shape(
                        chunk,
                        sub_shape.clone(),
                        arr_in.device(),
                    )?);
                }
                Ok(())
            })?;
        }
    }
    Ok(out)
}

pub fn stack(parts: &[Array]) -> Result<Array> {
    if parts.is_empty() {
        return Err(MinmlError::Other("stack: empty input".into()));
    }
    let base_shape = parts[0].shape().to_vec();
    let dev = parts[0].device();
    let dt = parts[0].dtype();
    for p in parts {
        if p.shape() != base_shape.as_slice() {
            return Err(MinmlError::Other("stack: shape mismatch".into()));
        }
        if p.device() != dev {
            return Err(MinmlError::Other("stack: device mismatch".into()));
        }
        if p.dtype() != dt {
            return Err(MinmlError::Other("stack: dtype mismatch".into()));
        }
    }
    if dev != Device::Cpu {
        return Err(MinmlError::Other("stack: CPU only for now".into()));
    }
    let mut out_shape = Vec::with_capacity(base_shape.len() + 1);
    out_shape.push(parts.len());
    out_shape.extend_from_slice(&base_shape);

    let per = parts[0].size();
    let bytes_per = per * dtype_bytes(dt);
    let total_bytes = bytes_per * parts.len();
    let mut buf = vec![0u8; total_bytes];

    for (i, p) in parts.iter().enumerate() {
        p.eval()?;
        let pbuf = p.buffer().expect("evaluated");
        let dst = &mut buf[i * bytes_per..(i + 1) * bytes_per];
        cpu_backend::copy_buffer_to_host(&*pbuf, dst)?;
    }

    match dt {
        DType::Float32 => {
            let data: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&buf).to_vec();
            Array::from_f32_with_shape(data, out_shape, dev)
        }
        DType::Int32 => {
            let data: Vec<i32> = bytemuck::cast_slice::<u8, i32>(&buf).to_vec();
            Array::from_i32_with_shape(data, out_shape, dev)
        }
    }
}

// Per-iteration callable. Receives:
//   * iter_index: which batch element we're on (used by binding shims to
//     look up language-native list inputs).
//   * args: Array arguments with batched ones already sliced (shape ==
//     orig.shape[1:]); unbatched ones passed through unchanged.
// Returns one Array per leaf of the function's logical return value.
pub type VmapCallable<'a> = dyn FnMut(usize, &[Array]) -> Result<Vec<Array>> + 'a;

// vmap_apply: orchestration loop. Direct port of transforms.cpp:94.
pub fn vmap_apply(
    big_n: usize,
    args: &[Array],
    in_axes: &[i32],
    f: &mut VmapCallable<'_>,
) -> Result<Vec<Array>> {
    if args.len() != in_axes.len() {
        return Err(MinmlError::Vmap("args/in_axes size mismatch"));
    }
    // Pre-slice batched inputs once.
    let mut sliced: Vec<Vec<Array>> = vec![Vec::new(); args.len()];
    for i in 0..args.len() {
        if in_axes[i] < 0 {
            continue;
        }
        if in_axes[i] != 0 {
            return Err(MinmlError::Vmap("only axis 0 supported"));
        }
        if args[i].shape().is_empty() {
            return Err(MinmlError::Vmap("cannot batch over a scalar"));
        }
        if args[i].shape()[0] != big_n {
            return Err(MinmlError::Vmap("batched dims disagree"));
        }
        sliced[i] = slice_axis0(&args[i])?;
    }

    let mut all_leaves: Vec<Vec<Array>> = Vec::with_capacity(big_n);
    for b in 0..big_n {
        let mut per_iter: Vec<Array> = Vec::with_capacity(args.len());
        for i in 0..args.len() {
            if in_axes[i] >= 0 {
                per_iter.push(sliced[i][b].clone());
            } else {
                per_iter.push(args[i].clone());
            }
        }
        all_leaves.push(f(b, &per_iter)?);
    }
    if all_leaves.is_empty() {
        return Err(MinmlError::Vmap("N=0"));
    }
    let n_leaves = all_leaves[0].len();
    for v in &all_leaves {
        if v.len() != n_leaves {
            return Err(MinmlError::Vmap("leaf count varies across iterations"));
        }
    }
    let mut stacked: Vec<Array> = Vec::with_capacity(n_leaves);
    for l in 0..n_leaves {
        let parts: Vec<Array> = all_leaves.iter().map(|v| v[l].clone()).collect();
        stacked.push(stack(&parts)?);
    }
    Ok(stacked)
}

// ============================================================================
// jit — kernel-fusion transform for elementwise add/mul chains.
//
// Mirrors the surface of vmap_apply: the caller hands in a closure that
// builds a (lazy) graph from a slice of input Arrays. `jit` traces it once
// — i.e. just runs it — and rewrites each output's producer DAG, replacing
// every contiguous chain of same-shape, same-device, Float32 add/mul ops
// with a single FusedElementwise primitive. The fused primitive holds an
// expression AST and dispatches per-device: for WebGPU and CUDA it emits
// one kernel and one launch; for CPU it walks the AST per element (which
// is still a win because no intermediate buffer is allocated).
// ============================================================================

pub type JitCallable<'a> = dyn FnMut(&[Array]) -> Result<Vec<Array>> + 'a;

pub fn jit(args: &[Array], f: &mut JitCallable<'_>) -> Result<Vec<Array>> {
    let outputs = f(args)?;
    Ok(outputs.iter().map(fuse_array).collect())
}

#[derive(Clone)]
enum FusedExpr {
    Input(usize),
    Add(Arc<FusedExpr>, Arc<FusedExpr>),
    Mul(Arc<FusedExpr>, Arc<FusedExpr>),
}

// Walk the DAG rooted at `arr` and pull contiguous Float32 add/mul chains
// into one FusedExpr. Anything else (already-evaluated buffer, non-fusable
// primitive, dtype/shape/device mismatch) becomes a leaf input.
fn fuse_array(arr: &Array) -> Array {
    if arr.evaluated() || arr.dtype() != DType::Float32 {
        return arr.clone();
    }
    let mut inputs: Vec<Array> = Vec::new();
    let expr = collect(arr, arr.shape(), arr.device(), &mut inputs);
    // No fusion possible (root was a leaf): hand back the original Array.
    if matches!(expr, FusedExpr::Input(_)) {
        return arr.clone();
    }
    Array::lazy(
        arr.shape().to_vec(),
        arr.dtype(),
        arr.device(),
        Arc::new(FusedElementwisePrim { expr }),
        inputs,
    )
}

fn collect(
    arr: &Array,
    target_shape: &[usize],
    target_device: Device,
    inputs: &mut Vec<Array>,
) -> FusedExpr {
    if arr.evaluated()
        || arr.shape() != target_shape
        || arr.device() != target_device
        || arr.dtype() != DType::Float32
    {
        return push_input(arr, inputs);
    }
    let Some((prim, prim_inputs)) = arr.lazy_state() else {
        return push_input(arr, inputs);
    };
    let kind = match prim.name() {
        "add" => FuseKind::Add,
        "mul" => FuseKind::Mul,
        _ => return push_input(arr, inputs),
    };
    if prim_inputs.len() != 2 {
        return push_input(arr, inputs);
    }
    let lhs = collect(&prim_inputs[0], target_shape, target_device, inputs);
    let rhs = collect(&prim_inputs[1], target_shape, target_device, inputs);
    match kind {
        FuseKind::Add => FusedExpr::Add(Arc::new(lhs), Arc::new(rhs)),
        FuseKind::Mul => FusedExpr::Mul(Arc::new(lhs), Arc::new(rhs)),
    }
}

enum FuseKind {
    Add,
    Mul,
}

// De-dup leaves by inner_id so `x*x` only allocates one input slot.
fn push_input(arr: &Array, inputs: &mut Vec<Array>) -> FusedExpr {
    let id = arr.inner_id();
    for (i, e) in inputs.iter().enumerate() {
        if e.inner_id() == id {
            return FusedExpr::Input(i);
        }
    }
    let idx = inputs.len();
    inputs.push(arr.clone());
    FusedExpr::Input(idx)
}

// ---- Source generation ----

fn render_cpu(expr: &FusedExpr, inputs: &[&[f32]], i: usize) -> f32 {
    match expr {
        FusedExpr::Input(k) => inputs[*k][i],
        FusedExpr::Add(l, r) => render_cpu(l, inputs, i) + render_cpu(r, inputs, i),
        FusedExpr::Mul(l, r) => render_cpu(l, inputs, i) * render_cpu(r, inputs, i),
    }
}

fn render_str(expr: &FusedExpr, names: &[String]) -> String {
    match expr {
        FusedExpr::Input(k) => format!("{}[i]", names[*k]),
        FusedExpr::Add(l, r) => format!("({} + {})", render_str(l, names), render_str(r, names)),
        FusedExpr::Mul(l, r) => format!("({} * {})", render_str(l, names), render_str(r, names)),
    }
}

fn input_names(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("in{i}")).collect()
}

#[cfg(feature = "webgpu")]
fn build_wgsl(expr: &FusedExpr, n_inputs: usize) -> String {
    let names = input_names(n_inputs);
    let mut s = String::new();
    for (i, name) in names.iter().enumerate() {
        s.push_str(&format!(
            "@group(0) @binding({i}) var<storage, read> {name} : array<f32>;\n"
        ));
    }
    s.push_str(&format!(
        "@group(0) @binding({}) var<storage, read_write> out : array<f32>;\n\n",
        n_inputs
    ));
    s.push_str("@compute @workgroup_size(64)\n");
    s.push_str("fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n");
    s.push_str("  let i = gid.x;\n");
    s.push_str("  if (i < arrayLength(&out)) {\n");
    s.push_str(&format!("    out[i] = {};\n", render_str(expr, &names)));
    s.push_str("  }\n}\n");
    s
}

#[cfg(feature = "cuda")]
fn build_cuda_c(expr: &FusedExpr, n_inputs: usize, fn_name: &str) -> String {
    let names = input_names(n_inputs);
    let mut s = String::new();
    s.push_str(&format!("extern \"C\" __global__\nvoid {fn_name}("));
    for name in &names {
        s.push_str(&format!("const float* __restrict__ {name}, "));
    }
    s.push_str("float* __restrict__ out, unsigned int n) {\n");
    s.push_str("    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;\n");
    s.push_str(&format!(
        "    if (i < n) out[i] = {};\n}}\n",
        render_str(expr, &names)
    ));
    s
}

// ---- The fused primitive ----

struct FusedElementwisePrim {
    expr: FusedExpr,
}

impl Primitive for FusedElementwisePrim {
    fn name(&self) -> &'static str {
        "fused_elementwise"
    }
    fn eval(&self, inputs: &[Array], out: &Array) -> Result<()> {
        match out.device() {
            Device::Cpu => eval_cpu(&self.expr, inputs, out),
            #[cfg(feature = "webgpu")]
            Device::WebGpu => eval_webgpu(&self.expr, inputs, out),
            #[cfg(not(feature = "webgpu"))]
            Device::WebGpu => Err(MinmlError::BackendNotBuilt("webgpu")),
            #[cfg(feature = "cuda")]
            Device::Cuda => eval_cuda(&self.expr, inputs, out),
            #[cfg(not(feature = "cuda"))]
            Device::Cuda => Err(MinmlError::BackendNotBuilt("cuda")),
        }
    }
}

fn eval_cpu(expr: &FusedExpr, inputs: &[Array], out: &Array) -> Result<()> {
    let n = out.size();
    // All inputs must already be evaluated by the time the primitive runs;
    // Array::eval drives the post-order walk for us.
    let in_bufs: Vec<_> = inputs
        .iter()
        .map(|a| a.buffer().expect("evaluated"))
        .collect();
    let buf_o = out.buffer().expect("evaluated");

    // Pull each input's f32 slice through cpu_backend::with_f32 and stash it
    // as a raw pointer + len. We only read while all the closures are live
    // (we re-enter with_f32 in a nested-closures fashion to keep aliasing
    // safe), but a flat loop is much easier — and for CPU buffers the
    // backing storage is a Vec<u8> behind a Mutex that with_f32 just locks
    // and exposes by pointer.
    cpu_backend::with_f32_mut(&*buf_o, |out_slice| {
        with_inputs_f32(&in_bufs, |inputs_f32| {
            for i in 0..n {
                out_slice[i] = render_cpu(expr, inputs_f32, i);
            }
        });
    });
    Ok(())
}

// Recursive helper: locks each input's f32 view in turn so the closure
// receives a slice-of-slices for the whole input list at once.
fn with_inputs_f32<F: FnOnce(&[&[f32]])>(bufs: &[Arc<dyn crate::buffer::Buffer>], f: F) {
    fn inner<F: FnOnce(&[&[f32]])>(
        bufs: &[Arc<dyn crate::buffer::Buffer>],
        acc: &mut Vec<*const f32>,
        lens: &mut Vec<usize>,
        f: F,
    ) {
        if let Some((head, rest)) = bufs.split_first() {
            cpu_backend::with_f32(&**head, |s| {
                acc.push(s.as_ptr());
                lens.push(s.len());
                inner(rest, acc, lens, f);
                acc.pop();
                lens.pop();
            });
        } else {
            // SAFETY: each pointer/len pair was produced by with_f32 in an
            // outer frame whose closure is still live, so the slice is valid
            // for the duration of `f`.
            let slices: Vec<&[f32]> = acc
                .iter()
                .zip(lens.iter())
                .map(|(p, l)| unsafe { std::slice::from_raw_parts(*p, *l) })
                .collect();
            f(&slices);
        }
    }
    let mut ptrs: Vec<*const f32> = Vec::with_capacity(bufs.len());
    let mut lens: Vec<usize> = Vec::with_capacity(bufs.len());
    inner(bufs, &mut ptrs, &mut lens, f);
}

#[cfg(feature = "webgpu")]
fn eval_webgpu(expr: &FusedExpr, inputs: &[Array], out: &Array) -> Result<()> {
    let wgsl = build_wgsl(expr, inputs.len());
    crate::webgpu::dispatch_jit_elementwise(&wgsl, inputs, out)
}

#[cfg(feature = "cuda")]
fn eval_cuda(expr: &FusedExpr, inputs: &[Array], out: &Array) -> Result<()> {
    let fn_name = "minml_jit_fused";
    let src = build_cuda_c(expr, inputs.len(), fn_name);
    crate::cuda::launch_fused_elementwise(&src, fn_name, inputs, out)
}
