// Transforms: slice_axis0, stack, vmap_apply.
//
// All sync, CPU-only — same scope as the C++ original (transforms.cpp:23
// throws on non-CPU). Slicing forces eval; the per-iter callable builds
// lazy graphs which are stacked back into a CPU array.

use crate::array::Array;
use crate::cpu::backend as cpu_backend;
use crate::device::Device;
use crate::dtype::{dtype_bytes, DType};
use crate::error::{MinmlError, Result};

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
