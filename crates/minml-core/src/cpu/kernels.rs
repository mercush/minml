// CPU kernels: add, mul, dot, ones, gather. Direct port of cpu_backend.cpp.
use crate::array::Array;
use crate::cpu::backend::{with_f32, with_f32_mut, with_i32, with_i32_mut};
use crate::dtype::DType;
use crate::error::{MinmlError, Result};

pub fn add(a: &Array, b: &Array, out: &Array) -> Result<()> {
    let n = out.size();
    let buf_a = a.buffer().expect("evaluated");
    let buf_b = b.buffer().expect("evaluated");
    let buf_out = out.buffer().expect("evaluated");
    with_f32(&*buf_a, |pa| {
        with_f32(&*buf_b, |pb| {
            with_f32_mut(&*buf_out, |po| {
                for i in 0..n {
                    po[i] = pa[i] + pb[i];
                }
            })
        })
    });
    Ok(())
}

pub fn mul(a: &Array, b: &Array, out: &Array) -> Result<()> {
    let n = out.size();
    let buf_a = a.buffer().expect("evaluated");
    let buf_b = b.buffer().expect("evaluated");
    let buf_out = out.buffer().expect("evaluated");
    with_f32(&*buf_a, |pa| {
        with_f32(&*buf_b, |pb| {
            with_f32_mut(&*buf_out, |po| {
                for i in 0..n {
                    po[i] = pa[i] * pb[i];
                }
            })
        })
    });
    Ok(())
}

pub fn dot(a: &Array, b: &Array, out: &Array) -> Result<()> {
    let n = a.size();
    let buf_a = a.buffer().expect("evaluated");
    let buf_b = b.buffer().expect("evaluated");
    let buf_out = out.buffer().expect("evaluated");
    let mut sum = 0.0f32;
    with_f32(&*buf_a, |pa| {
        with_f32(&*buf_b, |pb| {
            for i in 0..n {
                sum += pa[i] * pb[i];
            }
        })
    });
    with_f32_mut(&*buf_out, |po| po[0] = sum);
    Ok(())
}

pub fn ones(out: &Array) -> Result<()> {
    let n = out.size();
    let buf_out = out.buffer().expect("evaluated");
    match out.dtype() {
        DType::Float32 => with_f32_mut(&*buf_out, |p| {
            for v in &mut p[..n] {
                *v = 1.0;
            }
        }),
        DType::Int32 => with_i32_mut(&*buf_out, |p| {
            for v in &mut p[..n] {
                *v = 1;
            }
        }),
    }
    Ok(())
}

pub fn gather(table: &Array, indices: &Array, out: &Array) -> Result<()> {
    let big_n = table.shape()[0];
    let mut trail: usize = 1;
    for d in &table.shape()[1..] {
        trail *= *d;
    }
    let m = indices.size();
    let buf_t = table.buffer().expect("evaluated");
    let buf_idx = indices.buffer().expect("evaluated");
    let buf_out = out.buffer().expect("evaluated");

    let mut err: Option<MinmlError> = None;
    with_f32(&*buf_t, |t| {
        with_i32(&*buf_idx, |idx| {
            with_f32_mut(&*buf_out, |o| {
                for i in 0..m {
                    let k = idx[i];
                    if k < 0 || (k as usize) >= big_n {
                        err = Some(MinmlError::GatherOob);
                        return;
                    }
                    let src = (k as usize) * trail;
                    let dst = i * trail;
                    o[dst..dst + trail].copy_from_slice(&t[src..src + trail]);
                }
            })
        })
    });
    if let Some(e) = err {
        return Err(e);
    }
    Ok(())
}
