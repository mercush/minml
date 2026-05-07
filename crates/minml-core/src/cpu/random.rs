// CPU random / sampling — direct port of cpu_random.cpp.
//
// All draws come from Threefry-2x32. Per-element indexing uses
// (op_tag, key, output_index, sub_iter); op_tag prevents different ops
// from producing identical bits when given the same key.

use crate::array::Array;
use crate::cpu::backend::{with_f32, with_f32_mut, with_i32_mut};
use crate::error::{MinmlError, Result};
use crate::threefry::{threefry_2x32, threefry_u32, u32_to_unit_f32};

const TAG_RANDINT: u32 = 0x52414E44; // 'RAND'
const TAG_DIRICHLET: u32 = 0x44495243; // 'DIRC'
const TAG_CATEGORICAL_U: u32 = 0x43415455; // 'CATU'

fn uniform_for(k0: u32, k1: u32, tag: u32, i: u32, sub: u32) -> f32 {
    let (a, _) = threefry_2x32(k0 ^ tag, k1, i, sub);
    u32_to_unit_f32(a)
}

// Marsaglia & Tsang gamma sampler for shape >= 1, with the boost trick for
// shape < 1.
fn sample_gamma(k0: u32, k1: u32, shape: f32, base_idx: u32) -> f32 {
    if shape < 1.0 {
        let g = sample_gamma(k0, k1, shape + 1.0, base_idx);
        let mut u = uniform_for(k0, k1, 0xDEADBEEF, base_idx, 99);
        if u < 1e-30 {
            u = 1e-30;
        }
        return g * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    let mut sub: u32 = 0;
    loop {
        let mut u1 = uniform_for(k0, k1, 0xCAFE0001, base_idx, sub);
        sub += 1;
        let u2 = uniform_for(k0, k1, 0xCAFE0002, base_idx, sub);
        sub += 1;
        if u1 < 1e-30 {
            u1 = 1e-30;
        }
        let x = (-2.0 * u1.ln()).sqrt() * (6.2831853 * u2).cos();
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v3 = v * v * v;
        let u = uniform_for(k0, k1, 0xCAFE0003, base_idx, sub);
        sub += 1;
        if u < 1.0 - 0.0331 * x * x * x * x {
            return d * v3;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v3 + v3.ln()) {
            return d * v3;
        }
    }
}

pub fn randint(k0: u32, k1: u32, low: i32, high: i32, out: &Array) -> Result<()> {
    if high <= low {
        return Err(MinmlError::Other("randint: high <= low".into()));
    }
    let span = (high - low) as u32;
    let n = out.size();
    let buf = out.buffer().expect("evaluated");
    with_i32_mut(&*buf, |p| {
        for i in 0..n {
            let bits = threefry_u32(k0 ^ TAG_RANDINT, k1, i as u32);
            p[i] = low + (bits % span) as i32;
        }
    });
    Ok(())
}

pub fn dirichlet_sample(
    k0: u32,
    k1: u32,
    batch_shape: &[usize],
    alpha: &Array,
    out: &Array,
) -> Result<()> {
    let big_k = alpha.shape()[0];
    let mut big_b: usize = 1;
    for d in batch_shape {
        big_b *= *d;
    }
    let buf_a = alpha.buffer().expect("evaluated");
    let buf_o = out.buffer().expect("evaluated");
    with_f32(&*buf_a, |a| {
        with_f32_mut(&*buf_o, |o| {
            for b in 0..big_b {
                let row = &mut o[b * big_k..(b + 1) * big_k];
                let mut sum = 0.0f32;
                for k in 0..big_k {
                    let idx = (b * big_k + k) as u32;
                    row[k] = sample_gamma(k0 ^ TAG_DIRICHLET, k1, a[k], idx);
                    sum += row[k];
                }
                if sum > 0.0 {
                    for v in row.iter_mut() {
                        *v /= sum;
                    }
                }
            }
        })
    });
    Ok(())
}

pub fn categorical_sample(
    k0: u32,
    k1: u32,
    batch_shape: &[usize],
    probs: &Array,
    out: &Array,
) -> Result<()> {
    let big_k = probs.shape()[0];
    let mut big_b: usize = 1;
    for d in batch_shape {
        big_b *= *d;
    }
    if big_b == 0 {
        big_b = 1;
    }
    let buf_p = probs.buffer().expect("evaluated");
    let buf_o = out.buffer().expect("evaluated");

    let mut cdf = vec![0f32; big_k];
    let total = with_f32(&*buf_p, |p| {
        let mut t = 0.0f32;
        for k in 0..big_k {
            t += p[k];
            cdf[k] = t;
        }
        t
    });
    if total <= 0.0 {
        return Err(MinmlError::Other("categorical: probs sum to 0".into()));
    }
    with_i32_mut(&*buf_o, |o| {
        for b in 0..big_b {
            let u = uniform_for(k0, k1, TAG_CATEGORICAL_U, b as u32, 0) * total;
            let mut pick = (big_k as i32) - 1;
            for k in 0..big_k {
                if u < cdf[k] {
                    pick = k as i32;
                    break;
                }
            }
            o[b] = pick;
        }
    });
    Ok(())
}
