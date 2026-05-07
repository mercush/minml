// Op entry points + per-primitive eval dispatchers. Each constructor builds
// a lazy Array; eval() walks the DAG and runs per-backend kernels.

use crate::array::Array;
use crate::cpu::{kernels as cpu_kernels, random as cpu_random};
use crate::device::Device;
use crate::dtype::DType;
use crate::error::{MinmlError, Result};
use crate::primitive::Primitive;
use crate::prng::PRNGKey;
use std::sync::Arc;

fn check_same_shape(a: &Array, b: &Array) -> Result<()> {
    if a.shape() != b.shape() {
        return Err(MinmlError::ShapeMismatch);
    }
    if a.device() != b.device() {
        return Err(MinmlError::DeviceMismatch);
    }
    Ok(())
}

// ---- add ----

pub fn add(a: &Array, b: &Array) -> Result<Array> {
    check_same_shape(a, b)?;
    Ok(Array::lazy(
        a.shape().to_vec(),
        a.dtype(),
        a.device(),
        Arc::new(AddPrim),
        vec![a.clone(), b.clone()],
    ))
}

pub struct AddPrim;
impl Primitive for AddPrim {
    fn name(&self) -> &'static str {
        "add"
    }
    fn eval(&self, inputs: &[Array], out: &Array) -> Result<()> {
        match out.device() {
            Device::Cpu => cpu_kernels::add(&inputs[0], &inputs[1], out),
            #[cfg(feature = "webgpu")]
            Device::WebGpu => crate::webgpu::add(&inputs[0], &inputs[1], out),
            #[cfg(not(feature = "webgpu"))]
            Device::WebGpu => Err(MinmlError::BackendNotBuilt("webgpu")),
            #[cfg(feature = "cuda")]
            Device::Cuda => crate::cuda::add(&inputs[0], &inputs[1], out),
            #[cfg(not(feature = "cuda"))]
            Device::Cuda => Err(MinmlError::BackendNotBuilt("cuda")),
        }
    }
}

// ---- mul ----

pub fn mul(a: &Array, b: &Array) -> Result<Array> {
    check_same_shape(a, b)?;
    Ok(Array::lazy(
        a.shape().to_vec(),
        a.dtype(),
        a.device(),
        Arc::new(MulPrim),
        vec![a.clone(), b.clone()],
    ))
}

pub struct MulPrim;
impl Primitive for MulPrim {
    fn name(&self) -> &'static str {
        "mul"
    }
    fn eval(&self, inputs: &[Array], out: &Array) -> Result<()> {
        match out.device() {
            Device::Cpu => cpu_kernels::mul(&inputs[0], &inputs[1], out),
            #[cfg(feature = "webgpu")]
            Device::WebGpu => crate::webgpu::mul(&inputs[0], &inputs[1], out),
            #[cfg(not(feature = "webgpu"))]
            Device::WebGpu => Err(MinmlError::BackendNotBuilt("webgpu")),
            #[cfg(feature = "cuda")]
            Device::Cuda => crate::cuda::mul(&inputs[0], &inputs[1], out),
            #[cfg(not(feature = "cuda"))]
            Device::Cuda => Err(MinmlError::BackendNotBuilt("cuda")),
        }
    }
}

// ---- dot ----

pub fn dot(a: &Array, b: &Array) -> Result<Array> {
    check_same_shape(a, b)?;
    if a.shape().len() != 1 {
        return Err(MinmlError::DotRequires1D);
    }
    Ok(Array::lazy(
        vec![1],
        a.dtype(),
        a.device(),
        Arc::new(DotPrim),
        vec![a.clone(), b.clone()],
    ))
}

pub struct DotPrim;
impl Primitive for DotPrim {
    fn name(&self) -> &'static str {
        "dot"
    }
    fn eval(&self, inputs: &[Array], out: &Array) -> Result<()> {
        match out.device() {
            Device::Cpu => cpu_kernels::dot(&inputs[0], &inputs[1], out),
            #[cfg(feature = "webgpu")]
            Device::WebGpu => crate::webgpu::dot(&inputs[0], &inputs[1], out),
            #[cfg(not(feature = "webgpu"))]
            Device::WebGpu => Err(MinmlError::BackendNotBuilt("webgpu")),
            #[cfg(feature = "cuda")]
            Device::Cuda => crate::cuda::dot(&inputs[0], &inputs[1], out),
            #[cfg(not(feature = "cuda"))]
            Device::Cuda => Err(MinmlError::BackendNotBuilt("cuda")),
        }
    }
}

// ---- ones ----

pub fn ones(shape: Vec<usize>, dtype: DType, device: Device) -> Array {
    Array::lazy(shape, dtype, device, Arc::new(OnesPrim), Vec::new())
}

pub struct OnesPrim;
impl Primitive for OnesPrim {
    fn name(&self) -> &'static str {
        "ones"
    }
    fn eval(&self, _inputs: &[Array], out: &Array) -> Result<()> {
        match out.device() {
            Device::Cpu => cpu_kernels::ones(out),
            _ => Err(MinmlError::OpNotImplemented {
                op: "ones",
                device: out.device(),
            }),
        }
    }
}

// ---- randint ----

pub fn randint(k0: u32, k1: u32, low: i32, high: i32, shape: Vec<usize>, device: Device) -> Array {
    Array::lazy(
        shape,
        DType::Int32,
        device,
        Arc::new(RandIntPrim { k0, k1, low, high }),
        Vec::new(),
    )
}

pub struct RandIntPrim {
    pub k0: u32,
    pub k1: u32,
    pub low: i32,
    pub high: i32,
}
impl Primitive for RandIntPrim {
    fn name(&self) -> &'static str {
        "randint"
    }
    fn eval(&self, _inputs: &[Array], out: &Array) -> Result<()> {
        match out.device() {
            Device::Cpu => cpu_random::randint(self.k0, self.k1, self.low, self.high, out),
            _ => Err(MinmlError::OpNotImplemented {
                op: "randint",
                device: out.device(),
            }),
        }
    }
}

// ---- gather ----

pub fn gather(table: &Array, indices: &Array) -> Result<Array> {
    if indices.dtype() != DType::Int32 {
        return Err(MinmlError::GatherIndicesNotInt32);
    }
    if table.shape().is_empty() {
        return Err(MinmlError::GatherTableRank);
    }
    let mut out_shape = indices.shape().to_vec();
    for d in &table.shape()[1..] {
        out_shape.push(*d);
    }
    Ok(Array::lazy(
        out_shape,
        table.dtype(),
        table.device(),
        Arc::new(GatherPrim),
        vec![table.clone(), indices.clone()],
    ))
}

pub struct GatherPrim;
impl Primitive for GatherPrim {
    fn name(&self) -> &'static str {
        "gather"
    }
    fn eval(&self, inputs: &[Array], out: &Array) -> Result<()> {
        match out.device() {
            Device::Cpu => cpu_kernels::gather(&inputs[0], &inputs[1], out),
            _ => Err(MinmlError::OpNotImplemented {
                op: "gather",
                device: out.device(),
            }),
        }
    }
}

// ---- distribution sample primitives ----

pub fn dirichlet_sample(k0: u32, k1: u32, alpha: &Array, batch_shape: Vec<usize>) -> Result<Array> {
    if alpha.shape().len() != 1 {
        return Err(MinmlError::DirichletAlphaNot1D);
    }
    let mut out_shape = batch_shape.clone();
    out_shape.push(alpha.shape()[0]);
    Ok(Array::lazy(
        out_shape,
        DType::Float32,
        alpha.device(),
        Arc::new(DirichletSamplePrim {
            k0,
            k1,
            batch_shape,
        }),
        vec![alpha.clone()],
    ))
}

pub struct DirichletSamplePrim {
    pub k0: u32,
    pub k1: u32,
    pub batch_shape: Vec<usize>,
}
impl Primitive for DirichletSamplePrim {
    fn name(&self) -> &'static str {
        "dirichlet_sample"
    }
    fn eval(&self, inputs: &[Array], out: &Array) -> Result<()> {
        match out.device() {
            Device::Cpu => {
                cpu_random::dirichlet_sample(self.k0, self.k1, &self.batch_shape, &inputs[0], out)
            }
            _ => Err(MinmlError::OpNotImplemented {
                op: "dirichlet_sample",
                device: out.device(),
            }),
        }
    }
}

pub fn categorical_sample(
    k0: u32,
    k1: u32,
    probs: &Array,
    batch_shape: Vec<usize>,
) -> Result<Array> {
    if probs.shape().len() != 1 {
        return Err(MinmlError::CategoricalProbsNot1D);
    }
    Ok(Array::lazy(
        batch_shape.clone(),
        DType::Int32,
        probs.device(),
        Arc::new(CategoricalSamplePrim {
            k0,
            k1,
            batch_shape,
        }),
        vec![probs.clone()],
    ))
}

pub struct CategoricalSamplePrim {
    pub k0: u32,
    pub k1: u32,
    pub batch_shape: Vec<usize>,
}
impl Primitive for CategoricalSamplePrim {
    fn name(&self) -> &'static str {
        "categorical_sample"
    }
    fn eval(&self, inputs: &[Array], out: &Array) -> Result<()> {
        match out.device() {
            Device::Cpu => cpu_random::categorical_sample(
                self.k0,
                self.k1,
                &self.batch_shape,
                &inputs[0],
                out,
            ),
            _ => Err(MinmlError::OpNotImplemented {
                op: "categorical_sample",
                device: out.device(),
            }),
        }
    }
}

// ---- Distribution wrappers ----

pub struct Dirichlet {
    pub alpha: Array,
}

impl Dirichlet {
    pub fn new(alpha: Array) -> Self {
        Self { alpha }
    }
    pub fn sample(&self, key: PRNGKey, batch_shape: Vec<usize>) -> Result<Array> {
        dirichlet_sample(key.k0(), key.k1(), &self.alpha, batch_shape)
    }
}

pub struct Categorical {
    pub probs: Array,
}

impl Categorical {
    pub fn new(probs: Array) -> Self {
        Self { probs }
    }
    pub fn sample(&self, key: PRNGKey, batch_shape: Vec<usize>) -> Result<Array> {
        categorical_sample(key.k0(), key.k1(), &self.probs, batch_shape)
    }
}

pub struct Normal;

impl Normal {
    pub fn new() -> Self {
        Self
    }
    pub fn sample(&self, _key: PRNGKey, _batch_shape: Vec<usize>) -> Result<Array> {
        Err(MinmlError::Other(
            "Normal::sample: not implemented yet".into(),
        ))
    }
}

impl Default for Normal {
    fn default() -> Self {
        Self::new()
    }
}

