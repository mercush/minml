use crate::device::Device;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MinmlError {
    #[error("shape mismatch")]
    ShapeMismatch,
    #[error("device mismatch")]
    DeviceMismatch,
    #[error("dtype mismatch")]
    DtypeMismatch,
    #[error("data size {got} != product(shape)={expected}")]
    DataSize { got: usize, expected: usize },
    #[error("dot requires 1-D inputs")]
    DotRequires1D,
    #[error("gather: indices must be Int32")]
    GatherIndicesNotInt32,
    #[error("gather: table must have rank >= 1")]
    GatherTableRank,
    #[error("gather: index out of bounds")]
    GatherOob,
    #[error("dirichlet_sample: alpha must be 1-D")]
    DirichletAlphaNot1D,
    #[error("categorical_sample: probs must be 1-D")]
    CategoricalProbsNot1D,
    #[error("item() requires size==1")]
    ItemRequiresSize1,
    #[error("operation '{op}' not implemented for device {device}")]
    OpNotImplemented { op: &'static str, device: Device },
    #[error("backend '{0}' was not built into this binary")]
    BackendNotBuilt(&'static str),
    #[error("WebGPU not initialized; call init_webgpu() first")]
    WebGpuNotInitialized,
    #[error("WebGPU init failed: {0}")]
    WebGpuInitFailed(String),
    #[error("WebGPU readback failed")]
    WebGpuReadbackFailed,
    #[error("vmap: {0}")]
    Vmap(&'static str),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, MinmlError>;
