// minml-core
//
// Pure Rust port of minml. Lazy `Array` graph, `Primitive`-per-op,
// per-backend free functions (CPU, WebGPU, CUDA). The user-facing
// surface is mostly sync (graph builders); only `Array::tolist`,
// `Array::item` and `init_webgpu` are async,
// since those are the only places that can block on a GPU.
mod array;
mod buffer;
mod device;
mod device_dispatch;
mod dtype;
mod error;
mod ops;
mod primitive;
mod prng;
mod threefry;
mod transforms;

pub mod cpu;
#[cfg(feature = "webgpu")]
pub mod webgpu;
#[cfg(feature = "cuda")]
pub mod cuda;

pub use array::Array;
pub use buffer::Buffer;
pub use device::{default_device, set_default_device, Device};
pub use dtype::{dtype_bytes, DType};
pub use error::{MinmlError, Result as MinmlResult};
pub use ops::{
    add, categorical_sample, dirichlet_sample, dot, gather, mul, ones, randint, Categorical,
    Dirichlet, Normal,
};
pub use primitive::Primitive;
pub use prng::PRNGKey;
pub use transforms::{slice_axis0, stack, vmap_apply, VmapCallable};
