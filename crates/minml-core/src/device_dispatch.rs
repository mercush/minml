// Internal dispatch helpers — call the right backend for allocate / h2d /
// d2h. This is the Rust analog of the static helpers in src/array.cpp:22-47.
// Async d2h (the WebGPU-aware one) lives elsewhere; this file is sync only
// and is used by Array constructors and CPU readback.

use crate::buffer::Buffer;
use crate::cpu::backend as cpu_backend;
use crate::device::Device;
#[allow(unused_imports)]
use crate::error::{MinmlError, Result};
use std::sync::Arc;

pub(crate) fn allocate(d: Device, bytes: usize) -> Result<Arc<dyn Buffer>> {
    match d {
        Device::Cpu => Ok(cpu_backend::allocate(bytes)),
        #[cfg(feature = "webgpu")]
        Device::WebGpu => crate::webgpu::allocate(bytes),
        #[cfg(not(feature = "webgpu"))]
        Device::WebGpu => Err(MinmlError::BackendNotBuilt("webgpu")),
        #[cfg(feature = "cuda")]
        Device::Cuda => crate::cuda::allocate(bytes),
        #[cfg(not(feature = "cuda"))]
        Device::Cuda => Err(MinmlError::BackendNotBuilt("cuda")),
    }
}

pub(crate) fn h2d(d: Device, dst: &dyn Buffer, src: &[u8]) -> Result<()> {
    match d {
        Device::Cpu => cpu_backend::copy_host_to_buffer(dst, src),
        #[cfg(feature = "webgpu")]
        Device::WebGpu => crate::webgpu::copy_host_to_buffer(dst, src),
        #[cfg(not(feature = "webgpu"))]
        Device::WebGpu => Err(MinmlError::BackendNotBuilt("webgpu")),
        #[cfg(feature = "cuda")]
        Device::Cuda => crate::cuda::copy_host_to_buffer(dst, src),
        #[cfg(not(feature = "cuda"))]
        Device::Cuda => Err(MinmlError::BackendNotBuilt("cuda")),
    }
}

// Async d2h. CPU/CUDA wrap their sync d2h in `async { Ready(()) }`; only
// WebGPU has a real future (driven by Buffer::slice.map_async).
pub(crate) async fn d2h_async(d: Device, src: &dyn Buffer, dst: &mut [u8]) -> Result<()> {
    match d {
        Device::Cpu => cpu_backend::copy_buffer_to_host(src, dst),
        #[cfg(feature = "webgpu")]
        Device::WebGpu => crate::webgpu::copy_buffer_to_host_async(src, dst).await,
        #[cfg(not(feature = "webgpu"))]
        Device::WebGpu => Err(MinmlError::BackendNotBuilt("webgpu")),
        #[cfg(feature = "cuda")]
        Device::Cuda => crate::cuda::copy_buffer_to_host(src, dst),
        #[cfg(not(feature = "cuda"))]
        Device::Cuda => Err(MinmlError::BackendNotBuilt("cuda")),
    }
}
