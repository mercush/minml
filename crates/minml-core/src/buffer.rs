use crate::device::Device;
use std::any::Any;

// Opaque per-backend memory. Each backend has its own concrete type
// (CpuBuffer holds Vec<u8>; WebGpuBuffer holds wgpu::Buffer; CudaBuffer
// holds an FFI handle) with its own Drop. Array stores Arc<dyn Buffer>
// so it doesn't need to know which backend is in use.
pub trait Buffer: Any + Send + Sync {
    fn bytes(&self) -> usize;
    fn device(&self) -> Device;
    fn as_any(&self) -> &dyn Any;
}
