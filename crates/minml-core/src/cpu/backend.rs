use crate::buffer::Buffer;
use crate::device::Device;
use crate::error::{MinmlError, Result};
use parking_lot::RwLock;
use std::any::Any;
use std::sync::Arc;

// Plain heap-allocated bytes. RwLock (not Mutex) so multiple kernel reads
// over the same buffer don't deadlock — important when the same Array is
// passed twice to an op (e.g. dot(xy, xy)). Writes (h2d, output buffers)
// are exclusive. parking_lot::RwLock supports multiple readers on the
// same thread, which is exactly what we need.
pub struct CpuBuffer {
    pub(crate) data: RwLock<Vec<u8>>,
    bytes: usize,
}

impl CpuBuffer {
    pub fn new(bytes: usize) -> Self {
        Self {
            data: RwLock::new(vec![0u8; bytes]),
            bytes,
        }
    }
}

impl Buffer for CpuBuffer {
    fn bytes(&self) -> usize {
        self.bytes
    }
    fn device(&self) -> Device {
        Device::Cpu
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub fn allocate(bytes: usize) -> Arc<dyn Buffer> {
    Arc::new(CpuBuffer::new(bytes))
}

fn as_cpu(b: &dyn Buffer) -> &CpuBuffer {
    b.as_any().downcast_ref::<CpuBuffer>().expect("cpu buffer")
}

pub fn copy_host_to_buffer(dst: &dyn Buffer, src: &[u8]) -> Result<()> {
    let cpu = as_cpu(dst);
    let mut data = cpu.data.write();
    if src.len() > data.len() {
        return Err(MinmlError::Other(format!(
            "h2d size mismatch: src={} dst={}",
            src.len(),
            data.len()
        )));
    }
    data[..src.len()].copy_from_slice(src);
    Ok(())
}

pub fn copy_buffer_to_host(src: &dyn Buffer, dst: &mut [u8]) -> Result<()> {
    let cpu = as_cpu(src);
    let data = cpu.data.read();
    if dst.len() > data.len() {
        return Err(MinmlError::Other(format!(
            "d2h size mismatch: src={} dst={}",
            data.len(),
            dst.len()
        )));
    }
    dst.copy_from_slice(&data[..dst.len()]);
    Ok(())
}

// Internal helpers: typed views over the buffer bytes for kernels.
pub(crate) fn with_f32_mut<F, R>(buf: &dyn Buffer, f: F) -> R
where
    F: FnOnce(&mut [f32]) -> R,
{
    let cpu = as_cpu(buf);
    let mut bytes = cpu.data.write();
    f(bytemuck::cast_slice_mut::<u8, f32>(&mut bytes))
}

pub(crate) fn with_f32<F, R>(buf: &dyn Buffer, f: F) -> R
where
    F: FnOnce(&[f32]) -> R,
{
    let cpu = as_cpu(buf);
    let bytes = cpu.data.read();
    f(bytemuck::cast_slice::<u8, f32>(&bytes))
}

pub(crate) fn with_i32_mut<F, R>(buf: &dyn Buffer, f: F) -> R
where
    F: FnOnce(&mut [i32]) -> R,
{
    let cpu = as_cpu(buf);
    let mut bytes = cpu.data.write();
    f(bytemuck::cast_slice_mut::<u8, i32>(&mut bytes))
}

pub(crate) fn with_i32<F, R>(buf: &dyn Buffer, f: F) -> R
where
    F: FnOnce(&[i32]) -> R,
{
    let cpu = as_cpu(buf);
    let bytes = cpu.data.read();
    f(bytemuck::cast_slice::<u8, i32>(&bytes))
}
