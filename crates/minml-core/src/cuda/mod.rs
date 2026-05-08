// CUDA backend — extern "C" FFI to crates/minml-core/cuda/kernels.cu.
//
// Built only when feature="cuda". The .cu file is compiled by build.rs via
// the cc crate's CUDA support; cudart is linked at build time.

use crate::array::Array;
use crate::buffer::Buffer;
use crate::device::Device;
use crate::error::{MinmlError, Result};
use std::any::Any;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

#[repr(C)]
struct MinmlCudaBuf {
    _priv: [u8; 0],
}

unsafe extern "C" {
    fn minml_cuda_alloc(bytes: usize) -> *mut MinmlCudaBuf;
    fn minml_cuda_free(buf: *mut MinmlCudaBuf);
    fn minml_cuda_h2d(dst: *mut MinmlCudaBuf, src: *const c_void, bytes: usize) -> i32;
    fn minml_cuda_d2h(dst: *mut c_void, src: *const MinmlCudaBuf, bytes: usize) -> i32;
    fn minml_cuda_add(
        a: *const MinmlCudaBuf,
        b: *const MinmlCudaBuf,
        out: *mut MinmlCudaBuf,
        n: usize,
    ) -> i32;
    fn minml_cuda_mul(
        a: *const MinmlCudaBuf,
        b: *const MinmlCudaBuf,
        out: *mut MinmlCudaBuf,
        n: usize,
    ) -> i32;
    fn minml_cuda_dot(
        a: *const MinmlCudaBuf,
        b: *const MinmlCudaBuf,
        out: *mut MinmlCudaBuf,
        n: usize,
    ) -> i32;
}

pub struct CudaBuffer {
    handle: NonNull<MinmlCudaBuf>,
    bytes: usize,
}

unsafe impl Send for CudaBuffer {}
unsafe impl Sync for CudaBuffer {}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        unsafe { minml_cuda_free(self.handle.as_ptr()) }
    }
}

impl Buffer for CudaBuffer {
    fn bytes(&self) -> usize {
        self.bytes
    }
    fn device(&self) -> Device {
        Device::Cuda
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn as_cuda(b: &dyn Buffer) -> &CudaBuffer {
    b.as_any().downcast_ref::<CudaBuffer>().expect("cuda buffer")
}

fn check(rc: i32, op: &'static str) -> Result<()> {
    if rc == 0 {
        Ok(())
    } else {
        Err(MinmlError::Other(format!("cuda {op} failed (rc={rc})")))
    }
}

pub fn allocate(bytes: usize) -> Result<Arc<dyn Buffer>> {
    let raw = unsafe { minml_cuda_alloc(bytes) };
    let handle = NonNull::new(raw)
        .ok_or_else(|| MinmlError::Other("cudaMalloc returned null".into()))?;
    Ok(Arc::new(CudaBuffer { handle, bytes }))
}

pub fn copy_host_to_buffer(dst: &dyn Buffer, src: &[u8]) -> Result<()> {
    let buf = as_cuda(dst);
    let rc = unsafe {
        minml_cuda_h2d(
            buf.handle.as_ptr(),
            src.as_ptr() as *const c_void,
            src.len(),
        )
    };
    check(rc, "h2d")
}

pub fn copy_buffer_to_host(src: &dyn Buffer, dst: &mut [u8]) -> Result<()> {
    let buf = as_cuda(src);
    let rc = unsafe {
        minml_cuda_d2h(
            dst.as_mut_ptr() as *mut c_void,
            buf.handle.as_ptr(),
            dst.len(),
        )
    };
    check(rc, "d2h")
}

pub fn add(a: &Array, b: &Array, out: &Array) -> Result<()> {
    let buf_a = a.buffer().expect("evaluated");
    let buf_b = b.buffer().expect("evaluated");
    let buf_o = out.buffer().expect("evaluated");
    let rc = unsafe {
        minml_cuda_add(
            as_cuda(&*buf_a).handle.as_ptr(),
            as_cuda(&*buf_b).handle.as_ptr(),
            as_cuda(&*buf_o).handle.as_ptr(),
            out.size(),
        )
    };
    check(rc, "add")
}

pub fn mul(a: &Array, b: &Array, out: &Array) -> Result<()> {
    let buf_a = a.buffer().expect("evaluated");
    let buf_b = b.buffer().expect("evaluated");
    let buf_o = out.buffer().expect("evaluated");
    let rc = unsafe {
        minml_cuda_mul(
            as_cuda(&*buf_a).handle.as_ptr(),
            as_cuda(&*buf_b).handle.as_ptr(),
            as_cuda(&*buf_o).handle.as_ptr(),
            out.size(),
        )
    };
    check(rc, "mul")
}

pub fn dot(a: &Array, b: &Array, out: &Array) -> Result<()> {
    let buf_a = a.buffer().expect("evaluated");
    let buf_b = b.buffer().expect("evaluated");
    let buf_o = out.buffer().expect("evaluated");
    let rc = unsafe {
        minml_cuda_dot(
            as_cuda(&*buf_a).handle.as_ptr(),
            as_cuda(&*buf_b).handle.as_ptr(),
            as_cuda(&*buf_o).handle.as_ptr(),
            a.size(),
        )
    };
    check(rc, "dot")
}
