// CUDA backend, ported to NVlabs/cuda-oxide.
//
// Built only when feature = "cuda". `cuda-core` gives us RAII wrappers around
// the CUDA driver API (CudaContext, CudaStream, DeviceBuffer, CudaModule);
// we add a thin NVRTC FFI (`nvrtc.rs`) so the static ops and transforms::jit
// can compile their CUDA-C source to PTX at runtime without any nvcc / cubin
// in the build. The first call to allocate() spins up a global context bound
// to device 0 and JIT-compiles the static-ops module.
//
// Buffers are typed as DeviceBuffer<u8> (the rest of minml is byte-oriented),
// which keeps the public Buffer trait identical to the previous FFI version.

mod kernels;
mod nvrtc;

pub(crate) use nvrtc::compile_to_ptx;

use crate::array::Array;
use crate::buffer::Buffer;
use crate::device::Device;
use crate::error::{MinmlError, Result};

use cuda_bindings::CUdeviceptr;
use cuda_core::{
    launch_kernel_on_stream, CudaContext, CudaFunction, CudaModule, CudaStream, DeviceBuffer,
    LaunchConfig,
};
use parking_lot::Mutex;
use std::any::Any;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, OnceLock};

// ---------- global context / stream ----------

pub(crate) struct CudaBackend {
    pub(crate) ctx: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    // Compute capability, formatted as e.g. "sm_70". Used by NVRTC.
    pub(crate) arch: String,
    // Static ops module + memoized function handles.
    static_module: Arc<CudaModule>,
    functions: Mutex<HashMap<&'static str, CudaFunction>>,
    // JIT-compiled fused-kernel modules, keyed by an opaque cache key.
    pub(crate) jit_modules: Mutex<HashMap<String, Arc<CudaModule>>>,
}

static GLOBAL: OnceLock<Arc<CudaBackend>> = OnceLock::new();

fn ensure_backend() -> Result<Arc<CudaBackend>> {
    if let Some(b) = GLOBAL.get() {
        return Ok(b.clone());
    }
    let ctx = CudaContext::new(0)
        .map_err(|e| MinmlError::Other(format!("CudaContext::new(0): {e:?}")))?;
    let (major, minor) = ctx
        .compute_capability()
        .map_err(|e| MinmlError::Other(format!("compute_capability: {e:?}")))?;
    let arch = format!("sm_{}{}", major, minor);
    let stream = ctx.default_stream();

    let ptx = compile_to_ptx(kernels::KERNELS_SRC, "minml_static_ops", &arch)?;
    let static_module = ctx
        .load_module_from_ptx_src(&ptx)
        .map_err(|e| MinmlError::Other(format!("load_module_from_ptx_src: {e:?}")))?;

    let backend = Arc::new(CudaBackend {
        ctx,
        stream,
        arch,
        static_module,
        functions: Mutex::new(HashMap::new()),
        jit_modules: Mutex::new(HashMap::new()),
    });
    let _ = GLOBAL.set(backend.clone());
    Ok(GLOBAL.get().unwrap().clone())
}

impl CudaBackend {
    fn static_function(&self, name: &'static str) -> Result<CudaFunction> {
        if let Some(f) = self.functions.lock().get(name) {
            return Ok(f.clone());
        }
        let f = self
            .static_module
            .load_function(name)
            .map_err(|e| MinmlError::Other(format!("load_function({name}): {e:?}")))?;
        self.functions.lock().insert(name, f.clone());
        Ok(f)
    }
}

// ---------- Buffer ----------

pub struct CudaBuffer {
    inner: DeviceBuffer<u8>,
    // Cached so Buffer::bytes() avoids touching DeviceBuffer.
    bytes: usize,
}

impl CudaBuffer {
    pub(crate) fn cu_deviceptr(&self) -> CUdeviceptr {
        self.inner.cu_deviceptr()
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

pub(crate) fn as_cuda(b: &dyn Buffer) -> &CudaBuffer {
    b.as_any()
        .downcast_ref::<CudaBuffer>()
        .expect("cuda buffer")
}

// ---------- allocate / h2d / d2h ----------

pub fn allocate(bytes: usize) -> Result<Arc<dyn Buffer>> {
    let b = ensure_backend()?;
    let inner = DeviceBuffer::<u8>::zeroed(&b.stream, bytes)
        .map_err(|e| MinmlError::Other(format!("DeviceBuffer::zeroed({bytes}): {e:?}")))?;
    Ok(Arc::new(CudaBuffer { inner, bytes }))
}

pub fn copy_host_to_buffer(dst: &dyn Buffer, src: &[u8]) -> Result<()> {
    let b = ensure_backend()?;
    let dst = as_cuda(dst);
    if src.len() > dst.bytes {
        return Err(MinmlError::Other(format!(
            "h2d: src len {} > dst {}",
            src.len(),
            dst.bytes
        )));
    }
    unsafe {
        cuda_core::memory::memcpy_htod_async(
            dst.cu_deviceptr(),
            src.as_ptr(),
            src.len(),
            b.stream.cu_stream(),
        )
        .map_err(|e| MinmlError::Other(format!("memcpy_htod_async: {e:?}")))?;
    }
    b.stream
        .synchronize()
        .map_err(|e| MinmlError::Other(format!("stream sync after h2d: {e:?}")))?;
    Ok(())
}

pub fn copy_buffer_to_host(src: &dyn Buffer, dst: &mut [u8]) -> Result<()> {
    let b = ensure_backend()?;
    let src_buf = as_cuda(src);
    if dst.len() > src_buf.bytes {
        return Err(MinmlError::Other(format!(
            "d2h: dst len {} > src {}",
            dst.len(),
            src_buf.bytes
        )));
    }
    unsafe {
        cuda_core::memory::memcpy_dtoh_async(
            dst.as_mut_ptr(),
            src_buf.cu_deviceptr(),
            dst.len(),
            b.stream.cu_stream(),
        )
        .map_err(|e| MinmlError::Other(format!("memcpy_dtoh_async: {e:?}")))?;
    }
    b.stream
        .synchronize()
        .map_err(|e| MinmlError::Other(format!("stream sync after d2h: {e:?}")))?;
    Ok(())
}

// ---------- kernel launch helper ----------

pub(crate) fn launch_1d(
    func: &CudaFunction,
    stream: &CudaStream,
    n: u32,
    params: &mut [*mut c_void],
) -> Result<()> {
    let cfg = LaunchConfig::for_num_elems(n);
    unsafe {
        launch_kernel_on_stream(
            func,
            cfg.grid_dim,
            cfg.block_dim,
            cfg.shared_mem_bytes,
            stream,
            params,
        )
        .map_err(|e| MinmlError::Other(format!("launch_kernel_on_stream: {e:?}")))?;
    }
    Ok(())
}

// Run a kernel taking (a_ptr, b_ptr, out_ptr, n) on the default stream.
fn launch_binop(name: &'static str, a: &Array, b: &Array, out: &Array) -> Result<()> {
    let bk = ensure_backend()?;
    let func = bk.static_function(name)?;
    let buf_a = a.buffer().expect("evaluated");
    let buf_b = b.buffer().expect("evaluated");
    let buf_o = out.buffer().expect("evaluated");
    let mut a_ptr = as_cuda(&*buf_a).cu_deviceptr();
    let mut b_ptr = as_cuda(&*buf_b).cu_deviceptr();
    let mut o_ptr = as_cuda(&*buf_o).cu_deviceptr();
    let mut n: u32 = out.size() as u32;
    let mut params: [*mut c_void; 4] = [
        &mut a_ptr as *mut _ as *mut c_void,
        &mut b_ptr as *mut _ as *mut c_void,
        &mut o_ptr as *mut _ as *mut c_void,
        &mut n as *mut _ as *mut c_void,
    ];
    launch_1d(&func, &bk.stream, n, &mut params)?;
    bk.stream
        .synchronize()
        .map_err(|e| MinmlError::Other(format!("stream sync after {name}: {e:?}")))?;
    Ok(())
}

pub fn add(a: &Array, b: &Array, out: &Array) -> Result<()> {
    launch_binop("minml_add", a, b, out)
}

pub fn mul(a: &Array, b: &Array, out: &Array) -> Result<()> {
    launch_binop("minml_mul", a, b, out)
}

pub fn dot(a: &Array, b: &Array, out: &Array) -> Result<()> {
    let bk = ensure_backend()?;
    let func = bk.static_function("minml_dot")?;
    let buf_a = a.buffer().expect("evaluated");
    let buf_b = b.buffer().expect("evaluated");
    let buf_o = out.buffer().expect("evaluated");
    let mut a_ptr = as_cuda(&*buf_a).cu_deviceptr();
    let mut b_ptr = as_cuda(&*buf_b).cu_deviceptr();
    let mut o_ptr = as_cuda(&*buf_o).cu_deviceptr();
    let mut n: u32 = a.size() as u32;
    let mut params: [*mut c_void; 4] = [
        &mut a_ptr as *mut _ as *mut c_void,
        &mut b_ptr as *mut _ as *mut c_void,
        &mut o_ptr as *mut _ as *mut c_void,
        &mut n as *mut _ as *mut c_void,
    ];
    // dot uses a single block with 256 threads (shared-mem reduction).
    unsafe {
        cuda_core::launch_kernel_on_stream(
            &func,
            (1, 1, 1),
            (256, 1, 1),
            0,
            &bk.stream,
            &mut params,
        )
        .map_err(|e| MinmlError::Other(format!("launch dot: {e:?}")))?;
    }
    bk.stream
        .synchronize()
        .map_err(|e| MinmlError::Other(format!("stream sync after dot: {e:?}")))?;
    Ok(())
}

// ---------- Public entry for transforms::jit ----------

// Launch a JIT-compiled fused elementwise kernel. The CUDA-C source is
// produced by transforms::jit and used as the cache key — structurally
// identical fusions reuse the same compiled module. Kernel signature is
// fixed: `(in0_ptr, ..., inK_ptr, out_ptr, n)`, all f32 except n (u32).
pub(crate) fn launch_fused_elementwise(
    cuda_c_src: &str,
    fn_name: &str,
    inputs: &[Array],
    out: &Array,
) -> Result<()> {
    let bk = ensure_backend()?;
    let module = {
        let mut guard = bk.jit_modules.lock();
        if let Some(m) = guard.get(cuda_c_src) {
            m.clone()
        } else {
            let ptx = compile_to_ptx(cuda_c_src, fn_name, &bk.arch)?;
            let m = bk
                .ctx
                .load_module_from_ptx_src(&ptx)
                .map_err(|e| MinmlError::Other(format!("jit load_module: {e:?}")))?;
            guard.insert(cuda_c_src.to_string(), m.clone());
            m
        }
    };
    let func = module
        .load_function(fn_name)
        .map_err(|e| MinmlError::Other(format!("jit load_function({fn_name}): {e:?}")))?;

    // Hold inputs' buffers + a heap-stable Vec<CUdeviceptr> so the &mut to
    // each pointer stays valid for the duration of the launch.
    let in_bufs: Vec<_> = inputs
        .iter()
        .map(|a| a.buffer().expect("evaluated"))
        .collect();
    let mut in_ptrs: Vec<CUdeviceptr> =
        in_bufs.iter().map(|b| as_cuda(&**b).cu_deviceptr()).collect();
    let buf_o = out.buffer().expect("evaluated");
    let mut o_ptr = as_cuda(&*buf_o).cu_deviceptr();
    let mut n: u32 = out.size() as u32;

    let mut params: Vec<*mut c_void> = Vec::with_capacity(in_ptrs.len() + 2);
    for p in in_ptrs.iter_mut() {
        params.push(p as *mut _ as *mut c_void);
    }
    params.push(&mut o_ptr as *mut _ as *mut c_void);
    params.push(&mut n as *mut _ as *mut c_void);

    launch_1d(&func, &bk.stream, n, &mut params)?;
    bk.stream
        .synchronize()
        .map_err(|e| MinmlError::Other(format!("stream sync after fused: {e:?}")))?;
    Ok(())
}
