// WebGPU backend (wgpu).
//
// The same code compiles to native (Vulkan/Metal/DX12) and wasm32-unknown-
// unknown (navigator.gpu). Async surfaces are exactly two: `init()` (device
// acquisition) and `copy_buffer_to_host_async` (Buffer::slice.map_async).
// Everything else — pipeline creation, queue.write_buffer, queue.submit,
// dispatch — is sync.

mod shaders;

use crate::array::Array;
use crate::buffer::Buffer;
use crate::device::Device as MinmlDevice;
use crate::error::{MinmlError, Result};
use parking_lot::Mutex;
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

pub struct WebGpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Mutex<HashMap<&'static str, wgpu::ComputePipeline>>,
    // Pipelines built at runtime by transforms::jit. Keyed by the WGSL
    // source itself so the same fused expression reuses the same pipeline.
    jit_pipelines: Mutex<HashMap<String, wgpu::ComputePipeline>>,
}

static GLOBAL: OnceLock<Arc<WebGpuBackend>> = OnceLock::new();

fn ctx() -> Result<Arc<WebGpuBackend>> {
    GLOBAL
        .get()
        .cloned()
        .ok_or(MinmlError::WebGpuNotInitialized)
}

// Default init: ask wgpu for a default adapter + device. The user can also
// install a pre-built device with init_with_device (used by the wasm32
// binding when JS already has a GPUDevice handle).
pub async fn init() -> Result<()> {
    if GLOBAL.get().is_some() {
        return Ok(());
    }
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .map_err(|e| MinmlError::WebGpuInitFailed(format!("request_adapter: {e}")))?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("minml-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        })
        .await
        .map_err(|e| MinmlError::WebGpuInitFailed(format!("request_device: {e}")))?;
    install(device, queue);
    Ok(())
}

pub fn install(device: wgpu::Device, queue: wgpu::Queue) {
    let _ = GLOBAL.set(Arc::new(WebGpuBackend {
        device,
        queue,
        pipelines: Mutex::new(HashMap::new()),
        jit_pipelines: Mutex::new(HashMap::new()),
    }));
}

// ---- Buffer ----

pub struct WebGpuBuffer {
    pub(crate) handle: wgpu::Buffer,
    bytes: usize,
}

impl Buffer for WebGpuBuffer {
    fn bytes(&self) -> usize {
        self.bytes
    }
    fn device(&self) -> MinmlDevice {
        MinmlDevice::WebGpu
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn as_wgpu_buf(b: &dyn Buffer) -> &WebGpuBuffer {
    b.as_any()
        .downcast_ref::<WebGpuBuffer>()
        .expect("webgpu buffer")
}

// ---- Allocate / copies ----

pub fn allocate(bytes: usize) -> Result<Arc<dyn Buffer>> {
    let c = ctx()?;
    // wgpu requires buffer size > 0 and a multiple of 4 for COPY_DST. The
    // CPU semantics let n bytes be anything; we round up the allocation.
    let padded = bytes.max(4).next_multiple_of(4);
    let handle = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: padded as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    Ok(Arc::new(WebGpuBuffer { handle, bytes }))
}

pub fn copy_host_to_buffer(dst: &dyn Buffer, src: &[u8]) -> Result<()> {
    let c = ctx()?;
    let buf = as_wgpu_buf(dst);
    // queue.write_buffer requires src length to be a multiple of 4 (COPY_DST
    // alignment). Pad if needed.
    let padded_len = src.len().next_multiple_of(4);
    if padded_len == src.len() {
        c.queue.write_buffer(&buf.handle, 0, src);
    } else {
        let mut tmp = vec![0u8; padded_len];
        tmp[..src.len()].copy_from_slice(src);
        c.queue.write_buffer(&buf.handle, 0, &tmp);
    }
    Ok(())
}

// Async readback. Stages into a MapRead buffer, copies, maps it, memcpys
// into `dst`. The await suspends naturally on both native (poll the
// device) and wasm32 (the Promise resolves on the main JS thread).
pub async fn copy_buffer_to_host_async(src: &dyn Buffer, dst: &mut [u8]) -> Result<()> {
    let c = ctx()?;
    let src_buf = as_wgpu_buf(src);
    let bytes = dst.len();
    let padded = bytes.max(4).next_multiple_of(4);
    let staging = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("minml-readback-staging"),
        size: padded as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(&src_buf.handle, 0, &staging, 0, padded as u64);
    c.queue.submit(Some(enc.finish()));

    let slice = staging.slice(0..padded as u64);
    let (tx, rx) = futures_channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    // Native: poll until complete. wasm32: poll is a no-op; the promise
    // is driven by the JS event loop.
    #[cfg(not(target_arch = "wasm32"))]
    c.device.poll(wgpu::PollType::Wait).ok();

    rx.recv()
        .await
        .map_err(|_| MinmlError::WebGpuReadbackFailed)?
        .map_err(|_| MinmlError::WebGpuReadbackFailed)?;
    {
        let mapped = slice.get_mapped_range();
        dst.copy_from_slice(&mapped[..bytes]);
    }
    staging.unmap();
    Ok(())
}

// Tiny oneshot channel that works on both native and wasm32 without
// pulling in tokio. (futures::channel::oneshot would also work.)
fn futures_channel<T>() -> (
    futures_channel::Sender<T>,
    futures_channel::Receiver<T>,
) {
    futures_channel::oneshot()
}

mod futures_channel {
    use parking_lot::Mutex;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::task::{Context, Poll, Waker};

    pub struct Sender<T> {
        inner: Arc<Mutex<Inner<T>>>,
    }
    pub struct Receiver<T> {
        inner: Arc<Mutex<Inner<T>>>,
    }
    struct Inner<T> {
        value: Option<T>,
        waker: Option<Waker>,
        closed: bool,
    }

    pub fn oneshot<T>() -> (Sender<T>, Receiver<T>) {
        let inner = Arc::new(Mutex::new(Inner {
            value: None,
            waker: None,
            closed: false,
        }));
        (
            Sender {
                inner: inner.clone(),
            },
            Receiver { inner },
        )
    }

    impl<T> Sender<T> {
        pub fn send(self, v: T) -> std::result::Result<(), T> {
            let mut g = self.inner.lock();
            if g.closed {
                return Err(v);
            }
            g.value = Some(v);
            if let Some(w) = g.waker.take() {
                w.wake();
            }
            Ok(())
        }
    }

    impl<T> Receiver<T> {
        pub async fn recv(self) -> std::result::Result<T, ()> {
            ReceiverFut { inner: self.inner }.await
        }
    }

    struct ReceiverFut<T> {
        inner: Arc<Mutex<Inner<T>>>,
    }
    impl<T> Future for ReceiverFut<T> {
        type Output = std::result::Result<T, ()>;
        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
            let mut g = self.inner.lock();
            if let Some(v) = g.value.take() {
                Poll::Ready(Ok(v))
            } else if g.closed {
                Poll::Ready(Err(()))
            } else {
                g.waker = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }
}

// ---- Pipeline cache ----

fn pipeline(name: &'static str) -> Result<wgpu::ComputePipeline> {
    let c = ctx()?;
    if let Some(p) = c.pipelines.lock().get(name) {
        return Ok(p.clone());
    }
    let wgsl = match name {
        "add" => shaders::ADD_WGSL,
        "mul" => shaders::MUL_WGSL,
        "dot" => shaders::DOT_WGSL,
        _ => return Err(MinmlError::Other(format!("unknown pipeline: {name}"))),
    };
    let module = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(name),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let pipe = c
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
    c.pipelines.lock().insert(name, pipe.clone());
    Ok(pipe)
}

fn dispatch(
    kernel: &'static str,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    out: &wgpu::Buffer,
    workgroups: u32,
) -> Result<()> {
    let c = ctx()?;
    let pipe = pipeline(kernel)?;
    let layout = pipe.get_bind_group_layout(0);
    let bg = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out.as_entire_binding(),
            },
        ],
    });
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipe);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    c.queue.submit(Some(enc.finish()));
    Ok(())
}

// ---- Backend ops ----

pub fn add(a: &Array, b: &Array, out: &Array) -> Result<()> {
    let buf_a = a.buffer().expect("evaluated");
    let buf_b = b.buffer().expect("evaluated");
    let buf_o = out.buffer().expect("evaluated");
    let n = out.size() as u32;
    let wg = (n + 63) / 64;
    dispatch(
        "add",
        &as_wgpu_buf(&*buf_a).handle,
        &as_wgpu_buf(&*buf_b).handle,
        &as_wgpu_buf(&*buf_o).handle,
        wg,
    )
}

pub fn mul(a: &Array, b: &Array, out: &Array) -> Result<()> {
    let buf_a = a.buffer().expect("evaluated");
    let buf_b = b.buffer().expect("evaluated");
    let buf_o = out.buffer().expect("evaluated");
    let n = out.size() as u32;
    let wg = (n + 63) / 64;
    dispatch(
        "mul",
        &as_wgpu_buf(&*buf_a).handle,
        &as_wgpu_buf(&*buf_b).handle,
        &as_wgpu_buf(&*buf_o).handle,
        wg,
    )
}

pub fn dot(a: &Array, b: &Array, out: &Array) -> Result<()> {
    let c = ctx()?;
    let buf_a = a.buffer().expect("evaluated");
    let buf_b = b.buffer().expect("evaluated");
    let buf_o = out.buffer().expect("evaluated");
    // Output is a single f32 atomic; zero before kernel runs (kernel does
    // atomic adds into out[0]).
    let zero = [0u8; 4];
    c.queue.write_buffer(&as_wgpu_buf(&*buf_o).handle, 0, &zero);
    let n = a.size() as u32;
    let wg = (n + 63) / 64;
    dispatch(
        "dot",
        &as_wgpu_buf(&*buf_a).handle,
        &as_wgpu_buf(&*buf_b).handle,
        &as_wgpu_buf(&*buf_o).handle,
        wg,
    )
}

// ---- JIT runtime dispatch (used by transforms::jit) ----

// Compile-or-reuse a pipeline from arbitrary WGSL, then dispatch it with one
// storage binding per input plus the output. The kernel's `@workgroup_size`
// must be 64 — the JIT generator emits exactly that.
pub(crate) fn dispatch_jit_elementwise(
    wgsl: &str,
    inputs: &[Array],
    out: &Array,
) -> Result<()> {
    let c = ctx()?;
    let pipe = {
        let mut guard = c.jit_pipelines.lock();
        if let Some(p) = guard.get(wgsl) {
            p.clone()
        } else {
            let module = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("minml-jit"),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });
            let pipe = c
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("minml-jit"),
                    layout: None,
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });
            guard.insert(wgsl.to_string(), pipe.clone());
            pipe
        }
    };

    let in_bufs: Vec<_> = inputs
        .iter()
        .map(|a| a.buffer().expect("evaluated"))
        .collect();
    let out_buf = out.buffer().expect("evaluated");

    let layout = pipe.get_bind_group_layout(0);
    let mut entries: Vec<wgpu::BindGroupEntry> = Vec::with_capacity(inputs.len() + 1);
    for (i, b) in in_bufs.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry {
            binding: i as u32,
            resource: as_wgpu_buf(&**b).handle.as_entire_binding(),
        });
    }
    entries.push(wgpu::BindGroupEntry {
        binding: inputs.len() as u32,
        resource: as_wgpu_buf(&*out_buf).handle.as_entire_binding(),
    });
    let bg = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("minml-jit-bg"),
        layout: &layout,
        entries: &entries,
    });

    let n = out.size() as u32;
    let wg = n.div_ceil(64);
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipe);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(wg, 1, 1);
    }
    c.queue.submit(Some(enc.finish()));
    Ok(())
}
