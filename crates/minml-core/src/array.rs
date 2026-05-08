// Array — the user-facing tensor.
//
// Either evaluated (data is `Arc<dyn Buffer>`) or lazy (a `Primitive` plus a
// list of input Arrays). Calling `eval()` walks the input DAG iteratively
// in post-order, allocates output buffers, and runs each primitive. The
// only async surface is `tolist`/`item` (defined in ops.rs as free fns or
// here as methods); they invoke the backend's async d2h once the graph
// is materialized.
//
// Storage matches C++: contiguous row-major, dtype Float32 or Int32, no
// strides. The `batch_axis` field is set transiently by vmap.

use crate::buffer::Buffer;
use crate::device::Device;
use crate::dtype::{dtype_bytes, DType};
use crate::error::{MinmlError, Result};
use crate::primitive::Primitive;
use parking_lot::Mutex;
use std::sync::Arc;

#[derive(Clone)]
pub struct Array {
    shape: Vec<usize>,
    size: usize,
    device: Device,
    dtype: DType,
    batch_axis: Option<i32>,
    // Shared inner state so that all clones of the same logical Array see
    // the same evaluated buffer once eval() has run on any of them.
    inner: Arc<Mutex<ArrayInner>>,
}

enum ArrayInner {
    Evaluated(Arc<dyn Buffer>),
    Lazy {
        prim: Arc<dyn Primitive>,
        inputs: Vec<Array>,
    },
}

fn product(shape: &[usize]) -> usize {
    shape.iter().product()
}

impl Array {
    // ---- Eager constructors ----

    pub fn from_f32_with_shape(data: Vec<f32>, shape: Vec<usize>, device: Device) -> Result<Self> {
        let size = product(&shape);
        if data.len() != size {
            return Err(MinmlError::DataSize {
                got: data.len(),
                expected: size,
            });
        }
        let bytes = size * dtype_bytes(DType::Float32);
        let buf = crate::device_dispatch::allocate(device, bytes)?;
        crate::device_dispatch::h2d(device, &*buf, bytemuck::cast_slice(&data))?;
        Ok(Self {
            shape,
            size,
            device,
            dtype: DType::Float32,
            batch_axis: None,
            inner: Arc::new(Mutex::new(ArrayInner::Evaluated(buf))),
        })
    }

    pub fn from_i32_with_shape(data: Vec<i32>, shape: Vec<usize>, device: Device) -> Result<Self> {
        let size = product(&shape);
        if data.len() != size {
            return Err(MinmlError::DataSize {
                got: data.len(),
                expected: size,
            });
        }
        let bytes = size * dtype_bytes(DType::Int32);
        let buf = crate::device_dispatch::allocate(device, bytes)?;
        crate::device_dispatch::h2d(device, &*buf, bytemuck::cast_slice(&data))?;
        Ok(Self {
            shape,
            size,
            device,
            dtype: DType::Int32,
            batch_axis: None,
            inner: Arc::new(Mutex::new(ArrayInner::Evaluated(buf))),
        })
    }

    pub fn from_f32_1d(data: Vec<f32>, device: Device) -> Result<Self> {
        let n = data.len();
        Self::from_f32_with_shape(data, vec![n], device)
    }

    pub fn from_i32_1d(data: Vec<i32>, device: Device) -> Result<Self> {
        let n = data.len();
        Self::from_i32_with_shape(data, vec![n], device)
    }

    // ---- Lazy constructor ----

    pub fn lazy(
        shape: Vec<usize>,
        dtype: DType,
        device: Device,
        prim: Arc<dyn Primitive>,
        inputs: Vec<Array>,
    ) -> Self {
        let size = product(&shape);
        Self {
            shape,
            size,
            device,
            dtype,
            batch_axis: None,
            inner: Arc::new(Mutex::new(ArrayInner::Lazy { prim, inputs })),
        }
    }

    // ---- Accessors ----

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn size(&self) -> usize {
        self.size
    }
    pub fn device(&self) -> Device {
        self.device
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn batch_axis(&self) -> Option<i32> {
        self.batch_axis
    }

    pub fn evaluated(&self) -> bool {
        matches!(*self.inner.lock(), ArrayInner::Evaluated(_))
    }

    pub fn buffer(&self) -> Option<Arc<dyn Buffer>> {
        match &*self.inner.lock() {
            ArrayInner::Evaluated(b) => Some(b.clone()),
            _ => None,
        }
    }

    // ---- Vmap-axis tagging ----

    pub fn with_batch_axis(&self, axis: i32) -> Self {
        let mut out = self.clone();
        out.batch_axis = Some(axis);
        out
    }

    pub fn strip_batch_axis(&self) -> Self {
        let mut out = self.clone();
        out.batch_axis = None;
        out
    }

    // ---- Eval ----

    // Iterative post-order DFS. No recursion, no boxed futures: even on
    // WebGPU the dispatch is sync (queue.submit returns immediately); only
    // tolist/item actually await on d2h.
    pub fn eval(&self) -> Result<()> {
        if self.evaluated() {
            return Ok(());
        }
        let mut stack: Vec<(Array, bool)> = Vec::new();
        stack.push((self.clone(), false));
        while let Some((node, visited)) = stack.pop() {
            if node.evaluated() {
                continue;
            }
            if !visited {
                stack.push((node.clone(), true));
                let inputs = node.inputs_snapshot();
                for inp in inputs {
                    if !inp.evaluated() {
                        stack.push((inp, false));
                    }
                }
            } else {
                node.run_primitive()?;
            }
        }
        Ok(())
    }

    fn inputs_snapshot(&self) -> Vec<Array> {
        match &*self.inner.lock() {
            ArrayInner::Evaluated(_) => Vec::new(),
            ArrayInner::Lazy { inputs, .. } => inputs.clone(),
        }
    }

    fn run_primitive(&self) -> Result<()> {
        // Take the lazy state out, allocate, run, replace with Evaluated.
        let (prim, inputs) = {
            let guard = self.inner.lock();
            match &*guard {
                ArrayInner::Evaluated(_) => return Ok(()),
                ArrayInner::Lazy { prim, inputs } => (prim.clone(), inputs.clone()),
            }
            // guard dropped here; we'll re-acquire below
        };
        // Allocate output buffer.
        let bytes = self.size * dtype_bytes(self.dtype);
        let buf = crate::device_dispatch::allocate(self.device, bytes)?;
        // Install it on this Array so the primitive can write through.
        {
            let mut guard = self.inner.lock();
            *guard = ArrayInner::Evaluated(buf);
        }
        // Run the kernel. The primitive reads `output.buffer()`.
        prim.eval(&inputs, self)?;
        Ok(())
    }

    // Async readback. CPU/CUDA finish the work synchronously and resolve
    // immediately; WebGPU drives a real `map_async`. Bindings layer this
    // into Python coroutines / JS Promises.
    pub async fn tolist(&self) -> Result<Vec<f32>> {
        self.eval()?;
        let mut out = vec![0.0f32; self.size];
        let buf = self.buffer().expect("evaluated");
        crate::device_dispatch::d2h_async(self.device, &*buf, bytemuck::cast_slice_mut(&mut out))
            .await?;
        Ok(out)
    }

    pub async fn tolist_int(&self) -> Result<Vec<i32>> {
        self.eval()?;
        let mut out = vec![0i32; self.size];
        let buf = self.buffer().expect("evaluated");
        crate::device_dispatch::d2h_async(self.device, &*buf, bytemuck::cast_slice_mut(&mut out))
            .await?;
        Ok(out)
    }

    pub async fn item(&self) -> Result<f32> {
        if self.size != 1 {
            return Err(MinmlError::ItemRequiresSize1);
        }
        Ok(self.tolist().await?[0])
    }

}
