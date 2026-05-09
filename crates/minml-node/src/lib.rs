// minml-node — Node.js bindings for minml-core via napi-rs.
//
// Mirrors crates/minml-py/src/lib.rs at the level needed for
// "TypeScript frontend, native backend": Array, the three ops, async
// readbacks, and (optionally) the CUDA backend. The native addon (.node)
// is regular host code, so building with --features cuda links libcuda
// + libnvrtc directly — same path as the Python extension. This is how
// TypeScript reaches the CUDA backend; the wasm build can't, because
// browser/Node WASM has no FFI to host C libraries.
//
// Async surface: Array.tolist / item / eval and initWebGPU return
// Promises (napi-rs's tokio_rt feature drives them on Tokio). Graph
// builders are sync — they only build lazy graphs.
//
// transforms::jit isn't bound here: the closure-trace pattern needs to
// re-enter JS during Rust execution, which is awkward through the
// stable napi v2 API surface. The Python and wasm bindings already
// expose it; users who want JIT can compose the trace in Rust or wait
// for a follow-up. The basic ops compose fine for the CUDA-backend use
// case.

#![deny(clippy::all)]

use minml_core as ml;
use napi::bindgen_prelude::*;
use napi_derive::napi;

fn map_err(e: ml::MinmlError) -> Error {
    Error::new(Status::GenericFailure, e.to_string())
}

// ---- Enums (plain numbers on the JS side) ----

#[napi]
pub enum Device {
    CPU,
    CUDA,
    WebGPU,
}

impl From<Device> for ml::Device {
    fn from(d: Device) -> Self {
        match d {
            Device::CPU => ml::Device::Cpu,
            Device::CUDA => ml::Device::Cuda,
            Device::WebGPU => ml::Device::WebGpu,
        }
    }
}
impl From<ml::Device> for Device {
    fn from(d: ml::Device) -> Self {
        match d {
            ml::Device::Cpu => Device::CPU,
            ml::Device::Cuda => Device::CUDA,
            ml::Device::WebGpu => Device::WebGPU,
        }
    }
}

#[napi]
pub enum DType {
    Float32,
    Int32,
}

impl From<DType> for ml::DType {
    fn from(d: DType) -> Self {
        match d {
            DType::Float32 => ml::DType::Float32,
            DType::Int32 => ml::DType::Int32,
        }
    }
}
impl From<ml::DType> for DType {
    fn from(d: ml::DType) -> Self {
        match d {
            ml::DType::Float32 => DType::Float32,
            ml::DType::Int32 => DType::Int32,
        }
    }
}

// ---- Array ----

#[napi]
#[derive(Clone)]
pub struct MlArray {
    inner: ml::Array,
}

#[napi]
impl MlArray {
    #[napi(getter)]
    pub fn size(&self) -> u32 {
        self.inner.size() as u32
    }

    #[napi(getter)]
    pub fn shape(&self) -> Vec<u32> {
        self.inner.shape().iter().map(|&d| d as u32).collect()
    }

    #[napi(getter)]
    pub fn device(&self) -> Device {
        self.inner.device().into()
    }

    #[napi(getter)]
    pub fn dtype(&self) -> DType {
        self.inner.dtype().into()
    }

    // Force the lazy graph to evaluate. Returns Promise<void>.
    #[napi]
    pub async fn eval(&self) -> Result<()> {
        let arr = self.inner.clone();
        tokio::task::spawn_blocking(move || arr.eval())
            .await
            .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?
            .map_err(map_err)
    }

    // Promise<number[]>. Same shape across backends — CPU/CUDA finish
    // synchronously inside the Rust future; WebGPU truly suspends on
    // Buffer::slice.map_async.
    #[napi]
    pub async fn tolist(&self) -> Result<Vec<f64>> {
        let arr = self.inner.clone();
        match arr.dtype() {
            ml::DType::Float32 => {
                let v = arr.tolist().await.map_err(map_err)?;
                Ok(v.into_iter().map(|x| x as f64).collect())
            }
            ml::DType::Int32 => {
                let v = arr.tolist_int().await.map_err(map_err)?;
                Ok(v.into_iter().map(|x| x as f64).collect())
            }
        }
    }

    #[napi]
    pub async fn item(&self) -> Result<f64> {
        let arr = self.inner.clone();
        let v = arr.item().await.map_err(map_err)?;
        Ok(v as f64)
    }
}

// ---- Free functions ----

#[napi(js_name = "setDefaultDevice")]
pub fn set_default_device(d: Device) {
    ml::set_default_device(d.into());
}

#[napi(js_name = "defaultDevice")]
pub fn default_device() -> Device {
    ml::default_device().into()
}

#[napi]
pub fn array(data: Vec<f64>, device: Device) -> Result<MlArray> {
    let f32_data: Vec<f32> = data.into_iter().map(|x| x as f32).collect();
    Ok(MlArray {
        inner: ml::Array::from_f32_1d(f32_data, device.into()).map_err(map_err)?,
    })
}

#[napi(js_name = "arrayInt")]
pub fn array_int(data: Vec<i32>, device: Device) -> Result<MlArray> {
    Ok(MlArray {
        inner: ml::Array::from_i32_1d(data, device.into()).map_err(map_err)?,
    })
}

#[napi]
pub fn add(a: &MlArray, b: &MlArray) -> Result<MlArray> {
    Ok(MlArray {
        inner: ml::add(&a.inner, &b.inner).map_err(map_err)?,
    })
}

#[napi]
pub fn mul(a: &MlArray, b: &MlArray) -> Result<MlArray> {
    Ok(MlArray {
        inner: ml::mul(&a.inner, &b.inner).map_err(map_err)?,
    })
}

#[napi]
pub fn dot(a: &MlArray, b: &MlArray) -> Result<MlArray> {
    Ok(MlArray {
        inner: ml::dot(&a.inner, &b.inner).map_err(map_err)?,
    })
}

#[napi]
pub fn ones(shape: Vec<u32>, dtype: DType, device: Device) -> MlArray {
    let shape: Vec<usize> = shape.into_iter().map(|d| d as usize).collect();
    MlArray {
        inner: ml::ones(shape, dtype.into(), device.into()),
    }
}

// ---- WebGPU init ----

#[napi(js_name = "initWebGPU")]
pub async fn init_webgpu() -> Result<()> {
    // Tokio runtime is provided by napi-rs's tokio_rt feature.
    ml::webgpu::init().await.map_err(map_err)
}
