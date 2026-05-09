// minml-py — Python bindings for minml-core via pyo3.
//
// Async surface: `Array.tolist`, `Array.tolist_int`, `Array.item`,
// `Array.eval`, and `init_webgpu` are coroutines (Python `await`-able).
// Everything else (graph builders, distribution constructors,
// PRNGKey.split, vmap) is sync — they only build lazy graphs.
//
// vmap shim ports bindings/python/bind.cpp:35-140: walk `__dict__` to
// collect Array leaves, rebuild a fresh instance with stacked outputs.

use minml_core as ml;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

fn map_err(e: ml::MinmlError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

// ---- Enums ----

#[pyclass(name = "Device", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
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

#[pyclass(name = "DType", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
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

#[pyclass(name = "Array")]
#[derive(Clone)]
pub struct PyArray {
    inner: ml::Array,
}

#[pymethods]
impl PyArray {
    #[new]
    #[pyo3(signature = (data, device=Device::CPU))]
    fn new(data: Vec<f32>, device: Device) -> PyResult<Self> {
        Ok(Self {
            inner: ml::Array::from_f32_1d(data, device.into()).map_err(map_err)?,
        })
    }

    fn size(&self) -> usize {
        self.inner.size()
    }
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }
    fn device(&self) -> Device {
        self.inner.device().into()
    }
    fn dtype(&self) -> DType {
        self.inner.dtype().into()
    }

    // Async readback: returns a Python coroutine.
    fn eval<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let arr = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            tokio::task::spawn_blocking(move || arr.eval())
                .await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                .map_err(map_err)?;
            Ok(())
        })
    }

    fn tolist<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let arr = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            arr.tolist().await.map_err(map_err)
        })
    }

    fn tolist_int<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let arr = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            arr.tolist_int().await.map_err(map_err)
        })
    }

    fn item<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let arr = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            arr.item().await.map_err(map_err)
        })
    }
}

// ---- PRNGKey ----

#[pyclass(name = "PRNGKey")]
#[derive(Clone)]
pub struct PyPRNGKey {
    inner: ml::PRNGKey,
}

#[pymethods]
impl PyPRNGKey {
    #[new]
    fn new(k0: u32, k1: u32) -> Self {
        Self {
            inner: ml::PRNGKey::new(k0, k1),
        }
    }

    #[staticmethod]
    #[pyo3(name = "new")]
    fn from_seed(seed: u32) -> Self {
        Self {
            inner: ml::PRNGKey::from_seed(seed),
        }
    }

    fn k0(&self) -> u32 {
        self.inner.k0()
    }
    fn k1(&self) -> u32 {
        self.inner.k1()
    }

    fn split(&self, n: usize) -> Vec<PyPRNGKey> {
        self.inner
            .split(n)
            .into_iter()
            .map(|k| PyPRNGKey { inner: k })
            .collect()
    }
}

// ---- Distribution wrappers ----

#[pyclass(name = "Dirichlet")]
pub struct PyDirichlet {
    inner: ml::Dirichlet,
}

#[pymethods]
impl PyDirichlet {
    #[new]
    fn new(alpha: PyArray) -> Self {
        Self {
            inner: ml::Dirichlet::new(alpha.inner),
        }
    }
    fn sample(&self, key: PyPRNGKey, batch_shape: Vec<usize>) -> PyResult<PyArray> {
        Ok(PyArray {
            inner: self.inner.sample(key.inner, batch_shape).map_err(map_err)?,
        })
    }
}

#[pyclass(name = "Categorical")]
pub struct PyCategorical {
    inner: ml::Categorical,
}

#[pymethods]
impl PyCategorical {
    #[new]
    fn new(probs: PyArray) -> Self {
        Self {
            inner: ml::Categorical::new(probs.inner),
        }
    }
    fn sample(&self, key: PyPRNGKey, batch_shape: Vec<usize>) -> PyResult<PyArray> {
        Ok(PyArray {
            inner: self.inner.sample(key.inner, batch_shape).map_err(map_err)?,
        })
    }
}

#[pyclass(name = "Normal")]
pub struct PyNormal {
    inner: ml::Normal,
}

#[pymethods]
impl PyNormal {
    #[new]
    fn new() -> Self {
        Self {
            inner: ml::Normal::new(),
        }
    }
    fn sample(&self, key: PyPRNGKey, batch_shape: Vec<usize>) -> PyResult<PyArray> {
        Ok(PyArray {
            inner: self.inner.sample(key.inner, batch_shape).map_err(map_err)?,
        })
    }
}

// ---- Free functions ----

#[pyfunction]
fn set_default_device(d: Device) {
    ml::set_default_device(d.into());
}

#[pyfunction]
fn default_device() -> Device {
    ml::default_device().into()
}

#[pyfunction]
#[pyo3(signature = (data, device=Device::CPU))]
fn array(data: Vec<f32>, device: Device) -> PyResult<PyArray> {
    Ok(PyArray {
        inner: ml::Array::from_f32_1d(data, device.into()).map_err(map_err)?,
    })
}

#[pyfunction]
#[pyo3(signature = (data, device=Device::CPU))]
fn array_int(data: Vec<i32>, device: Device) -> PyResult<PyArray> {
    Ok(PyArray {
        inner: ml::Array::from_i32_1d(data, device.into()).map_err(map_err)?,
    })
}

#[pyfunction]
#[pyo3(signature = (data, shape, device=Device::CPU))]
fn array_shaped(data: Vec<f32>, shape: Vec<usize>, device: Device) -> PyResult<PyArray> {
    Ok(PyArray {
        inner: ml::Array::from_f32_with_shape(data, shape, device.into()).map_err(map_err)?,
    })
}

#[pyfunction]
#[pyo3(signature = (data, shape, device=Device::CPU))]
fn array_int_shaped(data: Vec<i32>, shape: Vec<usize>, device: Device) -> PyResult<PyArray> {
    Ok(PyArray {
        inner: ml::Array::from_i32_with_shape(data, shape, device.into()).map_err(map_err)?,
    })
}

#[pyfunction]
fn add(a: &PyArray, b: &PyArray) -> PyResult<PyArray> {
    Ok(PyArray {
        inner: ml::add(&a.inner, &b.inner).map_err(map_err)?,
    })
}

#[pyfunction]
fn mul(a: &PyArray, b: &PyArray) -> PyResult<PyArray> {
    Ok(PyArray {
        inner: ml::mul(&a.inner, &b.inner).map_err(map_err)?,
    })
}

#[pyfunction]
fn dot(a: &PyArray, b: &PyArray) -> PyResult<PyArray> {
    Ok(PyArray {
        inner: ml::dot(&a.inner, &b.inner).map_err(map_err)?,
    })
}

#[pyfunction]
#[pyo3(signature = (shape, dtype=DType::Float32, device=Device::CPU))]
fn ones(shape: Vec<usize>, dtype: DType, device: Device) -> PyArray {
    PyArray {
        inner: ml::ones(shape, dtype.into(), device.into()),
    }
}

#[pyfunction]
fn randint(
    k0: u32,
    k1: u32,
    low: i32,
    high: i32,
    shape: Vec<usize>,
    device: Device,
) -> PyArray {
    PyArray {
        inner: ml::randint(k0, k1, low, high, shape, device.into()),
    }
}

#[pyfunction]
fn gather(table: &PyArray, indices: &PyArray) -> PyResult<PyArray> {
    Ok(PyArray {
        inner: ml::gather(&table.inner, &indices.inner).map_err(map_err)?,
    })
}

#[pyfunction]
fn dirichlet_sample(k0: u32, k1: u32, alpha: &PyArray, batch_shape: Vec<usize>) -> PyResult<PyArray> {
    Ok(PyArray {
        inner: ml::dirichlet_sample(k0, k1, &alpha.inner, batch_shape).map_err(map_err)?,
    })
}

#[pyfunction]
fn categorical_sample(
    k0: u32,
    k1: u32,
    probs: &PyArray,
    batch_shape: Vec<usize>,
) -> PyResult<PyArray> {
    Ok(PyArray {
        inner: ml::categorical_sample(k0, k1, &probs.inner, batch_shape).map_err(map_err)?,
    })
}

// ---- vmap shim ----

fn is_array(h: &Bound<'_, PyAny>) -> bool {
    h.is_instance_of::<PyArray>()
}

fn collect_leaves<'py>(
    v: &Bound<'py, PyAny>,
    keys: Option<&mut Vec<String>>,
) -> PyResult<Vec<ml::Array>> {
    if is_array(v) {
        let arr: PyArray = v.extract()?;
        if let Some(k) = keys {
            k.push(String::new());
        }
        return Ok(vec![arr.inner]);
    }
    let dict = v.getattr("__dict__")?;
    let dict: Bound<'py, PyDict> = dict.downcast_into()?;
    let mut out = Vec::new();
    let mut local_keys = Vec::new();
    for (k, val) in dict.iter() {
        if is_array(&val) {
            let arr: PyArray = val.extract()?;
            out.push(arr.inner);
            local_keys.push(k.extract::<String>()?);
        }
    }
    if out.is_empty() {
        return Err(PyRuntimeError::new_err("vmap: function returned no Arrays"));
    }
    if let Some(k) = keys {
        *k = local_keys;
    }
    Ok(out)
}

fn rebuild_tree<'py>(
    py: Python<'py>,
    template: &Bound<'py, PyAny>,
    keys: &[String],
    stacked: Vec<ml::Array>,
) -> PyResult<Py<PyAny>> {
    if keys.len() == 1 && keys[0].is_empty() {
        return Ok(PyArray {
            inner: stacked.into_iter().next().unwrap(),
        }
        .into_pyobject(py)?
        .into_any()
        .unbind());
    }
    let cls = template.getattr("__class__")?;
    let new_method = cls.getattr("__new__")?;
    let out = new_method.call1((cls,))?;
    for (k, arr) in keys.iter().zip(stacked.into_iter()) {
        let py_arr = PyArray { inner: arr }.into_pyobject(py)?.into_any();
        out.setattr(k.as_str(), py_arr)?;
    }
    Ok(out.unbind())
}

#[pyfunction]
#[pyo3(name = "vmap_apply")]
fn vmap_apply_py<'py>(
    py: Python<'py>,
    f: Bound<'py, PyAny>,
    in_axes: Bound<'py, PyList>,
    args: Bound<'py, PyList>,
) -> PyResult<Py<PyAny>> {
    let n = in_axes.len();
    if args.len() != n {
        return Err(PyRuntimeError::new_err("vmap: in_axes length != args length"));
    }
    let in_axes_v: Vec<i32> = (0..n)
        .map(|i| -> PyResult<i32> {
            let h = in_axes.get_item(i)?;
            if h.is_none() {
                Ok(-1)
            } else {
                h.extract::<i32>()
            }
        })
        .collect::<PyResult<_>>()?;

    // Split inputs by kind: embind Array (batched/unbatched) vs JS list /
    // scalar (looked up by iter index).
    let mut c_args: Vec<ml::Array> = Vec::new();
    let mut c_in_axes: Vec<i32> = Vec::new();
    let mut c_index: Vec<i32> = vec![-1; n];
    let mut py_args: Vec<Py<PyAny>> = Vec::with_capacity(n);
    let mut batch_n: usize = 0;
    let mut found = false;

    for i in 0..n {
        let a = args.get_item(i)?;
        if is_array(&a) {
            let arr: PyArray = a.extract()?;
            c_index[i] = c_args.len() as i32;
            c_args.push(arr.inner.clone());
            c_in_axes.push(in_axes_v[i]);
            if in_axes_v[i] >= 0 && !found {
                if in_axes_v[i] != 0 {
                    return Err(PyRuntimeError::new_err(
                        "vmap: only axis 0 supported on Arrays",
                    ));
                }
                if arr.inner.shape().is_empty() {
                    return Err(PyRuntimeError::new_err("vmap: cannot batch over a scalar"));
                }
                batch_n = arr.inner.shape()[0];
                found = true;
            }
            py_args.push(a.unbind());
        } else if in_axes_v[i] >= 0 {
            if !found {
                batch_n = a.len()?;
                found = true;
            }
            py_args.push(a.unbind());
        } else {
            py_args.push(a.unbind());
        }
    }
    if !found {
        return Err(PyRuntimeError::new_err("vmap: no batched inputs"));
    }

    // Per-iter callable: build a Python tuple, call f, collect leaves.
    let mut leaf_keys: Vec<String> = Vec::new();
    let mut first_result: Option<Py<PyAny>> = None;
    let mut first = true;

    let mut callable = |b: usize, sliced: &[ml::Array]| -> ml::MinmlResult<Vec<ml::Array>> {
        let result: PyResult<Vec<ml::Array>> = (|| {
            let call_args = PyList::empty(py);
            for i in 0..n {
                if c_index[i] >= 0 {
                    let arr = sliced[c_index[i] as usize].clone();
                    let py_arr = PyArray { inner: arr }.into_pyobject(py)?.into_any();
                    call_args.append(py_arr)?;
                } else if in_axes_v[i] >= 0 {
                    let bound = py_args[i].bind(py);
                    call_args.append(bound.get_item(b)?)?;
                } else {
                    call_args.append(py_args[i].bind(py))?;
                }
            }
            let tup = PyTuple::new(py, call_args.iter())?;
            let r = f.call1(tup)?;
            if first {
                first_result = Some(r.clone().unbind());
                first = false;
                let mut local_keys = Vec::new();
                let leaves = collect_leaves(&r, Some(&mut local_keys))?;
                leaf_keys = local_keys;
                Ok(leaves)
            } else {
                collect_leaves(&r, None)
            }
        })();
        result.map_err(|e| ml::MinmlError::Other(format!("vmap: python callable: {e}")))
    };

    let stacked = ml::vmap_apply(batch_n, &c_args, &c_in_axes, &mut callable).map_err(map_err)?;
    let template = first_result
        .ok_or_else(|| PyRuntimeError::new_err("vmap: no iterations"))?;
    rebuild_tree(py, template.bind(py), &leaf_keys, stacked)
}

#[pyfunction]
fn vmap<'py>(
    py: Python<'py>,
    f: Bound<'py, PyAny>,
    in_axes: Bound<'py, PyList>,
) -> PyResult<Py<PyAny>> {
    let f = f.unbind();
    let in_axes = in_axes.unbind();
    let closure = pyo3::types::PyCFunction::new_closure(
        py,
        None,
        None,
        move |args: &Bound<'_, PyTuple>, _kw: Option<&Bound<'_, PyDict>>| -> PyResult<Py<PyAny>> {
            Python::attach(|py| {
                let arg_list = PyList::empty(py);
                for a in args.iter() {
                    arg_list.append(a)?;
                }
                vmap_apply_py(py, f.bind(py).clone(), in_axes.bind(py).clone(), arg_list)
            })
        },
    )?;
    Ok(closure.into_pyobject(py)?.into_any().unbind())
}

// ---- WebGPU init (async) ----

#[pyfunction]
fn init_webgpu(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        ml::webgpu::init().await.map_err(map_err)?;
        Ok::<(), PyErr>(())
    })
}

#[pymodule]
fn _minml(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Device>()?;
    m.add_class::<DType>()?;
    m.add_class::<PyArray>()?;
    m.add_class::<PyPRNGKey>()?;
    m.add_class::<PyDirichlet>()?;
    m.add_class::<PyCategorical>()?;
    m.add_class::<PyNormal>()?;

    m.add_function(wrap_pyfunction!(set_default_device, m)?)?;
    m.add_function(wrap_pyfunction!(default_device, m)?)?;
    m.add_function(wrap_pyfunction!(array, m)?)?;
    m.add_function(wrap_pyfunction!(array_int, m)?)?;
    m.add_function(wrap_pyfunction!(array_shaped, m)?)?;
    m.add_function(wrap_pyfunction!(array_int_shaped, m)?)?;
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(mul, m)?)?;
    m.add_function(wrap_pyfunction!(dot, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(randint, m)?)?;
    m.add_function(wrap_pyfunction!(gather, m)?)?;
    m.add_function(wrap_pyfunction!(dirichlet_sample, m)?)?;
    m.add_function(wrap_pyfunction!(categorical_sample, m)?)?;
    m.add_function(wrap_pyfunction!(vmap_apply_py, m)?)?;
    m.add_function(wrap_pyfunction!(vmap, m)?)?;
    m.add_function(wrap_pyfunction!(init_webgpu, m)?)?;
    Ok(())
}
