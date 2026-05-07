// minml-wasm — TypeScript/WASM bindings for minml-core via wasm-bindgen.
//
// Async surface: `Array.tolist`, `Array.item`, `Array.eval`, and
// `init_webgpu` return Promises (via wasm-bindgen-futures). Graph
// builders (add/mul/dot/...) are synchronous; they don't touch the GPU.
//
// vmap shim ports bindings/ts/bind.cpp:60-194: walks JS objects via
// js_sys::Reflect to collect Array leaves, rebuilds with stacked outputs.

use minml_core as ml;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

fn jserr(e: ml::MinmlError) -> JsValue {
    JsValue::from_str(&e.to_string())
}

#[wasm_bindgen(start)]
pub fn _start() {
    console_error_panic_hook::set_once();
}

// ---- Enums (plain numbers on the JS side) ----

#[wasm_bindgen]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Device {
    CPU = 0,
    CUDA = 1,
    WebGPU = 2,
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

#[wasm_bindgen]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Float32 = 0,
    Int32 = 1,
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

#[wasm_bindgen]
#[derive(Clone)]
pub struct Array {
    inner: ml::Array,
}

#[wasm_bindgen]
impl Array {
    #[wasm_bindgen(js_name = size)]
    pub fn size(&self) -> usize {
        self.inner.size()
    }

    #[wasm_bindgen(js_name = shape)]
    pub fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    #[wasm_bindgen(js_name = device)]
    pub fn device(&self) -> Device {
        self.inner.device().into()
    }

    #[wasm_bindgen(js_name = dtype)]
    pub fn dtype(&self) -> DType {
        self.inner.dtype().into()
    }

    // Promise<void>. Forces the lazy graph to run; useful to surface
    // backend errors at a known point.
    #[wasm_bindgen(js_name = eval)]
    pub fn eval_js(&self) -> js_sys::Promise {
        let arr = self.inner.clone();
        future_to_promise(async move {
            arr.eval().map_err(jserr)?;
            Ok(JsValue::UNDEFINED)
        })
    }

    // Promise<number[]>. On WebGPU this awaits map_async; on CPU/CUDA the
    // future resolves immediately. Same JS shape across backends.
    #[wasm_bindgen(js_name = tolist)]
    pub fn tolist_js(&self) -> js_sys::Promise {
        let arr = self.inner.clone();
        future_to_promise(async move {
            // Type dispatch: Float32 -> Vec<f32>; Int32 -> Vec<i32>.
            // Both mapped to a JS Array of plain numbers, matching the
            // current TS surface (bindings/ts/bind.cpp:39-46).
            match arr.dtype() {
                ml::DType::Float32 => {
                    let v = arr.tolist().await.map_err(jserr)?;
                    let arr_js = js_sys::Array::new_with_length(v.len() as u32);
                    for (i, x) in v.iter().enumerate() {
                        arr_js.set(i as u32, JsValue::from_f64(*x as f64));
                    }
                    Ok(arr_js.into())
                }
                ml::DType::Int32 => {
                    let v = arr.tolist_int().await.map_err(jserr)?;
                    let arr_js = js_sys::Array::new_with_length(v.len() as u32);
                    for (i, x) in v.iter().enumerate() {
                        arr_js.set(i as u32, JsValue::from_f64(*x as f64));
                    }
                    Ok(arr_js.into())
                }
            }
        })
    }

    // Promise<number>.
    #[wasm_bindgen(js_name = item)]
    pub fn item_js(&self) -> js_sys::Promise {
        let arr = self.inner.clone();
        future_to_promise(async move {
            let v = arr.item().await.map_err(jserr)?;
            Ok(JsValue::from_f64(v as f64))
        })
    }
}

// ---- PRNGKey ----

#[wasm_bindgen]
#[derive(Clone)]
pub struct PRNGKey {
    inner: ml::PRNGKey,
}

#[wasm_bindgen]
impl PRNGKey {
    #[wasm_bindgen(js_name = "new")]
    pub fn new_seed(seed: u32) -> Self {
        Self {
            inner: ml::PRNGKey::from_seed(seed),
        }
    }

    pub fn k0(&self) -> u32 {
        self.inner.k0()
    }
    pub fn k1(&self) -> u32 {
        self.inner.k1()
    }

    // Returns a real JS Array of PRNGKey, destructurable as
    //   const [k1, k2, k3] = key.split(3).
    pub fn split(&self, n: usize) -> js_sys::Array {
        let kids = self.inner.split(n);
        let out = js_sys::Array::new_with_length(kids.len() as u32);
        for (i, k) in kids.into_iter().enumerate() {
            out.set(i as u32, JsValue::from(PRNGKey { inner: k }));
        }
        out
    }
}

// ---- Distribution wrappers ----

#[wasm_bindgen]
pub struct Dirichlet {
    inner: ml::Dirichlet,
}

#[wasm_bindgen]
impl Dirichlet {
    #[wasm_bindgen(constructor)]
    pub fn new(alpha: Array) -> Self {
        Self {
            inner: ml::Dirichlet::new(alpha.inner),
        }
    }
    pub fn sample(&self, key: &PRNGKey, batch_shape: Vec<usize>) -> Result<Array, JsValue> {
        Ok(Array {
            inner: self.inner.sample(key.inner, batch_shape).map_err(jserr)?,
        })
    }
}

#[wasm_bindgen]
pub struct Categorical {
    inner: ml::Categorical,
}

#[wasm_bindgen]
impl Categorical {
    #[wasm_bindgen(constructor)]
    pub fn new(probs: Array) -> Self {
        Self {
            inner: ml::Categorical::new(probs.inner),
        }
    }
    pub fn sample(&self, key: &PRNGKey, batch_shape: Vec<usize>) -> Result<Array, JsValue> {
        Ok(Array {
            inner: self.inner.sample(key.inner, batch_shape).map_err(jserr)?,
        })
    }
}

#[wasm_bindgen]
pub struct Normal {
    inner: ml::Normal,
}

#[wasm_bindgen]
impl Normal {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: ml::Normal::new(),
        }
    }
    pub fn sample(&self, key: &PRNGKey, batch_shape: Vec<usize>) -> Result<Array, JsValue> {
        Ok(Array {
            inner: self.inner.sample(key.inner, batch_shape).map_err(jserr)?,
        })
    }
}

// ---- Free functions ----

#[wasm_bindgen(js_name = setDefaultDevice)]
pub fn set_default_device(d: Device) {
    ml::set_default_device(d.into());
}

#[wasm_bindgen(js_name = defaultDevice)]
pub fn default_device() -> Device {
    ml::default_device().into()
}

#[wasm_bindgen]
pub fn array(data: Vec<f32>, device: Device) -> Result<Array, JsValue> {
    Ok(Array {
        inner: ml::Array::from_f32_1d(data, device.into()).map_err(jserr)?,
    })
}

#[wasm_bindgen(js_name = arrayInt)]
pub fn array_int(data: Vec<i32>, device: Device) -> Result<Array, JsValue> {
    Ok(Array {
        inner: ml::Array::from_i32_1d(data, device.into()).map_err(jserr)?,
    })
}

#[wasm_bindgen]
pub fn add(a: &Array, b: &Array) -> Result<Array, JsValue> {
    Ok(Array {
        inner: ml::add(&a.inner, &b.inner).map_err(jserr)?,
    })
}

#[wasm_bindgen]
pub fn mul(a: &Array, b: &Array) -> Result<Array, JsValue> {
    Ok(Array {
        inner: ml::mul(&a.inner, &b.inner).map_err(jserr)?,
    })
}

#[wasm_bindgen]
pub fn dot(a: &Array, b: &Array) -> Result<Array, JsValue> {
    Ok(Array {
        inner: ml::dot(&a.inner, &b.inner).map_err(jserr)?,
    })
}

#[wasm_bindgen]
pub fn ones(shape: Vec<usize>, dtype: DType, device: Device) -> Array {
    Array {
        inner: ml::ones(shape, dtype.into(), device.into()),
    }
}

#[wasm_bindgen]
pub fn randint(
    k0: u32,
    k1: u32,
    low: i32,
    high: i32,
    shape: Vec<usize>,
    device: Device,
) -> Array {
    Array {
        inner: ml::randint(k0, k1, low, high, shape, device.into()),
    }
}

#[wasm_bindgen]
pub fn gather(table: &Array, indices: &Array) -> Result<Array, JsValue> {
    Ok(Array {
        inner: ml::gather(&table.inner, &indices.inner).map_err(jserr)?,
    })
}

#[wasm_bindgen(js_name = dirichletSample)]
pub fn dirichlet_sample(
    k0: u32,
    k1: u32,
    alpha: &Array,
    batch_shape: Vec<usize>,
) -> Result<Array, JsValue> {
    Ok(Array {
        inner: ml::dirichlet_sample(k0, k1, &alpha.inner, batch_shape).map_err(jserr)?,
    })
}

#[wasm_bindgen(js_name = categoricalSample)]
pub fn categorical_sample(
    k0: u32,
    k1: u32,
    probs: &Array,
    batch_shape: Vec<usize>,
) -> Result<Array, JsValue> {
    Ok(Array {
        inner: ml::categorical_sample(k0, k1, &probs.inner, batch_shape).map_err(jserr)?,
    })
}

// ---- WebGPU init ----

#[wasm_bindgen(js_name = initWebGPU)]
pub fn init_webgpu() -> js_sys::Promise {
    future_to_promise(async move {
        ml::webgpu::init().await.map_err(jserr)?;
        Ok(JsValue::UNDEFINED)
    })
}

// ---- vmap shim ----
//
// Wasm-side vmap supports two return shapes from the user callable:
//   1. A single Array.
//   2. A JS Array of Array (one per leaf).
//
// Pytree-style returns (a class instance whose properties are Arrays,
// preserving `instanceof Trace`) are NOT supported here — that would
// require detecting wasm-bindgen Array instances from a JsValue, which
// wasm-bindgen 0.2 doesn't expose without `extends`. The Python binding
// keeps the pytree shim because pyo3 lets us walk `__dict__` directly.
// JS code wanting pytree behavior can spread / destructure manually
// around the vmap call.
//
// Inputs may be Array (batched along axis 0 if in_axes[i] >= 0), or any
// JS array-like (batched by `.length`, indexed each iteration). Mirrors
// bindings/ts/bind.cpp:113-194 modulo the pytree handling.

// Take ownership of a wasm-bindgen Array from a JsValue if it really is
// one. The marker we use is the `__wbg_ptr` field present on every
// wasm-bindgen wrapper. We then read the pointer and reconstruct an
// owned `Array` by cloning the inner ml::Array (which is cheap — Arc).
fn try_take_array(v: &JsValue) -> Option<Array> {
    use wasm_bindgen::convert::RefFromWasmAbi;
    let ptr_val = js_sys::Reflect::get(v, &JsValue::from_str("__wbg_ptr")).ok()?;
    if ptr_val.is_undefined() {
        return None;
    }
    let ptr = ptr_val.as_f64()? as u32;
    if ptr == 0 {
        return None;
    }
    // Unsafe: assumes the JsValue is one of *our* Array wrappers, not some
    // other #[wasm_bindgen] struct. If users pass a different wrapper
    // here, the cast is UB. We only call this from the vmap shim, so the
    // contract is "pass Arrays and JS arrays only" — matches what the C++
    // bind shim does today.
    let r: <Array as RefFromWasmAbi>::Anchor = unsafe { Array::ref_from_abi(ptr) };
    Some(Array { inner: r.inner.clone() })
}

#[wasm_bindgen(js_name = vmapApply)]
pub fn vmap_apply_js(
    f: js_sys::Function,
    in_axes: js_sys::Array,
    args: js_sys::Array,
) -> Result<JsValue, JsValue> {
    let n = in_axes.length() as usize;
    if args.length() as usize != n {
        return Err(JsValue::from_str("vmap: in_axes length != args length"));
    }
    let in_axes_v: Vec<i32> = (0..n)
        .map(|i| {
            let a = in_axes.get(i as u32);
            if a.is_null() || a.is_undefined() {
                -1
            } else {
                a.as_f64().unwrap_or(-1.0) as i32
            }
        })
        .collect();

    let mut c_args: Vec<ml::Array> = Vec::new();
    let mut c_in_axes: Vec<i32> = Vec::new();
    let mut c_index: Vec<i32> = vec![-1; n];
    let mut js_args: Vec<JsValue> = Vec::with_capacity(n);
    let mut batch_n: usize = 0;
    let mut found = false;

    for i in 0..n {
        let a = args.get(i as u32);
        js_args.push(a.clone());
        if let Some(arr) = try_take_array(&a) {
            c_index[i] = c_args.len() as i32;
            c_args.push(arr.inner.clone());
            c_in_axes.push(in_axes_v[i]);
            if in_axes_v[i] >= 0 && !found {
                if in_axes_v[i] != 0 {
                    return Err(JsValue::from_str(
                        "vmap: only axis 0 supported on Arrays",
                    ));
                }
                if arr.inner.shape().is_empty() {
                    return Err(JsValue::from_str("vmap: cannot batch over a scalar"));
                }
                batch_n = arr.inner.shape()[0];
                found = true;
            }
        } else if in_axes_v[i] >= 0 && !found {
            let len = js_sys::Reflect::get(&a, &JsValue::from_str("length"))
                .map_err(|_| JsValue::from_str("vmap: batched non-Array missing .length"))?
                .as_f64()
                .ok_or_else(|| JsValue::from_str("vmap: .length not a number"))?
                as usize;
            batch_n = len;
            found = true;
        }
    }
    if !found {
        return Err(JsValue::from_str("vmap: no batched inputs"));
    }

    let mut callable_err: Option<JsValue> = None;
    let mut return_shape_n: Option<usize> = None; // None=Array; Some(n)=JS list of n

    let mut callable = |b: usize, sliced: &[ml::Array]| -> ml::MinmlResult<Vec<ml::Array>> {
        let call_args = js_sys::Array::new();
        for i in 0..n {
            if c_index[i] >= 0 {
                let arr = sliced[c_index[i] as usize].clone();
                let v: JsValue = JsValue::from(Array { inner: arr });
                call_args.push(&v);
            } else if in_axes_v[i] >= 0 {
                let v = js_sys::Reflect::get_u32(&js_args[i], b as u32)
                    .unwrap_or(JsValue::UNDEFINED);
                call_args.push(&v);
            } else {
                call_args.push(&js_args[i]);
            }
        }
        let r = match f.apply(&JsValue::NULL, &call_args) {
            Ok(v) => v,
            Err(e) => {
                callable_err = Some(e);
                return Err(ml::MinmlError::Other("vmap: js callable threw".into()));
            }
        };
        // Single Array return:
        if let Some(arr) = try_take_array(&r) {
            return_shape_n = None;
            return Ok(vec![arr.inner]);
        }
        // Otherwise treat as JS array of Array.
        if !r.is_object() {
            callable_err = Some(JsValue::from_str(
                "vmap: callable must return an Array or array-of-Array",
            ));
            return Err(ml::MinmlError::Other("vmap: bad return type".into()));
        }
        let len = js_sys::Reflect::get(&r, &JsValue::from_str("length"))
            .ok()
            .and_then(|v| v.as_f64())
            .map(|v| v as usize);
        match len {
            Some(l) => {
                let mut out = Vec::with_capacity(l);
                for i in 0..l {
                    let leaf = js_sys::Reflect::get_u32(&r, i as u32).map_err(|e| {
                        callable_err = Some(e);
                        ml::MinmlError::Other("vmap: leaf access failed".into())
                    })?;
                    match try_take_array(&leaf) {
                        Some(a) => out.push(a.inner),
                        None => {
                            callable_err = Some(JsValue::from_str(
                                "vmap: leaf is not an Array",
                            ));
                            return Err(ml::MinmlError::Other("vmap: bad leaf".into()));
                        }
                    }
                }
                return_shape_n = Some(l);
                Ok(out)
            }
            None => {
                callable_err = Some(JsValue::from_str(
                    "vmap: callable must return Array or list of Array",
                ));
                Err(ml::MinmlError::Other("vmap: bad return type".into()))
            }
        }
    };

    let stacked = match ml::vmap_apply(batch_n, &c_args, &c_in_axes, &mut callable) {
        Ok(v) => v,
        Err(e) => {
            if let Some(je) = callable_err {
                return Err(je);
            }
            return Err(jserr(e));
        }
    };

    match return_shape_n {
        None => {
            let v: JsValue = JsValue::from(Array {
                inner: stacked.into_iter().next().unwrap(),
            });
            Ok(v)
        }
        Some(_) => {
            let out = js_sys::Array::new();
            for arr in stacked {
                out.push(&JsValue::from(Array { inner: arr }));
            }
            Ok(out.into())
        }
    }
}
