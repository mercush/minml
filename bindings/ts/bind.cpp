// bindings/ts/bind.cpp
//
// Compiled to WASM by Emscripten. Produces minml_js.js (loader) and
// minml_js.wasm.
//
// initWebGPU() acquires a WebGPU adapter+device entirely inside WASM via
// the wgpuInstance{RequestAdapter} / wgpuAdapter{RequestDevice} C API.
// With emdawnwebgpu these calls map through to navigator.gpu under the
// hood. ASYNCIFY (see CMakeLists.txt) lets the spin-wait below yield to
// the JS event loop while the underlying Promises resolve; on the JS
// side this surfaces as `await m.initWebGPU()`.
#include <emscripten.h>
#include <emscripten/bind.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <webgpu/webgpu.h>

#include "minml/array.h"
#include "minml/device.h"
#include "minml/distributions.h"
#include "minml/dtype.h"
#include "minml/ops.h"
#include "minml/prng.h"
#include "minml/transforms.h"
#include "minml/webgpu.h"

using namespace emscripten;
using namespace minml;

namespace {

Array make_array(const val& js_arr, Device d) {
  std::vector<float> data = vecFromJSArray<float>(js_arr);
  return Array(std::move(data), d);
}

// Plain JS Array of numbers — better notebook ergonomics than VectorFloat.
val array_tolist(Array& self) {
  if (self.dtype() == DType::Int32) {
    std::vector<int32_t> data = self.tolist_int();
    return val::array(data.begin(), data.end());
  }
  std::vector<float> data = self.tolist();
  return val::array(data.begin(), data.end());
}

// Expose Array.shape as a JS array of plain numbers.
val array_shape(Array& self) {
  return val::array(self.shape().begin(), self.shape().end());
}

// Convenience: construct a 1-D Int32 Array from a JS array of numbers.
Array make_array_int(const val& js_arr, Device d) {
  std::vector<int32_t> data = vecFromJSArray<int32_t>(js_arr);
  size_t n = data.size();  // pre-eval; std::move may run first.
  return Array(std::move(data), std::vector<size_t>{n}, d);
}

// ---- vmap binding shim ---------------------------------------------------
//
// Translates JS callable + JS pytree into the language-agnostic
// minml::vmap_apply (src/transforms.cpp). The actual loop, slicing, and
// stacking live in C++; this shim only handles JS-specific concerns:
// holding the JS callable, indexing JS-array list inputs by iteration,
// and walking a JS object pytree returned from f.

val embind_array_class() {
  static val cls = val::module_property("Array");
  return cls;
}

bool is_embind_array(const val& v) {
  return v.instanceof(embind_array_class());
}

// Collect own-enumerable Array properties of a JS object into a flat
// vector; for a single Array, returns just it. Caller pairs the order
// here with a parallel `keys` list to rebuild later.
std::vector<Array> collect_leaves_js(val v, std::vector<std::string>* keys) {
  std::vector<Array> out;
  if (is_embind_array(v)) {
    out.push_back(v.as<Array&>());
    if (keys) keys->push_back("");
    return out;
  }
  val all_keys = val::global("Object").call<val>("keys", v);
  size_t n = all_keys["length"].as<size_t>();
  for (size_t i = 0; i < n; ++i) {
    std::string k = all_keys[i].as<std::string>();
    val child = v[k];
    if (is_embind_array(child)) {
      out.push_back(child.as<Array&>());
      if (keys) keys->push_back(k);
    }
  }
  if (out.empty())
    throw std::runtime_error("vmap: function returned no Arrays");
  return out;
}

val rebuild_tree_js(val template_val, const std::vector<std::string>& keys,
                    const std::vector<Array>& stacked) {
  if (keys.size() == 1 && keys[0].empty()) return val(stacked[0]);
  val out = val::object();
  for (size_t i = 0; i < keys.size(); ++i) out.set(keys[i], stacked[i]);
  // Match the prototype of the template so `instanceof Trace` holds.
  val proto = val::global("Object").call<val>("getPrototypeOf", template_val);
  val::global("Object").call<val>("setPrototypeOf", out, proto);
  return out;
}

val vmap_apply_js(val f, val in_axes_js, val args_js) {
  // Parse in_axes (null -> -1 = unbatched).
  size_t n = in_axes_js["length"].as<size_t>();
  std::vector<int> in_axes(n);
  for (size_t i = 0; i < n; ++i) {
    val a = in_axes_js[i];
    in_axes[i] = (a.isNull() || a.isUndefined()) ? -1 : a.as<int>();
  }
  if (args_js["length"].as<size_t>() != n)
    throw std::runtime_error("vmap: in_axes length != args length");

  // Split inputs by kind:
  //   - embind Array, batched: feed to vmap_apply
  //   - embind Array, unbatched: feed to vmap_apply with in_axes=-1
  //   - JS list / scalar (anything else): captured here, looked up by iter
  std::vector<Array> c_args;
  std::vector<int> c_in_axes;
  // For each original position: where to find its per-iter value at call time.
  // -1 in c_index means "from JS args (lookup in closure)".
  std::vector<int> c_index(n, -1);
  std::vector<val> js_args(n);  // closure for JS-side inputs
  size_t batch_N = 0;
  bool found_batch = false;

  for (size_t i = 0; i < n; ++i) {
    val a = args_js[i];
    js_args[i] = a;  // always store; only used for non-Array paths
    if (is_embind_array(a)) {
      Array& arr = a.as<Array&>();
      c_index[i] = static_cast<int>(c_args.size());
      c_args.push_back(arr);
      c_in_axes.push_back(in_axes[i]);
      if (in_axes[i] >= 0 && !found_batch) {
        if (in_axes[i] != 0)
          throw std::runtime_error("vmap: only axis 0 supported on Arrays");
        if (arr.shape().empty())
          throw std::runtime_error("vmap: cannot batch over a scalar");
        batch_N = arr.shape()[0];
        found_batch = true;
      }
    } else if (in_axes[i] >= 0) {
      // JS list (or array-like): batch by length.
      if (!found_batch) {
        batch_N = a["length"].as<size_t>();
        found_batch = true;
      }
    }
  }
  if (!found_batch) throw std::runtime_error("vmap: no batched inputs");

  // First-call leaf info, captured during the first f invocation.
  std::vector<std::string> leaf_keys;
  val first_result = val::undefined();
  bool first = true;

  VmapCallable callable = [&](size_t b, const std::vector<Array>& sliced) {
    val call_args = val::array();
    for (size_t i = 0; i < n; ++i) {
      val v;
      if (c_index[i] >= 0) {
        // From C++: either passed-through Array or sliced one.
        v = val(sliced[c_index[i]]);
      } else if (in_axes[i] >= 0) {
        // JS list: index by b.
        v = js_args[i][b];
      } else {
        v = js_args[i];
      }
      call_args.call<void>("push", v);
    }
    val r = f.call<val>("apply", val::null(), call_args);
    if (first) {
      first_result = r;
      first = false;
      return collect_leaves_js(r, &leaf_keys);
    }
    return collect_leaves_js(r, /*keys=*/nullptr);
  };

  std::vector<Array> stacked = vmap_apply(batch_N, c_args, c_in_axes, callable);
  return rebuild_tree_js(first_result, leaf_keys, stacked);
}

// ---- Wrappers for distribution sample (PRNGKey -> k0/k1) -----------------

Array dirichlet_sample_js(uint32_t k0, uint32_t k1, const Array& alpha,
                          const val& batch_shape_js) {
  std::vector<size_t> bs = vecFromJSArray<size_t>(batch_shape_js);
  return dirichlet_sample(k0, k1, alpha, std::move(bs));
}

Array categorical_sample_js(uint32_t k0, uint32_t k1, const Array& probs,
                            const val& batch_shape_js) {
  std::vector<size_t> bs = vecFromJSArray<size_t>(batch_shape_js);
  return categorical_sample(k0, k1, probs, std::move(bs));
}

Array ones_js(const val& shape_js, DType dt, Device dev) {
  std::vector<size_t> shape = vecFromJSArray<size_t>(shape_js);
  return ones(std::move(shape), dt, dev);
}

Array randint_js(uint32_t k0, uint32_t k1, int32_t low, int32_t high,
                 const val& shape_js, Device dev) {
  std::vector<size_t> shape = vecFromJSArray<size_t>(shape_js);
  return randint(k0, k1, low, high, std::move(shape), dev);
}

Array dirichlet_sample_method(const Dirichlet& self, const PRNGKey& key,
                              const val& batch_shape_js) {
  std::vector<size_t> bs = vecFromJSArray<size_t>(batch_shape_js);
  return self.sample(key, std::move(bs));
}

Array categorical_sample_method(const Categorical& self, const PRNGKey& key,
                                const val& batch_shape_js) {
  std::vector<size_t> bs = vecFromJSArray<size_t>(batch_shape_js);
  return self.sample(key, std::move(bs));
}

Array normal_sample_method(const Normal& self, const PRNGKey& key,
                           const val& batch_shape_js) {
  std::vector<size_t> bs = vecFromJSArray<size_t>(batch_shape_js);
  return self.sample(key, std::move(bs));
}

// Return a real JS Array (destructurable as `const [k1,k2,k3] = key.split(3)`)
// rather than an embind register_vector wrapper.
val prngkey_split(const PRNGKey& self, size_t n) {
  std::vector<PRNGKey> kids = self.split(n);
  val out = val::array();
  for (auto& k : kids) out.call<void>("push", val(k));
  return out;
}

// ---- WebGPU device acquisition -------------------------------------------

struct InitState {
  WGPUDevice device = nullptr;
  bool done = false;
  std::string error;
};

void append_message(std::string& out, WGPUStringView msg) {
  if (msg.data && msg.length) {
    out += ": ";
    out.append(msg.data, msg.length);
  }
}

void on_device(WGPURequestDeviceStatus status, WGPUDevice device,
               WGPUStringView message, void* userdata1, void* /*userdata2*/) {
  auto* s = static_cast<InitState*>(userdata1);
  if (status == WGPURequestDeviceStatus_Success) {
    s->device = device;
  } else {
    s->error = "wgpuAdapterRequestDevice failed";
    append_message(s->error, message);
  }
  s->done = true;
}

void on_adapter(WGPURequestAdapterStatus status, WGPUAdapter adapter,
                WGPUStringView message, void* userdata1, void* /*userdata2*/) {
  auto* s = static_cast<InitState*>(userdata1);
  if (status != WGPURequestAdapterStatus_Success) {
    s->error = "wgpuInstanceRequestAdapter failed";
    append_message(s->error, message);
    s->done = true;
    return;
  }
  WGPURequestDeviceCallbackInfo cb{};
  cb.mode = WGPUCallbackMode_AllowSpontaneous;
  cb.callback = on_device;
  cb.userdata1 = s;
  wgpuAdapterRequestDevice(adapter, nullptr, cb);
}

void init_webgpu() {
  InitState s;

  WGPUInstance instance = wgpuCreateInstance(nullptr);
  if (!instance) throw std::runtime_error("wgpuCreateInstance returned null");

  WGPURequestAdapterCallbackInfo cb{};
  cb.mode = WGPUCallbackMode_AllowSpontaneous;
  cb.callback = on_adapter;
  cb.userdata1 = &s;
  wgpuInstanceRequestAdapter(instance, nullptr, cb);

  // Yield to the JS event loop until the chained callbacks complete.
  while (!s.done) emscripten_sleep(0);

  if (!s.device) {
    throw std::runtime_error(s.error.empty() ? "WebGPU init failed" : s.error);
  }
  webgpu_init_with_device(s.device);
}

}  // namespace

EMSCRIPTEN_BINDINGS(minml_module) {
  enum_<Device>("Device")
      .value("CPU", Device::CPU)
      .value("CUDA", Device::CUDA)
      .value("WebGPU", Device::WebGPU);

  enum_<DType>("DType")
      .value("Float32", DType::Float32)
      .value("Int32", DType::Int32);

  class_<Array>("Array")
      .function("size", &Array::size)
      .function("shape", &array_shape)
      .function("device", &Array::device)
      .function("dtype", &Array::dtype)
      .function("eval", &Array::eval)
      .function("tolist", &array_tolist)
      .function("item", &Array::item);

  class_<PRNGKey>("PRNGKey")
      .class_function("new", &PRNGKey::from_seed)
      .function("split", &prngkey_split)
      .function("k0", &PRNGKey::k0)
      .function("k1", &PRNGKey::k1);

  class_<Dirichlet>("Dirichlet")
      .constructor<Array>()
      .function("sample", &dirichlet_sample_method);

  class_<Categorical>("Categorical")
      .constructor<Array>()
      .function("sample", &categorical_sample_method);

  class_<Normal>("Normal")
      .constructor<>()
      .function("sample", &normal_sample_method);

  function("array", &make_array);
  function("arrayInt", &make_array_int);
  function("add", &add);
  function("mul", &mul);
  function("dot", &dot);
  function("ones", &ones_js);
  function("randint", &randint_js);
  function("gather", &gather);
  function("dirichletSample", &dirichlet_sample_js);
  function("categoricalSample", &categorical_sample_js);
  function("vmapApply", &vmap_apply_js);
  function("setDefaultDevice", &set_default_device);
  function("defaultDevice", &default_device);
  function("initWebGPU", &init_webgpu);
}
