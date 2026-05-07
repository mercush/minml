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
#include "minml/dtype.h"
#include "minml/ops.h"
#include "minml/webgpu.h"

using namespace emscripten;
using namespace minml;

namespace {

Array make_array(const val& js_arr, Device d) {
  std::vector<float> data = vecFromJSArray<float>(js_arr);
  return Array(std::move(data), d);
}

// Return a plain JS Array instead of an embind register_vector wrapper.
// Plain arrays print and iterate naturally in REPLs/notebooks; users who
// need a typed buffer can call Float32Array.from() themselves.
val array_tolist(Array& self) {
  std::vector<float> data = self.tolist();
  return val::array(data.begin(), data.end());
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
      .value("Float32", DType::Float32);

  class_<Array>("Array")
      .function("size", &Array::size)
      .function("device", &Array::device)
      .function("dtype", &Array::dtype)
      .function("eval", &Array::eval)
      .function("tolist", &array_tolist)
      .function("item", &Array::item);

  function("array", &make_array);
  function("add", &add);
  function("mul", &mul);
  function("dot", &dot);
  function("setDefaultDevice", &set_default_device);
  function("initWebGPU", &init_webgpu);
}
