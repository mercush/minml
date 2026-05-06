// src/webgpu_backend.cpp
//
// WebGPU backend using the standard webgpu.h C API. The same source compiles
// against Dawn (native) and Emscripten (browser); the difference is which
// implementation provides webgpu.h.
//
// init_webgpu() must be called once before any op. In the browser the device
// is acquired asynchronously; here we expose a callback-based init the user
// awaits with a Promise on the JS side.
//
// This file is deliberately compact and skips advanced concerns:
//   * No staging-buffer pool: every host->device copy uses wgpuQueueWriteBuffer
//     and every device->host uses MapAsync each time.
//   * Pipelines are compiled lazily on first use and cached.
//   * Errors are reported via std::runtime_error; production code would route
//     them through wgpuDeviceSetUncapturedErrorCallback.
#include <webgpu/webgpu.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "backend.h"
#include "minml/buffer.h"
#include "webgpu_shaders.h"

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

namespace minml {

namespace {

// ---- Global device handle -------------------------------------------------

struct Ctx {
  WGPUDevice device = nullptr;
  WGPUQueue queue = nullptr;
  std::unordered_map<std::string, WGPUComputePipeline> pipelines;
};
Ctx& ctx() {
  static Ctx c;
  return c;
}

void require_init() {
  if (!ctx().device)
    throw std::runtime_error(
        "WebGPU not initialized; call minml::webgpu_init() first");
}

// ---- Buffer ---------------------------------------------------------------

struct WebGPUBuffer : Buffer {
  WGPUBuffer handle = nullptr;
  ~WebGPUBuffer() override {
    if (handle) wgpuBufferRelease(handle);
  }
};

WebGPUBuffer& as_wgpu(Buffer& b) { return static_cast<WebGPUBuffer&>(b); }
const WebGPUBuffer& as_wgpu(const Buffer& b) {
  return static_cast<const WebGPUBuffer&>(b);
}
WGPUBuffer handle(const Array& a) { return as_wgpu(*a.buffer()).handle; }

// ---- Pipeline cache -------------------------------------------------------

// Helper: webgpu.h now uses WGPUStringView (a {data, length} struct) for
// every string field. WGPU_STRLEN means "compute length from null terminator".
WGPUStringView sv(const char* s) {
  return WGPUStringView{s, WGPU_STRLEN};
}

WGPUComputePipeline make_pipeline(const char* wgsl, const char* entry) {
  // The chained struct that carries WGSL source is WGPUShaderSourceWGSL in
  // the modern API (was WGPUShaderModuleWGSLDescriptor).
  WGPUShaderSourceWGSL wgsl_source{};
  wgsl_source.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_source.code = sv(wgsl);

  WGPUShaderModuleDescriptor sm_desc{};
  sm_desc.nextInChain = &wgsl_source.chain;
  WGPUShaderModule sm = wgpuDeviceCreateShaderModule(ctx().device, &sm_desc);

  WGPUComputePipelineDescriptor desc{};
  desc.compute.module = sm;
  desc.compute.entryPoint = sv(entry);
  WGPUComputePipeline p = wgpuDeviceCreateComputePipeline(ctx().device, &desc);
  wgpuShaderModuleRelease(sm);
  return p;
}

WGPUComputePipeline pipeline(const std::string& name) {
  auto it = ctx().pipelines.find(name);
  if (it != ctx().pipelines.end()) return it->second;
  WGPUComputePipeline p = nullptr;
  if (name == "add") p = make_pipeline(kAddWgsl, "main");
  else if (name == "dot") p = make_pipeline(kDotWgsl, "main");
  else throw std::runtime_error("unknown pipeline: " + name);
  ctx().pipelines[name] = p;
  return p;
}

// ---- Dispatch helper ------------------------------------------------------

void dispatch(const std::string& kernel, WGPUBuffer a, WGPUBuffer b,
              WGPUBuffer out, uint32_t workgroups) {
  auto pipe = pipeline(kernel);

  WGPUBindGroupEntry entries[3]{};
  entries[0].binding = 0; entries[0].buffer = a;   entries[0].size = WGPU_WHOLE_SIZE;
  entries[1].binding = 1; entries[1].buffer = b;   entries[1].size = WGPU_WHOLE_SIZE;
  entries[2].binding = 2; entries[2].buffer = out; entries[2].size = WGPU_WHOLE_SIZE;

  WGPUBindGroupDescriptor bg_desc{};
  bg_desc.layout = wgpuComputePipelineGetBindGroupLayout(pipe, 0);
  bg_desc.entryCount = 3;
  bg_desc.entries = entries;
  WGPUBindGroup bg = wgpuDeviceCreateBindGroup(ctx().device, &bg_desc);

  WGPUCommandEncoder enc =
      wgpuDeviceCreateCommandEncoder(ctx().device, nullptr);
  WGPUComputePassEncoder pass =
      wgpuCommandEncoderBeginComputePass(enc, nullptr);
  wgpuComputePassEncoderSetPipeline(pass, pipe);
  wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
  wgpuComputePassEncoderDispatchWorkgroups(pass, workgroups, 1, 1);
  wgpuComputePassEncoderEnd(pass);
  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
  wgpuQueueSubmit(ctx().queue, 1, &cmd);

  wgpuCommandBufferRelease(cmd);
  wgpuCommandEncoderRelease(enc);
  wgpuComputePassEncoderRelease(pass);
  wgpuBindGroupRelease(bg);
}

// ---- Readback (sync via spin-wait on Dawn, Asyncify on Emscripten) -------

struct MapState { bool done = false; bool ok = false; };

// Modern callback signature: (status, message, userdata1, userdata2).
void on_mapped(WGPUMapAsyncStatus status, WGPUStringView /*message*/,
               void* userdata1, void* /*userdata2*/) {
  auto* s = static_cast<MapState*>(userdata1);
  s->ok = (status == WGPUMapAsyncStatus_Success);
  s->done = true;
}

void wait_for_map(MapState& s) {
#ifdef __EMSCRIPTEN__
  // Requires linking with -sASYNCIFY; yields to the JS event loop.
  while (!s.done) emscripten_sleep(0);
#else
  // Dawn: pump the device until the callback fires.
  while (!s.done) wgpuDeviceTick(ctx().device);
#endif
}

}  // namespace

// ---- Public init ----------------------------------------------------------

// Caller-provided device. The caller is responsible for acquiring a device
// via wgpuInstanceRequestAdapter() + wgpuAdapterRequestDevice() (works the
// same in Dawn native and emdawnwebgpu) before calling this.
void webgpu_init_with_device(WGPUDevice dev) {
  ctx().device = dev;
  ctx().queue = wgpuDeviceGetQueue(dev);
}

// ---- Backend entry points -------------------------------------------------

std::shared_ptr<Buffer> webgpu_allocate(size_t bytes) {
  require_init();
  auto b = std::make_shared<WebGPUBuffer>();
  b->bytes = bytes;
  b->device = Device::WebGPU;

  WGPUBufferDescriptor d{};
  d.size = bytes;
  d.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc |
            WGPUBufferUsage_CopyDst;
  b->handle = wgpuDeviceCreateBuffer(ctx().device, &d);
  return b;
}

void webgpu_copy_host_to_device(Buffer& dst, const float* src, size_t n) {
  require_init();
  wgpuQueueWriteBuffer(ctx().queue, as_wgpu(dst).handle, 0, src,
                       n * sizeof(float));
}

void webgpu_copy_device_to_host(const Buffer& src, float* dst, size_t n) {
  require_init();
  // Stage into a MapRead buffer, copy with a command, then map.
  WGPUBufferDescriptor d{};
  d.size = n * sizeof(float);
  d.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
  WGPUBuffer staging = wgpuDeviceCreateBuffer(ctx().device, &d);

  WGPUCommandEncoder enc =
      wgpuDeviceCreateCommandEncoder(ctx().device, nullptr);
  wgpuCommandEncoderCopyBufferToBuffer(enc, as_wgpu(src).handle, 0, staging, 0,
                                       d.size);
  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
  wgpuQueueSubmit(ctx().queue, 1, &cmd);

  MapState s;
  // wgpuBufferMapAsync now takes a WGPUBufferMapCallbackInfo struct that
  // bundles the callback + userdata + scheduling mode.
  WGPUBufferMapCallbackInfo cb_info{};
  cb_info.mode = WGPUCallbackMode_AllowSpontaneous;
  cb_info.callback = on_mapped;
  cb_info.userdata1 = &s;
  wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, d.size, cb_info);
  wait_for_map(s);
  if (!s.ok) {
    wgpuBufferRelease(staging);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);
    throw std::runtime_error("webgpu mapAsync failed");
  }
  const void* mapped = wgpuBufferGetConstMappedRange(staging, 0, d.size);
  std::memcpy(dst, mapped, d.size);
  wgpuBufferUnmap(staging);
  wgpuBufferRelease(staging);
  wgpuCommandBufferRelease(cmd);
  wgpuCommandEncoderRelease(enc);
}

void webgpu_add(const Array& a, const Array& b, Array& out) {
  require_init();
  uint32_t n = static_cast<uint32_t>(out.size());
  uint32_t wg = (n + 63) / 64;
  dispatch("add", handle(a), handle(b), handle(out), wg);
}

void webgpu_dot(const Array& a, const Array& b, Array& out) {
  require_init();
  // Output is size 1 — zero it first since the kernel does atomic_add into it.
  float zero = 0.f;
  wgpuQueueWriteBuffer(ctx().queue, handle(out), 0, &zero, sizeof(zero));
  uint32_t n = static_cast<uint32_t>(a.size());
  uint32_t wg = (n + 63) / 64;
  dispatch("dot", handle(a), handle(b), handle(out), wg);
}

}  // namespace minml
