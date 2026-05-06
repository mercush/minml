// minml/webgpu.h
//
// Public hook for handing a pre-acquired WebGPU device to the runtime.
// The header pulls in <webgpu/webgpu.h>, which only exists on builds
// configured with -DMINML_BUILD_WEBGPU=ON. Including this file outside
// such a build is a configuration error.
#pragma once

#include <webgpu/webgpu.h>

namespace minml {

// The caller is responsible for acquiring a device — typically via
// wgpuInstanceRequestAdapter + wgpuAdapterRequestDevice (works the same
// in Dawn native and emdawnwebgpu) — before calling this. The handle
// must remain valid for the lifetime of any subsequent WebGPU op.
void webgpu_init_with_device(WGPUDevice device);

}  // namespace minml
