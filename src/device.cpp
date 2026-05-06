// src/device.cpp
#include "minml/device.h"

namespace minml {

namespace {
Device g_default = Device::CPU;
}

Device default_device() { return g_default; }
void set_default_device(Device d) { g_default = d; }

const char* device_name(Device d) {
  switch (d) {
    case Device::CPU: return "cpu";
    case Device::CUDA: return "cuda";
    case Device::WebGPU: return "webgpu";
  }
  return "unknown";
}

}  // namespace minml
