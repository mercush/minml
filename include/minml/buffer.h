// minml/buffer.h
//
// A Buffer is opaque, ref-counted device memory. Each backend supplies its
// own subclass with its own destructor (free, cudaFree, wgpuBufferRelease).
// The base class lets Array hold a backend-agnostic shared_ptr.
#pragma once

#include <cstddef>
#include "minml/device.h"

namespace minml {

struct Buffer {
  size_t bytes = 0;
  Device device = Device::CPU;
  virtual ~Buffer() = default;
};

}  // namespace minml
