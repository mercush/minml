// src/cpu_backend.cpp
//
// The reference backend. Always built. Used to validate the architecture
// without GPU drivers.
#include <cstring>
#include <memory>

#include "backend.h"
#include "minml/buffer.h"

namespace minml {

namespace {
struct CpuBuffer : Buffer {
  float* ptr = nullptr;
  ~CpuBuffer() override { delete[] ptr; }
};

CpuBuffer& as_cpu(Buffer& b) { return static_cast<CpuBuffer&>(b); }
const CpuBuffer& as_cpu(const Buffer& b) {
  return static_cast<const CpuBuffer&>(b);
}
const float* data(const Array& a) { return as_cpu(*a.buffer()).ptr; }
float* data(Array& a) { return as_cpu(*a.buffer()).ptr; }
}  // namespace

std::shared_ptr<Buffer> cpu_allocate(size_t bytes) {
  auto b = std::make_shared<CpuBuffer>();
  b->bytes = bytes;
  b->device = Device::CPU;
  b->ptr = new float[bytes / sizeof(float)];
  return b;
}

void cpu_copy_host_to_device(Buffer& dst, const float* src, size_t n) {
  std::memcpy(as_cpu(dst).ptr, src, n * sizeof(float));
}

void cpu_copy_device_to_host(const Buffer& src, float* dst, size_t n) {
  std::memcpy(dst, as_cpu(src).ptr, n * sizeof(float));
}

void cpu_add(const Array& a, const Array& b, Array& out) {
  const float* pa = data(a);
  const float* pb = data(b);
  float* po = data(out);
  for (size_t i = 0; i < out.size(); ++i) po[i] = pa[i] + pb[i];
}

void cpu_dot(const Array& a, const Array& b, Array& out) {
  const float* pa = data(a);
  const float* pb = data(b);
  float sum = 0.f;
  for (size_t i = 0; i < a.size(); ++i) sum += pa[i] * pb[i];
  data(out)[0] = sum;
}

}  // namespace minml
