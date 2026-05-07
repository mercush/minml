// src/cpu_backend.cpp
//
// The reference backend. Always built. Used to validate the architecture
// without GPU drivers.
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "backend.h"
#include "minml/buffer.h"

namespace minml {

namespace {
struct CpuBuffer : Buffer {
  // Untyped raw bytes; reinterpret per dtype at the kernel.
  unsigned char* ptr = nullptr;
  ~CpuBuffer() override { delete[] ptr; }
};

CpuBuffer& as_cpu(Buffer& b) { return static_cast<CpuBuffer&>(b); }
const CpuBuffer& as_cpu(const Buffer& b) {
  return static_cast<const CpuBuffer&>(b);
}
}  // namespace

float* cpu_data_f32(Array& a) {
  return reinterpret_cast<float*>(as_cpu(*a.buffer()).ptr);
}
const float* cpu_data_f32(const Array& a) {
  return reinterpret_cast<const float*>(as_cpu(*a.buffer()).ptr);
}
int32_t* cpu_data_i32(Array& a) {
  return reinterpret_cast<int32_t*>(as_cpu(*a.buffer()).ptr);
}
const int32_t* cpu_data_i32(const Array& a) {
  return reinterpret_cast<const int32_t*>(as_cpu(*a.buffer()).ptr);
}

namespace {
const float* data(const Array& a) { return cpu_data_f32(a); }
float* data(Array& a) { return cpu_data_f32(a); }
int32_t* data_i32(Array& a) { return cpu_data_i32(a); }
const int32_t* data_i32(const Array& a) { return cpu_data_i32(a); }
}  // namespace

std::shared_ptr<Buffer> cpu_allocate(size_t bytes) {
  auto b = std::make_shared<CpuBuffer>();
  b->bytes = bytes;
  b->device = Device::CPU;
  b->ptr = new unsigned char[bytes];
  return b;
}

// h2d/d2h take float* but treat the source as raw bytes; callers pass
// reinterpret_cast<const float*>(int32_data) for non-float dtypes. n is
// element count, not bytes — buffer ownership and dtype are tracked at
// the Array level.
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

void cpu_mul(const Array& a, const Array& b, Array& out) {
  const float* pa = data(a);
  const float* pb = data(b);
  float* po = data(out);
  for (size_t i = 0; i < out.size(); ++i) po[i] = pa[i] * pb[i];
}

void cpu_dot(const Array& a, const Array& b, Array& out) {
  const float* pa = data(a);
  const float* pb = data(b);
  float sum = 0.f;
  for (size_t i = 0; i < a.size(); ++i) sum += pa[i] * pb[i];
  data(out)[0] = sum;
}

void cpu_ones(Array& out) {
  if (out.dtype() == DType::Float32) {
    float* p = data(out);
    for (size_t i = 0; i < out.size(); ++i) p[i] = 1.0f;
  } else {  // Int32
    int32_t* p = data_i32(out);
    for (size_t i = 0; i < out.size(); ++i) p[i] = 1;
  }
}

void cpu_gather(const Array& table, const Array& indices, Array& out) {
  // table: (N, ...trail). indices: (...batch). out: (...batch, ...trail).
  size_t N = table.shape()[0];
  size_t trail = 1;
  for (size_t i = 1; i < table.shape().size(); ++i) trail *= table.shape()[i];
  size_t M = indices.size();

  const float* t = data(table);
  const int32_t* idx = data_i32(indices);
  float* o = data(out);

  for (size_t i = 0; i < M; ++i) {
    int32_t k = idx[i];
    if (k < 0 || static_cast<size_t>(k) >= N)
      throw std::runtime_error("gather: index out of range");
    std::memcpy(o + i * trail, t + static_cast<size_t>(k) * trail,
                trail * sizeof(float));
  }
}

}  // namespace minml
