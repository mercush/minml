// src/stubs.cpp
//
// When a backend is disabled at configure time, its source isn't compiled but
// the symbols are still referenced from ops.cpp. We resolve them to functions
// that throw, so building without (say) CUDA still links and only fails if
// you try to use Device::CUDA.
//
// One of MINML_HAS_CPU / CUDA / WEBGPU is defined per backend that IS built.
// We supply stubs only for the ones that are NOT built.

#include <stdexcept>

#include "backend.h"

namespace minml {
namespace {
[[noreturn]] void fail(const char* dev) {
  throw std::runtime_error(std::string("minml: backend not built: ") + dev);
}
}  // namespace

#ifndef MINML_HAS_CPU
std::shared_ptr<Buffer> cpu_allocate(size_t) { fail("cpu"); }
void cpu_copy_host_to_device(Buffer&, const float*, size_t) { fail("cpu"); }
void cpu_copy_device_to_host(const Buffer&, float*, size_t) { fail("cpu"); }
void cpu_add(const Array&, const Array&, Array&) { fail("cpu"); }
void cpu_mul(const Array&, const Array&, Array&) { fail("cpu"); }
void cpu_dot(const Array&, const Array&, Array&) { fail("cpu"); }
#endif

#ifndef MINML_HAS_CUDA
std::shared_ptr<Buffer> cuda_allocate(size_t) { fail("cuda"); }
void cuda_copy_host_to_device(Buffer&, const float*, size_t) { fail("cuda"); }
void cuda_copy_device_to_host(const Buffer&, float*, size_t) { fail("cuda"); }
void cuda_add(const Array&, const Array&, Array&) { fail("cuda"); }
void cuda_mul(const Array&, const Array&, Array&) { fail("cuda"); }
void cuda_dot(const Array&, const Array&, Array&) { fail("cuda"); }
#endif

#ifndef MINML_HAS_WEBGPU
std::shared_ptr<Buffer> webgpu_allocate(size_t) { fail("webgpu"); }
void webgpu_copy_host_to_device(Buffer&, const float*, size_t) {
  fail("webgpu");
}
void webgpu_copy_device_to_host(const Buffer&, float*, size_t) {
  fail("webgpu");
}
void webgpu_add(const Array&, const Array&, Array&) { fail("webgpu"); }
void webgpu_mul(const Array&, const Array&, Array&) { fail("webgpu"); }
void webgpu_dot(const Array&, const Array&, Array&) { fail("webgpu"); }
#endif

}  // namespace minml
