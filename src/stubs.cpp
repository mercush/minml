// src/stubs.cpp
//
// When a backend is disabled at configure time, its source isn't compiled
// but the symbols are still referenced from ops.cpp / array.cpp. We resolve
// them to functions that throw, so building without (say) CUDA still links
// and only fails if you try to use Device::CUDA.

#include <stdexcept>
#include <string>

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
void cpu_ones(Array&) { fail("cpu"); }
void cpu_randint(uint32_t, uint32_t, int32_t, int32_t, Array&) { fail("cpu"); }
void cpu_gather(const Array&, const Array&, Array&) { fail("cpu"); }
void cpu_dirichlet_sample(uint32_t, uint32_t, const std::vector<size_t>&,
                          const Array&, Array&) { fail("cpu"); }
void cpu_categorical_sample(uint32_t, uint32_t, const std::vector<size_t>&,
                            const Array&, Array&) { fail("cpu"); }
#endif

#ifndef MINML_HAS_CUDA
std::shared_ptr<Buffer> cuda_allocate(size_t) { fail("cuda"); }
void cuda_copy_host_to_device(Buffer&, const float*, size_t) { fail("cuda"); }
void cuda_copy_device_to_host(const Buffer&, float*, size_t) { fail("cuda"); }
void cuda_add(const Array&, const Array&, Array&) { fail("cuda"); }
void cuda_mul(const Array&, const Array&, Array&) { fail("cuda"); }
void cuda_dot(const Array&, const Array&, Array&) { fail("cuda"); }
void cuda_ones(Array&) { fail("cuda"); }
void cuda_randint(uint32_t, uint32_t, int32_t, int32_t, Array&) { fail("cuda"); }
void cuda_gather(const Array&, const Array&, Array&) { fail("cuda"); }
void cuda_dirichlet_sample(uint32_t, uint32_t, const std::vector<size_t>&,
                           const Array&, Array&) { fail("cuda"); }
void cuda_categorical_sample(uint32_t, uint32_t, const std::vector<size_t>&,
                             const Array&, Array&) { fail("cuda"); }
#endif

#ifndef MINML_HAS_WEBGPU
std::shared_ptr<Buffer> webgpu_allocate(size_t) { fail("webgpu"); }
void webgpu_copy_host_to_device(Buffer&, const float*, size_t) { fail("webgpu"); }
void webgpu_copy_device_to_host(const Buffer&, float*, size_t) { fail("webgpu"); }
void webgpu_add(const Array&, const Array&, Array&) { fail("webgpu"); }
void webgpu_mul(const Array&, const Array&, Array&) { fail("webgpu"); }
void webgpu_dot(const Array&, const Array&, Array&) { fail("webgpu"); }
void webgpu_ones(Array&) { fail("webgpu"); }
void webgpu_randint(uint32_t, uint32_t, int32_t, int32_t, Array&) { fail("webgpu"); }
void webgpu_gather(const Array&, const Array&, Array&) { fail("webgpu"); }
void webgpu_dirichlet_sample(uint32_t, uint32_t, const std::vector<size_t>&,
                             const Array&, Array&) { fail("webgpu"); }
void webgpu_categorical_sample(uint32_t, uint32_t, const std::vector<size_t>&,
                               const Array&, Array&) { fail("webgpu"); }
#endif

// New ops haven't been wired to the WebGPU/CUDA backends; expose throwing
// definitions even when those backends ARE built so the link succeeds.
// Guard each by an ifdef so we don't double-define when a real impl shows
// up later.
#ifdef MINML_HAS_WEBGPU
void webgpu_ones(Array&) { fail("webgpu (ones)"); }
void webgpu_randint(uint32_t, uint32_t, int32_t, int32_t, Array&) { fail("webgpu (randint)"); }
void webgpu_gather(const Array&, const Array&, Array&) { fail("webgpu (gather)"); }
void webgpu_dirichlet_sample(uint32_t, uint32_t, const std::vector<size_t>&,
                             const Array&, Array&) { fail("webgpu (dirichlet)"); }
void webgpu_categorical_sample(uint32_t, uint32_t, const std::vector<size_t>&,
                               const Array&, Array&) { fail("webgpu (categorical)"); }
#endif

#ifdef MINML_HAS_CUDA
void cuda_ones(Array&) { fail("cuda (ones)"); }
void cuda_randint(uint32_t, uint32_t, int32_t, int32_t, Array&) { fail("cuda (randint)"); }
void cuda_gather(const Array&, const Array&, Array&) { fail("cuda (gather)"); }
void cuda_dirichlet_sample(uint32_t, uint32_t, const std::vector<size_t>&,
                           const Array&, Array&) { fail("cuda (dirichlet)"); }
void cuda_categorical_sample(uint32_t, uint32_t, const std::vector<size_t>&,
                             const Array&, Array&) { fail("cuda (categorical)"); }
#endif

}  // namespace minml
