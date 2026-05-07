// src/backend.h  (internal)
//
// Per-backend functions. Each backend (cpu_backend.cpp, cuda_backend.cu,
// webgpu_backend.cpp) defines its own. Backends compiled out are linked
// from stubs.cpp which throws.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "minml/array.h"
#include "minml/buffer.h"

namespace minml {

// CPU. Internal data accessors so backend code (cpu_backend.cpp,
// cpu_random.cpp) can interpret a CPU Array's buffer as float* / int32*
// without exposing the CpuBuffer subclass.
std::shared_ptr<Buffer> cpu_allocate(size_t bytes);
void cpu_copy_host_to_device(Buffer& dst, const float* src, size_t n);
void cpu_copy_device_to_host(const Buffer& src, float* dst, size_t n);
float* cpu_data_f32(Array& a);
const float* cpu_data_f32(const Array& a);
int32_t* cpu_data_i32(Array& a);
const int32_t* cpu_data_i32(const Array& a);
void cpu_add(const Array& a, const Array& b, Array& out);
void cpu_mul(const Array& a, const Array& b, Array& out);
void cpu_dot(const Array& a, const Array& b, Array& out);
void cpu_ones(Array& out);
void cpu_randint(uint32_t k0, uint32_t k1, int32_t low, int32_t high, Array& out);
void cpu_gather(const Array& table, const Array& indices, Array& out);
void cpu_dirichlet_sample(uint32_t k0, uint32_t k1,
                          const std::vector<size_t>& batch_shape,
                          const Array& alpha, Array& out);
void cpu_categorical_sample(uint32_t k0, uint32_t k1,
                            const std::vector<size_t>& batch_shape,
                            const Array& probs, Array& out);

// CUDA
std::shared_ptr<Buffer> cuda_allocate(size_t bytes);
void cuda_copy_host_to_device(Buffer& dst, const float* src, size_t n);
void cuda_copy_device_to_host(const Buffer& src, float* dst, size_t n);
void cuda_add(const Array& a, const Array& b, Array& out);
void cuda_mul(const Array& a, const Array& b, Array& out);
void cuda_dot(const Array& a, const Array& b, Array& out);
void cuda_ones(Array& out);
void cuda_randint(uint32_t k0, uint32_t k1, int32_t low, int32_t high, Array& out);
void cuda_gather(const Array& table, const Array& indices, Array& out);
void cuda_dirichlet_sample(uint32_t k0, uint32_t k1,
                           const std::vector<size_t>& batch_shape,
                           const Array& alpha, Array& out);
void cuda_categorical_sample(uint32_t k0, uint32_t k1,
                             const std::vector<size_t>& batch_shape,
                             const Array& probs, Array& out);

// WebGPU
std::shared_ptr<Buffer> webgpu_allocate(size_t bytes);
void webgpu_copy_host_to_device(Buffer& dst, const float* src, size_t n);
void webgpu_copy_device_to_host(const Buffer& src, float* dst, size_t n);
void webgpu_add(const Array& a, const Array& b, Array& out);
void webgpu_mul(const Array& a, const Array& b, Array& out);
void webgpu_dot(const Array& a, const Array& b, Array& out);
void webgpu_ones(Array& out);
void webgpu_randint(uint32_t k0, uint32_t k1, int32_t low, int32_t high, Array& out);
void webgpu_gather(const Array& table, const Array& indices, Array& out);
void webgpu_dirichlet_sample(uint32_t k0, uint32_t k1,
                             const std::vector<size_t>& batch_shape,
                             const Array& alpha, Array& out);
void webgpu_categorical_sample(uint32_t k0, uint32_t k1,
                               const std::vector<size_t>& batch_shape,
                               const Array& probs, Array& out);

}  // namespace minml
