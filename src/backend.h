// src/backend.h  (internal)
//
// Per-backend functions. Each backend (cpu_backend.cpp, cuda_backend.cu,
// webgpu_backend.cpp) defines its own. Backends compiled out are linked
// from stubs.cpp which throws.
#pragma once

#include <memory>
#include <vector>

#include "minml/array.h"
#include "minml/buffer.h"

namespace minml {

// CPU
std::shared_ptr<Buffer> cpu_allocate(size_t bytes);
void cpu_copy_host_to_device(Buffer& dst, const float* src, size_t n);
void cpu_copy_device_to_host(const Buffer& src, float* dst, size_t n);
void cpu_add(const Array& a, const Array& b, Array& out);
void cpu_mul(const Array& a, const Array& b, Array& out);
void cpu_dot(const Array& a, const Array& b, Array& out);

// CUDA
std::shared_ptr<Buffer> cuda_allocate(size_t bytes);
void cuda_copy_host_to_device(Buffer& dst, const float* src, size_t n);
void cuda_copy_device_to_host(const Buffer& src, float* dst, size_t n);
void cuda_add(const Array& a, const Array& b, Array& out);
void cuda_mul(const Array& a, const Array& b, Array& out);
void cuda_dot(const Array& a, const Array& b, Array& out);

// WebGPU
std::shared_ptr<Buffer> webgpu_allocate(size_t bytes);
void webgpu_copy_host_to_device(Buffer& dst, const float* src, size_t n);
void webgpu_copy_device_to_host(const Buffer& src, float* dst, size_t n);
void webgpu_add(const Array& a, const Array& b, Array& out);
void webgpu_mul(const Array& a, const Array& b, Array& out);
void webgpu_dot(const Array& a, const Array& b, Array& out);

}  // namespace minml
