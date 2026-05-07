// src/cuda_backend.cu
//
// Compiled by nvcc when MINML_BUILD_CUDA=ON. The host-side dispatch is the
// same shape as cpu_backend.cpp; only the kernels and allocate/copy use
// the CUDA runtime.
#include <cuda_runtime.h>

#include <memory>
#include <stdexcept>

#include "backend.h"
#include "minml/buffer.h"

namespace minml {

namespace {

#define CUDA_CHECK(expr)                                              \
  do {                                                                \
    cudaError_t err = (expr);                                         \
    if (err != cudaSuccess)                                           \
      throw std::runtime_error(std::string("CUDA: ") +                \
                               cudaGetErrorString(err));              \
  } while (0)

struct CudaBuffer : Buffer {
  void* ptr = nullptr;
  ~CudaBuffer() override {
    if (ptr) cudaFree(ptr);
  }
};

CudaBuffer& as_cuda(Buffer& b) { return static_cast<CudaBuffer&>(b); }
const CudaBuffer& as_cuda(const Buffer& b) {
  return static_cast<const CudaBuffer&>(b);
}
float* data(const Array& a) {
  return static_cast<float*>(as_cuda(*a.buffer()).ptr);
}

// ---- Kernels ---------------------------------------------------------------

__global__ void add_kernel(const float* a, const float* b, float* out,
                           size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}

__global__ void mul_kernel(const float* a, const float* b, float* out,
                           size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] * b[i];
}

// One-block reduction: enough for a minimal demo. For longer vectors, split
// into a multi-block first pass + a final reduction.
__global__ void dot_kernel(const float* a, const float* b, float* out,
                           size_t n) {
  __shared__ float scratch[256];
  int tid = threadIdx.x;
  float local = 0.f;
  for (size_t i = tid; i < n; i += blockDim.x) local += a[i] * b[i];
  scratch[tid] = local;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) scratch[tid] += scratch[tid + s];
    __syncthreads();
  }
  if (tid == 0) out[0] = scratch[0];
}

}  // namespace

std::shared_ptr<Buffer> cuda_allocate(size_t bytes) {
  auto b = std::make_shared<CudaBuffer>();
  b->bytes = bytes;
  b->device = Device::CUDA;
  CUDA_CHECK(cudaMalloc(&b->ptr, bytes));
  return b;
}

void cuda_copy_host_to_device(Buffer& dst, const float* src, size_t n) {
  CUDA_CHECK(cudaMemcpy(as_cuda(dst).ptr, src, n * sizeof(float),
                        cudaMemcpyHostToDevice));
}

void cuda_copy_device_to_host(const Buffer& src, float* dst, size_t n) {
  CUDA_CHECK(cudaMemcpy(dst, as_cuda(src).ptr, n * sizeof(float),
                        cudaMemcpyDeviceToHost));
}

void cuda_add(const Array& a, const Array& b, Array& out) {
  size_t n = out.size();
  int block = 256;
  int grid = static_cast<int>((n + block - 1) / block);
  add_kernel<<<grid, block>>>(data(a), data(b), data(out), n);
  CUDA_CHECK(cudaGetLastError());
}

void cuda_mul(const Array& a, const Array& b, Array& out) {
  size_t n = out.size();
  int block = 256;
  int grid = static_cast<int>((n + block - 1) / block);
  mul_kernel<<<grid, block>>>(data(a), data(b), data(out), n);
  CUDA_CHECK(cudaGetLastError());
}

void cuda_dot(const Array& a, const Array& b, Array& out) {
  dot_kernel<<<1, 256>>>(data(a), data(b), data(out), a.size());
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace minml
