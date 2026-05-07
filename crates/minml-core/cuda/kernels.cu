// minml CUDA shim — single .cu file, called from Rust via extern "C".
//
// Returns int error codes; the Rust side translates them into MinmlError.
// Opaque handle ownership: the shim allocates `minml_cuda_buf*` on the
// host heap (carrying a `void* device_ptr`); Rust calls minml_cuda_free
// when the corresponding CudaBuffer is dropped.
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

extern "C" {

struct minml_cuda_buf {
    void*  device_ptr;
    size_t bytes;
};

minml_cuda_buf* minml_cuda_alloc(size_t bytes) {
    minml_cuda_buf* h = (minml_cuda_buf*)malloc(sizeof(minml_cuda_buf));
    if (!h) return nullptr;
    if (cudaMalloc(&h->device_ptr, bytes) != cudaSuccess) {
        free(h);
        return nullptr;
    }
    h->bytes = bytes;
    return h;
}

void minml_cuda_free(minml_cuda_buf* buf) {
    if (!buf) return;
    if (buf->device_ptr) cudaFree(buf->device_ptr);
    free(buf);
}

int minml_cuda_h2d(minml_cuda_buf* dst, const void* src, size_t bytes) {
    return cudaMemcpy(dst->device_ptr, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess ? 0 : 1;
}

int minml_cuda_d2h(void* dst, const minml_cuda_buf* src, size_t bytes) {
    return cudaMemcpy(dst, src->device_ptr, bytes, cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : 1;
}

__global__ void add_kernel(const float* a, const float* b, float* out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void mul_kernel(const float* a, const float* b, float* out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

__global__ void dot_kernel(const float* a, const float* b, float* out, size_t n) {
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

int minml_cuda_add(const minml_cuda_buf* a, const minml_cuda_buf* b,
                   minml_cuda_buf* out, size_t n) {
    int block = 256;
    int grid  = static_cast<int>((n + block - 1) / block);
    add_kernel<<<grid, block>>>((const float*)a->device_ptr,
                                (const float*)b->device_ptr,
                                (float*)out->device_ptr, n);
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

int minml_cuda_mul(const minml_cuda_buf* a, const minml_cuda_buf* b,
                   minml_cuda_buf* out, size_t n) {
    int block = 256;
    int grid  = static_cast<int>((n + block - 1) / block);
    mul_kernel<<<grid, block>>>((const float*)a->device_ptr,
                                (const float*)b->device_ptr,
                                (float*)out->device_ptr, n);
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

int minml_cuda_dot(const minml_cuda_buf* a, const minml_cuda_buf* b,
                   minml_cuda_buf* out, size_t n) {
    dot_kernel<<<1, 256>>>((const float*)a->device_ptr,
                           (const float*)b->device_ptr,
                           (float*)out->device_ptr, n);
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

}  // extern "C"
