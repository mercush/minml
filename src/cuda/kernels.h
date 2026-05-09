// extern "C" interface between addon.cc and kernels.cu. The struct is
// opaque to the JS side (passed via Napi::External<minml_cuda_buf>).

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct minml_cuda_buf {
    void*  device_ptr;
    size_t bytes;
};

minml_cuda_buf* minml_cuda_alloc(size_t bytes);
void            minml_cuda_free(minml_cuda_buf* buf);
int             minml_cuda_h2d(minml_cuda_buf* dst, const void* src, size_t bytes);
int             minml_cuda_d2h(void* dst, const minml_cuda_buf* src, size_t bytes);
int             minml_cuda_add(const minml_cuda_buf* a, const minml_cuda_buf* b, minml_cuda_buf* out, size_t n);
int             minml_cuda_mul(const minml_cuda_buf* a, const minml_cuda_buf* b, minml_cuda_buf* out, size_t n);
int             minml_cuda_dot(const minml_cuda_buf* a, const minml_cuda_buf* b, minml_cuda_buf* out, size_t n);

#ifdef __cplusplus
}
#endif
