// minml CUDA shim — C ABI surface used from Rust via extern "C".
// Real implementation lands in step 7. This header is a placeholder so
// the build.rs cargo:rerun-if-changed doesn't fail when the feature is
// off.
#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct minml_cuda_buf minml_cuda_buf;

minml_cuda_buf* minml_cuda_alloc(size_t bytes);
void            minml_cuda_free(minml_cuda_buf* buf);
int             minml_cuda_h2d(minml_cuda_buf* dst, const void* src, size_t bytes);
int             minml_cuda_d2h(void* dst, const minml_cuda_buf* src, size_t bytes);
int             minml_cuda_add(const minml_cuda_buf* a, const minml_cuda_buf* b,
                               minml_cuda_buf* out, size_t n);
int             minml_cuda_mul(const minml_cuda_buf* a, const minml_cuda_buf* b,
                               minml_cuda_buf* out, size_t n);
int             minml_cuda_dot(const minml_cuda_buf* a, const minml_cuda_buf* b,
                               minml_cuda_buf* out, size_t n);

#ifdef __cplusplus
}
#endif
