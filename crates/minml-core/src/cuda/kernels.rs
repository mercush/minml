// CUDA-C source for the static ops. NVRTC-compiled to PTX at first use, then
// loaded once and shared across all evaluations through a Mutex<HashMap> in
// the parent module. Kernel signatures take raw `float*` / `size_t` rather
// than typed slices so the same module can serve any contiguous f32 buffer.

pub const KERNELS_SRC: &str = r#"
extern "C" __global__
void minml_add(const float* __restrict__ a,
               const float* __restrict__ b,
               float* __restrict__ out,
               unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

extern "C" __global__
void minml_mul(const float* __restrict__ a,
               const float* __restrict__ b,
               float* __restrict__ out,
               unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

extern "C" __global__
void minml_dot(const float* __restrict__ a,
               const float* __restrict__ b,
               float* __restrict__ out,
               unsigned int n) {
    __shared__ float scratch[256];
    int tid = threadIdx.x;
    float local = 0.0f;
    for (unsigned int i = tid; i < n; i += blockDim.x) {
        local += a[i] * b[i];
    }
    scratch[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[0] = scratch[0];
}
"#;
