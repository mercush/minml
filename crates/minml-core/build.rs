// Build script for the optional CUDA backend.
//
// The CUDA backend now goes through NVlabs/cuda-oxide (`cuda-core`) at
// runtime; kernels and JIT-fused expressions are compiled CUDA-C → PTX via
// NVRTC at first use, then loaded with `CudaContext::load_module_from_ptx_src`.
// Nothing is compiled here — we only declare the runtime link against
// `nvrtc` (the CUDA toolkit ships it next to `cudart`/the driver) when the
// feature is enabled.
fn main() {
    if std::env::var("CARGO_FEATURE_CUDA").is_ok() {
        // cuda-core links the CUDA driver itself; we add NVRTC for runtime
        // kernel compilation (used by both the static ops and transforms::jit).
        println!("cargo:rustc-link-lib=nvrtc");
    }
}
