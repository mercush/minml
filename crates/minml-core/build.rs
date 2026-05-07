// Build script for the optional CUDA shim.
//
// Only runs nvcc when the `cuda` feature is enabled. The shim is a single
// .cu file that exposes an extern "C" ABI of opaque-handle, error-code
// functions; the Rust side declares those externs in src/cuda/mod.rs.
fn main() {
    println!("cargo:rerun-if-changed=cuda/kernels.cu");
    println!("cargo:rerun-if-changed=cuda/kernels.h");

    if std::env::var("CARGO_FEATURE_CUDA").is_ok() {
        cc::Build::new()
            .cuda(true)
            .flag("-cudart=static")
            .file("cuda/kernels.cu")
            .compile("minml_cuda_kernels");
        println!("cargo:rustc-link-lib=cudart");
    }
}
