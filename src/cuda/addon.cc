// N-API bridge: exposes the kernels.cu functions + the jit's NVRTC
// pipeline to JS. Device buffers are wrapped as
// Napi::External<minml_cuda_buf>; compiled jit kernels are wrapped as
// Napi::External<FusedKernel>. Each External's finalizer cleans up the
// underlying CUDA resource.

#include <napi.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kernels.h"

static void buffer_finalizer(Napi::Env /*env*/, minml_cuda_buf* buf) {
    if (buf) minml_cuda_free(buf);
}

static minml_cuda_buf* unwrap(const Napi::Value& v) {
    return v.As<Napi::External<minml_cuda_buf>>().Data();
}

static Napi::Value Alloc(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    size_t bytes = info[0].As<Napi::Number>().Int64Value();
    minml_cuda_buf* h = minml_cuda_alloc(bytes);
    if (!h) {
        Napi::Error::New(env, "cudaMalloc failed").ThrowAsJavaScriptException();
        return env.Null();
    }
    return Napi::External<minml_cuda_buf>::New(env, h, buffer_finalizer);
}

static Napi::Value H2D(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    minml_cuda_buf* dst = unwrap(info[0]);
    Napi::Uint8Array src = info[1].As<Napi::Uint8Array>();
    if (minml_cuda_h2d(dst, src.Data(), src.ByteLength()) != 0) {
        Napi::Error::New(env, "cuda h2d failed").ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value D2H(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    minml_cuda_buf* src = unwrap(info[0]);
    Napi::Uint8Array dst = info[1].As<Napi::Uint8Array>();
    if (minml_cuda_d2h(dst.Data(), src, dst.ByteLength()) != 0) {
        Napi::Error::New(env, "cuda d2h failed").ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value Add(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    minml_cuda_buf* a = unwrap(info[0]);
    minml_cuda_buf* b = unwrap(info[1]);
    minml_cuda_buf* out = unwrap(info[2]);
    size_t n = info[3].As<Napi::Number>().Int64Value();
    if (minml_cuda_add(a, b, out, n) != 0) {
        Napi::Error::New(env, "cuda add failed").ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value Mul(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    minml_cuda_buf* a = unwrap(info[0]);
    minml_cuda_buf* b = unwrap(info[1]);
    minml_cuda_buf* out = unwrap(info[2]);
    size_t n = info[3].As<Napi::Number>().Int64Value();
    if (minml_cuda_mul(a, b, out, n) != 0) {
        Napi::Error::New(env, "cuda mul failed").ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value Dot(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    minml_cuda_buf* a = unwrap(info[0]);
    minml_cuda_buf* b = unwrap(info[1]);
    minml_cuda_buf* out = unwrap(info[2]);
    size_t n = info[3].As<Napi::Number>().Int64Value();
    if (minml_cuda_dot(a, b, out, n) != 0) {
        Napi::Error::New(env, "cuda dot failed").ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

// ---- jit: NVRTC compile + driver-API launch ----

struct FusedKernel {
    CUmodule   module;
    CUfunction func;
};

// Source-string -> FusedKernel cache. Owned by the addon; FusedKernels
// are leaked at process exit (CUDA already tears them down). The map
// keeps live pointers; the JS-side External wraps them but we don't
// finalize on GC because the cache may still hand the same kernel out
// again.
static std::unordered_map<std::string, FusedKernel*> g_kernel_cache;
static bool g_driver_inited = false;

static void ensure_driver_inited(Napi::Env env) {
    if (g_driver_inited) return;
    // Prime the runtime API's primary context so the driver API sees it.
    if (cudaSetDevice(0) != cudaSuccess) {
        Napi::Error::New(env, "cudaSetDevice(0) failed").ThrowAsJavaScriptException();
        return;
    }
    if (cuInit(0) != CUDA_SUCCESS) {
        Napi::Error::New(env, "cuInit(0) failed").ThrowAsJavaScriptException();
        return;
    }
    // Force a runtime-API call so the primary context is created and
    // current on this thread before we ask the driver API for it.
    void* dummy = nullptr;
    if (cudaMalloc(&dummy, 1) == cudaSuccess) cudaFree(dummy);
    g_driver_inited = true;
}

static FusedKernel* compile_or_lookup(Napi::Env env, const std::string& source) {
    auto it = g_kernel_cache.find(source);
    if (it != g_kernel_cache.end()) return it->second;

    nvrtcProgram prog;
    if (nvrtcCreateProgram(&prog, source.c_str(), "fused.cu", 0, nullptr, nullptr)
        != NVRTC_SUCCESS) {
        Napi::Error::New(env, "nvrtcCreateProgram failed").ThrowAsJavaScriptException();
        return nullptr;
    }
    nvrtcResult cres = nvrtcCompileProgram(prog, 0, nullptr);
    if (cres != NVRTC_SUCCESS) {
        size_t logsz = 0;
        nvrtcGetProgramLogSize(prog, &logsz);
        std::string log(logsz, '\0');
        if (logsz) nvrtcGetProgramLog(prog, &log[0]);
        nvrtcDestroyProgram(&prog);
        Napi::Error::New(env, std::string("NVRTC compile failed: ") + log)
            .ThrowAsJavaScriptException();
        return nullptr;
    }
    size_t ptxsz = 0;
    nvrtcGetPTXSize(prog, &ptxsz);
    std::string ptx(ptxsz, '\0');
    nvrtcGetPTX(prog, &ptx[0]);
    nvrtcDestroyProgram(&prog);

    FusedKernel* k = new FusedKernel{};
    if (cuModuleLoadData(&k->module, ptx.data()) != CUDA_SUCCESS) {
        delete k;
        Napi::Error::New(env, "cuModuleLoadData failed").ThrowAsJavaScriptException();
        return nullptr;
    }
    if (cuModuleGetFunction(&k->func, k->module, "fused") != CUDA_SUCCESS) {
        cuModuleUnload(k->module);
        delete k;
        Napi::Error::New(env, "cuModuleGetFunction(\"fused\") failed").ThrowAsJavaScriptException();
        return nullptr;
    }
    g_kernel_cache.emplace(source, k);
    return k;
}

static Napi::Value Compile(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    ensure_driver_inited(env);
    if (env.IsExceptionPending()) return env.Null();
    std::string source = info[0].As<Napi::String>().Utf8Value();
    FusedKernel* k = compile_or_lookup(env, source);
    if (!k) return env.Null();
    // Cache holds ownership; External is just a typed pointer.
    return Napi::External<FusedKernel>::New(env, k);
}

// Pull the device pointers from the JS handles array (each entry is an
// External<minml_cuda_buf>). The output buffer is the last entry.
static std::vector<void*> collect_device_ptrs(const Napi::Array& handles) {
    std::vector<void*> ptrs;
    uint32_t n = handles.Length();
    ptrs.reserve(n);
    for (uint32_t i = 0; i < n; i++) {
        Napi::Value v = handles[i];
        minml_cuda_buf* buf = v.As<Napi::External<minml_cuda_buf>>().Data();
        ptrs.push_back(buf->device_ptr);
    }
    return ptrs;
}

static Napi::Value LaunchElem(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    FusedKernel* k = info[0].As<Napi::External<FusedKernel>>().Data();
    Napi::Array handles = info[1].As<Napi::Array>();
    size_t n = info[2].As<Napi::Number>().Int64Value();

    std::vector<void*> dptrs = collect_device_ptrs(handles);

    // Build a void*[] of pointers to each arg location:
    //   &dptr0, &dptr1, ..., &dptr_out, &n
    std::vector<void*> args;
    args.reserve(dptrs.size() + 1);
    for (auto& p : dptrs) args.push_back(&p);
    args.push_back(&n);

    int block = 256;
    int grid  = static_cast<int>((n + block - 1) / block);
    if (cuLaunchKernel(k->func, grid, 1, 1, block, 1, 1, 0, nullptr,
                       args.data(), nullptr) != CUDA_SUCCESS) {
        Napi::Error::New(env, "cuLaunchKernel (fused elementwise) failed")
            .ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value LaunchReduce(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    FusedKernel* k = info[0].As<Napi::External<FusedKernel>>().Data();
    Napi::Array handles = info[1].As<Napi::Array>();
    size_t n = info[2].As<Napi::Number>().Int64Value();

    std::vector<void*> dptrs = collect_device_ptrs(handles);
    if (dptrs.empty()) {
        Napi::Error::New(env, "launch_reduce: no handles").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    // Reduce kernel does atomicAdd into out[0]; zero it first.
    if (cudaMemset(dptrs.back(), 0, sizeof(float)) != cudaSuccess) {
        Napi::Error::New(env, "cudaMemset(out, 0) failed").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    std::vector<void*> args;
    args.reserve(dptrs.size() + 1);
    for (auto& p : dptrs) args.push_back(&p);
    args.push_back(&n);

    int block = 256;
    int grid  = 1;
    if (cuLaunchKernel(k->func, grid, 1, 1, block, 1, 1, 0, nullptr,
                       args.data(), nullptr) != CUDA_SUCCESS) {
        Napi::Error::New(env, "cuLaunchKernel (fused reduce) failed")
            .ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Object Init(Napi::Env env, Napi::Object exports) {
    exports.Set("alloc",         Napi::Function::New(env, Alloc));
    exports.Set("h2d",           Napi::Function::New(env, H2D));
    exports.Set("d2h",           Napi::Function::New(env, D2H));
    exports.Set("add",           Napi::Function::New(env, Add));
    exports.Set("mul",           Napi::Function::New(env, Mul));
    exports.Set("dot",           Napi::Function::New(env, Dot));
    exports.Set("compile",       Napi::Function::New(env, Compile));
    exports.Set("launch_elem",   Napi::Function::New(env, LaunchElem));
    exports.Set("launch_reduce", Napi::Function::New(env, LaunchReduce));
    return exports;
}

NODE_API_MODULE(minml_cuda, Init)
