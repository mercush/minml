// N-API bridge: exposes the kernels.cu functions to JS. Pointers are
// wrapped as Napi::External<minml_cuda_buf>; the External finalizer
// (registered in alloc) calls minml_cuda_free when JS GC drops the
// reference, so JS owns the buffer lifetime.

#include <napi.h>

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

static Napi::Object Init(Napi::Env env, Napi::Object exports) {
    exports.Set("alloc", Napi::Function::New(env, Alloc));
    exports.Set("h2d",   Napi::Function::New(env, H2D));
    exports.Set("d2h",   Napi::Function::New(env, D2H));
    exports.Set("add",   Napi::Function::New(env, Add));
    exports.Set("mul",   Napi::Function::New(env, Mul));
    exports.Set("dot",   Napi::Function::New(env, Dot));
    return exports;
}

NODE_API_MODULE(minml_cuda, Init)
