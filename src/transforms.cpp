// src/transforms.cpp
#include "minml/transforms.h"

#include <cstring>
#include <stdexcept>

#include "backend.h"
#include "minml/buffer.h"

namespace minml {

namespace {

size_t product_after_first(const std::vector<size_t>& shape) {
  size_t n = 1;
  for (size_t i = 1; i < shape.size(); ++i) n *= shape[i];
  return n;
}

}  // namespace

std::vector<Array> slice_axis0(const Array& arr_in) {
  if (arr_in.device() != Device::CPU)
    throw std::runtime_error("slice_axis0: CPU only for now");
  Array arr = arr_in;
  arr.eval();
  if (arr.shape().empty())
    throw std::runtime_error("slice_axis0: cannot slice a scalar");
  size_t N = arr.shape()[0];
  std::vector<size_t> sub_shape(arr.shape().begin() + 1, arr.shape().end());
  size_t per = product_after_first(arr.shape());

  std::vector<Array> out;
  out.reserve(N);
  if (arr.dtype() == DType::Float32) {
    auto data = arr.tolist();
    for (size_t i = 0; i < N; ++i) {
      std::vector<float> chunk(data.begin() + i * per,
                               data.begin() + (i + 1) * per);
      out.emplace_back(std::move(chunk), sub_shape, arr.device());
    }
  } else {
    auto data = arr.tolist_int();
    for (size_t i = 0; i < N; ++i) {
      std::vector<int32_t> chunk(data.begin() + i * per,
                                 data.begin() + (i + 1) * per);
      out.emplace_back(std::move(chunk), sub_shape, arr.device());
    }
  }
  return out;
}

Array stack(const std::vector<Array>& parts) {
  if (parts.empty()) throw std::runtime_error("stack: empty input");
  const std::vector<size_t>& base_shape = parts.front().shape();
  Device dev = parts.front().device();
  DType dt = parts.front().dtype();
  for (const auto& p : parts) {
    if (p.shape() != base_shape) throw std::runtime_error("stack: shape mismatch");
    if (p.device() != dev)       throw std::runtime_error("stack: device mismatch");
    if (p.dtype() != dt)         throw std::runtime_error("stack: dtype mismatch");
  }

  std::vector<size_t> out_shape;
  out_shape.reserve(base_shape.size() + 1);
  out_shape.push_back(parts.size());
  for (size_t d : base_shape) out_shape.push_back(d);

  if (dev != Device::CPU)
    throw std::runtime_error("stack: CPU only for now");

  size_t per = parts.front().size();
  size_t bytes_per = per * dtype_bytes(dt);

  Array out = (dt == DType::Float32)
      ? Array(std::vector<float>(per * parts.size()), out_shape, dev)
      : Array(std::vector<int32_t>(per * parts.size()), out_shape, dev);

  unsigned char* dst = (dt == DType::Float32)
      ? reinterpret_cast<unsigned char*>(cpu_data_f32(out))
      : reinterpret_cast<unsigned char*>(cpu_data_i32(out));

  for (size_t i = 0; i < parts.size(); ++i) {
    Array p = parts[i];
    p.eval();
    const unsigned char* src = (dt == DType::Float32)
        ? reinterpret_cast<const unsigned char*>(cpu_data_f32(p))
        : reinterpret_cast<const unsigned char*>(cpu_data_i32(p));
    std::memcpy(dst + i * bytes_per, src, bytes_per);
  }
  return out;
}

std::vector<Array> vmap_apply(size_t N,
                              const std::vector<Array>& args,
                              const std::vector<int>& in_axes,
                              const VmapCallable& f) {
  if (args.size() != in_axes.size())
    throw std::runtime_error("vmap_apply: args/in_axes size mismatch");

  // Pre-slice batched Array inputs once.
  std::vector<std::vector<Array>> sliced(args.size());
  for (size_t i = 0; i < args.size(); ++i) {
    if (in_axes[i] < 0) continue;
    if (in_axes[i] != 0)
      throw std::runtime_error("vmap_apply: only axis 0 supported");
    if (args[i].shape().empty())
      throw std::runtime_error("vmap_apply: cannot batch over a scalar");
    if (args[i].shape()[0] != N)
      throw std::runtime_error("vmap_apply: batched dims disagree");
    sliced[i] = slice_axis0(args[i]);
  }

  // Loop, calling f per batch element. all_leaves[iter][leaf_idx] = Array.
  std::vector<std::vector<Array>> all_leaves;
  all_leaves.reserve(N);
  for (size_t b = 0; b < N; ++b) {
    std::vector<Array> per_iter;
    per_iter.reserve(args.size());
    for (size_t i = 0; i < args.size(); ++i) {
      per_iter.push_back((in_axes[i] >= 0) ? sliced[i][b] : args[i]);
    }
    all_leaves.push_back(f(b, per_iter));
  }
  if (all_leaves.empty())
    throw std::runtime_error("vmap_apply: N=0");

  // Stack each leaf position separately.
  size_t n_leaves = all_leaves.front().size();
  for (const auto& v : all_leaves) {
    if (v.size() != n_leaves)
      throw std::runtime_error("vmap_apply: leaf count varies across iterations");
  }
  std::vector<Array> stacked;
  stacked.reserve(n_leaves);
  for (size_t l = 0; l < n_leaves; ++l) {
    std::vector<Array> parts;
    parts.reserve(N);
    for (size_t b = 0; b < N; ++b) parts.push_back(all_leaves[b][l]);
    stacked.push_back(stack(parts));
  }
  return stacked;
}

}  // namespace minml
