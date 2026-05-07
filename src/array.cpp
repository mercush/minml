// src/array.cpp
#include "minml/array.h"

#include <cstdint>
#include <cstring>
#include <numeric>
#include <stdexcept>

#include "backend.h"
#include "primitives.h"

namespace minml {

namespace {

size_t product(const std::vector<size_t>& shape) {
  size_t n = 1;
  for (size_t d : shape) n *= d;
  return n;
}

std::shared_ptr<Buffer> allocate(size_t bytes, Device d) {
  switch (d) {
    case Device::CPU: return cpu_allocate(bytes);
    case Device::CUDA: return cuda_allocate(bytes);
    case Device::WebGPU: return webgpu_allocate(bytes);
  }
  throw std::runtime_error("unknown device");
}

void copy_h2d(Buffer& dst, const float* src, size_t n, Device d) {
  switch (d) {
    case Device::CPU: cpu_copy_host_to_device(dst, src, n); return;
    case Device::CUDA: cuda_copy_host_to_device(dst, src, n); return;
    case Device::WebGPU: webgpu_copy_host_to_device(dst, src, n); return;
  }
  throw std::runtime_error("unknown device");
}

void copy_d2h(const Buffer& src, float* dst, size_t n, Device d) {
  switch (d) {
    case Device::CPU: cpu_copy_device_to_host(src, dst, n); return;
    case Device::CUDA: cuda_copy_device_to_host(src, dst, n); return;
    case Device::WebGPU: webgpu_copy_device_to_host(src, dst, n); return;
  }
  throw std::runtime_error("unknown device");
}

}  // namespace

Array::Array(std::vector<float> data, std::vector<size_t> shape, Device device)
    : shape_(std::move(shape)),
      size_(product(shape_)),
      device_(device),
      dtype_(DType::Float32) {
  if (data.size() != size_)
    throw std::runtime_error("data size != product(shape)");
  data_ = allocate(size_ * dtype_bytes(dtype_), device_);
  copy_h2d(*data_, data.data(), size_, device_);
}

Array::Array(std::vector<int32_t> data, std::vector<size_t> shape, Device device)
    : shape_(std::move(shape)),
      size_(product(shape_)),
      device_(device),
      dtype_(DType::Int32) {
  if (data.size() != size_)
    throw std::runtime_error("data size != product(shape)");
  data_ = allocate(size_ * dtype_bytes(dtype_), device_);
  // copy_h2d takes float*, but the underlying memcpy on CPU treats it as
  // bytes. For non-CPU backends int32 isn't supported yet (stubs); fine.
  copy_h2d(*data_, reinterpret_cast<const float*>(data.data()), size_, device_);
}

Array::Array(std::vector<float> data, Device device)
    : shape_({data.size()}),
      size_(data.size()),
      device_(device),
      dtype_(DType::Float32) {
  data_ = allocate(size_ * dtype_bytes(dtype_), device_);
  copy_h2d(*data_, data.data(), size_, device_);
}

Array::Array(std::vector<size_t> shape, DType dtype, Device device,
             std::shared_ptr<Primitive> prim, std::vector<Array> inputs)
    : shape_(std::move(shape)),
      size_(product(shape_)),
      device_(device),
      dtype_(dtype),
      primitive_(std::move(prim)),
      inputs_(std::move(inputs)) {}

Array Array::with_batch_axis(int axis) const {
  Array out = *this;
  out.batch_axis_ = axis;
  return out;
}

Array Array::strip_batch_axis() const {
  Array out = *this;
  out.batch_axis_.reset();
  return out;
}

void Array::set_data(std::shared_ptr<Buffer> b) {
  data_ = std::move(b);
  primitive_.reset();
  inputs_.clear();
}

void Array::eval() {
  if (evaluated()) return;
  for (auto& in : inputs_) in.eval();
  data_ = allocate(size_ * dtype_bytes(dtype_), device_);
  primitive_->eval(inputs_, *this);
  primitive_.reset();
  inputs_.clear();
}

std::vector<float> Array::tolist() {
  eval();
  std::vector<float> out(size_);
  copy_d2h(*data_, out.data(), size_, device_);
  return out;
}

std::vector<int32_t> Array::tolist_int() {
  eval();
  std::vector<int32_t> out(size_);
  copy_d2h(*data_, reinterpret_cast<float*>(out.data()), size_, device_);
  return out;
}

float Array::item() {
  if (size_ != 1) throw std::runtime_error("item() requires size==1");
  return tolist()[0];
}

}  // namespace minml
