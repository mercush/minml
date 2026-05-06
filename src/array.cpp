// src/array.cpp
#include "minml/array.h"

#include <stdexcept>

#include "backend.h"
#include "primitives.h"

namespace minml {

namespace {

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

Array::Array(std::vector<float> data, Device device)
    : size_(data.size()), device_(device) {
  data_ = allocate(size_ * sizeof(float), device_);
  copy_h2d(*data_, data.data(), size_, device_);
}

Array::Array(size_t size, Device device, std::shared_ptr<Primitive> prim,
             std::vector<Array> inputs)
    : size_(size),
      device_(device),
      primitive_(std::move(prim)),
      inputs_(std::move(inputs)) {}

void Array::set_data(std::shared_ptr<Buffer> b) {
  data_ = std::move(b);
  primitive_.reset();
  inputs_.clear();
}

void Array::eval() {
  if (evaluated()) return;
  // Post-order: evaluate inputs first.
  for (auto& in : inputs_) in.eval();
  // Allocate the output buffer, then run the primitive.
  data_ = allocate(size_ * sizeof(float), device_);
  primitive_->eval(inputs_, *this);
  // Drop the lazy graph so dependents can be GC'd.
  primitive_.reset();
  inputs_.clear();
}

std::vector<float> Array::tolist() {
  eval();
  std::vector<float> out(size_);
  copy_d2h(*data_, out.data(), size_, device_);
  return out;
}

float Array::item() {
  if (size_ != 1) throw std::runtime_error("item() requires size==1");
  return tolist()[0];
}

}  // namespace minml
