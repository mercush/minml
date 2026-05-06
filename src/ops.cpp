// src/ops.cpp
#include "minml/ops.h"

#include <memory>
#include <stdexcept>

#include "backend.h"
#include "primitives.h"

namespace minml {

namespace {
void check_same(const Array& a, const Array& b) {
  if (a.size() != b.size())
    throw std::runtime_error("size mismatch");
  if (a.device() != b.device())
    throw std::runtime_error("device mismatch");
}
}  // namespace

Array add(const Array& a, const Array& b) {
  check_same(a, b);
  return Array(a.size(), a.device(), std::make_shared<AddPrim>(),
               std::vector<Array>{a, b});
}

Array dot(const Array& a, const Array& b) {
  check_same(a, b);
  return Array(/*size=*/1, a.device(), std::make_shared<DotPrim>(),
               std::vector<Array>{a, b});
}

// Primitive dispatch: same op, picks backend by device.
void AddPrim::eval(const std::vector<Array>& inputs, Array& out) {
  switch (out.device()) {
    case Device::CPU: cpu_add(inputs[0], inputs[1], out); return;
    case Device::CUDA: cuda_add(inputs[0], inputs[1], out); return;
    case Device::WebGPU: webgpu_add(inputs[0], inputs[1], out); return;
  }
}

void DotPrim::eval(const std::vector<Array>& inputs, Array& out) {
  switch (out.device()) {
    case Device::CPU: cpu_dot(inputs[0], inputs[1], out); return;
    case Device::CUDA: cuda_dot(inputs[0], inputs[1], out); return;
    case Device::WebGPU: webgpu_dot(inputs[0], inputs[1], out); return;
  }
}

}  // namespace minml
