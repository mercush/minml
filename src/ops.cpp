// src/ops.cpp
//
// Op entry points and per-primitive eval dispatchers. Each op builds a
// lazy Array; eval() walks the DAG and runs per-backend kernels.
//
// vmap is handled via JS-loop dispatch in the embind binding shim — the
// per-primitive Primitive::vmap rules here are aspirational (they let
// future versions push batches into kernels) and currently throw. The
// notebook works because the binding loop calls f once per batch element.
#include "minml/ops.h"

#include <memory>
#include <stdexcept>

#include "backend.h"
#include "primitives.h"

namespace minml {

namespace {

void check_same_shape(const Array& a, const Array& b) {
  if (a.shape() != b.shape())
    throw std::runtime_error("shape mismatch");
  if (a.device() != b.device())
    throw std::runtime_error("device mismatch");
}

}  // namespace

// ---- Elementwise add ------------------------------------------------------

Array add(const Array& a, const Array& b) {
  check_same_shape(a, b);
  return Array(a.shape(), a.dtype(), a.device(), std::make_shared<AddPrim>(),
               std::vector<Array>{a, b});
}

void AddPrim::eval(const std::vector<Array>& inputs, Array& out) {
  switch (out.device()) {
    case Device::CPU: cpu_add(inputs[0], inputs[1], out); return;
    case Device::CUDA: cuda_add(inputs[0], inputs[1], out); return;
    case Device::WebGPU: webgpu_add(inputs[0], inputs[1], out); return;
  }
}

VmapResult AddPrim::vmap(const std::vector<Array>&, const std::vector<int>&) {
  // Elementwise add over matching batch axes is just add on flat memory.
  // Future: implement here when in-kernel batching is wired up.
  throw std::runtime_error("AddPrim::vmap: use binding-level vmap loop");
}

// ---- Elementwise mul ------------------------------------------------------

Array mul(const Array& a, const Array& b) {
  check_same_shape(a, b);
  return Array(a.shape(), a.dtype(), a.device(), std::make_shared<MulPrim>(),
               std::vector<Array>{a, b});
}

void MulPrim::eval(const std::vector<Array>& inputs, Array& out) {
  switch (out.device()) {
    case Device::CPU: cpu_mul(inputs[0], inputs[1], out); return;
    case Device::CUDA: cuda_mul(inputs[0], inputs[1], out); return;
    case Device::WebGPU: webgpu_mul(inputs[0], inputs[1], out); return;
  }
}

VmapResult MulPrim::vmap(const std::vector<Array>&, const std::vector<int>&) {
  throw std::runtime_error("MulPrim::vmap: use binding-level vmap loop");
}

// ---- Reduction dot --------------------------------------------------------

Array dot(const Array& a, const Array& b) {
  check_same_shape(a, b);
  if (a.shape().size() != 1)
    throw std::runtime_error("dot requires 1-D inputs");
  return Array(std::vector<size_t>{1}, a.dtype(), a.device(),
               std::make_shared<DotPrim>(), std::vector<Array>{a, b});
}

void DotPrim::eval(const std::vector<Array>& inputs, Array& out) {
  switch (out.device()) {
    case Device::CPU: cpu_dot(inputs[0], inputs[1], out); return;
    case Device::CUDA: cuda_dot(inputs[0], inputs[1], out); return;
    case Device::WebGPU: webgpu_dot(inputs[0], inputs[1], out); return;
  }
}

// ---- Ones (constant) ------------------------------------------------------

Array ones(std::vector<size_t> shape, DType dtype, Device device) {
  return Array(std::move(shape), dtype, device, std::make_shared<OnesPrim>(),
               std::vector<Array>{});
}

void OnesPrim::eval(const std::vector<Array>&, Array& out) {
  switch (out.device()) {
    case Device::CPU: cpu_ones(out); return;
    case Device::CUDA: cuda_ones(out); return;
    case Device::WebGPU: webgpu_ones(out); return;
  }
}

VmapResult OnesPrim::vmap(const std::vector<Array>&, const std::vector<int>&) {
  throw std::runtime_error("OnesPrim::vmap: ones() has no batched input");
}

// ---- RandInt --------------------------------------------------------------

Array randint(uint32_t k0, uint32_t k1, int32_t low, int32_t high,
              std::vector<size_t> shape, Device device) {
  return Array(std::move(shape), DType::Int32, device,
               std::make_shared<RandIntPrim>(k0, k1, low, high),
               std::vector<Array>{});
}

void RandIntPrim::eval(const std::vector<Array>&, Array& out) {
  switch (out.device()) {
    case Device::CPU: cpu_randint(k0_, k1_, low_, high_, out); return;
    case Device::CUDA: cuda_randint(k0_, k1_, low_, high_, out); return;
    case Device::WebGPU: webgpu_randint(k0_, k1_, low_, high_, out); return;
  }
}

VmapResult RandIntPrim::vmap(const std::vector<Array>&,
                             const std::vector<int>&) {
  throw std::runtime_error("RandIntPrim::vmap: randint has no Array inputs");
}

// ---- Gather ---------------------------------------------------------------

Array gather(const Array& table, const Array& indices) {
  if (indices.dtype() != DType::Int32)
    throw std::runtime_error("gather: indices must be Int32");
  if (table.shape().empty())
    throw std::runtime_error("gather: table must have rank >= 1");
  // out shape = indices.shape ++ table.shape[1:]
  std::vector<size_t> out_shape = indices.shape();
  for (size_t i = 1; i < table.shape().size(); ++i)
    out_shape.push_back(table.shape()[i]);
  return Array(std::move(out_shape), table.dtype(), table.device(),
               std::make_shared<GatherPrim>(),
               std::vector<Array>{table, indices});
}

void GatherPrim::eval(const std::vector<Array>& inputs, Array& out) {
  switch (out.device()) {
    case Device::CPU: cpu_gather(inputs[0], inputs[1], out); return;
    case Device::CUDA: cuda_gather(inputs[0], inputs[1], out); return;
    case Device::WebGPU: webgpu_gather(inputs[0], inputs[1], out); return;
  }
}

VmapResult GatherPrim::vmap(const std::vector<Array>&,
                            const std::vector<int>&) {
  throw std::runtime_error("GatherPrim::vmap: use binding-level vmap loop");
}

// ---- Distribution sample primitives --------------------------------------

Array dirichlet_sample(uint32_t k0, uint32_t k1, const Array& alpha,
                       std::vector<size_t> batch_shape) {
  if (alpha.shape().size() != 1)
    throw std::runtime_error("dirichlet_sample: alpha must be 1-D");
  std::vector<size_t> out_shape = batch_shape;
  out_shape.push_back(alpha.shape()[0]);
  return Array(std::move(out_shape), DType::Float32, alpha.device(),
               std::make_shared<DirichletSamplePrim>(k0, k1, std::move(batch_shape)),
               std::vector<Array>{alpha});
}

void DirichletSamplePrim::eval(const std::vector<Array>& inputs, Array& out) {
  switch (out.device()) {
    case Device::CPU:    cpu_dirichlet_sample(k0_, k1_, batch_shape_, inputs[0], out); return;
    case Device::CUDA:   cuda_dirichlet_sample(k0_, k1_, batch_shape_, inputs[0], out); return;
    case Device::WebGPU: webgpu_dirichlet_sample(k0_, k1_, batch_shape_, inputs[0], out); return;
  }
}

VmapResult DirichletSamplePrim::vmap(const std::vector<Array>&,
                                     const std::vector<int>&) {
  throw std::runtime_error("DirichletSamplePrim::vmap: use binding-level vmap loop");
}

Array categorical_sample(uint32_t k0, uint32_t k1, const Array& probs,
                         std::vector<size_t> batch_shape) {
  if (probs.shape().size() != 1)
    throw std::runtime_error("categorical_sample: probs must be 1-D (use vmap to batch)");
  return Array(batch_shape, DType::Int32, probs.device(),
               std::make_shared<CategoricalSamplePrim>(k0, k1, std::move(batch_shape)),
               std::vector<Array>{probs});
}

void CategoricalSamplePrim::eval(const std::vector<Array>& inputs, Array& out) {
  switch (out.device()) {
    case Device::CPU:    cpu_categorical_sample(k0_, k1_, batch_shape_, inputs[0], out); return;
    case Device::CUDA:   cuda_categorical_sample(k0_, k1_, batch_shape_, inputs[0], out); return;
    case Device::WebGPU: webgpu_categorical_sample(k0_, k1_, batch_shape_, inputs[0], out); return;
  }
}

VmapResult CategoricalSamplePrim::vmap(const std::vector<Array>&,
                                       const std::vector<int>&) {
  throw std::runtime_error("CategoricalSamplePrim::vmap: use binding-level vmap loop");
}

}  // namespace minml
