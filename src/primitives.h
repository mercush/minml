// src/primitives.h  (internal)
//
// A Primitive is an op-specific node attached to a lazy Array. eval() in
// Array.cpp calls Primitive::eval(inputs, output). Each Primitive dispatches
// on output.device() to a backend-specific free function declared in
// backend.h. Backends not built in link in stub functions that throw.
//
// vmap() is the per-primitive batching rule. The default impl throws so
// unsupported ops surface a clear error rather than silently miscomputing.
// in_axes[i] = batch axis of inputs[i], or -1 if unbatched.
#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "minml/array.h"

namespace minml {

struct VmapResult {
  Array out;
  int out_axis;
};

class Primitive {
 public:
  virtual ~Primitive() = default;
  virtual const char* name() const = 0;
  virtual void eval(const std::vector<Array>& inputs, Array& output) = 0;
  virtual VmapResult vmap(const std::vector<Array>& inputs,
                          const std::vector<int>& in_axes) {
    throw std::runtime_error(std::string("vmap not implemented for ") + name());
  }
};

class AddPrim : public Primitive {
 public:
  const char* name() const override { return "add"; }
  void eval(const std::vector<Array>& inputs, Array& output) override;
  VmapResult vmap(const std::vector<Array>& inputs,
                  const std::vector<int>& in_axes) override;
};

class MulPrim : public Primitive {
 public:
  const char* name() const override { return "mul"; }
  void eval(const std::vector<Array>& inputs, Array& output) override;
  VmapResult vmap(const std::vector<Array>& inputs,
                  const std::vector<int>& in_axes) override;
};

class DotPrim : public Primitive {
 public:
  const char* name() const override { return "dot"; }
  void eval(const std::vector<Array>& inputs, Array& output) override;
};

class OnesPrim : public Primitive {
 public:
  const char* name() const override { return "ones"; }
  void eval(const std::vector<Array>& inputs, Array& output) override;
  VmapResult vmap(const std::vector<Array>& inputs,
                  const std::vector<int>& in_axes) override;
};

// RandIntPrim has no Array inputs; the parameters (key, low, high, shape)
// are stored on the primitive itself.
class RandIntPrim : public Primitive {
 public:
  RandIntPrim(uint32_t k0, uint32_t k1, int32_t low, int32_t high)
      : k0_(k0), k1_(k1), low_(low), high_(high) {}
  const char* name() const override { return "randint"; }
  void eval(const std::vector<Array>& inputs, Array& output) override;
  VmapResult vmap(const std::vector<Array>& inputs,
                  const std::vector<int>& in_axes) override;
  uint32_t k0() const { return k0_; }
  uint32_t k1() const { return k1_; }
  int32_t low() const { return low_; }
  int32_t high() const { return high_; }
 private:
  uint32_t k0_, k1_;
  int32_t low_, high_;
};

// GatherPrim: inputs = (table, indices). Always axis 0.
class GatherPrim : public Primitive {
 public:
  const char* name() const override { return "gather"; }
  void eval(const std::vector<Array>& inputs, Array& output) override;
  VmapResult vmap(const std::vector<Array>& inputs,
                  const std::vector<int>& in_axes) override;
};

// DirichletSamplePrim: inputs = (alpha,). batch_shape and key on the prim.
class DirichletSamplePrim : public Primitive {
 public:
  DirichletSamplePrim(uint32_t k0, uint32_t k1,
                      std::vector<size_t> batch_shape)
      : k0_(k0), k1_(k1), batch_shape_(std::move(batch_shape)) {}
  const char* name() const override { return "dirichlet_sample"; }
  void eval(const std::vector<Array>& inputs, Array& output) override;
  VmapResult vmap(const std::vector<Array>& inputs,
                  const std::vector<int>& in_axes) override;
  uint32_t k0() const { return k0_; }
  uint32_t k1() const { return k1_; }
  const std::vector<size_t>& batch_shape() const { return batch_shape_; }
 private:
  uint32_t k0_, k1_;
  std::vector<size_t> batch_shape_;
};

// CategoricalSamplePrim: inputs = (probs,).
class CategoricalSamplePrim : public Primitive {
 public:
  CategoricalSamplePrim(uint32_t k0, uint32_t k1,
                        std::vector<size_t> batch_shape)
      : k0_(k0), k1_(k1), batch_shape_(std::move(batch_shape)) {}
  const char* name() const override { return "categorical_sample"; }
  void eval(const std::vector<Array>& inputs, Array& output) override;
  VmapResult vmap(const std::vector<Array>& inputs,
                  const std::vector<int>& in_axes) override;
  uint32_t k0() const { return k0_; }
  uint32_t k1() const { return k1_; }
  const std::vector<size_t>& batch_shape() const { return batch_shape_; }
 private:
  uint32_t k0_, k1_;
  std::vector<size_t> batch_shape_;
};

}  // namespace minml
