// src/primitives.h  (internal)
//
// A Primitive is an op-specific node attached to a lazy Array. eval() in
// Array.cpp calls Primitive::eval(inputs, output). Each Primitive dispatches
// on output.device() to a backend-specific free function declared in
// backend.h. Backends not built in link in stub functions that throw.
#pragma once

#include <memory>
#include <stdexcept>
#include <vector>

#include "minml/array.h"

namespace minml {

class Primitive {
 public:
  virtual ~Primitive() = default;
  virtual const char* name() const = 0;
  virtual void eval(const std::vector<Array>& inputs, Array& output) = 0;
};

class AddPrim : public Primitive {
 public:
  const char* name() const override { return "add"; }
  void eval(const std::vector<Array>& inputs, Array& output) override;
};

class MulPrim : public Primitive {
 public:
  const char* name() const override { return "mul"; }
  void eval(const std::vector<Array>& inputs, Array& output) override;
};

class DotPrim : public Primitive {
 public:
  const char* name() const override { return "dot"; }
  void eval(const std::vector<Array>& inputs, Array& output) override;
};

}  // namespace minml
