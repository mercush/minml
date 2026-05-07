// minml/array.h
//
// Array is the user-facing tensor. It is either:
//   * evaluated  -> data_ points to a Buffer with concrete bytes.
//   * lazy       -> primitive_ + inputs_ describe a deferred computation.
//
// Calling eval() (directly or via tolist()/item()) walks the input DAG
// post-order and invokes each primitive on its target device, producing
// concrete Buffers and clearing the lazy fields.
//
// 1-D vectors only. dtype() exists but Float32 is the only value supported
// today; widening to a second dtype is a new enum value plus per-backend
// kernels for it.
#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "minml/buffer.h"
#include "minml/device.h"
#include "minml/dtype.h"

namespace minml {

class Primitive;  // defined in src/primitives.h

class Array {
 public:
  // Eager construct from host data. Allocates a device buffer and copies.
  Array(std::vector<float> data, Device device = default_device());

  // Lazy construct: a future result of `prim` applied to `inputs`.
  Array(size_t size, Device device, std::shared_ptr<Primitive> prim,
        std::vector<Array> inputs);

  // Accessors.
  size_t size() const { return size_; }
  Device device() const { return device_; }
  DType dtype() const { return dtype_; }
  bool evaluated() const { return data_ != nullptr; }
  const std::shared_ptr<Buffer>& buffer() const { return data_; }
  const std::shared_ptr<Primitive>& primitive() const { return primitive_; }
  const std::vector<Array>& inputs() const { return inputs_; }

  // Force computation. After this, evaluated() == true.
  void eval();

  // Convenience: forces eval, copies device->host.
  std::vector<float> tolist();
  float item();  // size() must be 1.

  // Backend-internal: install evaluated data and drop the lazy graph.
  void set_data(std::shared_ptr<Buffer> b);

 private:
  size_t size_ = 0;
  Device device_ = Device::CPU;
  DType dtype_ = DType::Float32;            // only Float32 today.
  std::shared_ptr<Buffer> data_;            // null until evaluated.
  std::shared_ptr<Primitive> primitive_;    // null once evaluated.
  std::vector<Array> inputs_;               // empty once evaluated.
};

}  // namespace minml
