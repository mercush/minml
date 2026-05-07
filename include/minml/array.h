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
// Storage is always C-contiguous (row-major); no strides yet. dtype is
// Float32 or Int32 today.
//
// batch_axis_ is set transiently by the vmap framework: an op entry point
// (add, gather, ...) that sees a batched input dispatches through the
// primitive's vmap rule and tags the output. Cleared by strip_batch_axis()
// once vmap unwinds.
#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "minml/buffer.h"
#include "minml/device.h"
#include "minml/dtype.h"

namespace minml {

class Primitive;  // defined in src/primitives.h

class Array {
 public:
  // Eager construct from float host data, with explicit shape.
  Array(std::vector<float> data, std::vector<size_t> shape,
        Device device = default_device());
  // Eager construct from int32 host data.
  Array(std::vector<int32_t> data, std::vector<size_t> shape,
        Device device = default_device());

  // 1-D float convenience: shape inferred from data.size().
  Array(std::vector<float> data, Device device = default_device());

  // Lazy construct: a future result of `prim` applied to `inputs`.
  Array(std::vector<size_t> shape, DType dtype, Device device,
        std::shared_ptr<Primitive> prim, std::vector<Array> inputs);

  // Accessors.
  const std::vector<size_t>& shape() const { return shape_; }
  size_t size() const { return size_; }
  Device device() const { return device_; }
  DType dtype() const { return dtype_; }
  std::optional<int> batch_axis() const { return batch_axis_; }
  bool evaluated() const { return data_ != nullptr; }
  const std::shared_ptr<Buffer>& buffer() const { return data_; }
  const std::shared_ptr<Primitive>& primitive() const { return primitive_; }
  const std::vector<Array>& inputs() const { return inputs_; }

  // Returns a sibling Array (shared buffer / primitive / inputs) carrying
  // the requested batch_axis tag. Used by vmap_apply at the start of a
  // trace.
  Array with_batch_axis(int axis) const;
  // Returns a sibling Array with batch_axis_ cleared.
  Array strip_batch_axis() const;

  // Force computation. After this, evaluated() == true.
  void eval();

  // Convenience: forces eval, copies device->host. Returned vector
  // contains float bits regardless of dtype; bindings reinterpret for
  // Int32 outputs.
  std::vector<float> tolist();
  std::vector<int32_t> tolist_int();
  float item();  // size() must be 1.

  // Backend-internal: install evaluated data and drop the lazy graph.
  void set_data(std::shared_ptr<Buffer> b);

 private:
  std::vector<size_t> shape_;
  size_t size_ = 0;                          // product of shape_, cached.
  Device device_ = Device::CPU;
  DType dtype_ = DType::Float32;
  std::optional<int> batch_axis_;            // set transiently under vmap.
  std::shared_ptr<Buffer> data_;             // null until evaluated.
  std::shared_ptr<Primitive> primitive_;     // null once evaluated.
  std::vector<Array> inputs_;                // empty once evaluated.
};

}  // namespace minml
