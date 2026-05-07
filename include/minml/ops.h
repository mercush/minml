// minml/ops.h
//
// User-facing ops. Each builds a lazy Array on the device of the inputs (or
// the default device for nullary ops). They do not allocate output buffers
// or run kernels until eval().
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "minml/array.h"
#include "minml/dtype.h"

namespace minml {

// Elementwise: out[i] = a[i] + b[i]. Shapes must match.
Array add(const Array& a, const Array& b);

// Elementwise: out[i] = a[i] * b[i]. Shapes must match.
Array mul(const Array& a, const Array& b);

// Reduction: out = sum(a[i] * b[i]). 1-D inputs of equal size. Result size 1.
Array dot(const Array& a, const Array& b);

// Constant. Produces a new Array filled with 1 (1.0 for Float32, 1 for Int32).
Array ones(std::vector<size_t> shape, DType dtype = DType::Float32,
           Device device = default_device());

// Uniform integer sampling on [low, high). Output shape `shape`, Int32.
Array randint(uint32_t k0, uint32_t k1, int32_t low, int32_t high,
              std::vector<size_t> shape, Device device = default_device());

// Index `table` along axis 0 by `indices` (Int32). table shape (N, ...trail);
// indices shape (M,) or (...batch, M); output shape (...batch, M, ...trail).
Array gather(const Array& table, const Array& indices);

// Sample from Dirichlet(alpha). alpha shape (K,) Float32. batch_shape may be
// empty. Output shape (...batch_shape, K) Float32.
Array dirichlet_sample(uint32_t k0, uint32_t k1, const Array& alpha,
                       std::vector<size_t> batch_shape);

// Sample from Categorical(probs). probs shape (K,) Float32. batch_shape
// may be empty. Output shape `batch_shape`, Int32.
Array categorical_sample(uint32_t k0, uint32_t k1, const Array& probs,
                         std::vector<size_t> batch_shape);

}  // namespace minml
