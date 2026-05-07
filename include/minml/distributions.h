// minml/distributions.h
//
// Thin C++ classes that hold distribution parameters and expose .sample().
// Each .sample() builds a lazy Array via the matching *_sample primitive
// (see ops.h).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "minml/array.h"
#include "minml/prng.h"

namespace minml {

class Dirichlet {
 public:
  explicit Dirichlet(Array alpha) : alpha_(std::move(alpha)) {}
  Array sample(PRNGKey key, std::vector<size_t> batch_shape) const;
  const Array& alpha() const { return alpha_; }
 private:
  Array alpha_;
};

class Categorical {
 public:
  explicit Categorical(Array probs) : probs_(std::move(probs)) {}
  Array sample(PRNGKey key, std::vector<size_t> batch_shape) const;
  const Array& probs() const { return probs_; }
 private:
  Array probs_;
};

// Standard Normal(0, 1). Box-Muller from two uniforms; sample() draws
// `batch_shape` independent values. Not used by the notebook today but
// kept for parity with the original Mantle import surface.
class Normal {
 public:
  Normal() = default;
  Array sample(PRNGKey key, std::vector<size_t> batch_shape) const;
};

}  // namespace minml
