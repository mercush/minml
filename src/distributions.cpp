#include "minml/distributions.h"

#include <stdexcept>

#include "minml/ops.h"

namespace minml {

Array Dirichlet::sample(PRNGKey key, std::vector<size_t> batch_shape) const {
  return dirichlet_sample(key.k0(), key.k1(), alpha_, std::move(batch_shape));
}

Array Categorical::sample(PRNGKey key, std::vector<size_t> batch_shape) const {
  return categorical_sample(key.k0(), key.k1(), probs_,
                            std::move(batch_shape));
}

Array Normal::sample(PRNGKey, std::vector<size_t>) const {
  throw std::runtime_error(
      "Normal::sample: not implemented yet. Add a normal_sample primitive "
      "if you need this.");
}

}  // namespace minml
