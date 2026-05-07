// minml/prng.h
//
// PRNGKey — a splittable pseudo-random key, JAX-style. Two uint32s of state
// plus deterministic split. No global state: every random op takes a key.
//
// Splits derive children via Threefry on (parent, child_index), so the
// same parent always produces the same children. Reseeding requires
// PRNGKey::from_seed(new_seed).
#pragma once

#include <cstdint>
#include <vector>

namespace minml {

class PRNGKey {
 public:
  PRNGKey() : k0_(0), k1_(0) {}
  PRNGKey(uint32_t k0, uint32_t k1) : k0_(k0), k1_(k1) {}

  static PRNGKey from_seed(uint32_t seed);

  // Returns n derived keys. Same parent always yields the same children.
  std::vector<PRNGKey> split(size_t n) const;

  uint32_t k0() const { return k0_; }
  uint32_t k1() const { return k1_; }

 private:
  uint32_t k0_;
  uint32_t k1_;
};

}  // namespace minml
