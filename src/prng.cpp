#include "minml/prng.h"

#include "threefry.h"

namespace minml {

PRNGKey PRNGKey::from_seed(uint32_t seed) {
  // Hash the seed once so seed=0 doesn't produce a degenerate key.
  uint32_t a, b;
  threefry_2x32(seed, 0xCAFEBABE, 0, 0, a, b);
  return PRNGKey(a, b);
}

std::vector<PRNGKey> PRNGKey::split(size_t n) const {
  std::vector<PRNGKey> out;
  out.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    uint32_t a, b;
    threefry_2x32(k0_, k1_, static_cast<uint32_t>(i), 0xC0FFEE, a, b);
    out.emplace_back(a, b);
  }
  return out;
}

}  // namespace minml
