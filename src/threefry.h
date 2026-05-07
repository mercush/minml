// src/threefry.h  (internal)
//
// Threefry-2x32: a counter-based, splittable PRNG. Pure function from
// (key, counter) -> uniform u32. Same hash on every backend so a given
// PRNGKey produces the same draws on CPU, CUDA, WebGPU.
//
// Reference: Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3"
// (SC'11). The 2x32 variant uses 20 rounds.
#pragma once

#include <cstdint>

namespace minml {

inline uint32_t rotl32(uint32_t x, int k) {
  return (x << k) | (x >> (32 - k));
}

// Hash (k0, k1, ctr0, ctr1) -> (out0, out1) via Threefry-2x32-20.
inline void threefry_2x32(uint32_t k0, uint32_t k1, uint32_t ctr0, uint32_t ctr1,
                          uint32_t& out0, uint32_t& out1) {
  // Rotation constants from the reference.
  static const int R[8] = {13, 15, 26, 6, 17, 29, 16, 24};

  uint32_t k2 = 0x1BD11BDA ^ k0 ^ k1;
  uint32_t x0 = ctr0 + k0;
  uint32_t x1 = ctr1 + k1;

  // 20 rounds, key injected every 4 rounds.
  for (int round = 0; round < 20; ++round) {
    x0 += x1;
    x1 = rotl32(x1, R[round % 8]);
    x1 ^= x0;
    if ((round + 1) % 4 == 0) {
      int s = (round + 1) / 4;
      uint32_t ks0 = (s % 3 == 0) ? k0 : (s % 3 == 1) ? k1 : k2;
      uint32_t ks1 = (s % 3 == 0) ? k1 : (s % 3 == 1) ? k2 : k0;
      x0 += ks0;
      x1 += ks1 + static_cast<uint32_t>(s);
    }
  }
  out0 = x0;
  out1 = x1;
}

// Convenience: scalar hash (key, i) -> u32. Combines the two outputs
// into one usable uint32.
inline uint32_t threefry_u32(uint32_t k0, uint32_t k1, uint32_t i) {
  uint32_t a, b;
  threefry_2x32(k0, k1, i, 0, a, b);
  return a;
}

// u32 -> uniform float in [0, 1). 24-bit mantissa is enough; clear the
// top 8 bits and divide.
inline float u32_to_unit_f32(uint32_t u) {
  return static_cast<float>(u >> 8) * (1.0f / 16777216.0f);
}

}  // namespace minml
