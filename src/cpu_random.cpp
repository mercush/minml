// src/cpu_random.cpp
//
// CPU implementations of the random / sampling primitives. All draws come
// from Threefry-2x32 — see src/threefry.h. Same key + same op = same draws,
// independent of allocation order or thread count.
//
// Per-element indexing convention: each output element gets a counter
// derived from (op_tag, key, output_index, sub_iter). op_tag prevents
// different ops from producing identical bits when given the same key.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "backend.h"
#include "minml/buffer.h"
#include "threefry.h"

namespace minml {

namespace {

// Data accessors live in cpu_backend.cpp; we just call cpu_data_*.

// Op tags so different ops with the same key produce different bits.
constexpr uint32_t TAG_RANDINT       = 0x52414E44;  // 'RAND'
constexpr uint32_t TAG_DIRICHLET     = 0x44495243;  // 'DIRC'
constexpr uint32_t TAG_CATEGORICAL_U = 0x43415455;  // 'CATU' (uniforms)

float uniform_for(uint32_t k0, uint32_t k1, uint32_t tag, uint32_t i,
                  uint32_t sub) {
  uint32_t a, b;
  threefry_2x32(k0 ^ tag, k1, i, sub, a, b);
  return u32_to_unit_f32(a);
}

// Marsaglia & Tsang (2000) gamma sampler for shape >= 1. For shape < 1 we
// use the boost trick: draw G(shape+1, 1), multiply by U^(1/shape).
float sample_gamma(uint32_t k0, uint32_t k1, float shape, uint32_t base_idx) {
  if (shape < 1.0f) {
    float g = sample_gamma(k0, k1, shape + 1.0f, base_idx);
    float u = uniform_for(k0, k1, 0xDEADBEEF, base_idx, 99);
    if (u < 1e-30f) u = 1e-30f;
    return g * std::pow(u, 1.0f / shape);
  }
  float d = shape - 1.0f / 3.0f;
  float c = 1.0f / std::sqrt(9.0f * d);
  uint32_t sub = 0;
  while (true) {
    // Box-Muller for one normal sample from two uniforms.
    float u1 = uniform_for(k0, k1, 0xCAFE0001, base_idx, sub++);
    float u2 = uniform_for(k0, k1, 0xCAFE0002, base_idx, sub++);
    if (u1 < 1e-30f) u1 = 1e-30f;
    float x = std::sqrt(-2.0f * std::log(u1)) *
              std::cos(6.2831853f * u2);
    float v = 1.0f + c * x;
    if (v <= 0.0f) continue;
    v = v * v * v;
    float u = uniform_for(k0, k1, 0xCAFE0003, base_idx, sub++);
    if (u < 1.0f - 0.0331f * x * x * x * x) return d * v;
    if (std::log(u) < 0.5f * x * x + d * (1.0f - v + std::log(v)))
      return d * v;
  }
}

}  // namespace

void cpu_randint(uint32_t k0, uint32_t k1, int32_t low, int32_t high,
                 Array& out) {
  if (high <= low) throw std::runtime_error("randint: high <= low");
  uint32_t span = static_cast<uint32_t>(high - low);
  int32_t* p = cpu_data_i32(out);
  for (size_t i = 0; i < out.size(); ++i) {
    uint32_t bits = threefry_u32(k0 ^ TAG_RANDINT, k1,
                                 static_cast<uint32_t>(i));
    p[i] = low + static_cast<int32_t>(bits % span);
  }
}

void cpu_dirichlet_sample(uint32_t k0, uint32_t k1,
                          const std::vector<size_t>& batch_shape,
                          const Array& alpha, Array& out) {
  // out shape = batch_shape ++ (K,). Each output row of length K is one
  // Dirichlet draw: K independent gammas, normalized.
  size_t K = alpha.shape()[0];
  size_t B = 1;
  for (size_t d : batch_shape) B *= d;
  const float* a = cpu_data_f32(alpha);
  float* o = cpu_data_f32(out);
  for (size_t b = 0; b < B; ++b) {
    float* row = o + b * K;
    float sum = 0.0f;
    for (size_t k = 0; k < K; ++k) {
      uint32_t idx = static_cast<uint32_t>(b * K + k);
      row[k] = sample_gamma(k0 ^ TAG_DIRICHLET, k1, a[k], idx);
      sum += row[k];
    }
    if (sum > 0.0f) {
      for (size_t k = 0; k < K; ++k) row[k] /= sum;
    }
  }
}

void cpu_categorical_sample(uint32_t k0, uint32_t k1,
                            const std::vector<size_t>& batch_shape,
                            const Array& probs, Array& out) {
  // probs is (K,). Sample one categorical index per output element.
  // Inverse CDF: draw u ~ U[0,1), find the smallest k s.t. cumsum[k] > u.
  size_t K = probs.shape()[0];
  size_t B = 1;
  for (size_t d : batch_shape) B *= d;
  if (B == 0) B = 1;  // empty batch_shape means a scalar draw.

  const float* p = cpu_data_f32(probs);
  // Build the (un-normalized) CDF once.
  std::vector<float> cdf(K);
  float total = 0.0f;
  for (size_t k = 0; k < K; ++k) { total += p[k]; cdf[k] = total; }
  if (total <= 0.0f) throw std::runtime_error("categorical: probs sum to 0");

  int32_t* o = cpu_data_i32(out);
  for (size_t b = 0; b < B; ++b) {
    float u = uniform_for(k0, k1, TAG_CATEGORICAL_U,
                          static_cast<uint32_t>(b), 0) * total;
    int32_t pick = static_cast<int32_t>(K) - 1;
    for (size_t k = 0; k < K; ++k) {
      if (u < cdf[k]) { pick = static_cast<int32_t>(k); break; }
    }
    o[b] = pick;
  }
}

}  // namespace minml
