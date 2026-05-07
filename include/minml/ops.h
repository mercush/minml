// minml/ops.h
//
// User-facing ops. Both build a lazy Array on the device of the inputs.
// They do not allocate output buffers or run kernels until eval().
#pragma once

#include "minml/array.h"

namespace minml {

// Elementwise: out[i] = a[i] + b[i]. Sizes must match.
Array add(const Array& a, const Array& b);

// Elementwise: out[i] = a[i] * b[i]. Sizes must match.
Array mul(const Array& a, const Array& b);

// Reduction: out = sum(a[i] * b[i]). Sizes must match. Result is size==1.
Array dot(const Array& a, const Array& b);

}  // namespace minml
