// minml/dtype.h
//
// Element type for an Array. Only Float32 is implemented today; the enum
// exists so the Array surface (`.dtype`) is stable and a second dtype can
// be added without changing call sites. Adding one means: a new enum value,
// a `dtype_bytes()` arm, and per-backend kernels for it.
#pragma once

#include <cstddef>

namespace minml {

enum class DType { Float32, Int32 };

inline constexpr size_t dtype_bytes(DType t) {
  switch (t) {
    case DType::Float32: return 4;
    case DType::Int32:   return 4;
  }
  return 0;
}

const char* dtype_name(DType t);

}  // namespace minml
