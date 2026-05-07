#include "minml/dtype.h"

namespace minml {

const char* dtype_name(DType t) {
  switch (t) {
    case DType::Float32: return "float32";
  }
  return "unknown";
}

}  // namespace minml
