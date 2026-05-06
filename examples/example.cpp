// examples/example.cpp
#include <cstdio>

#include "minml/array.h"
#include "minml/device.h"
#include "minml/ops.h"

int main() {
  using namespace minml;

  // Default: CPU. Switch with set_default_device(Device::CUDA) etc.
  Array x({1, 2, 3, 4});
  Array y({10, 20, 30, 40});

  // Lazy: nothing has run yet.
  Array s = add(x, y);
  Array d = dot(x, y);

  // tolist()/item() force evaluation post-order.
  auto r = s.tolist();
  std::printf("add  -> [%g, %g, %g, %g]\n", r[0], r[1], r[2], r[3]);
  std::printf("dot  -> %g\n", d.item());

  // Composed lazy graph: dot(x+y, x+y).
  Array z = dot(add(x, y), add(x, y));
  std::printf("dot(x+y, x+y) -> %g\n", z.item());
  return 0;
}
