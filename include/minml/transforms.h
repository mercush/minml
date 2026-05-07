// minml/transforms.h
//
// Function transformations.
//
// `vmap_apply` is the language-agnostic core of vmap: it slices each
// batched Array input along axis 0, calls a user-provided per-iteration
// callable, and stacks the resulting per-iteration leaf arrays along a
// new leading axis.
//
// Bindings (embind / nanobind) wrap this with a small adapter that
// translates the language's native callable + pytree into the
// `VmapCallable` signature below. Language-native heterogeneous inputs
// (e.g. a JS list of PRNGKey) live in the binding's closure and are
// indexed by the iteration counter the C++ loop hands back.
//
// `stack` is exposed too — it's the building block for vmap_apply but
// also handy on its own.
#pragma once

#include <cstddef>
#include <functional>
#include <vector>

#include "minml/array.h"

namespace minml {

// Slice an Array along axis 0 into N pieces, each shape == arr.shape[1:].
// Forces eval. Used by vmap_apply and exposed for direct binding use.
std::vector<Array> slice_axis0(const Array& arr);

// Stack N Arrays of identical shape/dtype/device along a new leading axis.
// Output shape: (N, ...orig_shape).
Array stack(const std::vector<Array>& parts);

// Per-iteration callable. Receives:
//   * iter_index: which batch element (0 .. N-1) we're on. Bindings use
//     this to look up language-native list inputs (e.g. JS PRNGKey lists).
//   * args: Array arguments with batched ones already sliced (shape ==
//     orig.shape[1:]); unbatched ones passed through unchanged.
// Returns one Array per leaf of the function's logical return value
// (single-Array returns yield {arr}; pytree returns yield one per leaf).
using VmapCallable =
    std::function<std::vector<Array>(size_t iter_index,
                                     const std::vector<Array>& args)>;

// vmap_apply: the orchestration loop.
//
// N is the batch size — bindings determine it from whatever batched input
// they have (Array shape[axis] or list length).
// args / in_axes have the same length. in_axes[i] >= 0 means args[i] is
// batched on that axis (only axis 0 supported today); -1 means it's
// passed through unchanged. Bindings that handle non-Array batched inputs
// (lists) keep those out of `args` and capture them in their closure.
//
// Returns one stacked Array per leaf, in the same order the callable
// emitted them. Bindings rebuild their pytree from this list.
std::vector<Array> vmap_apply(size_t N,
                              const std::vector<Array>& args,
                              const std::vector<int>& in_axes,
                              const VmapCallable& f);

}  // namespace minml
