// bindings/python/bind.cpp
//
// Builds a Python extension named _minml.
//
// nanobind handles std::vector<T>, std::optional, etc. via the headers
// in nanobind/stl/. Distribution classes and PRNGKey are bound as plain
// nb::class_ types.
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "minml/array.h"
#include "minml/device.h"
#include "minml/distributions.h"
#include "minml/dtype.h"
#include "minml/ops.h"
#include "minml/prng.h"
#include "minml/transforms.h"

namespace nb = nanobind;
using namespace minml;

namespace {

// vmap shim — translates Python callable + Python pytree into the
// language-agnostic minml::vmap_apply.
//
// Inputs may be Arrays (batched along axis 0) or Python lists / scalars
// (batched by length, looked up by iter index). Returns a single Array
// or, for pytree returns, a fresh Python class instance with the same
// attributes — matching the Deno/embind behavior.

bool is_array(nb::handle h) { return nb::isinstance<Array>(h); }

std::vector<Array> collect_leaves_py(nb::object v,
                                     std::vector<std::string>* keys) {
  std::vector<Array> out;
  if (is_array(v)) {
    out.push_back(nb::cast<Array>(v));
    if (keys) keys->push_back("");
    return out;
  }
  // Walk __dict__ in insertion order.
  nb::object d = v.attr("__dict__");
  for (nb::handle key : d) {
    nb::object val = d[key];
    if (is_array(val)) {
      out.push_back(nb::cast<Array>(val));
      if (keys) keys->push_back(nb::cast<std::string>(key));
    }
  }
  if (out.empty())
    throw std::runtime_error("vmap: function returned no Arrays");
  return out;
}

nb::object rebuild_tree_py(nb::object template_val,
                           const std::vector<std::string>& keys,
                           const std::vector<Array>& stacked) {
  if (keys.size() == 1 && keys[0].empty()) return nb::cast(stacked[0]);
  // Build a fresh instance of template_val's class via cls.__new__(cls).
  nb::object cls = template_val.attr("__class__");
  nb::object out = cls.attr("__new__")(cls);
  for (size_t i = 0; i < keys.size(); ++i) {
    out.attr(keys[i].c_str()) = nb::cast(stacked[i]);
  }
  return out;
}

nb::object vmap_apply_py(nb::callable f, nb::list in_axes_py,
                         nb::list args_py) {
  size_t n = nb::len(in_axes_py);
  if (nb::len(args_py) != n)
    throw std::runtime_error("vmap: in_axes length != args length");

  std::vector<int> in_axes(n);
  for (size_t i = 0; i < n; ++i) {
    nb::handle a = in_axes_py[i];
    in_axes[i] = a.is_none() ? -1 : nb::cast<int>(a);
  }

  std::vector<Array> c_args;
  std::vector<int> c_in_axes;
  std::vector<int> c_index(n, -1);
  std::vector<nb::object> py_args(n);
  size_t batch_N = 0;
  bool found_batch = false;

  for (size_t i = 0; i < n; ++i) {
    nb::object a = nb::cast<nb::object>(args_py[i]);
    py_args[i] = a;
    if (is_array(a)) {
      Array arr = nb::cast<Array>(a);
      c_index[i] = static_cast<int>(c_args.size());
      c_args.push_back(arr);
      c_in_axes.push_back(in_axes[i]);
      if (in_axes[i] >= 0 && !found_batch) {
        if (in_axes[i] != 0)
          throw std::runtime_error("vmap: only axis 0 supported on Arrays");
        if (arr.shape().empty())
          throw std::runtime_error("vmap: cannot batch over a scalar");
        batch_N = arr.shape()[0];
        found_batch = true;
      }
    } else if (in_axes[i] >= 0) {
      if (!found_batch) {
        batch_N = nb::len(a);
        found_batch = true;
      }
    }
  }
  if (!found_batch) throw std::runtime_error("vmap: no batched inputs");

  std::vector<std::string> leaf_keys;
  nb::object first_result;
  bool first = true;

  VmapCallable callable = [&](size_t b, const std::vector<Array>& sliced) {
    nb::list call_args;
    for (size_t i = 0; i < n; ++i) {
      if (c_index[i] >= 0) {
        call_args.append(nb::cast(sliced[c_index[i]]));
      } else if (in_axes[i] >= 0) {
        call_args.append(py_args[i][b]);
      } else {
        call_args.append(py_args[i]);
      }
    }
    nb::object r = f(*nb::tuple(call_args));
    if (first) {
      first_result = r;
      first = false;
      return collect_leaves_py(r, &leaf_keys);
    }
    return collect_leaves_py(r, /*keys=*/nullptr);
  };

  std::vector<Array> stacked = vmap_apply(batch_N, c_args, c_in_axes, callable);
  return rebuild_tree_py(first_result, leaf_keys, stacked);
}

}  // namespace

NB_MODULE(_minml, m) {
  nb::enum_<Device>(m, "Device")
      .value("CPU", Device::CPU)
      .value("CUDA", Device::CUDA)
      .value("WebGPU", Device::WebGPU);

  nb::enum_<DType>(m, "DType")
      .value("Float32", DType::Float32)
      .value("Int32", DType::Int32);

  m.def("set_default_device", &set_default_device);
  m.def("default_device", &default_device);

  nb::class_<Array>(m, "Array")
      .def(nb::init<std::vector<float>, Device>(), nb::arg("data"),
           nb::arg("device") = Device::CPU)
      .def("size", &Array::size)
      .def("shape", &Array::shape)
      .def("device", &Array::device)
      .def("dtype", &Array::dtype)
      .def("eval", &Array::eval)
      .def("tolist", &Array::tolist)
      .def("tolist_int", &Array::tolist_int)
      .def("item", &Array::item);

  // 1-D Float32 from a Python list.
  m.def("array",
        [](std::vector<float> data, Device d) {
          return Array(std::move(data), d);
        },
        nb::arg("data"), nb::arg("device") = Device::CPU);
  // 1-D Int32 from a Python list of ints.
  m.def("array_int",
        [](std::vector<int32_t> data, Device d) {
          size_t n = data.size();
          return Array(std::move(data), std::vector<size_t>{n}, d);
        },
        nb::arg("data"), nb::arg("device") = Device::CPU);
  // N-D constructors, used by the Python-side vmap helper to stack
  // per-iteration outputs along a new leading axis.
  m.def("array_shaped",
        [](std::vector<float> data, std::vector<size_t> shape, Device d) {
          return Array(std::move(data), std::move(shape), d);
        },
        nb::arg("data"), nb::arg("shape"), nb::arg("device") = Device::CPU);
  m.def("array_int_shaped",
        [](std::vector<int32_t> data, std::vector<size_t> shape, Device d) {
          return Array(std::move(data), std::move(shape), d);
        },
        nb::arg("data"), nb::arg("shape"), nb::arg("device") = Device::CPU);

  m.def("add", &add);
  m.def("mul", &mul);
  m.def("dot", &dot);
  m.def("ones", &ones, nb::arg("shape"),
        nb::arg("dtype") = DType::Float32,
        nb::arg("device") = Device::CPU);
  m.def("randint", &randint);
  m.def("gather", &gather);
  m.def("dirichlet_sample", &dirichlet_sample);
  m.def("categorical_sample", &categorical_sample);

  nb::class_<PRNGKey>(m, "PRNGKey")
      .def(nb::init<uint32_t, uint32_t>())
      .def_static("new", &PRNGKey::from_seed)
      .def("split", &PRNGKey::split)
      .def("k0", &PRNGKey::k0)
      .def("k1", &PRNGKey::k1);

  nb::class_<Dirichlet>(m, "Dirichlet")
      .def(nb::init<Array>())
      .def("sample", &Dirichlet::sample);

  nb::class_<Categorical>(m, "Categorical")
      .def(nb::init<Array>())
      .def("sample", &Categorical::sample);

  nb::class_<Normal>(m, "Normal")
      .def(nb::init<>())
      .def("sample", &Normal::sample);

  // Direct: vmap_apply(f, in_axes, args) — three positional args.
  m.def("vmap_apply", &vmap_apply_py, nb::arg("f"), nb::arg("in_axes"),
        nb::arg("args"));

  // Curried: vmap(f, in_axes)(args...) — matches the JAX/MLX call shape.
  m.def("vmap", [](nb::callable f, nb::list in_axes) -> nb::object {
    return nb::cpp_function(
        [f, in_axes](nb::args args) -> nb::object {
          nb::list args_list;
          for (auto h : args) args_list.append(h);
          return vmap_apply_py(f, in_axes, args_list);
        });
  }, nb::arg("f"), nb::arg("in_axes"));
}
