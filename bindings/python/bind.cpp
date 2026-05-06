// bindings/python/bind.cpp
//
// Builds a Python extension named _minml. Usage from Python:
//
//   import _minml as m
//   x = m.array([1.0, 2.0, 3.0])
//   y = m.array([4.0, 5.0, 6.0])
//   print(m.add(x, y).tolist())   # [5, 7, 9]
//   print(m.dot(x, y).item())     # 32.0
//
// nanobind is preferred over pybind11: smaller, faster, fewer dependencies.
// The same code structure works under pybind11 by swapping `nb::` for `py::`.
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "minml/array.h"
#include "minml/device.h"
#include "minml/ops.h"

namespace nb = nanobind;
using namespace minml;

NB_MODULE(_minml, m) {
  nb::enum_<Device>(m, "Device")
      .value("CPU", Device::CPU)
      .value("CUDA", Device::CUDA)
      .value("WebGPU", Device::WebGPU);

  m.def("set_default_device", &set_default_device);
  m.def("default_device", &default_device);

  nb::class_<Array>(m, "Array")
      .def(nb::init<std::vector<float>, Device>(), nb::arg("data"),
           nb::arg("device") = Device::CPU)
      .def("size", &Array::size)
      .def("device", &Array::device)
      .def("eval", &Array::eval)
      .def("tolist", &Array::tolist)
      .def("item", &Array::item);

  m.def("array",
        [](std::vector<float> data, Device d) { return Array(std::move(data), d); },
        nb::arg("data"), nb::arg("device") = Device::CPU);

  m.def("add", &add);
  m.def("dot", &dot);
}
