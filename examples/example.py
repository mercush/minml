"""Example using the minml Python bindings on the CUDA backend.

Requires the build to have been configured with -DMINML_BUILD_CUDA=ON;
otherwise the CUDA stubs will throw at the first allocation.
"""
import _minml as m

device = m.Device.CUDA

x = m.array([1.0, 2.0, 3.0, 4.0], device=device)
y = m.array([10.0, 20.0, 30.0, 40.0], device=device)

print("add ->", m.add(x, y).tolist())
print("dot ->", m.dot(x, y).item())

# Lazy graphs are transparent to the caller; eval is forced by tolist/item.
print("dot(x+y, x+y) ->", m.dot(m.add(x, y), m.add(x, y)).item())
