"""minml Python example — async API.

Mirrors examples/example.cpp / the old sync version. Every readback
(`tolist`, `item`) is a coroutine because WebGPU readback is async, and
the surface is uniform across backends. CPU and CUDA finish synchronously
but still return awaitables.
"""
import asyncio
import _minml as m


async def main() -> None:
    device = m.Device.CPU  # set to m.Device.CUDA on a CUDA box

    x = m.array([1.0, 2.0, 3.0, 4.0], device=device)
    y = m.array([10.0, 20.0, 30.0, 40.0], device=device)

    print("add ->", await m.add(x, y).tolist())
    print("dot ->", await m.dot(x, y).item())

    # Lazy graph: add evaluated transparently when item() forces eval.
    print("dot(x+y, x+y) ->", await m.dot(m.add(x, y), m.add(x, y)).item())


if __name__ == "__main__":
    asyncio.run(main())
