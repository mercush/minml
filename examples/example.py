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

    # ---- jit: kernel-fusion transform ----
    # Without jit, `mul(add(x, y), add(x, x))` is three lazy nodes that eval
    # as three kernel launches with two intermediate buffers. After jit, the
    # whole expression becomes one FusedElementwise primitive: on WebGPU /
    # CUDA that's one runtime-compiled kernel and one launch; on CPU the
    # tree is walked per element with no intermediates.
    fused = m.jit(lambda a, b: m.mul(m.add(a, b), m.add(a, a)))
    print("jit (x+y)*(x+x) ->", await fused(x, y).tolist())

    # jit also accepts pytree-style returns (any object whose __dict__
    # values are Arrays) — same shape goes in, same shape comes out.
    class Pair:
        def __init__(self, s: m.Array, d: m.Array) -> None:
            self.sum = s
            self.diff = d

    def both(a: m.Array, b: m.Array) -> Pair:
        return Pair(
            m.add(m.mul(a, b), m.mul(a, a)),   # a*b + a*a
            m.add(m.mul(a, a), m.mul(a, a)),   # 2 * a*a
        )

    pair = m.jit(both)(x, y)
    print("jit pair.sum  ->", await pair.sum.tolist())
    print("jit pair.diff ->", await pair.diff.tolist())


if __name__ == "__main__":
    asyncio.run(main())
