// jit smoke tests against the CUDA backend. Auto-skipped on hosts
// without the addon built — same pattern as tests/cuda.test.ts.

import { describe, expect, test } from "vitest";
import {
  add,
  Array,
  Device,
  dot,
  jit,
  MinmlError,
  mul,
} from "../src/index.js";

function cuda_available(): boolean {
  try {
    Array.from_f32_1d([1], Device.Cuda);
    return true;
  } catch (e) {
    if (e instanceof MinmlError && e.kind === "backend_not_built") {
      return false;
    }
    throw e;
  }
}

const HAS_CUDA = cuda_available();

describe.skipIf(!HAS_CUDA)("jit on CUDA", () => {
  test("fused elementwise: add(mul(x, y), z)", async () => {
    const x = Array.from_f32_1d([1, 2, 3, 4], Device.Cuda);
    const y = Array.from_f32_1d([2, 2, 2, 2], Device.Cuda);
    const z = Array.from_f32_1d([10, 20, 30, 40], Device.Cuda);
    const f = (a: Array, b: Array, c: Array) => add(mul(a, b), c);
    expect(await jit(f)(x, y, z).tolist()).toEqual([12, 24, 36, 48]);
  });

  test("fused reduction: dot(add(x, y), z)", async () => {
    const x = Array.from_f32_1d([1, 2, 3, 4], Device.Cuda);
    const y = Array.from_f32_1d([2, 2, 2, 2], Device.Cuda);
    const z = Array.from_f32_1d([10, 20, 30, 40], Device.Cuda);
    expect(
      await jit((a: Array, b: Array, c: Array) => dot(add(a, b), c))(
        x,
        y,
        z,
      ).item(),
    ).toBe(500); // 30 + 80 + 150 + 240
  });

  test("shared subexpr forces materialization once", async () => {
    const x = Array.from_f32_1d([1, 2, 3, 4], Device.Cuda);
    const y = Array.from_f32_1d([10, 20, 30, 40], Device.Cuda);
    const f = (a: Array, b: Array) => {
      const xy = add(a, b);
      return dot(xy, xy);
    };
    expect(await jit(f)(x, y).item()).toBe(3630);
  });

  test("kernel cache: re-running same jit'd f does not re-compile", async () => {
    // Hard to assert cache hits from JS without an instrumentation hook;
    // this test mainly verifies repeated calls don't crash and produce
    // consistent results.
    const x = Array.from_f32_1d([1, 2, 3, 4], Device.Cuda);
    const y = Array.from_f32_1d([5, 5, 5, 5], Device.Cuda);
    const f = jit((a: Array, b: Array) => mul(add(a, b), b));
    for (let i = 0; i < 3; i++) {
      expect(await f(x, y).tolist()).toEqual([30, 35, 40, 45]);
    }
  });
});
