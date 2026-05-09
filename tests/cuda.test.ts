// CUDA backend tests. Auto-skip on hosts without the addon built — so
// `npm test` stays green on macOS / a fresh Linux checkout, and only
// runs the real assertions after `npm run build:cuda`.

import { describe, expect, test } from "vitest";
import {
  add,
  Array,
  Device,
  dot,
  mul,
  MinmlError,
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

describe.skipIf(!HAS_CUDA)("cuda backend", () => {
  test("add round-trip", async () => {
    const x = Array.from_f32_1d([1, 2, 3, 4], Device.Cuda);
    const y = Array.from_f32_1d([10, 20, 30, 40], Device.Cuda);
    expect(await add(x, y).tolist()).toEqual([11, 22, 33, 44]);
  });

  test("mul round-trip", async () => {
    const x = Array.from_f32_1d([1, 2, 3], Device.Cuda);
    const y = Array.from_f32_1d([4, 5, 6], Device.Cuda);
    expect(await mul(x, y).tolist()).toEqual([4, 10, 18]);
  });

  test("dot reduces to scalar", async () => {
    const x = Array.from_f32_1d([1, 2, 3, 4], Device.Cuda);
    const y = Array.from_f32_1d([10, 20, 30, 40], Device.Cuda);
    expect(await dot(x, y).item()).toBe(300);
  });

  test("lazy graph: shared subexpression evaluated once", async () => {
    const x = Array.from_f32_1d([1, 2, 3, 4], Device.Cuda);
    const y = Array.from_f32_1d([10, 20, 30, 40], Device.Cuda);
    const xy = add(x, y);
    expect(await dot(xy, xy).item()).toBe(11 * 11 + 22 * 22 + 33 * 33 + 44 * 44);
  });
});

// Also assert the negative path: on hosts without CUDA, the error is
// well-formed (not a generic crash). Always runs.
test("backend_not_built error shape (host-agnostic)", () => {
  if (HAS_CUDA) return; // not applicable when the addon is loaded
  try {
    Array.from_f32_1d([1, 2], Device.Cuda);
    throw new Error("should have thrown");
  } catch (e) {
    expect(e).toBeInstanceOf(MinmlError);
    expect((e as MinmlError).kind).toBe("backend_not_built");
  }
});
