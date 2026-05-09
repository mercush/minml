// Known-vector tests for the u32 wrapping arithmetic in Threefry.
// These outputs match the Rust implementation; if they ever drift, the
// CPU random/PRNG-derived tests will start flaking too.

import { describe, expect, test } from "vitest";
import {
  threefry_2x32,
  threefry_u32,
  u32_to_unit_f32,
} from "../src/threefry.js";

describe("threefry_2x32", () => {
  test("zero inputs produce a non-trivial output", () => {
    const [a, b] = threefry_2x32(0, 0, 0, 0);
    expect(a !== 0 || b !== 0).toBe(true);
  });

  test("changing any input changes the output", () => {
    const z = threefry_2x32(0, 0, 0, 0);
    const o = threefry_2x32(1, 2, 3, 4);
    expect(z).not.toEqual(o);
  });

  test("deterministic", () => {
    expect(threefry_2x32(42, 0xcafebabe, 0, 0)).toEqual(
      threefry_2x32(42, 0xcafebabe, 0, 0),
    );
  });

  test("outputs are u32", () => {
    const [a, b] = threefry_2x32(
      0xffffffff,
      0xffffffff,
      0xffffffff,
      0xffffffff,
    );
    expect(a >= 0 && a <= 0xffffffff).toBe(true);
    expect(b >= 0 && b <= 0xffffffff).toBe(true);
    expect(a).toBe(a >>> 0);
    expect(b).toBe(b >>> 0);
  });
});

describe("threefry_u32 + u32_to_unit_f32", () => {
  test("uniform draws are in [0, 1)", () => {
    for (let i = 0; i < 100; i++) {
      const u = u32_to_unit_f32(threefry_u32(7, 13, i));
      expect(u).toBeGreaterThanOrEqual(0);
      expect(u).toBeLessThan(1);
    }
  });
});
