// jit transform tests, CPU-only path. CPU is a fusion barrier, so
// jit(f) on CPU should be a *correctness no-op*: same outputs as
// running f directly. These tests exercise the trace + rewrite
// machinery (refcount, build_expr, opaque pass-through) without
// depending on a GPU.

import { expect, test } from "vitest";
import {
  add,
  Array,
  Device,
  dot,
  jit,
  mul,
} from "../src/index.js";

test("jit is a correctness no-op on CPU: pure add", async () => {
  const x = Array.from_f32_1d([1, 2, 3, 4], Device.Cpu);
  const y = Array.from_f32_1d([10, 20, 30, 40], Device.Cpu);
  const f = (a: Array, b: Array) => add(a, b);
  expect(await jit(f)(x, y).tolist()).toEqual([11, 22, 33, 44]);
});

test("jit elementwise chain on CPU: add(mul(x, y), z)", async () => {
  const x = Array.from_f32_1d([1, 2, 3, 4], Device.Cpu);
  const y = Array.from_f32_1d([2, 2, 2, 2], Device.Cpu);
  const z = Array.from_f32_1d([10, 20, 30, 40], Device.Cpu);
  const f = (a: Array, b: Array, c: Array) => add(mul(a, b), c);
  expect(await jit(f)(x, y, z).tolist()).toEqual([12, 24, 36, 40 + 8]);
});

test("jit reduction on CPU: dot(add(x, y), z)", async () => {
  const x = Array.from_f32_1d([1, 2, 3, 4], Device.Cpu);
  const y = Array.from_f32_1d([2, 2, 2, 2], Device.Cpu);
  const z = Array.from_f32_1d([10, 20, 30, 40], Device.Cpu);
  // (1+2)*10 + (2+2)*20 + (3+2)*30 + (4+2)*40 = 30 + 80 + 150 + 240 = 500
  const f = (a: Array, b: Array, c: Array) => dot(add(a, b), c);
  expect(await jit(f)(x, y, z).item()).toBe(500);
});

test("jit handles shared subexpression: dot(xy, xy) where xy=add(x,y)", async () => {
  const x = Array.from_f32_1d([1, 2, 3, 4], Device.Cpu);
  const y = Array.from_f32_1d([10, 20, 30, 40], Device.Cpu);
  const f = (a: Array, b: Array) => {
    const xy = add(a, b);
    return dot(xy, xy);
  };
  // 11^2 + 22^2 + 33^2 + 44^2 = 121 + 484 + 1089 + 1936 = 3630
  expect(await jit(f)(x, y).item()).toBe(3630);
});

test("jit returning multiple outputs", async () => {
  const x = Array.from_f32_1d([1, 2, 3], Device.Cpu);
  const y = Array.from_f32_1d([4, 5, 6], Device.Cpu);
  const f = (a: Array, b: Array): [Array, Array] => [add(a, b), mul(a, b)];
  const [s, p] = jit(f)(x, y);
  expect(await s.tolist()).toEqual([5, 7, 9]);
  expect(await p.tolist()).toEqual([4, 10, 18]);
});

test("jit pytree-style return: class instance with Array fields", async () => {
  const x = Array.from_f32_1d([1, 2, 3], Device.Cpu);
  const y = Array.from_f32_1d([4, 5, 6], Device.Cpu);

  class Pair {
    constructor(
      public sum: Array,
      public prod: Array,
    ) {}
  }

  const both = jit(
    (a: Array, b: Array) => new Pair(add(a, b), mul(a, b)),
  );
  const out = both(x, y);
  // Class identity is preserved on the way out.
  expect(out instanceof Pair).toBe(true);
  expect(await out.sum.tolist()).toEqual([5, 7, 9]);
  expect(await out.prod.tolist()).toEqual([4, 10, 18]);
});

test("jit pytree-style return: nested object", async () => {
  const x = Array.from_f32_1d([1, 2, 3], Device.Cpu);
  const f = jit((a: Array) => ({ wrapped: { value: add(a, a) } }));
  const out = f(x);
  expect(await out.wrapped.value.tolist()).toEqual([2, 4, 6]);
});

test("jit pre-evaluated input passes through", async () => {
  const x = Array.from_f32_1d([1, 2, 3], Device.Cpu);
  const y = Array.from_f32_1d([4, 5, 6], Device.Cpu);
  // Evaluate x's source so the lazy DAG ends at an eager Array.
  await x.tolist();
  const f = (a: Array, b: Array) => add(a, b);
  expect(await jit(f)(x, y).tolist()).toEqual([5, 7, 9]);
});
