// Integration tests: round-trip the CPU backend through the lazy graph.
// Direct port of crates/minml-core/tests/cpu.rs.

import { expect, test } from "vitest";
import {
  add,
  Array,
  categorical_sample,
  dirichlet_sample,
  dot,
  DType,
  Device,
  gather,
  mul,
  ones,
  PRNGKey,
  randint,
  slice_axis0,
  stack,
  vmap_apply,
} from "../src/index.js";

test("example_match", async () => {
  // x = [1, 2, 3, 4]; y = [10, 20, 30, 40]
  // add(x, y) = [11, 22, 33, 44]
  // dot(x, y) = 1*10 + 2*20 + 3*30 + 4*40 = 300
  // dot(x+y, x+y) = 11^2 + 22^2 + 33^2 + 44^2 = 3630.
  const x = Array.from_f32_1d([1, 2, 3, 4], Device.Cpu);
  const y = Array.from_f32_1d([10, 20, 30, 40], Device.Cpu);

  const s = add(x, y);
  expect(await s.tolist()).toEqual([11, 22, 33, 44]);

  const d = dot(x, y);
  expect(await d.item()).toBe(300);

  // Lazy graph: add evaluated twice via shared subgraph.
  const xy = add(x, y);
  const dd = dot(xy, xy);
  expect(await dd.item()).toBe(11 * 11 + 22 * 22 + 33 * 33 + 44 * 44);
});

test("mul_works", async () => {
  const a = Array.from_f32_1d([1, 2, 3], Device.Cpu);
  const b = Array.from_f32_1d([4, 5, 6], Device.Cpu);
  expect(await mul(a, b).tolist()).toEqual([4, 10, 18]);
});

test("ones_float", async () => {
  expect(await ones([5], DType.Float32, Device.Cpu).tolist()).toEqual([
    1, 1, 1, 1, 1,
  ]);
});

test("ones_int", async () => {
  expect(await ones([3], DType.Int32, Device.Cpu).tolist_int()).toEqual([
    1, 1, 1,
  ]);
});

test("randint_in_range_and_deterministic", async () => {
  const r = randint(123, 456, 10, 20, [64], Device.Cpu);
  const v = await r.tolist_int();
  expect(v).toHaveLength(64);
  for (const x of v) {
    expect(x).toBeGreaterThanOrEqual(10);
    expect(x).toBeLessThan(20);
  }
  // Determinism: same key + op + shape -> same draws.
  const r2 = randint(123, 456, 10, 20, [64], Device.Cpu);
  expect(await r2.tolist_int()).toEqual(v);
});

test("gather_works", async () => {
  const table = Array.from_f32_with_shape(
    [10, 20, 30, 40, 50, 60],
    [3, 2],
    Device.Cpu,
  );
  const idx = Array.from_i32_1d([2, 0, 1], Device.Cpu);
  const r = gather(table, idx);
  expect(r.shape()).toEqual([3, 2]);
  expect(await r.tolist()).toEqual([50, 60, 10, 20, 30, 40]);
});

test("dirichlet_sums_to_one", async () => {
  const alpha = Array.from_f32_1d([1, 1, 1], Device.Cpu);
  const key = PRNGKey.from_seed(7);
  const s = dirichlet_sample(key.k0, key.k1, alpha, [4]);
  expect(s.shape()).toEqual([4, 3]);
  const v = await s.tolist();
  for (let r = 0; r < 4; r++) {
    const row = v.slice(r * 3, (r + 1) * 3);
    const sum = row.reduce((a, b) => a + b, 0);
    expect(Math.abs(sum - 1)).toBeLessThan(1e-4);
    for (const x of row) expect(x).toBeGreaterThanOrEqual(0);
  }
});

test("categorical_in_range", async () => {
  const probs = Array.from_f32_1d([0.1, 0.7, 0.2], Device.Cpu);
  const key = PRNGKey.from_seed(13);
  const s = categorical_sample(key.k0, key.k1, probs, [100]);
  const v = await s.tolist_int();
  for (const x of v) {
    expect(x).toBeGreaterThanOrEqual(0);
    expect(x).toBeLessThan(3);
  }
});

test("prng_split_deterministic", () => {
  const parent = PRNGKey.from_seed(1234);
  const a = parent.split(5);
  const b = parent.split(5);
  for (let i = 0; i < 5; i++) {
    expect(a[i].equals(b[i])).toBe(true);
  }
  expect(a[0].equals(a[1])).toBe(false);
});

test("slice_and_stack_roundtrip", async () => {
  const a = Array.from_f32_with_shape(
    [1, 2, 3, 4, 5, 6],
    [3, 2],
    Device.Cpu,
  );
  const parts = slice_axis0(a);
  expect(parts).toHaveLength(3);
  expect(parts[0].shape()).toEqual([2]);
  const restacked = stack(parts);
  expect(restacked.shape()).toEqual([3, 2]);
  expect(await restacked.tolist()).toEqual([1, 2, 3, 4, 5, 6]);
});

test("vmap_add_per_row", async () => {
  const xs = Array.from_f32_with_shape(
    [1, 2, 3, 4, 5, 6],
    [3, 2],
    Device.Cpu,
  );
  const ys = Array.from_f32_with_shape(
    [10, 10, 20, 20, 30, 30],
    [3, 2],
    Device.Cpu,
  );
  const out = vmap_apply(
    3,
    [xs, ys],
    [0, 0],
    (_iter, args) => [add(args[0], args[1])],
  );
  expect(out).toHaveLength(1);
  expect(out[0].shape()).toEqual([3, 2]);
  expect(await out[0].tolist()).toEqual([11, 12, 23, 24, 35, 36]);
});
