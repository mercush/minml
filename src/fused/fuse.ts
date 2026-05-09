// DAG rewrite pass: take an output Array, produce a new Array whose
// lazy nodes are FusedElemPrim / FusedReducePrim wherever a maximal
// {add, mul, dot} sub-DAG exists on a fusable backend (CUDA / WebGPU).
//
// Boundaries:
//   * eager arrays
//   * arrays on CPU device (CPU is a hard fusion barrier)
//   * lazy nodes with refcount > 1 (shared subexpressions are
//     materialized once and read by both consumers)
//   * non-elementwise primitives
//
// `dot` (the only reduction we have) is handled specially: a dot at the
// root of fuse() produces a reduce_sum Plan whose body is
// mul(buildExpr(P), buildExpr(Q)). Inside buildExpr, dot is treated as
// a boundary (recursively fused as its own kernel and read as an input).

import { Array } from "../array.js";
import { Device } from "../device.js";
import { MinmlError } from "../error.js";
import { fusion_class_of, type Primitive } from "../primitive.js";
import { type Expr } from "./expr.js";
import { FusedElemPrim, FusedReducePrim } from "./prim.js";

type RefMap = Map<Array, number>;
// Memoize fuse_with results so a shared lazy subexpression (refcount > 1)
// is rewritten once and the same wrapped Array is reused as a boundary
// in every consumer's fused kernel.
type FuseCache = Map<Array, Array>;

function compute_refcounts(root: Array): RefMap {
  const refs: RefMap = new Map();
  const seen = new Set<Array>();
  const visit = (arr: Array): void => {
    if (seen.has(arr)) return;
    seen.add(arr);
    const state = arr.lazy_state();
    if (!state) return;
    for (const inp of state.inputs) {
      refs.set(inp, (refs.get(inp) ?? 0) + 1);
      visit(inp);
    }
  };
  visit(root);
  return refs;
}

function is_fusable_device(d: Device): boolean {
  return d === Device.WebGpu || d === Device.Cuda;
}

// True iff `arr` can be absorbed directly into a parent fused expression
// (i.e., not a boundary).
function is_fusable_inline(arr: Array, refcount: RefMap): boolean {
  if (arr.evaluated()) return false;
  if (!is_fusable_device(arr.device())) return false;
  if ((refcount.get(arr) ?? 0) > 1) return false;
  const state = arr.lazy_state();
  if (!state) return false;
  return fusion_class_of(state.prim) === "elementwise";
}

// Children of an absorbed node: recurse with boundary rules.
function build_expr(
  arr: Array,
  refcount: RefMap,
  boundaries: Array[],
  cache: FuseCache,
): Expr {
  if (!is_fusable_inline(arr, refcount)) {
    const fused = fuse_with(arr, refcount, cache);
    // Dedupe by reference: if the same Array is already a boundary,
    // reuse its index instead of binding the same buffer twice.
    for (let i = 0; i < boundaries.length; i++) {
      if (boundaries[i] === fused) return { kind: "input", index: i };
    }
    const idx = boundaries.length;
    boundaries.push(fused);
    return { kind: "input", index: idx };
  }
  return build_expr_root(arr, refcount, boundaries, cache);
}

// `arr` is known to be a fusable elementwise prim — absorb its op + children.
function build_expr_root(
  arr: Array,
  refcount: RefMap,
  boundaries: Array[],
  cache: FuseCache,
): Expr {
  const state = arr.lazy_state()!;
  const op = state.prim.name();
  const a = build_expr(state.inputs[0], refcount, boundaries, cache);
  const b = build_expr(state.inputs[1], refcount, boundaries, cache);
  if (op === "add") return { kind: "add", a, b };
  if (op === "mul") return { kind: "mul", a, b };
  throw MinmlError.other(`build_expr_root: unexpected elementwise op '${op}'`);
}

// Internal: fuse using a precomputed refcount map + memo cache.
function fuse_with(arr: Array, refcount: RefMap, cache: FuseCache): Array {
  const cached = cache.get(arr);
  if (cached) return cached;

  let result: Array;

  if (arr.evaluated() || !is_fusable_device(arr.device())) {
    result = arr;
  } else {
    const state = arr.lazy_state();
    if (!state) {
      result = arr;
    } else {
      const cls = fusion_class_of(state.prim);
      if (cls === "elementwise") {
        const boundaries: Array[] = [];
        const expr = build_expr_root(arr, refcount, boundaries, cache);
        result = Array.lazy(
          arr.shape().slice(),
          arr.dtype(),
          arr.device(),
          new FusedElemPrim(expr, arr.size()),
          boundaries,
        );
      } else if (cls === "reduction") {
        if (state.prim.name() !== "dot") {
          throw MinmlError.other(
            `fuse: unknown reduction primitive '${state.prim.name()}'`,
          );
        }
        const boundaries: Array[] = [];
        const left = build_expr(state.inputs[0], refcount, boundaries, cache);
        const right = build_expr(state.inputs[1], refcount, boundaries, cache);
        const body: Expr = { kind: "mul", a: left, b: right };
        result = Array.lazy(
          arr.shape().slice(),
          arr.dtype(),
          arr.device(),
          new FusedReducePrim(body, state.inputs[0].size()),
          boundaries,
        );
      } else {
        // Opaque: rewrite inputs but keep the original primitive.
        const new_inputs = state.inputs.map((inp) =>
          fuse_with(inp, refcount, cache),
        );
        let same = true;
        for (let i = 0; i < new_inputs.length; i++) {
          if (new_inputs[i] !== state.inputs[i]) {
            same = false;
            break;
          }
        }
        result = same
          ? arr
          : Array.lazy(
              arr.shape().slice(),
              arr.dtype(),
              arr.device(),
              state.prim,
              new_inputs,
            );
      }
    }
  }

  cache.set(arr, result);
  return result;
}

// Public entry point: rewrite the DAG rooted at `arr` with fused
// primitives. Returns the (possibly new) Array.
export function fuse(arr: Array): Array {
  return fuse_with(arr, compute_refcounts(arr), new Map());
}

// Re-exported here so prim.ts can avoid the import cycle by going
// through this module if it ever needs to. Currently unused — left as
// a stub for future symmetry.
export type { Primitive };
