// Expression AST for fused kernels.
//
// `Expr` is a tree of `add` / `mul` operating on N input arrays. `Plan`
// wraps an Expr with the kernel form ('elementwise' = out[i] = body;
// 'reduce_sum' = out[0] = sum_i body) and the iteration count `size`.
//
// `emit(e, name)` stringifies an Expr using a per-input naming callback,
// e.g. `i => \`in${i}[i]\`` for both WGSL and CUDA C.

export type Expr =
  | { kind: "input"; index: number }
  | { kind: "add"; a: Expr; b: Expr }
  | { kind: "mul"; a: Expr; b: Expr };

export type Plan =
  | { kind: "elementwise"; body: Expr; size: number }
  | { kind: "reduce_sum"; body: Expr; size: number };

export function input(index: number): Expr {
  return { kind: "input", index };
}

export function add(a: Expr, b: Expr): Expr {
  return { kind: "add", a, b };
}

export function mul(a: Expr, b: Expr): Expr {
  return { kind: "mul", a, b };
}

export function emit(e: Expr, name: (idx: number) => string): string {
  switch (e.kind) {
    case "input":
      return name(e.index);
    case "add":
      return `(${emit(e.a, name)} + ${emit(e.b, name)})`;
    case "mul":
      return `(${emit(e.a, name)} * ${emit(e.b, name)})`;
  }
}
