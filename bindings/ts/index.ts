// bindings/ts/index.ts
//
// Thin re-export shim. Awaits createMinml() once at module load and
// re-exposes the embind module's properties (PRNGKey, Dirichlet, etc.) as
// named ESM exports. This file exists only because ESM `import { X }`
// can't pull names out of a module-instance object — the embind module
// returns one big object rather than itself being an ESM with named
// exports.
//
// Adjust the relative path if you move the build directory.
import createMinml from "../../build/minml_js.js";

export const m = await createMinml();

// Re-export the constructors and enums as named values.
export const Device = m.Device;
export const DType = m.DType;
export const PRNGKey = m.PRNGKey;
export const Dirichlet = m.Dirichlet;
export const Categorical = m.Categorical;
export const Normal = m.Normal;

// vmap sugar: turn `(...args) => m.vmapApply(f, in_axes, args)` into
// `vmap(f, in_axes)(...args)` so it reads naturally in the notebook.
export const vmap = (f: Function, in_axes: number[]) =>
  (...args: unknown[]) => m.vmapApply(f, in_axes, args);

// Tensor is the embind Array class. Re-exported under the more
// notebook-friendly name (avoids shadowing the JS built-in Array).
export const Tensor = (m as unknown as { Array: unknown }).Array;
