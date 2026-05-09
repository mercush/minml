// CUDA backend on Node, via the N-API addon at build/Release/minml_cuda.node.
//
// The addon is *optional*: if it isn't built (e.g., on macOS or a host
// without the CUDA toolkit), all backend functions throw
// MinmlError.backend_not_built("cuda"). CPU and WebGPU stay fully usable
// without it. Build the addon with `npm run build:cuda` on a CUDA host.
//
// Node-only modules (`module`, `path`, `url`) are imported *dynamically*
// inside a `typeof process !== "undefined"` guard so this file is safe
// to load in a browser — bare specifiers like "module" would otherwise
// fail to resolve and break the whole module graph.

import type { Array } from "../array.js";
import type { Buffer } from "../buffer.js";
import { Device } from "../device.js";
import { MinmlError } from "../error.js";

// Opaque handle to a `minml_cuda_buf*` (Napi::External). Never inspected
// from JS — passed back to the addon for every op.
type CudaHandle = unknown;

interface Addon {
  alloc(bytes: number): CudaHandle;
  h2d(dst: CudaHandle, src: Uint8Array): void;
  d2h(src: CudaHandle, dst: Uint8Array): void;
  add(a: CudaHandle, b: CudaHandle, out: CudaHandle, n: number): void;
  mul(a: CudaHandle, b: CudaHandle, out: CudaHandle, n: number): void;
  dot(a: CudaHandle, b: CudaHandle, out: CudaHandle, n: number): void;
  // jit-fused kernels.
  // `compile` runs NVRTC, caches by source, returns an opaque kernel handle.
  // `launch_elem`/`launch_reduce` call cuLaunchKernel via the driver API.
  // The output buffer's handle is the last entry in `handles`.
  compile(source: string): CudaHandle;
  launch_elem(kernel: CudaHandle, handles: CudaHandle[], n: number): void;
  launch_reduce(kernel: CudaHandle, handles: CudaHandle[], n: number): void;
}

// Try common addon locations. dist/src/cuda is the published path;
// src/cuda is hit when vitest imports source directly. The dynamic
// `await import("module")` etc. only run in Node — in a browser the
// `typeof process` check short-circuits so the module never even tries
// to resolve those bare specifiers.
async function load_addon(): Promise<Addon | null> {
  if (typeof process === "undefined") return null;
  try {
    const { createRequire } = await import("module");
    const { dirname, resolve } = await import("path");
    const { fileURLToPath } = await import("url");
    const here = dirname(fileURLToPath(import.meta.url));
    const candidates = [
      resolve(here, "../../../build/Release/minml_cuda.node"),
      resolve(here, "../../build/Release/minml_cuda.node"),
    ];
    const req = createRequire(import.meta.url);
    for (const c of candidates) {
      try {
        return req(c) as Addon;
      } catch {
        // try next
      }
    }
  } catch {
    // any failure (browser, missing module, etc.) -> no addon.
  }
  return null;
}

// Top-level await: in Node this resolves before any user code runs. In
// the browser the function returns null synchronously (no `process`),
// so this still completes promptly with addon=null.
const addon: Addon | null = await load_addon();

function ctx(): Addon {
  if (addon === null) throw MinmlError.backend_not_built("cuda");
  return addon;
}

// ---- Buffer ----

export class CudaBuffer implements Buffer {
  readonly handle: CudaHandle;
  private readonly _bytes: number;

  constructor(handle: CudaHandle, bytes: number) {
    this.handle = handle;
    this._bytes = bytes;
  }

  bytes(): number {
    return this._bytes;
  }
  device(): Device {
    return Device.Cuda;
  }
}

function as_cuda(b: Buffer): CudaBuffer {
  if (!(b instanceof CudaBuffer)) {
    throw MinmlError.other("expected CudaBuffer");
  }
  return b;
}

// ---- Allocate / copies ----

export function allocate(bytes: number): Buffer {
  const a = ctx();
  return new CudaBuffer(a.alloc(bytes), bytes);
}

export function copy_host_to_buffer(dst: Buffer, src: Uint8Array): void {
  const a = ctx();
  a.h2d(as_cuda(dst).handle, src);
}

// CUDA d2h is sync; wrapped by deviceDispatch.d2h_async into a resolved
// promise.
export function copy_buffer_to_host(src: Buffer, dst: Uint8Array): void {
  const a = ctx();
  a.d2h(as_cuda(src).handle, dst);
}

// ---- Backend ops ----

export function add(arr_a: Array, arr_b: Array, out: Array): void {
  const a = ctx();
  a.add(
    as_cuda(arr_a.buffer()!).handle,
    as_cuda(arr_b.buffer()!).handle,
    as_cuda(out.buffer()!).handle,
    out.size(),
  );
}

export function mul(arr_a: Array, arr_b: Array, out: Array): void {
  const a = ctx();
  a.mul(
    as_cuda(arr_a.buffer()!).handle,
    as_cuda(arr_b.buffer()!).handle,
    as_cuda(out.buffer()!).handle,
    out.size(),
  );
}

export function dot(arr_a: Array, arr_b: Array, out: Array): void {
  const a = ctx();
  a.dot(
    as_cuda(arr_a.buffer()!).handle,
    as_cuda(arr_b.buffer()!).handle,
    as_cuda(out.buffer()!).handle,
    arr_a.size(),
  );
}

// ---- Fused kernels (used by src/cuda/fused.ts) ----

export function compile_kernel(source: string): CudaHandle {
  return ctx().compile(source);
}

export function launch_elem(
  kernel: CudaHandle,
  inputs: Array[],
  out: Array,
  n: number,
): void {
  const a = ctx();
  const handles: CudaHandle[] = inputs.map(
    (arr) => as_cuda(arr.buffer()!).handle,
  );
  handles.push(as_cuda(out.buffer()!).handle);
  a.launch_elem(kernel, handles, n);
}

export function launch_reduce(
  kernel: CudaHandle,
  inputs: Array[],
  out: Array,
  n: number,
): void {
  const a = ctx();
  const handles: CudaHandle[] = inputs.map(
    (arr) => as_cuda(arr.buffer()!).handle,
  );
  handles.push(as_cuda(out.buffer()!).handle);
  a.launch_reduce(kernel, handles, n);
}
