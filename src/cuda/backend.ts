// CUDA backend on Node, via the N-API addon at build/Release/minml_cuda.node.
//
// The addon is *optional*: if it isn't built (e.g., on macOS or a host
// without the CUDA toolkit), all backend functions throw
// MinmlError.backend_not_built("cuda"). CPU and WebGPU stay fully usable
// without it. Build the addon with `npm run build:cuda` on a CUDA host.

import { createRequire } from "module";
import { dirname, resolve } from "path";
import { fileURLToPath } from "url";

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
}

// Try common addon locations. dist/src/cuda is the published path;
// src/cuda is hit when vitest imports source directly.
function load_addon(): Addon | null {
  if (typeof process === "undefined") return null;
  let here: string;
  try {
    here = dirname(fileURLToPath(import.meta.url));
  } catch {
    return null;
  }
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
  return null;
}

const addon = load_addon();

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
