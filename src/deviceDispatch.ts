// Internal dispatch helpers — call the right backend for allocate / h2d /
// d2h.

import type { Buffer } from "./buffer.js";
import * as cpu_backend from "./cpu/backend.js";
import * as cuda from "./cuda/backend.js";
import { Device } from "./device.js";
import * as webgpu from "./webgpu/backend.js";

export function allocate(d: Device, bytes: number): Buffer {
  switch (d) {
    case Device.Cpu:
      return cpu_backend.allocate(bytes);
    case Device.Cuda:
      return cuda.allocate(bytes);
    case Device.WebGpu:
      return webgpu.allocate(bytes);
  }
}

export function h2d(d: Device, dst: Buffer, src: Uint8Array): void {
  switch (d) {
    case Device.Cpu:
      cpu_backend.copy_host_to_buffer(dst, src);
      return;
    case Device.Cuda:
      cuda.copy_host_to_buffer(dst, src);
      return;
    case Device.WebGpu:
      webgpu.copy_host_to_buffer(dst, src);
      return;
  }
}

// Async d2h. CPU and CUDA finish synchronously and resolve immediately;
// WebGPU drives a real mapAsync.
export async function d2h_async(
  d: Device,
  src: Buffer,
  dst: Uint8Array,
): Promise<void> {
  switch (d) {
    case Device.Cpu:
      cpu_backend.copy_buffer_to_host(src, dst);
      return;
    case Device.Cuda:
      cuda.copy_buffer_to_host(src, dst);
      return;
    case Device.WebGpu:
      await webgpu.copy_buffer_to_host_async(src, dst);
      return;
  }
}
