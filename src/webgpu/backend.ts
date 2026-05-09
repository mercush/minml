// WebGPU backend on the browser-native navigator.gpu API.
//
// init() acquires a device. Pipelines for add/mul/dot are lazily compiled
// and cached. Buffer reads are async (mapAsync); kernel dispatches are
// sync (queue.submit returns immediately).

import type { Array } from "../array.js";
import type { Buffer } from "../buffer.js";
import { Device } from "../device.js";
import { MinmlError } from "../error.js";
import { ADD_WGSL, DOT_WGSL, MUL_WGSL } from "./shaders.js";

interface Backend {
  device: GPUDevice;
  queue: GPUQueue;
  pipelines: Map<string, GPUComputePipeline>;
}

let GLOBAL: Backend | null = null;

function ctx(): Backend {
  if (GLOBAL === null) throw MinmlError.webgpu_not_initialized();
  return GLOBAL;
}

export async function init(): Promise<void> {
  if (GLOBAL !== null) return;
  if (typeof navigator === "undefined" || !navigator.gpu) {
    throw MinmlError.webgpu_init_failed("navigator.gpu is not available");
  }
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (!adapter) {
    throw MinmlError.webgpu_init_failed("requestAdapter returned null");
  }
  const device = await adapter.requestDevice({ label: "minml-device" });
  install(device);
}

// Allow the binding layer (or a test harness) to install a pre-built device.
export function install(device: GPUDevice): void {
  if (GLOBAL !== null) return;
  GLOBAL = {
    device,
    queue: device.queue,
    pipelines: new Map(),
  };
}

// ---- Buffer ----

export class WebGpuBuffer implements Buffer {
  readonly handle: GPUBuffer;
  private readonly _bytes: number;

  constructor(handle: GPUBuffer, bytes: number) {
    this.handle = handle;
    this._bytes = bytes;
  }

  bytes(): number {
    return this._bytes;
  }
  device(): Device {
    return Device.WebGpu;
  }
}

function as_wgpu(b: Buffer): WebGpuBuffer {
  if (!(b instanceof WebGpuBuffer)) {
    throw MinmlError.other("expected WebGpuBuffer");
  }
  return b;
}

function next_multiple_of_4(n: number): number {
  return (n + 3) & ~3;
}

// ---- Allocate / copies ----

export function allocate(bytes: number): Buffer {
  const c = ctx();
  // wgpu requires buffer size > 0 and a multiple of 4 for COPY_DST. The
  // CPU semantics let n bytes be anything; we round up the allocation.
  const padded = next_multiple_of_4(Math.max(bytes, 4));
  const handle = c.device.createBuffer({
    size: padded,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
    mappedAtCreation: false,
  });
  return new WebGpuBuffer(handle, bytes);
}

export function copy_host_to_buffer(dst: Buffer, src: Uint8Array): void {
  const c = ctx();
  const buf = as_wgpu(dst);
  // writeBuffer requires src length to be a multiple of 4 (COPY_DST
  // alignment). Pad if needed.
  const padded_len = next_multiple_of_4(src.length);
  const data = padded_len === src.length ? src : (() => {
    const tmp = new Uint8Array(padded_len);
    tmp.set(src);
    return tmp;
  })();
  c.queue.writeBuffer(buf.handle, 0, data as Uint8Array<ArrayBuffer>);
}

// Async readback. Stages into a MapRead buffer, copies, maps, memcpys
// into dst. mapAsync is already a Promise so no oneshot channel needed.
export async function copy_buffer_to_host_async(
  src: Buffer,
  dst: Uint8Array,
): Promise<void> {
  const c = ctx();
  const src_buf = as_wgpu(src);
  const bytes = dst.length;
  const padded = next_multiple_of_4(Math.max(bytes, 4));
  const staging = c.device.createBuffer({
    label: "minml-readback-staging",
    size: padded,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    mappedAtCreation: false,
  });
  const enc = c.device.createCommandEncoder();
  enc.copyBufferToBuffer(src_buf.handle, 0, staging, 0, padded);
  c.queue.submit([enc.finish()]);

  await staging.mapAsync(GPUMapMode.READ, 0, padded);
  const mapped = staging.getMappedRange(0, padded);
  dst.set(new Uint8Array(mapped, 0, bytes));
  staging.unmap();
}

// ---- Pipeline cache ----

function pipeline(name: string): GPUComputePipeline {
  let wgsl: string;
  switch (name) {
    case "add":
      wgsl = ADD_WGSL;
      break;
    case "mul":
      wgsl = MUL_WGSL;
      break;
    case "dot":
      wgsl = DOT_WGSL;
      break;
    default:
      throw MinmlError.other(`unknown pipeline: ${name}`);
  }
  return compute_pipeline(wgsl, name);
}

// Source-keyed pipeline cache. Used both by the static add/mul/dot kernels
// (via `pipeline(name)`) and by the jit's fused-kernel emitter (which
// generates WGSL on the fly).
export function compute_pipeline(
  wgsl: string,
  label: string,
): GPUComputePipeline {
  const c = ctx();
  const cached = c.pipelines.get(wgsl);
  if (cached) return cached;
  const module = c.device.createShaderModule({ label, code: wgsl });
  const pipe = c.device.createComputePipeline({
    label,
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  c.pipelines.set(wgsl, pipe);
  return pipe;
}

// Internal helper for fused.ts so it can build dynamic bind groups
// without re-implementing context plumbing.
export function get_device_and_queue(): { device: GPUDevice; queue: GPUQueue } {
  const c = ctx();
  return { device: c.device, queue: c.queue };
}

function dispatch(
  kernel: string,
  a: GPUBuffer,
  b: GPUBuffer,
  out: GPUBuffer,
  workgroups: number,
): void {
  const c = ctx();
  const pipe = pipeline(kernel);
  const layout = pipe.getBindGroupLayout(0);
  const bg = c.device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: { buffer: a } },
      { binding: 1, resource: { buffer: b } },
      { binding: 2, resource: { buffer: out } },
    ],
  });
  const enc = c.device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipe);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(workgroups, 1, 1);
  pass.end();
  c.queue.submit([enc.finish()]);
}

// ---- Backend ops ----

export function add(a: Array, b: Array, out: Array): void {
  const buf_a = a.buffer()!;
  const buf_b = b.buffer()!;
  const buf_o = out.buffer()!;
  const n = out.size();
  const wg = Math.ceil(n / 64);
  dispatch("add", as_wgpu(buf_a).handle, as_wgpu(buf_b).handle, as_wgpu(buf_o).handle, wg);
}

export function mul(a: Array, b: Array, out: Array): void {
  const buf_a = a.buffer()!;
  const buf_b = b.buffer()!;
  const buf_o = out.buffer()!;
  const n = out.size();
  const wg = Math.ceil(n / 64);
  dispatch("mul", as_wgpu(buf_a).handle, as_wgpu(buf_b).handle, as_wgpu(buf_o).handle, wg);
}

export function dot(a: Array, b: Array, out: Array): void {
  const c = ctx();
  const buf_a = a.buffer()!;
  const buf_b = b.buffer()!;
  const buf_o = out.buffer()!;
  // Output is a single f32 atomic; zero before kernel runs (kernel does
  // atomic adds into out[0]).
  const zero = new Uint8Array(4);
  c.queue.writeBuffer(as_wgpu(buf_o).handle, 0, zero as Uint8Array<ArrayBuffer>);
  const n = a.size();
  const wg = Math.ceil(n / 64);
  dispatch("dot", as_wgpu(buf_a).handle, as_wgpu(buf_b).handle, as_wgpu(buf_o).handle, wg);
}
