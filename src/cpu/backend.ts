import type { Buffer } from "../buffer.js";
import { Device } from "../device.js";
import { MinmlError } from "../error.js";

// Plain heap-allocated bytes. JS is single-threaded so no locking is
// needed (Rust used parking_lot::RwLock to allow multiple read views of
// the same buffer when the same Array is passed twice to an op like
// dot(xy, xy)).
export class CpuBuffer implements Buffer {
  readonly data: ArrayBuffer;
  private readonly _bytes: number;

  constructor(bytes: number) {
    this._bytes = bytes;
    this.data = new ArrayBuffer(bytes);
  }

  bytes(): number {
    return this._bytes;
  }
  device(): Device {
    return Device.Cpu;
  }
}

export function allocate(bytes: number): Buffer {
  return new CpuBuffer(bytes);
}

function as_cpu(b: Buffer): CpuBuffer {
  if (!(b instanceof CpuBuffer)) {
    throw MinmlError.other("expected CpuBuffer");
  }
  return b;
}

export function copy_host_to_buffer(dst: Buffer, src: Uint8Array): void {
  const cpu = as_cpu(dst);
  if (src.length > cpu.bytes()) {
    throw MinmlError.other(
      `h2d size mismatch: src=${src.length} dst=${cpu.bytes()}`,
    );
  }
  new Uint8Array(cpu.data).set(src);
}

export function copy_buffer_to_host(src: Buffer, dst: Uint8Array): void {
  const cpu = as_cpu(src);
  if (dst.length > cpu.bytes()) {
    throw MinmlError.other(
      `d2h size mismatch: src=${cpu.bytes()} dst=${dst.length}`,
    );
  }
  dst.set(new Uint8Array(cpu.data, 0, dst.length));
}

// Typed-array views over the underlying ArrayBuffer. Multiple views over
// the same buffer share memory — same semantics as Rust's bytemuck::cast_slice.
export function f32_view(buf: Buffer): Float32Array {
  return new Float32Array(as_cpu(buf).data);
}

export function i32_view(buf: Buffer): Int32Array {
  return new Int32Array(as_cpu(buf).data);
}
