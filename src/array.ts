// Array — the user-facing tensor.
//
// Either evaluated (data is a Buffer) or lazy (a Primitive plus a list of
// input Arrays). Calling eval() walks the input DAG iteratively in
// post-order, allocates output buffers, and runs each primitive. The only
// async surface is tolist/tolist_int/item, which await on d2h.

import type { Buffer } from "./buffer.js";
import * as device_dispatch from "./deviceDispatch.js";
import { Device } from "./device.js";
import { DType, dtype_bytes } from "./dtype.js";
import { MinmlError } from "./error.js";
import type { Primitive } from "./primitive.js";

type State =
  | { kind: "evaluated"; buffer: Buffer }
  | { kind: "lazy"; prim: Primitive; inputs: Array[] };

// Boxed state so that all clones of the same logical Array see the same
// evaluated buffer once eval() has run on any of them. Mirrors Rust's
// Arc<Mutex<ArrayInner>>.
interface StateBox {
  state: State;
}

function product(shape: number[]): number {
  let p = 1;
  for (const d of shape) p *= d;
  return p;
}

export class Array {
  private readonly _shape: number[];
  private readonly _size: number;
  private readonly _device: Device;
  private readonly _dtype: DType;
  private readonly _batch_axis: number | null;
  private readonly _state_box: StateBox;

  private constructor(
    shape: number[],
    device: Device,
    dtype: DType,
    batch_axis: number | null,
    state_box: StateBox,
  ) {
    this._shape = shape;
    this._size = product(shape);
    this._device = device;
    this._dtype = dtype;
    this._batch_axis = batch_axis;
    this._state_box = state_box;
  }

  // ---- Eager constructors ----

  static from_f32_with_shape(
    data: Float32Array | number[],
    shape: number[],
    device: Device,
  ): Array {
    const size = product(shape);
    const arr = data instanceof Float32Array ? data : new Float32Array(data);
    if (arr.length !== size) {
      throw MinmlError.data_size(arr.length, size);
    }
    const bytes = size * dtype_bytes(DType.Float32);
    const buf = device_dispatch.allocate(device, bytes);
    device_dispatch.h2d(
      device,
      buf,
      new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength),
    );
    return new Array(shape, device, DType.Float32, null, {
      state: { kind: "evaluated", buffer: buf },
    });
  }

  static from_i32_with_shape(
    data: Int32Array | number[],
    shape: number[],
    device: Device,
  ): Array {
    const size = product(shape);
    const arr = data instanceof Int32Array ? data : new Int32Array(data);
    if (arr.length !== size) {
      throw MinmlError.data_size(arr.length, size);
    }
    const bytes = size * dtype_bytes(DType.Int32);
    const buf = device_dispatch.allocate(device, bytes);
    device_dispatch.h2d(
      device,
      buf,
      new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength),
    );
    return new Array(shape, device, DType.Int32, null, {
      state: { kind: "evaluated", buffer: buf },
    });
  }

  static from_f32_1d(data: Float32Array | number[], device: Device): Array {
    const arr = data instanceof Float32Array ? data : new Float32Array(data);
    return Array.from_f32_with_shape(arr, [arr.length], device);
  }

  static from_i32_1d(data: Int32Array | number[], device: Device): Array {
    const arr = data instanceof Int32Array ? data : new Int32Array(data);
    return Array.from_i32_with_shape(arr, [arr.length], device);
  }

  // ---- Lazy constructor ----

  static lazy(
    shape: number[],
    dtype: DType,
    device: Device,
    prim: Primitive,
    inputs: Array[],
  ): Array {
    return new Array(shape, device, dtype, null, {
      state: { kind: "lazy", prim, inputs },
    });
  }

  // ---- Accessors ----

  shape(): number[] {
    return this._shape;
  }
  size(): number {
    return this._size;
  }
  device(): Device {
    return this._device;
  }
  dtype(): DType {
    return this._dtype;
  }
  batch_axis(): number | null {
    return this._batch_axis;
  }

  evaluated(): boolean {
    return this._state_box.state.kind === "evaluated";
  }

  buffer(): Buffer | null {
    const s = this._state_box.state;
    return s.kind === "evaluated" ? s.buffer : null;
  }

  // ---- Vmap-axis tagging ----

  with_batch_axis(axis: number): Array {
    return new Array(
      this._shape,
      this._device,
      this._dtype,
      axis,
      this._state_box,
    );
  }

  strip_batch_axis(): Array {
    return new Array(
      this._shape,
      this._device,
      this._dtype,
      null,
      this._state_box,
    );
  }

  // ---- Eval ----

  // Iterative post-order DFS. No recursion. WebGPU dispatch is sync
  // (queue.submit returns immediately); only tolist/item actually await
  // on d2h.
  eval(): void {
    if (this.evaluated()) return;
    const stack: [Array, boolean][] = [[this, false]];
    while (stack.length > 0) {
      const [node, visited] = stack.pop()!;
      if (node.evaluated()) continue;
      if (!visited) {
        stack.push([node, true]);
        for (const inp of node._inputs_snapshot()) {
          if (!inp.evaluated()) stack.push([inp, false]);
        }
      } else {
        node._run_primitive();
      }
    }
  }

  private _inputs_snapshot(): Array[] {
    const s = this._state_box.state;
    return s.kind === "lazy" ? s.inputs.slice() : [];
  }

  private _run_primitive(): void {
    const s = this._state_box.state;
    if (s.kind === "evaluated") return;
    const { prim, inputs } = s;
    const bytes = this._size * dtype_bytes(this._dtype);
    const buf = device_dispatch.allocate(this._device, bytes);
    // Install on this Array first so the primitive can write through.
    this._state_box.state = { kind: "evaluated", buffer: buf };
    prim.eval(inputs, this);
  }

  // ---- Async readback ----

  async tolist(): Promise<number[]> {
    this.eval();
    const out = new Float32Array(this._size);
    const buf = this.buffer()!;
    await device_dispatch.d2h_async(
      this._device,
      buf,
      new Uint8Array(out.buffer, 0, out.byteLength),
    );
    return [...out];
  }

  async tolist_int(): Promise<number[]> {
    this.eval();
    const out = new Int32Array(this._size);
    const buf = this.buffer()!;
    await device_dispatch.d2h_async(
      this._device,
      buf,
      new Uint8Array(out.buffer, 0, out.byteLength),
    );
    return [...out];
  }

  async item(): Promise<number> {
    if (this._size !== 1) throw MinmlError.item_requires_size_1();
    return (await this.tolist())[0];
  }
}
