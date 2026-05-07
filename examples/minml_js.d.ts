// Ambient type declarations for the embind-generated minml WASM module.
// embind does not emit .d.ts files, so we describe the surface by hand to
// match bindings/ts/bind.cpp.

declare module "*/minml_js.js" {
  export type Device = { readonly value: number };
  export type DType = { readonly value: number };

  export interface DeviceEnum {
    readonly CPU: Device;
    readonly CUDA: Device;
    readonly WebGPU: Device;
  }

  export interface DTypeEnum {
    readonly Float32: DType;
  }

  // Renamed from "Array" so it doesn't shadow the JS built-in.
  // tolist/item return Promises because, on WebGPU, readback suspends the
  // WASM stack (ASYNCIFY) waiting on wgpuBufferMapAsync. On CPU the call
  // resolves immediately, so `await` is a no-op and the same shape works.
  export interface MinmlArray {
    size(): number;
    device(): Device;
    dtype(): DType;
    eval(): void;
    tolist(): Promise<number[]>;
    item(): Promise<number>;
  }

  export interface MinmlModule {
    Device: DeviceEnum;
    DType: DTypeEnum;
    array(data: number[], device: Device): MinmlArray;
    add(a: MinmlArray, b: MinmlArray): MinmlArray;
    mul(a: MinmlArray, b: MinmlArray): MinmlArray;
    dot(a: MinmlArray, b: MinmlArray): MinmlArray;
    setDefaultDevice(d: Device): void;
    // Acquires a WebGPU adapter+device inside WASM and registers it with
    // the runtime. Suspends until both navigator.gpu requests resolve.
    initWebGPU(): Promise<void>;
  }

  const createMinml: () => Promise<MinmlModule>;
  export default createMinml;
}
