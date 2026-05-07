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
    readonly Int32: DType;
  }

  // tolist/item return Promises because, on WebGPU, readback suspends the
  // WASM stack (ASYNCIFY) waiting on wgpuBufferMapAsync. On CPU the call
  // resolves immediately, so `await` is a no-op and the same shape works.
  export interface MinmlArray {
    size(): number;
    shape(): number[];
    device(): Device;
    dtype(): DType;
    eval(): void;
    tolist(): Promise<number[]>;
    item(): Promise<number>;
  }

  export interface PRNGKey {
    split(n: number): PRNGKey[];
    k0(): number;
    k1(): number;
  }

  export interface PRNGKeyCtor {
    new(seed: number): PRNGKey;
  }

  // Distribution classes hold their parameters and dispatch sample() to a
  // C++ primitive.
  export interface Dirichlet {
    sample(key: PRNGKey, batch_shape: number[]): MinmlArray;
  }
  export interface DirichletCtor {
    new(alpha: MinmlArray): Dirichlet;
  }

  export interface Categorical {
    sample(key: PRNGKey, batch_shape: number[]): MinmlArray;
  }
  export interface CategoricalCtor {
    new(probs: MinmlArray): Categorical;
  }

  export interface Normal {
    sample(key: PRNGKey, batch_shape: number[]): MinmlArray;
  }
  export interface NormalCtor {
    new(): Normal;
  }

  export interface MinmlModule {
    Device: DeviceEnum;
    DType: DTypeEnum;
    PRNGKey: PRNGKeyCtor & { new: (seed: number) => PRNGKey };
    Dirichlet: DirichletCtor;
    Categorical: CategoricalCtor;
    Normal: NormalCtor;

    array(data: number[], device: Device): MinmlArray;
    arrayInt(data: number[], device: Device): MinmlArray;
    add(a: MinmlArray, b: MinmlArray): MinmlArray;
    mul(a: MinmlArray, b: MinmlArray): MinmlArray;
    dot(a: MinmlArray, b: MinmlArray): MinmlArray;
    ones(shape: number[], dtype: DType, device: Device): MinmlArray;
    randint(k0: number, k1: number, low: number, high: number,
            shape: number[], device: Device): MinmlArray;
    gather(table: MinmlArray, indices: MinmlArray): MinmlArray;
    dirichletSample(k0: number, k1: number, alpha: MinmlArray,
                    batch_shape: number[]): MinmlArray;
    categoricalSample(k0: number, k1: number, probs: MinmlArray,
                      batch_shape: number[]): MinmlArray;
    // vmap orchestration: take a JS callable, in_axes, args; iterate per
    // batch element; stack outputs (recursing into pytree-shaped returns).
    vmapApply(f: Function, in_axes: number[], args: unknown[]): unknown;
    setDefaultDevice(d: Device): void;
    defaultDevice(): Device;
    // Acquires a WebGPU adapter+device inside WASM and registers it with
    // the runtime. Suspends until both navigator.gpu requests resolve.
    initWebGPU(): Promise<void>;
  }

  const createMinml: () => Promise<MinmlModule>;
  export default createMinml;
}
