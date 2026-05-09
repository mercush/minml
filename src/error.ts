import { Device } from "./device.js";

export type MinmlErrorKind =
  | "shape_mismatch"
  | "device_mismatch"
  | "dtype_mismatch"
  | "data_size"
  | "dot_requires_1d"
  | "gather_indices_not_int32"
  | "gather_table_rank"
  | "gather_oob"
  | "dirichlet_alpha_not_1d"
  | "categorical_probs_not_1d"
  | "item_requires_size_1"
  | "op_not_implemented"
  | "backend_not_built"
  | "webgpu_not_initialized"
  | "webgpu_init_failed"
  | "webgpu_readback_failed"
  | "vmap"
  | "other";

export class MinmlError extends Error {
  readonly kind: MinmlErrorKind;

  private constructor(kind: MinmlErrorKind, message: string) {
    super(message);
    this.kind = kind;
    this.name = "MinmlError";
  }

  static shape_mismatch(): MinmlError {
    return new MinmlError("shape_mismatch", "shape mismatch");
  }
  static device_mismatch(): MinmlError {
    return new MinmlError("device_mismatch", "device mismatch");
  }
  static dtype_mismatch(): MinmlError {
    return new MinmlError("dtype_mismatch", "dtype mismatch");
  }
  static data_size(got: number, expected: number): MinmlError {
    return new MinmlError(
      "data_size",
      `data size ${got} != product(shape)=${expected}`,
    );
  }
  static dot_requires_1d(): MinmlError {
    return new MinmlError("dot_requires_1d", "dot requires 1-D inputs");
  }
  static gather_indices_not_int32(): MinmlError {
    return new MinmlError(
      "gather_indices_not_int32",
      "gather: indices must be Int32",
    );
  }
  static gather_table_rank(): MinmlError {
    return new MinmlError(
      "gather_table_rank",
      "gather: table must have rank >= 1",
    );
  }
  static gather_oob(): MinmlError {
    return new MinmlError("gather_oob", "gather: index out of bounds");
  }
  static dirichlet_alpha_not_1d(): MinmlError {
    return new MinmlError(
      "dirichlet_alpha_not_1d",
      "dirichlet_sample: alpha must be 1-D",
    );
  }
  static categorical_probs_not_1d(): MinmlError {
    return new MinmlError(
      "categorical_probs_not_1d",
      "categorical_sample: probs must be 1-D",
    );
  }
  static item_requires_size_1(): MinmlError {
    return new MinmlError("item_requires_size_1", "item() requires size==1");
  }
  static op_not_implemented(op: string, device: Device): MinmlError {
    return new MinmlError(
      "op_not_implemented",
      `operation '${op}' not implemented for device ${device}`,
    );
  }
  static backend_not_built(name: string): MinmlError {
    return new MinmlError(
      "backend_not_built",
      `backend '${name}' was not built into this binary`,
    );
  }
  static webgpu_not_initialized(): MinmlError {
    return new MinmlError(
      "webgpu_not_initialized",
      "WebGPU not initialized; call init_webgpu() first",
    );
  }
  static webgpu_init_failed(detail: string): MinmlError {
    return new MinmlError(
      "webgpu_init_failed",
      `WebGPU init failed: ${detail}`,
    );
  }
  static webgpu_readback_failed(): MinmlError {
    return new MinmlError(
      "webgpu_readback_failed",
      "WebGPU readback failed",
    );
  }
  static vmap(detail: string): MinmlError {
    return new MinmlError("vmap", `vmap: ${detail}`);
  }
  static other(message: string): MinmlError {
    return new MinmlError("other", message);
  }
}
