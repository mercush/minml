export enum DType {
  Float32 = "Float32",
  Int32 = "Int32",
}

export function dtype_bytes(t: DType): number {
  switch (t) {
    case DType.Float32:
      return 4;
    case DType.Int32:
      return 4;
  }
}
