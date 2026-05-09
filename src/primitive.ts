import type { Array } from "./array.js";

export type FusionClass = "elementwise" | "reduction" | "opaque";

// A Primitive is the op-specific node attached to a lazy Array. Its single
// job is to dispatch on the output device. eval() is sync — even WebGPU's
// kernel launches are sync (queue.submit returns immediately); only
// device->host readback is async, and that lives outside this trait.
//
// fusion_class is consulted by the jit transform: 'elementwise' prims can
// be absorbed into a fused kernel; 'reduction' prims can sit at the root
// of a fused subgraph (with elementwise prologue); 'opaque' prims are
// hard boundaries. Default 'opaque' if not implemented.
export interface Primitive {
  name(): string;
  eval(inputs: Array[], output: Array): void;
  fusion_class?(): FusionClass;
}

export function fusion_class_of(p: Primitive): FusionClass {
  return p.fusion_class ? p.fusion_class() : "opaque";
}
