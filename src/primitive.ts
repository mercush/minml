import type { Array } from "./array.js";

// A Primitive is the op-specific node attached to a lazy Array. Its single
// job is to dispatch on the output device. eval() is sync — even WebGPU's
// kernel launches are sync (queue.submit returns immediately); only
// device->host readback is async, and that lives outside this trait.
export interface Primitive {
  name(): string;
  eval(inputs: Array[], output: Array): void;
}
