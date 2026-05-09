// Node entry point — mirrors examples/example.ts but for stdout instead
// of the DOM. Picks Device.Cuda if the addon is built, otherwise falls
// back to Device.Cpu so it runs anywhere.
import { add, Array, Device, dot, jit, MinmlError, mul, set_default_device, } from "../dist/src/index.js";
function pick_device() {
    try {
        Array.from_f32_1d([1], Device.Cuda);
        return Device.Cuda;
    }
    catch (e) {
        if (e instanceof MinmlError && e.kind === "backend_not_built") {
            return Device.Cpu;
        }
        throw e;
    }
}
const device = pick_device();
set_default_device(device);
console.log("device =", device);
const x = Array.from_f32_1d([1, 2, 3, 4], device);
const y = Array.from_f32_1d([10, 20, 30, 40], device);
console.log("add ->", await add(x, y).tolist());
console.log("dot ->", await dot(x, y).item());
console.log("dot(x+y, x+y) ->", await dot(add(x, y), add(x, y)).item());
// ---- jit: kernel fusion ----
// On CUDA, this is one runtime-compiled kernel + one launch (vs three
// without jit). On CPU, jit is a correctness no-op — the graph is
// walked one primitive at a time.
const fused = jit((a, b) => mul(add(a, b), add(a, a)));
console.log("jit (x+y)*(x+x) ->", await fused(x, y).tolist());
// jit also accepts pytree-style returns — class instances whose
// enumerable fields are Arrays. Same shape goes in, same shape comes
// out, with each leaf separately fused.
class Pair {
    sum;
    diff;
    constructor(sum, diff) {
        this.sum = sum;
        this.diff = diff;
    }
}
const both = jit((a, b) => new Pair(add(mul(a, b), mul(a, a)), // a*b + a*a
add(mul(a, a), mul(a, a))));
const pair = both(x, y);
console.log("jit pair.sum  ->", await pair.sum.tolist());
console.log("jit pair.diff ->", await pair.diff.tolist());
//# sourceMappingURL=example.node.js.map