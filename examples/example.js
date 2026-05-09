// Browser entry point for the minml WASM demo on the WebGPU backend.
//
// The wasm-bindgen output lives at ../crates/minml-wasm/pkg/. `await init()`
// instantiates the wasm module; `await initWebGPU()` acquires the
// adapter+device through wgpu (which uses navigator.gpu under the hood —
// no Asyncify, no spin loop). Readbacks (tolist, item) are real Promises
// driven by Buffer::slice.map_async.
import init, { Device, array, add, mul, dot, initWebGPU, jitApply, setDefaultDevice, } from "../crates/minml-wasm/pkg/minml_wasm.js";
const out = document.getElementById("out");
out.textContent = "";
const log = (s) => {
    out.textContent += s + "\n";
};
try {
    if (!navigator.gpu) {
        throw new Error("WebGPU is not available in this browser");
    }
    await init();
    await initWebGPU();
    setDefaultDevice(Device.WebGPU);
    const x = array(new Float32Array([1, 2, 3, 4]), Device.WebGPU);
    const y = array(new Float32Array([10, 20, 30, 40]), Device.WebGPU);
    log("add -> " + (await add(x, y).tolist()).join(", "));
    log("dot -> " + (await dot(x, y).item()));
    log("dot(x+y, x+y) -> " +
        (await dot(add(x, y), add(x, y)).item()));
    // ---- jit: kernel fusion ----
    // Without jit, `mul(add(a, b), add(a, a))` runs as three WGSL dispatches
    // with two intermediate storage buffers. jitApply rewrites the lazy DAG
    // so the whole expression becomes one runtime-generated WGSL kernel and
    // one dispatch — no intermediates touch global memory.
    const fused = jitApply((a, b) => mul(add(a, b), add(a, a)), [x, y]);
    log("jit (x+y)*(x+x) -> " + (await fused.tolist()).join(", "));
    // jit also accepts multiple outputs. The Python binding walks
    // `__dict__` so users can return a class instance whose fields are
    // Arrays (the `Pair` shape in example.py); the wasm binding has no
    // such reflection, so the equivalent is a JS array of Arrays. Each
    // leaf becomes its own fused kernel — one dispatch per output.
    const [pairSum, pairDiff] = jitApply((a, b) => [
        add(mul(a, b), mul(a, a)), // a*b + a*a
        add(mul(a, a), mul(a, a)), // 2 * a*a
    ], [x, y]);
    log("jit pair.sum  -> " + (await pairSum.tolist()).join(", "));
    log("jit pair.diff -> " + (await pairDiff.tolist()).join(", "));
}
catch (err) {
    log("error: " + err);
    console.error(err);
}
