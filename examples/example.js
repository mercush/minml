// Browser entry point for the minml WASM demo on the WebGPU backend.
//
// The wasm-bindgen output lives at ../crates/minml-wasm/pkg/. `await init()`
// instantiates the wasm module; `await initWebGPU()` acquires the
// adapter+device through wgpu (which uses navigator.gpu under the hood —
// no Asyncify, no spin loop). Readbacks (tolist, item) are real Promises
// driven by Buffer::slice.map_async.
import init, { Device, array, add, dot, initWebGPU, setDefaultDevice, } from "../crates/minml-wasm/pkg/minml_wasm.js";
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
}
catch (err) {
    log("error: " + err);
    console.error(err);
}
