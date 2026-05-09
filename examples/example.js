// Browser entry point for the minml WebGPU demo.
//
// Imports the compiled TS library from ../dist (run `npm run build` in the
// repo root first). Readbacks (tolist, item) are Promises driven by
// GPUBuffer.mapAsync.
import { add, Array, Device, dot, init_webgpu, set_default_device, } from "../dist/src/index.js";
const out = document.getElementById("out");
out.textContent = "";
const log = (s) => {
    out.textContent += s + "\n";
};
try {
    if (!navigator.gpu) {
        throw new Error("WebGPU is not available in this browser");
    }
    await init_webgpu();
    set_default_device(Device.WebGpu);
    const x = Array.from_f32_1d([1, 2, 3, 4], Device.WebGpu);
    const y = Array.from_f32_1d([10, 20, 30, 40], Device.WebGpu);
    log("add -> " + (await add(x, y).tolist()).join(", "));
    log("dot -> " + (await dot(x, y).item()));
    log("dot(x+y, x+y) -> " + (await dot(add(x, y), add(x, y)).item()));
}
catch (err) {
    log("error: " + err);
    console.error(err);
}
//# sourceMappingURL=example.js.map