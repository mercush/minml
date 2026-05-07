// Browser entry point for the minml WASM demo on the WebGPU backend.
//
// The build outputs minml_js.{js,wasm} to ../build. If you'd rather keep
// the page self-contained, `cp build/minml_js.* examples/` and change
// the import path below.
//
// initWebGPU() suspends inside WASM (via ASYNCIFY) while it acquires an
// adapter and device through navigator.gpu. tolist() and item() also
// suspend on WebGPU readback (mapAsync), so they're awaited on the JS side.
// Everything else (array construction, add, dot building the lazy graph)
// stays synchronous.
import createMinml from "../build/minml_js.js";

const out = document.getElementById("out")!;
out.textContent = "";
const log = (s: string): void => {
  out.textContent += s + "\n";
};

try {
  if (!navigator.gpu) {
    throw new Error("WebGPU is not available in this browser");
  }

  const m = await createMinml();
  await m.initWebGPU();
  m.setDefaultDevice(m.Device.WebGPU);

  const x = m.array([1, 2, 3, 4], m.Device.WebGPU);
  const y = m.array([10, 20, 30, 40], m.Device.WebGPU);

  log("add -> " + (await m.add(x, y).tolist()).join(", "));
  log("dot -> " + (await m.dot(x, y).item()));
  log(
    "dot(x+y, x+y) -> " +
      (await m.dot(m.add(x, y), m.add(x, y)).item()),
  );
} catch (err) {
  log("error: " + err);
  console.error(err);
}
