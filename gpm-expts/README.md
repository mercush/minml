# gpm-expts (Deno notebooks)

Scratch space for poking at `minml` interactively from a Deno-kernel
Jupyter notebook.

## One-time setup

1. **Install Deno**:

   ```bash
   brew install deno
   ```

2. **Register the Deno Jupyter kernel** (writes a kernelspec to your user dir):

   ```bash
   deno jupyter --install
   ```

3. **Build minml's TypeScript/WASM bindings** so `build/minml_js.js` exists.
   This requires the Emscripten toolchain (`emcmake`/`emcc`):

   ```bash
   source /path/to/emsdk/emsdk_env.sh
   emcmake cmake -S .. -B ../build \
       -DMINML_BUILD_WEBGPU=ON \
       -DMINML_BUILD_TS=ON
   emmake cmake --build ../build -j
   ```

   Re-run after any change in `src/` or `bindings/ts/`.

   Note: `CMakeLists.txt` sets `-sENVIRONMENT=web,node` so the resulting
   module loads under both browsers and Deno's Node-compatible runtime.

## Running a notebook

Cursor's built-in kernel picker only auto-discovers Python kernels, so for
Deno you need to attach to a running Jupyter server:

```bash
jupyter lab --no-browser --ip=127.0.0.1 --port=8888 --ServerApp.root_dir=..
```

Then in Cursor: kernel picker -> **Existing Jupyter Server** -> paste the
`http://127.0.0.1:8888/lab?token=...` URL printed at startup -> pick the
**Deno** kernel.

Cells use top-level `await` and ESM imports. The last expression of a cell is
auto-rendered, so returning a plain JS object gives you an inspectable view.

## Why an absolute `file://` import?

Deno's Jupyter kernel walks up from CWD looking for `deno.json` /
`package.json` to resolve bare specifiers. Depending on where the kernel
boots, the local `deno.json` here may or may not be picked up -- so the
notebook sidesteps that by importing `build/minml_js.js` via a fully-qualified
`file://` URL. The local `deno.json` here remains useful for `deno run` from
this directory (which does pick up the import map).

## Layout

| File | Purpose |
|------|---------|
| `deno.json` | Import map for `deno run`/`deno test` from this directory. |
| `minml_intro.ipynb` | Tour of minml: tensors, randomness, distributions, autodiff. Cells will fail until minml's surface grows to match -- the notebook documents the gap. |
