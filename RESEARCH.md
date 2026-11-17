# Research notes: running upstream tinygrad in the browser

Research for tinygrad-web, done in November 2026 inside a Linux container with headless Chromium 153. Everything
marked **tested** was run here. Everything else comes from reading source code or package metadata, and says so.

Network note: the container could reach PyPI, npm, GitHub git/raw/release downloads and `api.github.com` search. It
could **not** load `mesozoic-egg.github.io`, `pyodide.org`, `cdn.jsdelivr.net` or the GitHub HTML UI. Those sites were
studied through their git repositories and package metadata instead.

## 1. The existing "tinygrad in the browser" demo (mesozoic-egg)

The site `https://mesozoic-egg.github.io/tinygrad/` is served from the `gh-pages` branch of
[`mesozoic-egg/tinygrad`](https://github.com/mesozoic-egg/tinygrad), a fork of tinygrad. Found with
`gh api search/repositories q=user:mesozoic-egg` and `git ls-remote`.

* **`gh-pages` (commit `6079013`, "Deployed e3a3764 with MkDocs version: 1.6.1", 2025-09-13).** This is a MkDocs build of
  the tinygrad documentation; its `CNAME` is `docs.tinygrad.org`. It ships the stock `markdown-exec` Pyodide helper
  (`assets/_markdown_exec_pyodide.js`: `loadPyodide()` from the CDN, `micropip`, `runPythonAsync`). No page calls
  `setupPyodide(...)`, so the site contains no runnable tinygrad code and nothing that uses WebGPU. Pyodide there
  would only run on the CPU.
* **`browser-device` branch (commits `9bff074`…`1c64f0f`, 2024-10-13/14, author "Mesozoic Egg").** This is the
  actual browser-GPU prototype:
  * `tinygrad/runtime/ops_web.py`: a hand-written WebSocket server (`localhost:8766`) in **native CPython**.
  * `extra/webgpu/index.html`: a page that connects to that socket, calls `navigator.gpu.requestAdapter()` and
    `requestDevice()`, and has a stand-alone Collatz WGSL example. Messages are a small text/binary framing
    (`initDevice`, `PUT_BUFFER`).
  * The branch deletes upstream's `ops_webgpu.py`. The new runtime has no tinygrad `Compiled` device yet: it
    stops at "two way comm", "text and binary data" and "some tweaks for demo".

Answers to the brief's questions:

| Question | Finding |
|---|---|
| 1. How does it run tinygrad inside Pyodide? | It doesn't. tinygrad runs in native CPython; the browser only lends its WebGPU device over a WebSocket. |
| 2. How does Python call the JS WebGPU API? | Through JSON/binary messages over a WebSocket to a page that runs the WebGPU calls. |
| 3. JSPI / Asyncify / other? | None. Python blocks on `socket.recv`, so the network is the sync boundary. |
| 4. tinygrad / Pyodide versions | A tinygrad fork as of Oct 2024 (~0.9.2/0.10 era, using `requestAdapterInfo()`, which has since been removed from browsers). No Pyodide. |
| 5. Open source / license | Yes. MIT, inherited from tinygrad ("Copyright (c) 2024, the tiny corp"). |
| 6. Reusable in a maintained library? | No. It needs a Python server (our hard requirement is no server), is unfinished, and targets a 2-year-old internal API. The useful idea is that Python drives the browser's WebGPU through a narrow command protocol. tinygrad-web's handle-based JS bridge follows the same principle, but in-process. |

### Other prior art found

* **[c4ffein/NNVP](https://github.com/c4ffein/NNVP)** (`nnvp-client-vue/src/lib/TinygradRuntime`, Oct 2026). Pyodide
  plus `micropip.install("tinygrad==0.13.0")` plus `sqlite3`. It traces a training step on tinygrad's GPU-less
  `NULL:WGSL` device, exports a self-contained **JavaScript** WebGPU runner with a vendored `extra/export_model.py`,
  and then loops on that runner in JS: "Python never touches the GPU." That's good for fixed graphs, but it isn't
  interactive tinygrad, and each export is tied to one tinygrad version. **License: CC BY-NC 4.0**, which isn't
  compatible with an MIT library, so no code was reused. It does confirm that tinygrad 0.13 imports in Pyodide.
* **Upstream's own export path** (`extra/export_model.py`, used by tinygrad's WebGPU browser demos such as EfficientNet and
  Stable Diffusion): compile ahead of time to WGSL + JS, and run without Python. Same trade-off as NNVP.
* **[CERBSim/webgpu](https://github.com/CERBSim/webgpu)**: a Python wrapper around WebGPU for Pyodide and local
  Python. License: **LGPL-2.1-or-later**. 588 commits, no releases. Its README doesn't describe the sync mechanism.
  **Not tested**: the license makes bundling into an MIT npm package awkward, and wgpudawn (below) covers the same
  need with a permissive license.

## 2. Upstream tinygrad's WebGPU backend

`tinygrad/runtime/ops_webgpu.py` (v0.14.0, also v0.12/0.13) drives **Dawn through ctypes**. It uses
`tinygrad.runtime.autogen.webgpu`, which loads `libwebgpu_dawn` (installed by `pydawn`/`wgpudawn`). It makes async
WebGPU calls synchronous with `wgpuInstanceWaitAny(..., timedWaitAnyEnable=True)`.

**Tested in Pyodide 314.0.8:** importing upstream's module fails with
`AttributeError: failed to load library webgpu: try setting WEBGPU_PATH?`, and `ctypes.CDLL("libwebgpu_dawn.so")` fails
with `OSError: could not load dynamic lib`. The native backend can't run in Pyodide, as expected: no native Dawn, and
there's no way to block on a GPU future from the browser's main loop.

Reused unchanged from upstream:

| Component | Reused? |
|---|---|
| `Tensor`, autograd, `nn`, optimizers, `TinyJit`, scheduler, codegen | yes, unmodified |
| `tinygrad/renderer/wgsl.py` (`WGSLRenderer`) | yes, unmodified. Its kernels go straight to `GPUDevice.createShaderModule`. |
| Kernel ABI: binding 0 = `INFINITY` uniform, storage buffers for tensors, uniforms for symbolic variables, `@workgroup_size` baked into the shader | yes, the JS bridge implements it |
| `LRUAllocator` (buffer reuse) | yes |
| WEBGPU-specific scheduler rules (`DEVICE_MAX_BUFS["WEBGPU"] = 8`, no buffer views, dtype table) | yes, because the browser device keeps the name `WEBGPU` |
| `ops_webgpu.py` (Dawn/ctypes) | **no**, replaced by `python/tinygrad_web/webgpu.py` + `src/bridge.ts` |
| `ops_cpu.py` (clang/LLVM/x86 asm) | **no**. In wasm, `CPU` is tinygrad's own pure-Python device (`ops_python.PythonDevice`). |

tinygrad finds devices with `importlib.import_module(f"tinygrad.runtime.ops_{name}")`. tinygrad-web puts its modules
into `sys.modules` under those names, so **no tinygrad file is modified**.

## 3. wgpudawn

[`wgpudawn`](https://pypi.org/project/wgpudawn/) 0.32.0.4 (BSD-2-Clause; authors Mark Harfouche, Almar Klein, Korijn van
Golen; status "Early"; first release 2026-10-13). It's the wgpu-py API on top of Dawn. Besides native wheels it publishes
`wgpudawn-0.32.0.4-cp314-cp314-pyemscripten_2026_0_wasm32.whl`, which runs **in Pyodide 314** on the browser's WebGPU through
Dawn's *Emdawnwebgpu* and needs **JSPI**.

**Tested** (Pyodide 314.0.8, Chromium 153, SwiftShader adapter): after `loadPackage(["numpy", "cffi", <wheel>])`,
`wgpu.gpu.request_adapter_sync()`, `request_device_sync()`, buffer creation, a WGSL compute pipeline, dispatch and
`queue.read_buffer()` all worked. A vector add returned `[5. 7. 9.]`. Import took 0.12 s.

It's a viable bridge, but tinygrad-web doesn't use it as its primary path:

* It doesn't expose the `webgpu.h` ctypes ABI that upstream's `ops_webgpu.py` expects. We would still have to write a
  tinygrad device against the wgpu-py API, about the same amount of code as against the JS API.
* Every call would go through Python → cffi → wasm C (Emdawnwebgpu) → JS. The direct route is Python → JS.
* It brings in cffi and an early-stage binary package, and only supports Pyodide 314.
* It needs JSPI too, so it doesn't widen browser support.

Reconsider it if a later tinygrad release makes its WEBGPU backend target the wgpu-py API, or if wgpudawn starts
exposing the C ABI to ctypes. Then upstream's backend could run in Pyodide with no tinygrad-web device code at all.

## 4. Bridging synchronous Python and asynchronous WebGPU

tinygrad's API is synchronous (`.numpy()`, `.item()`, `Device.synchronize()`), but WebGPU readback (`mapAsync`), queue
completion (`onSubmittedWorkDone`) and error scopes return promises.

| Option | Assessment |
|---|---|
| **JSPI** (`pyodide.ffi.run_sync`) | **Chosen.** Documented Pyodide API and a standardized WebAssembly feature. Suspends the wasm stack while a promise settles; ~0.1 ms per round trip (measured). Needs a JSPI-capable browser and code started via `runPythonAsync`/`eval_code_async` (the worker does this). |
| Asyncify | Pyodide isn't built with Asyncify. It would need a custom Pyodide build with a large size and speed cost. Rejected. |
| SharedArrayBuffer + `Atomics.wait` + a second "GPU worker" | Works without JSPI, but needs cross-origin isolation, and every GPU call becomes a cross-thread message (the GPUDevice must live in the other worker). A possible future fallback for browsers without JSPI. **Not implemented.** |
| Async Python API (`await t.numpy_async()`) | Breaks tinygrad's API. Rejected. |

Only synchronization points block: readback, `synchronize`, and pipeline compilation (`createComputePipelineAsync`, so
WGSL errors become Python exceptions). Dispatches and uploads are synchronous JS calls recorded into a batched compute pass.

**Browser support (as of this writing).** JSPI and WebGPU-in-workers were **verified only in Chromium 153** (this environment).
JSPI shipped by default in Chrome/Edge 137. Check https://webassembly.org/features/ ("JS Promise Integration") and MDN's
WebGPU compatibility table for Firefox and Safari. They were **not tested here**. If JSPI or a WebGPU adapter is missing,
`createTinygrad({device: "webgpu"})` fails with a descriptive error, and `device: "auto"` falls back to the interpreted
CPU device.

**WebGPU in workers:** the spec exposes `navigator.gpu` in dedicated workers (`WorkerNavigator.gpu`). Verified in Chromium. We
did not implement a main-thread fallback, because running Python on the main thread would freeze the page during compute.

## 5. Versions

| Component | Version | Notes |
|---|---|---|
| Pyodide | 314.0.8 (Python 3.14.2, emscripten 5.0.3, ABI `pyemscripten_2026_0`) | numpy 2.4.6 from the same distribution |
| tinygrad (default) | 0.14.0 | Latest PyPI release. `Tensor(..., requires_grad=...)` and `Tensor.train()` are gone upstream (use `Context(TRAINING=1)`). |
| tinygrad (also supported) | 0.12.0 | Different device API (`CompilerSet`, `Program(dev, name, lib)`), which the backend detects. Still has `requires_grad=` and `Tensor.train()`. |
| tinygrad 0.13.0 | not tested | |

Findings while porting (all handled without editing tinygrad's files):

1. tinygrad ≥ 0.13 compiles kernels in a `multiprocessing` pool (`PARALLEL`), and Pyodide has no `_multiprocessing`. Fixed with
   tinygrad's own knob: `PARALLEL=0`.
2. tinygrad ≥ 0.13 realizes device-less tensors (e.g. `Tensor.arange(...)` combined only with Python scalars) on a hard-coded
   `"CPU"` device. That device needs clang/LLVM, and its x86 assembler backend rejects `wasm32`. tinygrad-web provides `CPU` as
   tinygrad's Python interpreter device, and adds one documented runtime patch so device-less tensors are realized on
   `Device.DEFAULT`. See [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md).
3. tinygrad's RNG (`Tensor.randn`) goes through Box–Muller (`log`, `cos`). WGSL leaves the precision of transcendental
   functions implementation-defined, so `randn` output on WebGPU differs from CPU by up to ~7e-4 (measured). Matmuls of exact
   inputs (1×1 up to 130×130) matched native CPU bit for bit (max abs error 0). The other exact-input ops in the tests stay
   within rtol 1e-4 / atol 1e-5.
