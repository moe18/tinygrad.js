# tinygrad-web

Run the **real, upstream [tinygrad](https://github.com/tinygrad/tinygrad)** in the browser. Python runs in
**[Pyodide](https://pyodide.org)** inside a Web Worker, and tinygrad's kernels run as **WGSL compute shaders on the
browser's WebGPU** device. No Python server, CUDA or native install.

```ts
import { createTinygrad } from "tinygrad-web";

const runtime = await createTinygrad({ device: "webgpu" });
const result = await runtime.run(`
from tinygrad import Tensor
a = Tensor.randn(100, 100)
b = Tensor.randn(100, 100)
c = a @ b
print(c.numpy())
`);
console.log(result.stdout);
await runtime.destroy();
```

This isn't a reimplementation. tinygrad-web installs the unmodified tinygrad wheel from PyPI and adds a browser
`WEBGPU` device to it. `Tensor`, autograd, `nn`, the optimizers, `TinyJit`, the scheduler and the WGSL code generator
are all upstream code.

```
browser app ──► tinygrad-web (TypeScript API) ──► Web Worker
                                                     │
                                         Pyodide (CPython 3.14 → wasm)
                                                     │
                                  upstream tinygrad (PyPI wheel, unmodified)
                                                     │  WGSLRenderer → WGSL kernels
                                tinygrad_web.webgpu (WEBGPU device, Python)
                                                     │  pyodide FFI, integer handles, JSPI for awaits
                                      src/bridge.ts (JS WebGPU bridge)
                                                     │
                                      navigator.gpu ──► your GPU
```

## Status

What is verified, and where (see [Testing](#testing)):

| | |
|---|---|
| ✅ verified | Chromium 153 (headless), WebGPU through Dawn's **SwiftShader** adapter: the real WebGPU API and real WGSL compute shaders, executed on the CPU because the test machine has no GPU. 55 Playwright tests pass: 35 on tinygrad 0.14.0 (including the demo) and 20 on tinygrad 0.12.0 (1 skipped because it doesn't apply to 0.12). Results are compared against native upstream tinygrad on CPU. |
| ⚠️ not verified | Hardware GPUs, Firefox, Safari, mobile browsers. The code uses only standard WebGPU + JSPI and should work wherever both exist, but this hasn't been tested. |
| ❌ not supported | Browsers without JSPI (WebAssembly JavaScript Promise Integration). See [Browser support](#browser-support). |

## Install

```sh
npm install tinygrad-web
```

The package contains the TypeScript API (`dist/index.js`), the worker (`dist/worker.js`) and the pinned tinygrad wheels
(`dist/assets/*.whl`). Pyodide itself (~16 MB with numpy) loads at runtime, from the jsDelivr CDN by default, or from
your own server (see [Self-hosting Pyodide](#self-hosting-pyodide)).

### Bundlers

The library finds its worker and the tinygrad wheel with `new URL("...", import.meta.url)`, which Vite, webpack 5,
Rollup and esbuild handle. With **Vite**, exclude it from dependency pre-bundling:

```ts
// vite.config.ts
export default defineConfig({
  optimizeDeps: { exclude: ["tinygrad-web"] },
  server: { headers: { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" } },
});
```

The COOP/COEP headers are optional but recommended (see [Stopping execution](#stopping-execution)). If your setup can't
resolve the files, pass `workerURL` and `tinygradWheelURL` explicitly.

## API

```ts
createTinygrad(options?: CreateTinygradOptions): Promise<TinygradRuntime>
checkSupport(): Promise<{ webgpu, adapter, jspi, crossOriginIsolated, reason? }>

runtime.run(code: string, options?: { timeoutMs?, onStdout?, onStderr? }): Promise<RunResult>
runtime.interrupt(graceMs = 3000): Promise<"signal" | "terminate" | "idle">
runtime.restart(): Promise<RuntimeInfo>     // fresh worker + Python environment
runtime.reset(): Promise<void>              // clear user globals, keep the worker
runtime.getDeviceInfo(): RuntimeInfo
runtime.destroy(): Promise<void>
```

**`createTinygrad` options:**

| option | default | |
|---|---|---|
| `device` | `"webgpu"` | `"webgpu"`: fail if WebGPU/JSPI is unavailable. `"auto"`: fall back to the CPU device. `"cpu"`: tinygrad's pure-Python CPU device (slow). |
| `tinygradVersion` | `"0.14.0"` | `"0.14.0"` or `"0.12.0"` (bundled wheels). See [Compatibility](#tinygrad-versions). |
| `pyodideIndexURL` | jsDelivr `v314.0.8/full/` | Base URL of a Pyodide 314.0.8 distribution that includes numpy. |
| `tinygradWheelURL` | bundled wheel | Use another tinygrad wheel. |
| `workerURL` | bundled `worker.js` | |
| `packages` | `[]` | Extra Pyodide packages or wheel URLs to load at startup. |
| `network` | `"none"` | `"none"` removes `fetch`/XHR/WebSocket/… from the worker after startup. `"allow"` keeps them. See [Safety](#safety-and-isolation). |
| `maxGpuMemoryBytes` | unlimited | Cap on GPU buffer memory. Exceeding it raises `MemoryError` in Python. |
| `timeoutMs` | `0` (none) | Default per-`run()` time limit. |
| `initTimeoutMs` | `120000` | |
| `onStdout`, `onStderr`, `onStatus` | | Streaming callbacks. |

**`RunResult`:** `{ ok, status: "completed" | "error" | "interrupted" | "terminated", stdout, stderr, error?: { type, message,
traceback }, result?, durationMs, gpu: { kernelsCompiled, kernelsDispatched, bytesUploaded, bytesDownloaded, readbacks } | null }`.
`run()` never rejects because of a Python error; check `ok`. `result` is the `repr()` of a trailing expression.

**`RuntimeInfo`:** tinygrad device (`"WEBGPU"`/`"PYTHON"`), adapter info (vendor, architecture, `isFallbackAdapter`, features,
limits), tinygrad/Pyodide/Python versions, `syncMode`, `interruptMode`, init timings, and `diagnostics` (fallbacks, software
adapter warnings, active tinygrad patches).

Python state persists between `run()` calls: variables, imports and compiled kernels. Top-level `await` works.

### Inside Python

```python
from tinygrad import Tensor, Device   # Device.DEFAULT == "WEBGPU"
import tinygrad_web
tinygrad_web.stats()          # kernels compiled/dispatched, bytes moved, live GPU memory, bind-group cache hits
tinygrad_web.device_info()    # adapter info and limits
```

`Device["CPU"]` exists too. In the browser it is tinygrad's pure-Python interpreter device, which is correct but very
slow (see [Benchmarks](#benchmarks)).

## Examples

* [`examples/vite-demo`](examples/vite-demo): Vite + TypeScript app with an editor, Run/Stop, console, device info,
  timings and preloaded examples (tensor basics, matmul throughput, autograd, MLP training, generated WGSL, WebGPU vs CPU).
  ```sh
  npm install && npm run fetch:assets -- --pyodide && npm run build
  cd examples/vite-demo && npm install && npm run dev     # http://localhost:5173
  ```
* [`examples/python/mlp_training.py`](examples/python/mlp_training.py): trains an MLP with `nn.Linear`, `Adam` and `TinyJit`
  on a synthetic dataset. It runs both in the browser and natively.

## Browser support

tinygrad-web needs **WebGPU in a dedicated worker** and **JSPI**. JSPI lets Python block on WebGPU's asynchronous readback
via Pyodide's `run_sync`. Without JSPI, tinygrad's synchronous API (`.numpy()`, `.item()`) can't wait for the GPU.

* Verified: Chromium 153 on Linux (headless, SwiftShader). Chrome/Edge have shipped JSPI by default since version 137.
* Firefox and Safari: not verified. Check current JSPI status at https://webassembly.org/features/. Use `checkSupport()` to
  decide at runtime, and `device: "auto"` to fall back to the CPU device.
* WebGPU needs a secure context (HTTPS or `localhost`).

## Stopping execution

* **Cross-origin isolated pages** (COOP `same-origin` + COEP `require-corp`): `interrupt()` raises `KeyboardInterrupt`
  through a SharedArrayBuffer. The Python environment survives (measured latency ~20–40 ms; tested on infinite loops and
  GPU training loops).
* **Otherwise**, `interrupt()` terminates the worker and starts a fresh one (~3 s). All Python state is lost, and the
  interrupted `run()` resolves with `status: "terminated"`.
* `timeoutMs` uses the same mechanism and reports `error.type === "TimeoutError"`.
* If Python doesn't respond to the signal within `graceMs` (e.g. stuck inside one very long GPU wait), the worker is
  terminated.

## Safety and isolation

tinygrad-web is designed to run user-submitted code inside a larger app, but **a Web Worker is not a security boundary
equivalent to a server-side sandbox.** Code in the worker runs with your page's origin.

* Python runs in a dedicated worker. It can't touch the DOM, `localStorage`, cookies or your page's JS objects, and only
  sees what you send in messages. Don't put secrets in the worker's environment or in code you pass to `run()`.
* `network: "none"` (default) removes `fetch`, `XMLHttpRequest`, `WebSocket`, `EventSource`, `importScripts`, `Worker`,
  `SharedWorker`, `WebTransport` and `caches` from the worker scope after startup. This is **best effort**: it can't block
  everything (e.g. dynamic `import()` through `js.eval`). For real enforcement, serve `worker.js` with a
  `Content-Security-Policy` such as `connect-src 'none'` (workers take their CSP from their own script response), and/or
  host the runtime in a sandboxed cross-origin iframe.
* Runaway code: `timeoutMs`, `interrupt()`, and as a last resort `restart()`, which kills the worker.
* GPU: `maxGpuMemoryBytes` caps tinygrad's buffer allocations. Device loss (`GPUDevice.lost`) and WebGPU validation errors become
  Python `RuntimeError`s, and `restart()` gets a new device. WebAssembly memory is capped at 4 GiB by the platform. Browsers don't
  offer a finer-grained CPU/memory quota for workers.
* Browsers already sandbox WebGPU: shaders are validated and buffer access is bounds-checked.

## tinygrad versions

| | 0.14.0 (default) | 0.12.0 |
|---|---|---|
| `Tensor(x, requires_grad=True)` | removed upstream: every tensor can receive `.grad` | supported (and required for `.grad`) |
| training mode | `with Context(TRAINING=1):` | `with Tensor.train():` |
| test suite | all pass | all pass |

So the brief's autograd example `Tensor([2.0], requires_grad=True)` runs verbatim with `tinygradVersion: "0.12.0"`. On 0.14.0 it
raises upstream's `TypeError`, and `x = Tensor([2.0])` is the equivalent. The [compatibility strategy](docs/COMPATIBILITY.md)
explains how tinygrad is integrated without modifying it, the one runtime patch, and how to bump versions.

## Self-hosting Pyodide

```sh
npm run fetch:assets -- --pyodide    # verified (sha256) download -> vendor/pyodide/ (core + numpy, ~16 MB)
```

Serve that directory and pass `pyodideIndexURL: "/pyodide/"`. Pyodide 314.0.8 is required, because the backend and numpy wheel are
built for its ABI.

## Benchmarks

`npm run build && npm run bench -- [--quick] [--native /path/to/python-with-tinygrad]` writes `benchmarks/results/<label>.{json,md}`.
Results from the test machine: [benchmarks/results/swiftshader-chromium153-linux.md](benchmarks/results/swiftshader-chromium153-linux.md).
**The adapter there is SwiftShader (a CPU emulating a GPU), so absolute GPU numbers say nothing about real hardware.** On a real
GPU, run the same command with a headed or GPU-enabled Chromium (`--headed`, `TGW_CHROMIUM_ARGS`).

| measured (median) | value |
|---|---|
| cold init: Pyodide / numpy+tinygrad / WebGPU / `import tinygrad` / total | 1.89 s / 0.46 s / 0.01 s (in parallel) / 0.66 s / **3.2 s** |
| first kernel (codegen + WGSL compile + run) vs warm run | 54 ms vs 5.3 ms |
| JSPI round trip (`run_sync` on a resolved promise) | ~0.1 ms |
| matmul 512×512 / 1024×1024 (SwiftShader) | 5.6 / 5.9 GFLOP/s |
| MLP training step (2→64→1, batch 256, Adam) without / with `TinyJit` | 36.7 ms / **4.35 ms** |
| same step on the in-browser CPU device (interpreted) | ~5000 ms |

Takeaways:

* **Small ops are latency-bound.** A warm tiny kernel costs ~5 ms. That's tinygrad's Python scheduling in wasm (~1.9 ms) plus
  the submit→completion round trip (~3–4 ms on SwiftShader; expect less on hardware). WebGPU is not faster for tiny tensors.
* **Use `TinyJit`** for loops. It skips scheduling and replays cached kernels: 8× faster per training step here.
* Already implemented: kernel caching (tinygrad's own), pipeline creation once per kernel, `LRUAllocator` buffer reuse,
  bind-group and uniform caching (~98% hit rate in training loops), batched dispatches (one compute pass and one submit per
  sync point), pooled readback staging buffers, and WebGPU init overlapped with Pyodide startup.
* Not done yet: persisting compiled kernels across page loads (tinygrad's disk cache is in-memory in Pyodide), and
  timestamp-query-based kernel timing.

## Testing

```sh
npm install
npm run fetch:assets -- --pyodide   # tinygrad wheels + self-hosted Pyodide (needed: tests never use a CDN)
npm run build
npx playwright test                 # both tinygrad versions; add --project=tinygrad-0.14.0 for one
```

Tests run in headless Chromium with `--enable-unsafe-webgpu`, against a **real WebGPU adapter** (on GPU-less machines,
SwiftShader). Nothing is mocked. Each GPU test checks independently that the work ran on WebGPU: the JS bridge's dispatch
counters must be non-zero, no kernel may be compiled for tinygrad's CPU/PYTHON device, and one test reads a tensor's
`GPUBuffer` straight through WebGPU, bypassing tinygrad.

Numerical results are compared with **native upstream tinygrad on CPU**. Fixtures come from `tests/reference/generate.py`,
which runs the same `tests/programs/*.py` natively with `DEV=CPU`. Tolerances: rtol 1e-4 / atol 1e-5 for exact-input ops
(matmuls matched bit for bit). Values derived from `Tensor.randn` get 2e-3, because WGSL transcendental precision is
implementation-defined. Training losses get rtol 1e-3 over 60 steps.

| area | tests |
|---|---|
| Pyodide init, tinygrad import, versions, timings | `init.test.ts` |
| tensor add/mul/…, broadcasting, matmul (incl. spec examples) | `tensor.test.ts` |
| GPU buffer transfers (sizes, dtypes, padding, assign) | `tensor.test.ts` |
| autograd (spec example verbatim on 0.12, adapted on 0.14) | `autograd.test.ts` |
| NN training (SGD+momentum and Adam; TinyJit example) | `training.test.ts` |
| kernel caching, alloc/free, WGSL compile errors, device loss + restart, memory limit | `webgpu.test.ts` |
| repeated runs, stdout/stderr, interrupt (signal and terminate), timeouts, network policy, reset, destroy | `init.test.ts` |
| unsupported environment (no adapter): clear error, `auto` → CPU fallback, `cpu` mode | `init.test.ts` |
| Vite demo production build: every example plus Stop | `demo.test.ts` (needs `examples/vite-demo` built) |

Not verified by automated tests: browsers without JSPI (Chromium can't disable JSPI, so that error path is untested), hardware GPUs,
and non-Chromium browsers.

## Repository layout

```
src/            TypeScript: index.ts (API), runtime.ts (main thread), worker.ts, bridge.ts (WebGPU), protocol.ts, types.ts
python/tinygrad_web/   webgpu.py (WEBGPU device), cpu.py, patches.py, _runner.py, __init__.py (install)
tests/          Playwright tests, shared Python programs, native CPU reference fixtures
benchmarks/     bench.py (browser + native), run.mjs, results/
examples/       vite-demo/, python/mlp_training.py
scripts/        build.mjs, fetch-assets.mjs (sha256-pinned downloads), pins.mjs, serve.mjs
docs/COMPATIBILITY.md, RESEARCH.md
```

The build is reproducible. Every third-party artifact (tinygrad wheels, Pyodide release) is pinned by URL and SHA-256 in
`scripts/pins.mjs`, and npm dependencies are locked in `package-lock.json`.

## License

MIT for tinygrad-web's own code ([LICENSE](LICENSE)). The package redistributes unmodified tinygrad wheels (MIT). Pyodide (MPL-2.0)
and NumPy (BSD-3-Clause) load at runtime. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
