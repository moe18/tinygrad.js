# tinygrad.js

Run the **real, upstream [tinygrad](https://github.com/tinygrad/tinygrad)** in the browser. Python runs in
**[Pyodide](https://pyodide.org)** inside a Web Worker, and tinygrad's kernels run as **WGSL compute shaders on the
browser's WebGPU** device. No Python server, CUDA or native install.

**[Try the playground](https://moe18.github.io/tinygrad.js/)** (Chrome or Edge 137+): paint a dataset and watch tinygrad train an
MLP on your GPU.

```ts
import { createTinygrad } from "tinygrad.js";

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

This isn't a reimplementation. tinygrad.js installs the unmodified tinygrad wheel from PyPI and adds a browser
`WEBGPU` device to it. `Tensor`, autograd, `nn`, the optimizers, `TinyJit`, the scheduler and the WGSL code generator
are all upstream code.

**PyTorch-style code runs too.** `import torch` gives a PyTorch-compatible API implemented on tinygrad (it is not PyTorch itself):
tensors, autograd, `torch.nn`, `torch.optim`, `DataLoader`, `torch.save`/`load`. Eleven tutorial-style PyTorch programs, up to a
nanoGPT-style transformer, run unmodified on WebGPU and match real PyTorch's numbers. See [`import torch`](#import-torch).

```
browser app ──► tinygrad.js (TypeScript API) ──► Web Worker
                                                     │
                                         Pyodide (CPython 3.14 → wasm)
                                                     │
                                  upstream tinygrad (PyPI wheel, unmodified)
                                                     │  WGSLRenderer → WGSL kernels
                                tinygrad_js.webgpu (WEBGPU device, Python)
                                                     │  pyodide FFI, integer handles, JSPI for awaits
                                      src/bridge.ts (JS WebGPU bridge)
                                                     │
                                      navigator.gpu ──► your GPU
```

## Status

What is verified, and where (see [Testing](#testing)):

| | |
|---|---|
| ✅ verified | Chromium 153 (headless), WebGPU through Dawn's **SwiftShader** adapter: the real WebGPU API and real WGSL compute shaders, executed on the CPU because the test machine has no GPU. 77 Playwright tests pass: 54 on tinygrad 0.14.0 (including the demo and the [`import torch`](#import-torch) programs) and 23 on tinygrad 0.12.0 (14 skipped there: 13 need the torch layer, which requires 0.14, and 1 doesn't apply to 0.12). Results are compared against native upstream tinygrad on CPU, and for `import torch` against real PyTorch 2.15. |
| ⚠️ not verified | Hardware GPUs, Firefox, Safari, mobile browsers. The code uses only standard WebGPU + JSPI and should work wherever both exist, but this hasn't been tested. |
| ❌ not supported | Browsers without JSPI (WebAssembly JavaScript Promise Integration). See [Browser support](#browser-support). |

## Install

```sh
npm install tinygrad.js
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
  optimizeDeps: { exclude: ["tinygrad.js"] },
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
| `torch` | `true` | Make `import torch` resolve to the PyTorch-compatible API ([`import torch`](#import-torch)). Costs nothing until imported. |
| `network` | `"none"` | `"none"` removes `fetch`/XHR/WebSocket/… from the worker after startup. `"allow"` keeps them. See [Safety](#safety-and-isolation). |
| `loadPackagesOnImport` | `false` | Before each `run()`, load the Pyodide packages the code imports (pandas, matplotlib, scipy, scikit-learn, …) from `pyodideIndexURL`. Each is downloaded the first time it is imported; progress goes to `onStatus`. Works with `network: "none"`: only the runtime's own package loader can fetch, and only from the Pyodide distribution. |
| `maxGpuMemoryBytes` | unlimited | Cap on GPU buffer memory. Exceeding it raises `MemoryError` in Python. |
| `timeoutMs` | `0` (none) | Default per-`run()` time limit. |
| `initTimeoutMs` | `120000` | |
| `onStdout`, `onStderr`, `onStatus` | | Streaming callbacks. |

**`RunResult`:** `{ ok, status: "completed" | "error" | "interrupted" | "terminated", stdout, stderr, error?: { type, message,
traceback }, result?, durationMs, gpu: { kernelsCompiled, kernelsDispatched, bytesUploaded, bytesDownloaded, readbacks } | null }`.
`run()` never rejects because of a Python error; check `ok`. `result` is the `repr()` of a trailing expression.

**`RuntimeInfo`:** tinygrad device (`"WEBGPU"`/`"PYTHON"`), adapter info (vendor, architecture, `isFallbackAdapter`, features,
limits), tinygrad/Pyodide/Python versions, `syncMode`, `interruptMode`, init timings, `torchShim` (whether `import torch` is
available), and `diagnostics` (fallbacks, software adapter warnings, active tinygrad patches).

Python state persists between `run()` calls: variables, imports and compiled kernels. Top-level `await` works.

### Inside Python

```python
from tinygrad import Tensor, Device   # Device.DEFAULT == "WEBGPU"
import tinygrad_js
tinygrad_js.stats()          # kernels compiled/dispatched, bytes moved, live GPU memory, bind-group cache hits
tinygrad_js.device_info()    # adapter info and limits
```

### Training loops: `run_training`

`tinygrad_js.run_training` runs a plain step function fast without the author having to learn
[TinyJit](https://docs.tinygrad.org)'s rules. It's meant for exercise harnesses: the learner writes an ordinary step, and the harness
calls the helper.

```python
from tinygrad_js import run_training

def train_step(x, y):            # ordinary tinygrad code
  opt.zero_grad()
  loss = model(x).sparse_categorical_crossentropy(y)
  loss.backward()
  opt.step()
  return loss

result = run_training(train_step, steps=200, args=(X, Y))   # or args=lambda i: next_minibatch(i)
result.warmup_losses   # losses of the first `warmup` (default 2) plain steps: grade correctness on these
result.final_loss, result.jit_used, result.jit_fallback_reason, result.ms_per_step_jit
print(result.summary())  # "200 steps in 1.10s, TinyJit, loss 0.6931 -> 0.0412, 38.0 ms/step plain, 4.9 ms/step jit"
```

* The first `warmup` steps run plainly, then the rest run under `TinyJit`. Losses are read every `record_every` (10) steps and on the
  last one. `print_every` logs progress. Training mode (`Tensor.train()` / `Context(TRAINING=1)`) is applied for you on both
  tinygrad versions.
* Safety: TinyJit replays recorded kernels without rerunning your Python, and upstream it is silently wrong in two cases we
  measured. It freezes Python-number arguments, and on tinygrad 0.14 it returns a wrong-shaped result if input shapes change. The
  helper checks every call's argument shapes, dtypes and Python values against the recorded ones and switches to plain execution
  on any change. If TinyJit raises (e.g. `.item()` inside the step on tinygrad ≥ 0.13), it also falls back, and reports why in
  `jit_fallback_reason`.
* Not detectable: Python logic inside the step that depends on outside state (e.g. a global epoch counter). TinyJit freezes it at
  its recorded value, so keep such logic out of the step and grade on `warmup_losses`.
* Tested (both tinygrad versions, on WebGPU): JIT and plain losses agree to 1e-5. In the browser here, steps are ~8× faster with
  TinyJit. The fallbacks are tested too, and `interrupt()` stops it.

`Device["CPU"]` exists too. In the browser it is tinygrad's pure-Python interpreter device, which is correct but very
slow (see [Benchmarks](#benchmarks)).

### `import torch`

```python
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1, self.fc2 = nn.Linear(2, 64), nn.Linear(64, 2)
  def forward(self, x): return self.fc2(F.relu(self.fc1(x)))

model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for xb, yb in DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True):
  optimizer.zero_grad()
  loss = F.cross_entropy(model(xb), yb)
  loss.backward()
  optimizer.step()
print(loss.item(), next(model.parameters()).device)   # ... webgpu
```

`torch` here is a PyTorch-compatible layer that tinygrad.js implements on top of tinygrad
([`python/tinygrad_js/_shims/torch`](python/tinygrad_js/_shims/torch)). **It is not PyTorch** (PyTorch has no WebAssembly build),
and `torch.__version__` says so: `0.1.0+tinygrad.js`.

* **What runs:** tutorial-level PyTorch code, unmodified: tensor operations, autograd, `nn.Module` and the common layers (linear,
  convolution, normalization, recurrent, attention, transformer), losses, the standard optimizers and LR schedulers, `Dataset` /
  `DataLoader`, `torch.save` / `torch.load` (PyTorch's checkpoint format, interchangeable with real PyTorch in both directions).
* **How it is tested:** [eleven PyTorch programs](tests/torch_programs) (tensor basics, autograd, the "Learning PyTorch with Examples"
  polynomial, layers and losses, optimizers, a Quickstart-style classifier, the CIFAR-10 tutorial CNN and a small ResNet, a
  nanoGPT-style transformer, GAN/DQN/policy-gradient training patterns, checkpoints, learner-style snippets) run unmodified under real PyTorch 2.15 and under
  the shim in the browser on WebGPU, and the results must agree (rtol 2e-3 / atol 2e-4).
* **What does not run:** libraries that depend on PyTorch internals (`torchvision`, `transformers`, `lightning`, ...), custom
  `autograd.Function`, TorchScript, sparse/complex tensors, most of `torch.linalg`. These raise clear errors.
* **What behaves differently:** random streams differ from PyTorch's for the same seed; WebGPU has no 64-bit types, so `long` is
  stored as int32 and `double` as float32; a view does not see later in-place changes of its base tensor; evaluation is lazy, so
  a step costs tens of milliseconds unless it runs under `run_training` (TinyJit), which accepts PyTorch-style step functions.

[docs/TORCH.md](docs/TORCH.md) has the full coverage list and every known difference. Pass `torch: false` to `createTinygrad` to
turn it off. It needs tinygrad ≥ 0.14 (the default).

## CUDA kernels

tinygrad.js also compiles **CUDA C/C++ kernels** and runs them on WebGPU (CUDA → WGSL), on the CPU (CUDA → WebAssembly,
with a worker pool), or, when you explicitly configure one, on a remote NVIDIA GPU (the original source, via NVRTC).
They run standalone from JavaScript or on tinygrad tensors from Python, where they join the lazy graph and autograd.

```js
import { CUDA } from "tinygrad.js/cuda";
const kernel = await CUDA.compile(`__global__ void vector_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { c[i] = a[i] + b[i]; }
}`);
const ctx = kernel.ctx, a = ctx.rand([1024]), b = ctx.rand([1024]), c = ctx.zeros([1024]);
await kernel.launch({ grid: [4, 1, 1], block: [256, 1, 1], args: [a, b, c, 1024] });
```

```python
from tinygrad.cuda import CUDAKernel          # inside the tinygrad.js runtime (tinygrad >= 0.14)
kernel = CUDAKernel(open("vector_add.cu").read())
await kernel.launch(grid=(4, 1, 1), block=(256, 1, 1), args=[a, b, c, 1024])   # a, b, c: tinygrad Tensors
```

Supported CUDA:
- `__global__`/`__device__` functions, `__shared__`/`__constant__` memory, templates, structs with methods, macros
- the math library, half and bfloat16, atomics, barriers, warp intrinsics, cooperative-group block and warp tiles
- `printf`/`assert`, a kernel library (matmul, reductions, softmax, norms, conv, flash attention, quantized, MoE,
  optimizers) and cuBLAS/cuDNN/cuRAND-style APIs

Constructs a backend can't run exactly (`double` on WebGPU, 64-bit atomics on WebGPU, recursion, …) are rejected with an
explanation instead of being approximated. Guide, compatibility matrix and differences from native CUDA:
**[docs/CUDA.md](docs/CUDA.md)**. Remote GPU worker (Docker / Modal / RunPod): [server/cuda-worker](server/cuda-worker/README.md).

## Examples

* [`examples/vite-demo`](examples/vite-demo): Vite + TypeScript app with an editor, Run/Stop, console, device info,
  timings and preloaded examples (tensor basics, matmul throughput, autograd, MLP training, generated WGSL, WebGPU vs CPU).
  ```sh
  npm install && npm run fetch:assets -- --pyodide && npm run build
  cd examples/vite-demo && npm install && npm run dev     # http://localhost:5173
  ```
* [`examples/playground`](examples/playground) ([**try it live**](https://moe18.github.io/tinygrad.js/), deployed to GitHub
  Pages by `.github/workflows/pages.yml`): a single HTML page, no build step. Paint a 2-D dataset (or pick circle,
  XOR, moons, spiral) and watch an MLP's decision boundary update live as tinygrad trains it on WebGPU. You can change the
  depth, width, activation and learning rate, print the WGSL kernels of a forward pass, and run Python against the same
  session (`model`, `X`, `Y`, `opt` are live).
  ```sh
  npm install && npm run fetch:assets -- --pyodide && npm run build
  COI=1 npm run serve      # http://127.0.0.1:8080/examples/playground/
  ```
* [`examples/python/mlp_training.py`](examples/python/mlp_training.py): trains an MLP with `nn.Linear`, `Adam` and `TinyJit`
  on a synthetic dataset. It runs both in the browser and natively.
* [`examples/cuda-playground`](examples/cuda-playground): CUDA playground, a single HTML page. It has an editor and
  examples, runs on WebGPU or the CPU, shows outputs, timings, printf and errors, the kernel IR and generated
  WGSL/WebAssembly, and compares backends. It explains indexing, coalescing, reductions and tiling interactively, and
  can be embedded with a `postMessage` API. `COI=1 npm run serve`, then open `/examples/cuda-playground/`.
* [`examples/cuda/vector_add.mjs`](examples/cuda/vector_add.mjs) (`node examples/cuda/vector_add.mjs`),
  [`examples/python/cuda_vector_add.py`](examples/python/cuda_vector_add.py),
  [`examples/python/cuda_custom_autograd.py`](examples/python/cuda_custom_autograd.py): CUDA kernels from Node.js, and on
  tinygrad tensors with custom autograd.
* [`examples/python/torch_mlp.py`](examples/python/torch_mlp.py): plain PyTorch code (`nn.Module`, `DataLoader`, `Adam`, an LR
  scheduler, `torch.save`). The same file runs in the browser through [`import torch`](#import-torch) and with real PyTorch.

## Browser support

tinygrad.js needs **WebGPU in a dedicated worker** and **JSPI**. JSPI lets Python block on WebGPU's asynchronous readback
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

tinygrad.js is designed to run user-submitted code inside a larger app, but **a Web Worker is not a security boundary
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
| [`import torch`](#import-torch) | supported | not available (`ModuleNotFoundError`) |
| [`tinygrad.cuda`](#cuda-kernels) (JS `tinygrad.js/cuda` works without tinygrad) | supported | importable, but launches raise `CUDAError` (needs `Tensor.custom_kernel`) |
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
npm run fetch:assets -- --pyodide six   # tinygrad wheels + self-hosted Pyodide (needed: tests never use a CDN);
                                        # `six` is the package tests/packages.test.ts loads (skipped without it)
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
| `run_training` helper: JIT vs plain agreement, speed, fallbacks, interrupt | `run_training.test.ts` |
| `import torch`: eleven PyTorch programs compared with real PyTorch, `run_training` with PyTorch-style steps, unsupported APIs, opt-out | `torch.test.ts` (0.14 only) |
| kernel caching, alloc/free, WGSL compile errors, device loss + restart, memory limit | `webgpu.test.ts` |
| repeated runs, stdout/stderr, interrupt (signal and terminate, incl. during GPU waits), timeouts, network policy, reset, destroy | `init.test.ts` |
| unsupported environment (no adapter): clear error, `auto` → CPU fallback, `cpu` mode | `init.test.ts` |
| Vite demo production build: every example plus Stop | `demo.test.ts` (needs `examples/vite-demo` built) |
| playground: trains all four datasets, painting mid-training, shared Python session, WGSL, Stop | `playground.test.ts` |
| CUDA compiler: lexer, preprocessor, parser, lowering, passes, WGSL/WASM codegen | `cuda/compiler.node.test.ts` (`--project=cuda-node`) |
| CUDA kernel library on the CPU backend (inline and 4 workers), runtime API, randomized differential tests | `cuda/cpu.node.test.ts` |
| CUDA remote worker end to end (tokens, CORS, limits, timeouts, isolation) | `cuda/remote.node.test.ts` |
| CUDA on WebGPU (all cases, differential tests, bounds checks, AUTO selection), CPU pool in a browser | `cuda/webgpu.browser.test.ts` (`--project=cuda-browser`) |
| `tinygrad.cuda` on WEBGPU and CPU: views, autograd, MLP training, transformer block | `cuda/tinygrad.browser.test.ts` |
| CUDA playground and its embedding API | `cuda/playground.browser.test.ts` |

Not verified by automated tests: browsers without JSPI (Chromium can't disable JSPI, so that error path is untested), hardware GPUs,
and non-Chromium browsers.

## Repository layout

```
src/            TypeScript: index.ts (API), runtime.ts (main thread), worker.ts, bridge.ts (WebGPU), protocol.ts, types.ts
src/cuda/       CUDA: compiler/ (preprocess, parser, lower, passes, wgsl, wasm/), runtime/ (webgpu, wasm, remote, context),
                kernels/ (library), libs/ (cuBLAS/cuDNN/cuRAND subsets), device.ts, api.ts, tinygrad-host.ts
server/cuda-worker/   remote native CUDA worker (NVRTC + driver API), Docker / Modal / RunPod deployment
python/tinygrad_js/   webgpu.py (WEBGPU device), cpu.py, patches.py, training.py (run_training), _runner.py, __init__.py (install)
python/tinygrad_js/_shims/torch/   the PyTorch-compatible API (`import torch`): _tensor.py, nn/, optim/, utils/data, serialization.py
tests/          Playwright tests, shared Python programs, native CPU reference fixtures, torch_programs/ + PyTorch references
benchmarks/     bench.py (browser + native), run.mjs, cuda.mjs + cuda-bench.js + cuda_tinygrad.py (CUDA), dashboard.html, results/
examples/       vite-demo/, playground/, cuda-playground/, cuda/, python/
scripts/        build.mjs, fetch-assets.mjs (sha256-pinned downloads), pins.mjs, serve.mjs
docs/COMPATIBILITY.md, docs/TORCH.md, docs/CUDA.md, RESEARCH.md
```

The build is reproducible. Every third-party artifact (tinygrad wheels, Pyodide release) is pinned by URL and SHA-256 in
`scripts/pins.mjs`, and npm dependencies are locked in `package-lock.json`.

## License

MIT for tinygrad.js's own code ([LICENSE](LICENSE)). The package redistributes unmodified tinygrad wheels (MIT). Pyodide (MPL-2.0)
and NumPy (BSD-3-Clause) load at runtime. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
