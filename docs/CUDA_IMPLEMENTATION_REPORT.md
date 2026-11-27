# CUDA-compatible GPU and CPU execution in tinygrad.js: implementation report

Branch `claude/cuda-gpu-cpu-tinygrad-eo6g71`: 97 files changed, about 20k lines added (13k of them in `src/cuda`). The
user guide is [docs/CUDA.md](CUDA.md); this report covers what was built, how it was verified, and what is still missing.

## What was implemented

| area | result |
|---|---|
| **CUDA compiler** (`src/cuda/compiler`, 9.5k lines) | Preprocessor with built-in CUDA headers; parser for a C++ subset (templates, structs with methods and operators, enums, references, `auto`, switch, …). Semantic analysis lowers to a structured kernel IR, with inlining, overloads, template deduction, and pointers resolved to (buffer, offset). Optimization passes: exact C-semantics folding, constant and copy propagation, CSE, DCE, unrolling, LICM, scalar promotion, barrier simplification; `direct` mode skips them. Two backends: **WGSL** and **WebAssembly**. |
| **WebGPU backend** (`runtime/webgpu.ts`) | Capability detection (f16, subgroups, timestamps, limits). Exact emulation of i64, f16 and bf16; float atomics through CAS loops; warp ops through subgroups when the size is 32, otherwise emulated. A diagnostics buffer carries `printf`, `assert` and optional bounds checks. Grids beyond the per-dimension limit are split, and aliased pointer arguments share a binding. |
| **CPU backend** (`runtime/wasm.ts`, `compiler/wasm`) | WebAssembly codegen that splits each block into phases at barriers and runs threads as masked loops, so CUDA threads are never OS threads. Blocks run on a worker pool (Web Workers or Node `worker_threads`) over shared memory, with a calling-thread fallback when the page isn't cross-origin isolated. Native f64 and i64, and always-on bounds checks. |
| **Remote native CUDA** (`runtime/remote.ts`, `server/cuda-worker`) | Client and a standard-library Python gateway (NVRTC + driver API through ctypes). Short-lived HMAC session tokens and CORS allowlists; one child process per session, killed on timeout; memory, time, session and launch limits; device memory zeroed on allocation; sticky faults. Deployment as Docker image, Modal app and RunPod pod. |
| **Devices** (`device.ts`) | `auto`/`webgpu`/`cpu`/`wasm`/`cuda_remote`, with reasons for every choice. AUTO prefers a hardware GPU, uses the CPU instead of a software adapter, checks `requirements` (f64, block size), and never selects remote. |
| **Runtime API** (`runtime/context.ts`, `api.ts`) | Modules and functions, `launch` and `exec("k<<<g,b>>>(…)")`, tensors and pointers with offsets, `memcpy`/`memset`/`malloc`/`free` with lifetime checks, `__constant__` symbols, streams, events, `cuda*` wrappers with `cudaError_t` codes, and `inspect()` for IR and generated code. |
| **tinygrad integration** (`python/tinygrad_js/cuda.py`, `src/cuda/tinygrad-host.ts`) | `from tinygrad.cuda import CUDAKernel, CUDAFunction`. Kernels become precompiled-program nodes in tinygrad's graph, so they are lazy, `TinyJit`-capturable and mix with ops. They are zero-copy on WEBGPU and use the WebAssembly backend on the CPU device. Supports in-place writes, writes through views of stored tensors, aliasing, structs, templates, custom autograd, and `printf` and errors as `CUDAError`. |
| **Kernel library** (`src/cuda/kernels`) | Vector ops, three matmuls, transpose, reductions, scans, histograms, activations, softmax, layer and RMS norm, conv and pooling, embedding, cross-entropy, naive and flash attention, RoPE, KV cache, int8/int4/f16 matmuls, sparse, MoE, SGD and Adam(W), clipping, mixed-precision helpers. |
| **Library compatibility** (`src/cuda/libs`) | cuBLAS (`Sgemm`, strided batched, `gemv`, `axpy`, `scal`, `copy`, `dot`, `nrm2`, `asum`), cuDNN (activations, softmax, conv2d forward, pooling), cuRAND (Philox host API plus the device API in `curand_kernel.h`). |
| **Utilities** | `autotune` (cached per device), tensor and kernel-bundle serialization with sha256 integrity, device reports. |
| **Benchmarks** (`benchmarks/cuda*.{mjs,js,py}`, `dashboard.html`) | Verified-then-timed measurements with separate compile, transfer, steady-state, latency and GPU-timestamp numbers. Compares CUDA→WGSL in both compiler modes, CUDA→WASM in both thread modes, handwritten WGSL, and tinygrad ops. |
| **Playground** (`examples/cuda-playground`) | Editor, examples, backend and mode selection, arguments, outputs, IR/WGSL/WASM views, errors with line links, backend comparison, four interactive lessons (indexing and warps, coalescing, reduction with barriers, tiling), and an embedding API over `postMessage`. |
| **Docs and examples** | `docs/CUDA.md`, a README section, the worker README, `examples/cuda/vector_add.mjs`, and `examples/python/cuda_vector_add.py` / `cuda_custom_autograd.py`. Every example was run on both devices. |

## Architecture

```
CUDA C++ ─▶ preprocess ─▶ parse ─▶ lower (types, templates, inlining, pointers → buffer+offset) ─▶ kernel IR ─▶ passes
                                                                                                    ├─▶ WGSL ─▶ WebGPU (own device, or tinygrad's)
                                                                                                    └─▶ WASM ─▶ worker pool over shared memory
original source ──────────────────────────────────────────────────────────────▶ remote gateway ─▶ NVRTC + CUDA driver
JS: CudaContext/CudaModule/DeviceTensor ── Backend interface ── WebGPUBackend | WasmBackend | RemoteCudaBackend
Python: tinygrad.cuda ─▶ Tensor.custom_kernel(PROGRAM) ─▶ tinygrad WEBGPU device (dispatch into its encoder) | CPU device (WASM)
```

Every kernel instance is compiled once per (block size, dynamic shared size) and once per backend specialization:
aliasing pattern and zero-offset arguments. Pipelines and WASM modules are reused.

## Test results (final run, this machine)

| suite | result |
|---|---|
| `--project=cuda-node`: compiler units, 28 numerical cases × (inline, 4 workers), runtime API, 200-kernel randomized differential test, remote worker end to end | **92 passed** |
| `--project=cuda-browser`: 28 cases on WebGPU, 150-kernel differential test, bounds checks, grid splitting, aliasing, timestamps, AUTO selection, CPU pool in browser (isolated and not), `tinygrad.cuda` × {WEBGPU, CPU} (README example, integration, autograd, MLP training, transformer block), playground and embedding | **52 passed** |
| existing tinygrad.js suites (`tinygrad-0.14.0`, `tinygrad-0.12.0`), rerun after the shared-file changes | **80 passed, 15 skipped** (pre-existing conditional skips), 0 failed |
| `npx tsc --noEmit` | clean |

Some numerical results from these suites:
- MLP training with a CUDA Linear+ReLU layer and CUDA Adam matches pure tinygrad's 25-step losses to within 1.1e-5.
- CUDA SiLU gradients match tinygrad's to 1.2e-7.
- Integer results agree bit for bit with exact constant folding on both backends.
- WebGPU float results agree to within 4 ulp. Shader compilers may reassociate, as observed with SwiftShader.

Bugs found and fixed along the way:
- CPU-device results were lost when numpy was imported. Pyodide's `getBuffer` can hand out a copy, so the write-back
  moved to the Python side.
- The training test read the lazy loss after the optimizer step.
- `storageBarrier` was emitted unconditionally. The new elision rule brought `reduce_sum` from 65 ms to 51 ms, against
  49 ms for handwritten WGSL.
- Unscoped `enum` types weren't resolved.
- Template arguments in function return types used the wrong environment.
- Explicit template instantiations weren't accepted.
- Writes through views of `arange`-derived tensors were silently lost. They now raise `CUDAError` with a fix.
- The gateway's session-limit check could race.

## Benchmarks (measured)

The full data is in `benchmarks/results/cuda-swiftshader-chromium-linux.md` / `.json`. The machine was a 4-core Xeon
(2.1 GHz) running Chromium 158. Its only WebGPU adapter is **SwiftShader, a CPU emulation**, so the "GPU" numbers here
say nothing about GPU hardware.

| workload | CUDA→WGSL opt | CUDA→WGSL direct | handwritten WGSL | CUDA→WASM 4 thr | CUDA→WASM 1 thr |
|---|---|---|---|---|---|
| vector_add 4M | 36.7 ms | 41.6 | 32.0 | 5.2 | 8.7 |
| matmul 256 naive / tiled / regtile | 63.0 / 162.9 / 41.4 | 84.4 / 173.8 / 157.1 | – / 132.2 / – | 15.9 / 11.6 / 5.2 | 44.8 / 31.6 / 12.5 |
| matmul 512 naive / tiled / regtile | 505 / 1103 / 281 | 657 / 1220 / 1050 | – / 822 / – | 129 / 88 / 39 | – |
| reduce_sum 4M | 69.4 | 80.6 | 58.0 | 2.0 | 5.1 |
| softmax 1024² | 151 | 172 | – | 16.9 | 50.2 |
| layernorm 1024² | 128 | 118 | – | 2.8 | 7.1 |
| causal attention H=8 T=512 D=16: naive / flash | 282 / 54 | 293 / 155 | – | 59 / 27.7 | – |

What these numbers show:
- **The optimizing passes pay off.** Optimized mode beats direct mode in every workload except layernorm. The biggest
  gains are on register-blocked matmul (3.7×) and flash attention (2.9×).
- **Translated CUDA against handwritten WGSL, on the same algorithm:** vector_add takes 1.15× as long as handwritten,
  tiled matmul 1.23–1.34×, and reduce_sum 1.2×.
- **Against tinygrad's own operations:**
  - On WEBGPU, tinygrad's generated kernels were faster than the library's CUDA kernels in every pair measured, by
    1.45× (vector add) to 12× (matmul 256 against `matmul_tiled`).
  - On the interpreted CPU (PYTHON) device, CUDA kernels on the WebAssembly backend were **68–283× faster** than
    tinygrad's ops.

  No claim is made that translated CUDA beats tinygrad on WebGPU; on this machine it doesn't.
- **SwiftShader penalizes barriers heavily:** shared-memory tiling is slower than the naive matmul there, both for the
  translated kernel and for handwritten WGSL. On the CPU backend, tiling and register blocking help as they would on a
  GPU.

## Follow-up: hardware GPU (NVIDIA RTX 4070 Laptop GPU)

Run in Chromium 153 on Windows (D3D12), with `TGW_CHANNEL=chromium TGW_CHROMIUM_ARGS=--force-high-performance-gpu`. The
`cuda-browser` suite passes (53 tests); results in `benchmarks/results/cuda-rtx4070-laptop-chromium-win.md`, the native
comparison in [CUDA.md](CUDA.md#benchmarks). Running on real hardware found four bugs, now fixed:

1. **The suite's `exec` column measured synchronization, not kernels.** It divided one run of 10 launches by 10, and a
   browser needs ~2–8 ms to report that the GPU finished, so every fast kernel showed ~0.8 ms. The tinygrad comparison
   synchronized after every call, so all its pairs came out ~8.2 ms. Both now queue enough calls to fill ~200 ms. For
   example, vector_add 4M went from 0.80 ms (63 GB/s) to 0.20 ms (247 GB/s).
2. **GPU timestamps of split grids covered only the last piece.** A grid over 65,535 blocks becomes several dispatches,
   and each piece got its own timing entry, so callers reading one entry per launch got the last, smallest piece
   (vector_add 16M: 0.003 ms instead of 0.9 ms). A launch now reports one entry, the sum of its pieces.
3. **Optimized mode was slower than direct mode on the GPU.** Copy propagation left uniform-buffer reads such as
   `bitcast<i32>(P[2].z)` inside loops, and D3D12/NVIDIA did not hoist them: matmul_naive ran 1.7–2.3× slower than
   direct mode (4096: 371 ms against 162 ms). Every uniform word is now read once into a `let` at kernel entry; that
   kernel takes 158 ms in optimized mode. Optimized is now within 10% of direct mode or faster on every workload measured
(faster on flash attention D=64 by 1.3× and naive matmul 4096 by 1.14×). The
   "optimizing passes pay off" result above came from SwiftShader.
4. **Three library sources weren't plain CUDA.** `vector_add4` used `float4 + float4` (defined only by the SDK's
   `helper_math.h`), and `mixedPrecision` and `quantized` used `__half` without `<cuda_fp16.h>`. All 20 library sources now
   compile with nvcc 12.2; `TGJS_NVCC=... npx playwright test tests/cuda/nvcc.node.test.ts` checks this.

Against tinygrad's own operations on this GPU (TinyJit-captured calls, default tinygrad settings, measured before fix 3):
the library's CUDA kernels were faster on softmax and layernorm 4096² (3.8×), sum of 16M values (2.1×), causal attention
D=64 (1.5×) and `matmul_regtile` at 2048 and 4096 (1.35–1.44×). tinygrad was faster on vector add (1.4×) and than
`matmul_tiled` (2.1×). They tied on matmul 1024. This reverses the SwiftShader comparison above.

Still open: the first launch of some large kernels took 17–29 s (pipeline creation; cause not found), and a batch of
twenty ~170 ms launches once triggered the Windows GPU watchdog (`DXGI_ERROR_DEVICE_HUNG`); it didn't reproduce.

## Follow-up: code review fixes

A review of the branch found ten bugs. All are fixed, and each fix has a test that fails without it.

**Remote worker (`server/cuda-worker`).** Tests: `tests/test_gateway.py`, which needs no GPU.
1. **Kernel source could read host files.** NVRTC opens any file an `#include` names and quotes it in its error log. On
   the RTX 4070, `#include "/proc/self/environ"` returned the session process's environment in the compile error. The
   gateway's own `/proc/<pid>/environ`, which holds `TGJS_CUDA_TOKEN_SECRET`, was readable the same way whenever the two
   processes share a uid. Includes are now limited to plain relative header names (`tgjs_cuda_worker/source.py`). The
   gateway checks them and so does the NVRTC engine.
2. **Closing a session could hand its pipes to another session.** `kill()` closed the fds without the session lock. A
   request in flight could then write to, and read from, fd numbers already reused by a new session. The fds are now
   closed only while that lock is held. The child is signalled first, so the request in flight ends right away.
3. **A negative `Content-Length` skipped the body limit.** It did so before authentication: `rfile.read(-1)` reads until
   the client disconnects. `Content-Length` must now be digits only, and the token is checked before any of the body is
   read. A refused request's unread body closes the connection.
4. **Concurrent allocations overshot the memory limit.** Each one checked the quota before any of them was counted.
   Bytes are now reserved under a lock before the child is called, and a concurrent double free is refused instead of
   answering 500.

**Library.**
5. **The executable cache ignored the dynamic shared size.** Its key had the block size but not `sharedMem`, so a second
   launch with more shared memory reused code sized for less. That gave `cudaErrorIllegalAddress` on the CPU backend and
   silently clamped accesses on WebGPU.
6. **A CPU worker script that failed to load hung every launch.** The load error was taken as "ready". Workers now have to
   answer `ready` within 30 s, or the pool falls back to the calling thread. A worker that dies later fails the launch
   instead of hanging it.
7. **Unaligned uploads on WebGPU landed late.** Their edge words were merged on the CPU after an asynchronous read, so
   launches and writes recorded after them ran first. The aligned middle is now written directly, and the edge words
   are merged by a tiny compute dispatch, in queue order.
8. **A tensor passed twice was two copies on the CPU device.** Each copy was written back separately (WebGPU shares one
   buffer). Each distinct buffer is now sent once, with a map from parameter to buffer.
9. **`inf` and `NaN` scalars broke launches from Python.** `json.dumps` writes `Infinity`, which `JSON.parse` rejects.
   They are now tagged on the Python side and turned back into numbers in JavaScript.
10. **`kernel(...)` defaulted to `lazy=False`.** So a direct call inside `CUDAFunction.backward` ran eagerly, although the
    docs say launches there are deferred. It now defaults to the context's laziness, like `.launch()`.

## Hardware and browser compatibility

| | status |
|---|---|
| Chromium with WebGPU (SwiftShader) and Node.js 22 | **verified** by the automated suites |
| hardware GPUs through WebGPU (Chrome/Edge, Firefox 141+, Safari 26+) | **verified on one GPU**: NVIDIA RTX 4070 Laptop GPU, Chromium on Windows (see the follow-up above). Other GPUs and browsers aren't verified; features are detected per adapter (`ctx.info().features`). |
| NVIDIA GPUs through the remote worker | the protocol, isolation, limits and errors are verified with a test engine. The NVRTC + driver engine (`tgjs_cuda_worker/cuda.py`) **passes the remote suite on one GPU**: an RTX 4070 Laptop GPU under WSL 2 with CUDA 12.2 (`TGJS_CUDA_TEST_NATIVE=1`, 6/6 tests). Without a GPU it fails cleanly with `cudaErrorNoDevice`. Run `TGJS_CUDA_TEST_NATIVE=1` on a GPU machine to validate it. |

## Limitations

- **Unsupported CUDA, rejected with explanations:**
  - dynamic parallelism, recursion, function pointers, virtual functions, lambdas
  - pointers stored in memory, pointer-to-pointer
  - inline PTX, textures and surfaces, `wmma`, grid-wide cooperative groups, `memcpy_async`
  - inheritance, partial specialization, variadic templates, bit-fields, `goto`, range-based `for`
  - `printf("%s"/"%p")`, `clock()`
- **WebGPU:**
  - no `double` (`fp64: "demote"` opt-in with a warning) and no 64-bit atomics
  - float results within a few ulp of native, not bit-exact
  - out-of-bounds accesses clamp unless `boundsCheck` is on
  - the GPU watchdog may kill very long kernels
- **The CPU backend is scalar WebAssembly** (no SIMD vectorization yet). Lanes within a warp run in lockstep, so
  independent thread scheduling isn't modelled. Streams run in submission order.
- **`tinygrad.cuda`:**
  - needs tinygrad ≥ 0.14
  - a written view must view a stored tensor (lazy bases raise `CUDAError`)
  - the remote device isn't exposed to Python
- **The remote backend** doesn't support `__constant__`/`__device__` symbols or by-value struct parameters yet; it
  raises `cudaErrorNotSupported`.
- **Distributed training** is covered by serialization primitives only: tensors, kernel bundles and device reports.
  There are no collectives or multi-device execution.

## Remaining work

1. Add the remote symbol and struct-parameter support. The native engine passes the remote suite on one GPU; it still needs a run on a data-center GPU in the Docker image.
2. Benchmark on more hardware GPUs (done on one NVIDIA GPU), and find why some first launches take 17–29 s on D3D12.
3. WASM SIMD: vectorize across 4 threads of a block in the CPU codegen.
4. Close the gap to native CUDA on compute-bound kernels (matmul 1.5–1.8×, flash attention with D=64 3.7× on the RTX
   4070): vectorized loads, better register tiling defaults, and autotuned library launch parameters.
5. Grow the language coverage: partial specialization, inheritance, lambdas (inlinable), `cg::reduce`, `%s` with
   literal arguments.
6. Multi-device and collective primitives on top of the serialization APIs.
