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

## Hardware and browser compatibility

| | status |
|---|---|
| Chromium with WebGPU (SwiftShader) and Node.js 22 | **verified** by the automated suites |
| hardware GPUs through WebGPU (Chrome/Edge, Firefox 141+, Safari 26+) | not verified here: no GPU on this machine. The same standard WebGPU API is used; features are detected per adapter (`ctx.info().features`). |
| NVIDIA GPUs through the remote worker | the protocol, isolation, limits and errors are verified with a test engine. The NVRTC + driver engine (`tgjs_cuda_worker/cuda.py`) has **not run on NVIDIA hardware**. Without a GPU it fails cleanly with `cudaErrorNoDevice`. Run `TGJS_CUDA_TEST_NATIVE=1` on a GPU machine to validate it. |

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

1. Validate the native remote engine on NVIDIA hardware and add the remote symbol and struct-parameter support.
2. Benchmark on hardware GPUs. Real GPUs may well rank the variants differently from SwiftShader.
3. WASM SIMD: vectorize across 4 threads of a block in the CPU codegen.
4. Close the WebGPU performance gap to tinygrad's kernels: vectorized loads, better register tiling defaults, and
   autotuned library launch parameters.
5. Grow the language coverage: partial specialization, inheritance, lambdas (inlinable), `cg::reduce`, `%s` with
   literal arguments.
6. Multi-device and collective primitives on top of the serialization APIs.
