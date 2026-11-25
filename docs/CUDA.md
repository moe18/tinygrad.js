# CUDA in tinygrad.js

tinygrad.js compiles **CUDA C/C++ kernels** and runs them in the browser and in Node.js, on three kinds of hardware:

| device | where kernels run | how |
|---|---|---|
| `webgpu` | the user's GPU, through WebGPU | CUDA → kernel IR → **WGSL** compute shaders |
| `cpu` (alias `wasm`) | the CPU cores | CUDA → kernel IR → **WebAssembly**, one instance per worker (Web Workers or Node `worker_threads`) over shared memory |
| `cuda_remote` | a real NVIDIA GPU on a server you run | the **original source**, compiled by NVRTC and run by the CUDA driver ([`server/cuda-worker`](../server/cuda-worker/README.md)) |

You can use it standalone from JavaScript (`tinygrad.js/cuda`) or on tinygrad tensors from Python inside the tinygrad.js
runtime (`from tinygrad.cuda import CUDAKernel`). On tensors, kernels join tinygrad's lazy graph and its autograd. The
compiler never changes numerical precision silently. When a kernel needs something a backend can't do exactly (`double`
on WebGPU, 64-bit atomics on WebGPU, `clock()`, …), compilation fails with a message that names the construct and the
alternatives.

Contents: [Getting started](#getting-started) · [JavaScript API](#javascript-api) · [Python API](#python-api-tinygradcuda) ·
[Devices](#devices-and-selection) · [Memory](#memory) · [Tensors and autograd](#tensors-and-autograd) ·
[Compiler](#the-compiler) · [Debugging](#debugging) · [Performance](#performance-and-optimization) ·
[Kernel library](#kernel-library) · [Compatibility matrix](#compatibility-matrix) · [Differences from native CUDA](#differences-from-native-cuda) ·
[Browser support](#browser-and-hardware-support) · [Security](#security) · [Testing](#testing) · [Benchmarks](#benchmarks)

## Getting started

### JavaScript

```js
import { CUDA } from "tinygrad.js/cuda";

const kernel = await CUDA.compile(`
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { c[i] = a[i] + b[i]; }
}`);
const ctx = kernel.ctx;
const a = ctx.rand([1024]), b = ctx.rand([1024]), c = ctx.zeros([1024]);
await kernel.launch({ grid: [4, 1, 1], block: [256, 1, 1], args: [a, b, c, 1024] });
console.log(await c.toArray());
```

`CUDA.compile` uses the default context, created on first use with AUTO device selection. `CUDA.configure({ device })`
changes that context's options. Bundlers that can't resolve the CPU backend's worker script need
`workerURL: new URL("tinygrad.js/cuda-worker.js", import.meta.url)`. Run [`examples/cuda/vector_add.mjs`](../examples/cuda/vector_add.mjs)
with plain `node` to try the CPU backend.

### Python (tinygrad tensors, inside the tinygrad.js runtime)

```python
from tinygrad import Tensor
from tinygrad.cuda import CUDAKernel

kernel = CUDAKernel("""
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
""")
a = Tensor.rand(1024); b = Tensor.rand(1024); c = Tensor.zeros(1024)
await kernel.launch(grid=(4, 1, 1), block=(256, 1, 1), args=[a, b, c, 1024])
print(c.numpy())
```

On `WEBGPU` the kernel reads and writes tinygrad's own GPU buffers (zero copy). On the `CPU` device it runs on the
WebAssembly backend over tinygrad's buffers. `tinygrad.cuda` needs tinygrad ≥ 0.14 (`Tensor.custom_kernel`), and
`await` is optional because launches complete synchronously. More examples: [`examples/python/cuda_*.py`](../examples/python/).

### Playground

[`examples/cuda-playground/`](../examples/cuda-playground/) is a single HTML page with no build step. It has:
- an editor, examples, backend and compiler-mode selection, grid and block, and arguments
- outputs, timings, `printf`, and errors with clickable line numbers
- the kernel IR and the generated WGSL or WebAssembly
- a benchmark that compares backends
- interactive explanations of thread indexing and warps, memory coalescing, shared-memory reductions with barriers, and
  tiling

Serve the repository (`COI=1 node scripts/serve.mjs`) and open `/examples/cuda-playground/`. To embed it, use an
`<iframe src=".../examples/cuda-playground/?embed=1">` and drive it with `postMessage`:

```js
frame.contentWindow.postMessage({ type: "tgjs-cuda:set", id: 1, source, kernel: "k", grid: [4], block: [256],
  args: { x: { n: 1024, init: "rand" }, y: { n: 1024, init: "zeros" }, n: 1024 } }, origin);   // -> { type: "tgjs-cuda:ready", kernels }
frame.contentWindow.postMessage({ type: "tgjs-cuda:run", id: 2 }, origin);                     // -> { type: "tgjs-cuda:result", ok, output, printf, ms | error }
```

Limit which parent origins it accepts with `?origin=https://a.example`. Other query parameters are `backend=webgpu|cpu`
and `example=<name>`.

## JavaScript API

```ts
import { createCudaContext, CUDA, cublas, cudnn, curand, kernels } from "tinygrad.js/cuda";

const ctx = await createCudaContext({ device: "auto" });   // "auto" | "webgpu" | "cpu" | "wasm" | "cuda_remote"
ctx.selection   // { requested, selected, reasons: ["…why this backend…"] }
ctx.info()      // DeviceInfo: name, limits (maxThreadsPerBlock, sharedMemPerBlock, …), features (fp16, fp64, int64, atomics64, warps, timestamps…)

const m = ctx.compile(source, { mode: "optimized", boundsCheck: false, defines: { TILE: 16 }, fp64: "error" });
m.kernels                                             // signatures: name, template parameters, parameter types
await m.launch({ kernel: "matmul_tiled<16>", grid: [8, 8], block: [16, 16], args: [A, B, C, 256, 256, 256], sharedMem: 0, timing: true });
await m.exec("matmul_tiled<16><<<dim3(8, 8), dim3(16, 16)>>>(A, B, C, n, n, n)", { A, B, C, n: 256 });   // launch-statement syntax
const f = m.getFunction("vector_add"); await f.launch({ grid, block, args });
const { ir, code, notes } = await m.inspect("vector_add", [256], "all");   // kernel IR + WGSL / WebAssembly for a block size
await m.memcpyToSymbol("coef", new Float32Array([1, 2, 3])); await m.memcpyFromSymbol(out, "coef");

const rep = await ctx.synchronize();   // waits; throws CudaError for device-side faults; rep.printf, rep.kernelTimes
```

The launch arguments are matched to the kernel's parameters:

| parameter type | argument |
|---|---|
| pointer (`T*`) | a `DeviceTensor`, a `DevicePtr` (`t.ptr.add(bytes)` for offsets), or a host typed array (copied in and back) |
| scalar (int, float, …) | a number, bigint or boolean, checked: an `int` parameter rejects `1.5` |
| vector (`float4`, …) | an array of components |
| struct passed by value | an object with field names |

Tensors and memory:

```ts
ctx.tensor(data, shape?, dtype?)                         // upload
ctx.zeros(shape, dtype?), ctx.ones(…), ctx.full(…), ctx.arange(n), ctx.rand(shape, seed?), ctx.empty(…)
await t.toArray(); await t.toList(); await t.write(data); t.reshape(shape); t.free()
const p = ctx.malloc(bytes); await ctx.memcpy(dst, src, bytes?); ctx.memset(p, value, bytes); ctx.free(p)
```

The dtypes are `float32 float16 bfloat16 float64 int8 uint8 int16 uint16 int32 uint32 int64 uint64 bool`.

#### Runtime API equivalents

These are thin wrappers over the default context, with CUDA names and the `cudaError_t` numbering:

- devices: `cudaInitDevice`, `cudaGetDeviceCount`, `cudaSetDevice`, `cudaGetDevice`, `cudaGetDeviceProperties`, `cudaDeviceReset`
- memory: `cudaMalloc`, `cudaFree`, `cudaMemcpy` (with `cudaMemcpyKind`), `cudaMemcpyAsync`, `cudaMemset`, `cudaMemGetInfo`
- synchronization and errors: `cudaDeviceSynchronize`, `cudaGetLastError`, `cudaPeekAtLastError`, `cudaGetErrorName`, `cudaGetErrorString`
- streams and events: `cudaStreamCreate`, `cudaStreamSynchronize`, `cudaStreamDestroy`, `cudaEventCreate`, `cudaEventRecord`,
  `cudaEventSynchronize`, `cudaEventElapsedTime`, `cudaEventDestroy`

Errors are `CudaError` objects with `errorName` (for example `"cudaErrorIllegalAddress"`) and `code` (700).

#### Library compatibility subsets

These run the kernel library underneath, with the familiar call shapes (column-major for cuBLAS):

- **cuBLAS:** `cublasCreate`, `cublasSgemm`, `cublasSgemmStridedBatched`, `cublasSgemv`, `cublasSaxpy`, `cublasSscal`,
  `cublasScopy`, `cublasSdot`, `cublasSnrm2`, `cublasSasum`
- **cuDNN:** `cudnnCreate`, `cudnnActivationForward` (sigmoid, relu, tanh, clipped relu, elu), `cudnnSoftmaxForward`
  (accurate and log), `cudnnConvolutionForward` and `cudnnGetConvolution2dForwardOutputDim` (with dilation),
  `cudnnPoolingForward` (max and average)
- **cuRAND, host API:** `curandCreateGenerator` (Philox4x32-10), `curandSetPseudoRandomGeneratorSeed`,
  `curandSetGeneratorOffset`, `curandGenerateUniform`, `curandGenerateNormal`, `curandGenerate`
- **cuRAND, device API:** `#include <curand_kernel.h>` provides `curand_init`, `curand`, `curand_uniform`,
  `curand_normal`, `curand_normal2` and `curand_uniform4` on Philox states. `curandState` (XORWOW in CUDA) is mapped to
  Philox, with a compile warning, because the random streams differ from CUDA's.

Utilities:

- `autotune(ctx, module, key, candidates)` times configurations with GPU timestamps when available. It caches the
  winner per (key, source hash, device) in memory and in `localStorage`.
- `serializeTensors` / `deserializeTensors` handle a binary container for checkpoints and transfer.
- `serializeKernel` / `deserializeKernel` produce a kernel bundle with source, launch configuration and a sha256
  integrity check.
- `deviceReport(ctx)` describes the device's capabilities, for coordinating distributed jobs.

## Python API (`tinygrad.cuda`)

```python
from tinygrad.cuda import CUDAKernel, CUDAFunction, CUDAError, library, device_info, synchronize, serialize_tensors, deserialize_tensors

k = CUDAKernel(source, "kernel_name", options={"defines": {"TILE": 16}, "mode": "optimized", "boundsCheck": False})
k2 = k["other_kernel<64>"]                          # another kernel (or template instantiation) of the same module
k.launch(grid, block, [tensor_a, tensor_b, 1024], shared_mem=0, lazy=False)
await k.launch(grid=..., block=..., args=[...])     # the README form; the result is awaitable
k(a, b, n, grid=(4,), block=(256,))                 # call syntax
print(k.inspect((256,), "code", device="webgpu"))   # WGSL / "wasm" / "ir"
library("matmul")                                   # CUDA source of a kernel library group
```

- **Arguments.** Pointer parameters take `Tensor`s and other parameters take Python numbers (dicts and lists for structs
  and vectors). The same tensor passed twice is one aliased buffer.
- **Writes.** Tensors the kernel writes are updated in place, like `Tensor.assign`. A written slice or transpose updates
  the tensor it views when that tensor is stored in a buffer: from `Tensor(numpy)`, `.clone().realize()`, or an earlier
  kernel. A view of a lazy result (e.g. `Tensor.arange(8).reshape(2, 4)[1]`) can't be written through in tinygrad, so
  launch raises `CUDAError` instead of losing the write; call `.clone().realize()` on the base first.
- **Laziness.** Launches realize immediately by default, which matches CUDA semantics. `lazy=True` only records the
  kernel in tinygrad's graph, and it runs when its outputs are needed. This also works under `TinyJit`.
- **Devices.** Tensors on `WEBGPU` run on tinygrad's GPU device and share its buffers and command stream. Tensors on
  `CPU`/`PYTHON` run on the WebAssembly backend.
- **Errors.** `synchronize()` returns device `printf` output and raises `CUDAError` for device-side faults.

### Custom autograd

```python
class Square(CUDAFunction):
  @staticmethod
  def forward(ctx, x):
    y = Tensor.empty(*x.shape)
    square_fwd.launch((x.numel() + 255) // 256, 256, [x, y, x.numel()])
    ctx.save_for_backward(x)
    return y
  @staticmethod
  def backward(ctx, gy):
    (x,) = ctx.saved_tensors
    gx = Tensor.empty(*x.shape)
    square_bwd.launch((x.numel() + 255) // 256, 256, [x, gy.contiguous(), gx, x.numel()])
    return gx

y = Square.apply(x); y.sum().backward()   # x.grad comes from the CUDA backward kernel
```

The output joins tinygrad's graph through `Tensor.custom_kernel` with a `grad_fxn`, so CUDA layers mix freely with
tinygrad ops, optimizers and `TinyJit`. The test suite trains an MLP whose hidden layer and Adam optimizer are CUDA
kernels and gets the same losses as pure tinygrad to within 1.1e-5
([`tests/cuda/programs/training.py`](../tests/cuda/programs/training.py)).

## Devices and selection

`device: "auto"` (the default) chooses as follows:

1. `webgpu` if a **hardware** WebGPU adapter is available and every kernel listed in `requirements` fits it. A kernel
   that needs double precision, or a block larger than the adapter's workgroup limit, selects the CPU instead.
2. If the only adapter is a software implementation (SwiftShader, `isFallbackAdapter`), the CPU backend is used,
   because it is faster than an emulated GPU. Pass `allowSoftwareWebGPU: true` to use the software adapter anyway.
3. Otherwise `cpu`: a worker pool when cross-origin isolated (`SharedArrayBuffer`), else the calling thread.

`ctx.selection.reasons` explains every choice, for example `"the only WebGPU adapter is a software implementation (…)"`.
AUTO **never** selects `cuda_remote`, which has to be requested explicitly with an endpoint and a token from your
backend. An explicit `device: "webgpu"` doesn't fall back: it fails with a reason if WebGPU is unavailable.

| feature | WebGPU | CPU (WASM) | remote CUDA |
|---|---|---|---|
| `float` | native (WGSL f32) | native | native |
| `double` | **error**; `fp64: "demote"` computes in float, with a warning | native f64 | native |
| `half` / `__half` | native with `shader-f16`, otherwise exact software rounding | exact software rounding (RNE) | native |
| `__nv_bfloat16` | exact software rounding | exact software rounding | native |
| 64-bit integers | emulated exactly (pairs of u32) | native i64 | native |
| 32-bit atomics (int, unsigned, float add, CAS, …) | native, or CAS loops | native (shared memory, workers) | native |
| 64-bit atomics | **error** (WGSL has none) | native | native |
| warp intrinsics (`__shfl*_sync`, `__ballot_sync`, `__reduce_*_sync`, …) | subgroups when the subgroup size is exactly 32, otherwise emulated in workgroup memory | emulated (warp = 32 lanes) | native |
| max threads per block | adapter limit (usually 256) | 1024 | 1024 |
| dynamic shared memory | up to the adapter's workgroup storage (usually 16–32 KB) | 48 KB | device limit |
| `printf`, `assert`, `__trap` | yes (diagnostics buffer, reported at synchronize) | yes | yes |
| bounds checking | optional (`boundsCheck: true`); WebGPU otherwise clamps silently | always on | no (native behaviour) |
| kernel timing | GPU timestamps with `timestamp-query`, otherwise wall-clock | wall-clock | CUDA events |

## Memory

Device memory is a set of allocations; a `DevicePtr` is an (allocation, byte offset) pair, so pointer arithmetic on the
host (`t.ptr.add(64)`) and in kernels works as in CUDA. Allocations are bounds- and lifetime-checked on the host:

- Copies that exceed an allocation throw.
- Double frees, use after free, and pointers from another device context throw.
- A `memoryLimit` per context gives `cudaErrorMemoryAllocation` when exceeded.

On WebGPU each allocation is a `GPUBuffer`; on the CPU backend allocations live in one shared `WebAssembly.Memory`.

Inside kernels, each pointer must resolve at compile time to one buffer: a kernel parameter, a `__shared__` array, a
`__constant__`/`__device__` variable, or a local array. Pointer offsets can be dynamic. Pointers stored in memory,
pointer-to-pointer parameters and function pointers are rejected with an explanation. This is what lets the same
kernel become WGSL, which has no raw pointers.

`__constant__` and `__device__` variables are allocated per module, with their initializers. They're readable and
writable from the host with `memcpyToSymbol` / `memcpyFromSymbol`.

## Tensors and autograd

See [Python API](#python-api-tinygradcuda). The integration is a precompiled-program node in tinygrad's graph:

- tinygrad schedules it with its other kernels, which allows lazy execution, fusion around it and `TinyJit` capture;
- on `WEBGPU` it dispatches into tinygrad's own command encoder on tinygrad's buffers, so no data moves;
- autograd uses the `CUDAFunction.backward` you write.

On the CPU device, the program runs on the WebAssembly backend over tinygrad's host buffers.

## The compiler

```
CUDA source ─▶ preprocessor ─▶ parser (AST) ─▶ semantic analysis + lowering ─▶ kernel IR ─▶ passes ─┬▶ WGSL ─▶ WebGPU
                (#define, #if,    (C++ subset:   (types, overloads, templates,    (structured,       ├▶ WebAssembly ─▶ CPU workers
                 #include of       templates,     inlining, structs → scalars,     SSA-like)          └▶ (original source) ─▶ NVRTC on a remote GPU
                 built-in headers) structs, …)   pointers → buffer + offset)
```

- **Preprocessor:** object-like and function-like macros, `#`, `##`, `__VA_ARGS__`, conditionals, and `#pragma unroll`.
  Built-in headers cover `cuda_runtime.h`, `cuda_fp16.h`, `cuda_bf16.h`, `curand_kernel.h`, `<cmath>`, `<cstdint>`,
  `<cooperative_groups.h>` (`this_thread_block()`, `thread_block::sync/thread_rank/size/…`, and `tiled_partition<N>`
  with `shfl*`, `ballot`, `any` and `all`), and others.
- **Language:**
  - functions: `__global__`, `__device__` and `__host__ __device__` functions, with overloading and default arguments
  - templates: function and class templates (type and non-type parameters, explicit and deduced arguments, explicit
    instantiation, member templates), `constexpr`, `static constexpr` members
  - types: structs with methods and operators, `enum` and `enum class`, references, `auto`, casts, vector types
    (`float4`, `int2`, …, `make_*`)
  - statements: `switch`, `do`/`while`/`for`, `break`/`continue`, early `return`, and the ternary operator
  - memory spaces and qualifiers: `__shared__` (static and `extern` dynamic), `__constant__`, `__device__` globals,
    `__restrict__`, `__launch_bounds__`
- **Kernel IR:** structured statements (let, store, atomic, if, loop, barrier, warp op, printf) over pure expressions.
  Memory is addressed as (root buffer, byte offset). Device functions are inlined and structs are split into scalars.
  View it with `inspect(…, "ir")`.
- **Passes** (`mode: "optimized"`, the default): exact constant folding with C semantics, constant and copy
  propagation, CSE, dead-code elimination, loop unrolling (`#pragma unroll` and small constant loops), loop-invariant
  code motion, scalar promotion of private arrays, and barrier simplification. `mode: "direct"` skips them, which helps
  when comparing or debugging. Both are tested to give the same results.
- **WGSL backend:**
  - Buffers bind as typed arrays or words, as each buffer's accesses require, with atomic layouts where needed. Aliased
    pointer arguments share one binding.
  - Scalar parameters go through one uniform. Grids larger than `maxComputeWorkgroupsPerDimension` are split into
    several dispatches.
  - Barriers are uniform as WGSL requires: early returns before `__syncthreads()` are predicated.
  - `storageBarrier` is emitted only when global memory can carry data between threads.
- **WebAssembly backend:**
  - Each block runs as a loop over its threads, split into phases at barriers and warp operations; threads are not OS
    threads, so blocks of 1024 threads are cheap.
  - Divergent control flow is masked, and values that live across a barrier are spilled per thread.
  - Blocks are distributed over workers through an atomic counter in shared memory. The code is scalar WebAssembly:
    vectorizing across threads with WASM SIMD is not implemented yet.
- **Caching:** modules are parsed once. Each kernel is instantiated once per block size and dynamic-shared size, then
  compiled once per backend specialization (alias pattern, zero offsets); GPU pipelines and WASM modules are reused.
  `COMPILER_VERSION` is part of the cache keys.

## Debugging

- **Compile errors** are `CompileError`s with file:line:col diagnostics, and warnings are collected on the module
  (`m.warnings`). Unsupported constructs say what isn't supported and what to do instead.
- **Device `printf`** output is delivered at `synchronize()` (`ctx.onPrintf` receives it as it arrives). It supports
  `%d %i %u %x %X %o %c %f %e %g %%` with flags, width, precision and length modifiers. Literal text works, but `%s`
  arguments don't.
- **`assert(cond)`** and **`__trap()`** produce `cudaErrorAssert` / `cudaErrorLaunchFailure`, naming the kernel, thread
  and block.
- **Out-of-bounds accesses:** WebGPU silently clamps them, so for debugging compile with `{ boundsCheck: true }` to get
  `cudaErrorIllegalAddress: k: out-of-bounds store to 'a' at element 16 by thread (16,0,0) in block (0,0,0)`. The CPU
  backend always checks. Errors are sticky until `resetError()`, as in CUDA.
- **`inspect()`** shows the IR after the passes and the generated WGSL or WebAssembly, with notes (for example
  "warp operations emulated: subgroup size is not 32").

## Performance and optimization

- The usual CUDA advice applies to both backends: coalesced accesses, shared-memory tiling and register blocking. In
  the benchmarks below, `matmul_regtile` runs 3.1–3.6× faster than `matmul_naive` on the CPU backend and 1.5–1.8×
  faster on WebGPU. `flash_attention_fwd` beats `sdpa_naive` by 5.2× on WebGPU and 2.1× on the CPU.
- Block size matters differently per backend. GPUs want 64–256 threads. On the CPU backend, larger blocks amortize the
  per-block overhead.
- On WebGPU a software adapter (SwiftShader) is an emulation, so measure on real GPUs before drawing conclusions. AUTO
  avoids it.
- `autotune()` picks launch configurations per device.
- Don't assume a translated CUDA kernel is faster than tinygrad's own operations; measure it. On the WEBGPU device,
  tinygrad's generated vector add, matmul and softmax kernels were 1.5–12× faster than the library's CUDA kernels in
  our run. On the interpreted CPU device, CUDA kernels on the WebAssembly backend were 68–283× faster. See
  [Benchmarks](#benchmarks).

## Kernel library

`kernels.sources.<group>` (JS) or `library("<group>")` (Python) return CUDA source. Every kernel has a numerical test
against a host reference on both backends ([`tests/cuda/cases.ts`](../tests/cuda/cases.ts)).

| group | kernels |
|---|---|
| vectorAdd | `vector_add`, `vector_add4` (float4), `saxpy`, `elementwise`, `fill`, `scale` |
| matmul | `matmul_naive`, `matmul_tiled<TILE>`, `matmul_regtile<BM,BN,BK,TM,TN>`, `matmul_batched` |
| transpose | `transpose_naive`, `transpose_tiled` (padded shared tile) |
| reduction | `reduce_sum<BLOCK>`, `reduce_sum_warp<BLOCK>` (shuffles), `row_max<BLOCK>` |
| scan | `scan_block<BLOCK>` + `scan_add<BLOCK>` (multi-block), `scan_blelloch<BLOCK>` |
| histogram | `histogram256` (shared-memory atomics), `histogram_float` |
| activations | relu, gelu, silu forward/backward, `bias_act` |
| softmax | `softmax_rows<BLOCK>`, `softmax_rows_bwd<BLOCK>` |
| norms | `layernorm_fwd/bwd<BLOCK>`, `rmsnorm_fwd/bwd<BLOCK>` |
| conv | `conv2d_fwd`, `maxpool2d_fwd/bwd`, `avgpool2d_fwd` |
| embedding, crossEntropy | `embedding_fwd/bwd` (atomic scatter-add), `cross_entropy_fwd_bwd<BLOCK>` |
| attention, rope, kvcache | `sdpa_naive` (causal, GQA), `flash_attention_fwd<BR,BC,DMAX>`, `rope_apply/bwd`, `kv_cache_append`, `add_layernorm<BLOCK>` |
| quantized, sparse, moe | `int8_matmul`, `int8_matmul_dp4a`, `quantize_rows_int8`, `int4_matmul` (grouped scales), `matmul_f16`, `csr_spmv`, `csr_spmm`, `moe_topk`, `moe_dispatch`, `moe_combine` |
| optimizers, mixedPrecision | `sgd_step` (momentum, Nesterov, weight decay), `adam_step` (Adam/AdamW), `sumsq` + `clip_by_norm`, fp16/bf16 casts, `unscale_check`, `master_step` |

## Compatibility matrix

**Supported:** everything listed under [The compiler](#the-compiler) and [Devices](#devices-and-selection). That
includes:
- the math library (`sinf`, `expf`, `__expf`, `rsqrtf`, `erff`, `lgammaf`, `fmaf`, `frexpf`, `sincospif`, …, in float
  and double) and IEEE conversions with explicit rounding (`__float2int_rn/rz/rd/ru`, …)
- integer intrinsics (`__popc`, `__clz`, `__ffs`, `__brev`, `__byte_perm`, `__funnelshift_*`, `__mulhi`, `__dp4a`, …)
- half and bfloat16 arithmetic (`__hadd`, `__hfma`, `__half2`, `__hmul2`, …)
- atomics (`atomicAdd` / `Sub` / `Exch` / `Min` / `Max` / `Inc` / `Dec` / `And` / `Or` / `Xor` / `CAS`)
- barriers (`__syncthreads`, `__syncthreads_count/and/or`, `__syncwarp`) and fences (`__threadfence*`)
- warp intrinsics, both `_sync` forms and lane masks
- the cache hints `__ldg`, `__ldcg`, …, which are accepted as plain loads

**Rejected, each with an explanation:**

| feature | why / instead |
|---|---|
| host code (`main`, `cudaMalloc` in C++) | host functions are skipped; launch from JS or Python (or `m.exec("k<<<g, b>>>(…)")`) |
| dynamic parallelism (launching kernels from kernels) | not expressible on WebGPU |
| recursion, function pointers, virtual functions, lambdas in device code | everything is inlined |
| pointers stored in memory, pointers to pointers, pointer-to-integer casts, a pointer that may refer to different buffers | each pointer must map to one buffer at compile time (WGSL has no raw pointers) |
| `double` on WebGPU (unless `fp64: "demote"`), 64-bit atomics on WebGPU | WGSL has no f64 or 64-bit atomics; use the CPU or remote backend |
| `clock()`, `clock64()`, `__globaltimer` | browsers expose no per-thread cycle counter; time launches with events |
| legacy warp intrinsics without `_sync` (`__shfl`, `__ballot`, `__any`, `__all`) | removed in CUDA 9 for sm_70+; use the `_sync` forms |
| `%p` and `%s` with non-literal arguments in `printf`, `*` width | no pointers or strings at runtime |
| inheritance, bit-fields, variadic templates, partial specialization, range-based `for`, `goto`, `static` locals, destructors | not implemented in this compiler |
| inline PTX (`asm(...)`), texture and surface objects, grid-wide cooperative groups (`this_grid().sync()`), `cg::reduce`, tensor cores (`wmma`), `cuda::pipeline`/`memcpy_async` | not implemented |

## Differences from native CUDA

- **Floating point on WebGPU.** WGSL guarantees fewer things than CUDA:
  - Division and some transcendentals are not correctly rounded (WGSL allows 2.5 ulp for `/`).
  - Shader compilers may reassociate and contract float arithmetic.
  - Implementations may ignore the sign of zero.

  Integer results are exact; float results can differ in the last bits (the differential tests allow 4 ulp on WebGPU).
  The CPU backend computes IEEE single and double exactly as written: no reassociation, no FMA contraction.
- **Scheduling.** Blocks run in some order with no forward-progress guarantee between blocks, as on GPUs. A kernel that
  spins waiting for another block can hang (the remote worker kills such launches after its time limit).
  Inter-block spin-waits are undefined behaviour in CUDA too.
- **Warps** are 32 lanes everywhere. Independent thread scheduling within a warp isn't modelled: lanes of a warp
  execute in lockstep with masking, so code that relies on `__syncwarp` for correctness works, and code that relies on
  divergent lanes progressing independently does not.
- **Streams** run in submission order: correct, but with no concurrency between streams.
- **Errors** are reported at the next synchronization, as with asynchronous CUDA launches.

## Browser and hardware support

| environment | WebGPU backend | CPU backend |
|---|---|---|
| Chrome / Edge 113+ (Windows, macOS, ChromeOS, Android) | yes; subgroups where the `subgroups` feature is available | yes; worker pool needs cross-origin isolation (COOP/COEP) |
| Firefox 141+ (Windows), Safari 26+ | yes (feature support varies: check `ctx.info().features`) | yes |
| other browsers, workers without WebGPU | no: AUTO picks the CPU | yes |
| Node.js 18+ | no WebGPU | yes (`worker_threads`) |
| servers with NVIDIA GPUs | — | — (remote backend: NVRTC + driver, sm_50+) |

Tested in CI: Chromium with SwiftShader (software WebGPU) and Node.js. Real-GPU adapters behave the same through the
same WebGPU API, but their numbers aren't covered by the benchmarks below.

## Security

- **Kernels in the browser** run inside WebGPU's and WebAssembly's sandboxes:
  - WebGPU clamps out-of-bounds accesses, and WebAssembly traps on them, so a kernel can't reach memory outside its
    buffers.
  - Long-running GPU work can be killed by the browser's GPU watchdog.
  - Memory per context is capped with `memoryLimit`.
- **Remote execution** is off unless you configure an endpoint, and AUTO never selects a remote (possibly paid) GPU.
  Browsers hold only short-lived session tokens issued by your backend, never cloud credentials. The worker isolates
  each session in its own process with limits (see [`server/cuda-worker/README.md`](../server/cuda-worker/README.md)).
- Kernel bundles from elsewhere are checked against their sha256 (`deserializeKernel`).

## Testing

```bash
npm run build
npx playwright test --project=cuda-node      # compiler unit tests, CPU backend (inline + 4 workers), runtime API, remote worker end to end
npx playwright test --project=cuda-browser   # WebGPU backend, CPU worker pool in a browser, tinygrad.cuda on WEBGPU and CPU, playground
```

The suites cover the following:

- **Numerical cases:** 27 cases with tolerances, covering the whole kernel library, every run on WebGPU, on the CPU
  inline, and on 4 workers.
- **Randomized differential tests:** 200 kernels on the CPU and 150 on WebGPU. Random integer and float expression
  kernels run with runtime inputs and must equal the compiler's exact constant folding of the same expressions.
- **Compiler units:** lexer, preprocessor, parser, lowering, passes, WGSL memory-ordering rules, and printf formatting.
- **tinygrad integration:**
  - views and in-place writes, aliasing, templates, structs, lazy launches, `TinyJit`, printf and errors
  - custom autograd against tinygrad's autograd
  - MLP training against pure tinygrad
  - transformer-block inference with flash attention, RoPE and RMSNorm
- **Remote worker:** a real gateway process covering tokens, CORS, limits, the timeout kill, sticky faults and session
  ownership. Kernels execute on the WebAssembly engine in tests; `TGJS_CUDA_TEST_NATIVE=1` runs the same tests against
  a GPU.

## Benchmarks

`npm run bench:cuda` (`benchmarks/cuda.mjs`) measures every library workload through CUDA→WGSL (optimized and direct),
CUDA→WASM (pool and one thread), and handwritten WGSL. It then compares CUDA kernels on tinygrad tensors with the
equivalent tinygrad operations. Results are verified before they're timed. It reports compile time, transfer time,
steady-state execution, single-launch latency and GPU timestamps separately. Results go to
`benchmarks/results/cuda-*.{json,md}`; browse them with [`benchmarks/dashboard.html`](../benchmarks/dashboard.html).

The measured results from the development machine are in
[`benchmarks/results/cuda-swiftshader-chromium-linux.md`](../benchmarks/results/cuda-swiftshader-chromium-linux.md).
That machine has only a software WebGPU adapter, so its "GPU" numbers measure SwiftShader running on the CPU, not GPU
hardware.
