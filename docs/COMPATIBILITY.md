# Upstream compatibility strategy

tinygrad.js uses **unmodified upstream tinygrad wheels from PyPI**. It never forks or vendors tinygrad's source. Everything
browser-specific lives in `python/tinygrad_js/` and is plugged in at runtime.

## Integration points

| What | How | Where |
|---|---|---|
| `WEBGPU` device | `sys.modules["tinygrad.runtime.ops_webgpu"] = tinygrad_js.webgpu`. tinygrad resolves devices with `importlib.import_module("tinygrad.runtime.ops_<name>")`. | `__init__.py`, `webgpu.py` |
| `CPU` device | `sys.modules["tinygrad.runtime.ops_cpu"] = tinygrad_js.cpu`, which is `CpuDevice(PythonDevice)`: tinygrad's own pure-Python device under the name `CPU`. Upstream's CPU backend needs clang/LLVM, which wasm doesn't have. | `cpu.py` |
| Default device | Sets the `DEV` environment variable (or `helpers.DEV.value` if tinygrad is already imported). | `__init__.py` |
| No multiprocessing | `PARALLEL=0` (tinygrad's own knob). | `__init__.py` |
| **Behavioural patch** (1) | `Tensor._buffer` is wrapped so that device-less tensors are realized on `Device.DEFAULT` instead of the hard-coded `"CPU"`. It's only applied when upstream's source contains `clone("CPU")` (≥ 0.13). Numerics are identical; only where the kernel runs changes. | `patches.py` |

`tinygrad_js.applied_patches()` (and `RuntimeInfo.diagnostics` in JS) reports which patches are active.

### The device contract the backend relies on

From `tinygrad.device`: `Compiled.__init__`, `LRUAllocator` (`_alloc`, `_free`, `_copyin`, `_copyout`), `Program.__call__(*bufs,
global_size, local_size, vals, wait)`, and `WGSLRenderer`'s kernel ABI (binding 0 = `INFINITY` uniform, then one binding per buffer
argument, then one `i32` uniform per symbolic variable). Two shapes of this contract are supported:

| tinygrad | `Compiled.__init__` | Program constructor |
|---|---|---|
| 0.12.0 | `(device, allocator, CompilerSet([...]), runtime, ...)` | `runtime(name, lib)` |
| 0.13+ / 0.14.0 | `(device, allocator, renderers, runtime_t, arch=...)` | `runtime_t(dev, TinyELF)` |

`webgpu.py` picks one by inspecting `Compiled.__init__`'s parameters.

## Supported versions

| tinygrad | Status |
|---|---|
| **0.14.0** | default; full test suite passes |
| **0.12.0** | supported (`tinygradVersion: "0.12.0"`); full tensor/autograd/training/WebGPU suite passes |
| 0.13.0 | not tested |
| master | not tested; see "Bumping tinygrad" |

Which to use: 0.14.0 follows current upstream docs. 0.12.0 is the newest release that still accepts
`Tensor(..., requires_grad=True)` and `Tensor.train()`, which older tutorials and exercise code use.

## Bumping tinygrad

1. Add the wheel (URL + sha256) to `scripts/pins.mjs` and a `bundledWheelURL` case in `src/runtime.ts`.
2. `npm run fetch:assets && npm run build`.
3. Generate CPU references with native upstream tinygrad:
   `python -m venv v && v/bin/pip install tinygrad==X numpy && DEV=CPU v/bin/python tests/reference/generate.py`.
4. Add a Playwright project for the version in `playwright.config.ts` and run `npm test`.
   For the torch shim also run `DEV=CPU python tests/reference/check_torch_native.py` with the new tinygrad (see the section below).
5. Things that changed between releases before and are worth checking: `Compiled.__init__` and `Program` signatures,
   `TinyELF`, `Tensor._buffer` (patch guard), new hard-coded device names (`grep -rn '"CPU"' tinygrad`), new process/thread
   usage (`multiprocessing`, `subprocess`), and `WGSLRenderer`'s binding layout (`render_kernel`). The bridge parses bindings
   from the WGSL source, so a layout change shows up as a clear error, not silent corruption.

## The torch shim's dependencies on tinygrad

`import torch` ([docs/TORCH.md](TORCH.md)) is built on tinygrad's public `Tensor` API plus a few behaviours that are not part of a
stable contract. They are the first things to check when bumping tinygrad; `tests/reference/check_torch_native.py` and
`tests/torch.test.ts` exercise all of them.

| used by the shim | why |
|---|---|
| `Tensor.gradient(*targets, gradient=)` | `backward()` and `torch.autograd.grad` |
| `Ops.DETACH` in the UOp graph | finding the leaves a result depends on without crossing `detach()` |
| `Tensor.realize` replaces the realized node in every tensor, and realizing `x.contiguous()` leaves `x`'s graph intact | values can be read before `backward()` without losing the graph |
| `Tensor.assign` + one `realize` orders reads of the old value before the write | optimizer steps and in-place parameter updates |
| `UOp.has_buffer_identity()`, `UOp.is_virtual`, `Tensor.clone()` returning a tensor with its own identity | giving every leaf its own storage (equal constants share one graph node) |
| `Tensor._getitem(indices, value)` | functional `x[idx] = value` |
| `Device[...].renderer.supported_dtypes()` | narrowing 64-bit dtypes on WebGPU |

The shim checks for `has_buffer_identity`/`is_virtual` and reports "needs tinygrad >= 0.14" otherwise (tinygrad 0.12).

## Possible upstream contributions

* Let `Tensor._buffer` realize device-less tensors on `Device.DEFAULT`, or make the host device configurable. That would remove the
  only behavioural patch.
* Make `ops_webgpu.py`'s Dawn access pluggable (e.g. a small "WebGPU API" interface with ctypes-Dawn and JS implementations).
  The browser device could then live upstream.
* Skip the `multiprocessing` worker pool automatically when `sys.platform == "emscripten"`.
