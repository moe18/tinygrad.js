"""Browser WebGPU device for upstream tinygrad, running inside Pyodide.

This module replaces ``tinygrad.runtime.ops_webgpu`` at runtime (see ``tinygrad_web.install``). Upstream's
native module drives Dawn through ctypes, which cannot work in WebAssembly; this one reuses everything else
from tinygrad (scheduler, codegen, ``WGSLRenderer``, ``LRUAllocator``) and sends the final WGSL kernels and
buffer operations to the browser's ``navigator.gpu`` through a small JavaScript bridge (``src/bridge.ts``).

The device keeps the name ``WEBGPU`` so tinygrad's existing WEBGPU-specific rules still apply (8 storage
buffers per kernel, no buffer views, the dtype support table).

Synchronous points (buffer readback, ``synchronize``, pipeline compilation) await JavaScript promises with
``pyodide.ffi.run_sync``, which needs JSPI (WebAssembly JavaScript Promise Integration) and Python code that
is running inside ``runPythonAsync``.
"""
from __future__ import annotations
import functools, inspect, time
from tinygrad.device import Compiled, LRUAllocator, BufferSpec
from tinygrad.renderer.wgsl import WGSLRenderer
from tinygrad.helpers import round_up

_bridge = None  # the JS bridge object, set by tinygrad_web.install()

def set_bridge(bridge) -> None:
  global _bridge
  _bridge = bridge

def get_bridge():
  if _bridge is None: raise RuntimeError("tinygrad_web: no WebGPU bridge installed; call tinygrad_web.install(bridge) first")
  return _bridge

def _await(promise):
  """Block on a JS promise. Requires JSPI and being called from code started by runPythonAsync."""
  from pyodide.ffi import run_sync
  try: return run_sync(promise)
  except Exception as e:
    # surface the JS error message as a normal Python RuntimeError
    msg = str(e)
    if "can_run_sync" in msg or "stack switching" in msg.lower() or "JSPI" in msg:
      raise RuntimeError("tinygrad_web: blocking on WebGPU requires JSPI (WebAssembly JavaScript Promise Integration) and code "
                         f"executed through runPythonAsync. Original error: {msg}") from None
    raise RuntimeError(f"WebGPU: {msg}") from None

# tinygrad's device API changed between releases; detect which one we are talking to.
_COMPILED_PARAMS = inspect.signature(Compiled.__init__).parameters
NEW_DEVICE_API = "renderers" in _COMPILED_PARAMS        # tinygrad >= 0.13: Compiled(device, allocator, renderers, runtime_t)

class WebGPUProgram:
  """One compiled WGSL kernel. Constructed as (dev, TinyELF) on tinygrad >= 0.13, or (dev, name, lib) on 0.12."""
  def __init__(self, dev:WebGpuDevice, name_or_obj, lib:bytes|None=None):
    if lib is None: name, lib = name_or_obj.name, name_or_obj.lib
    else: name = name_or_obj
    self.dev, self.name = dev, name
    self.src = lib.decode() if isinstance(lib, (bytes, bytearray)) else str(lib)
    self.pid = dev.compile(name, self.src)

  def __call__(self, *bufs, global_size=(1,1,1), local_size=(1,1,1), vals=(), wait=False, **kw) -> float|None:
    gx, gy, gz = (int(g) for g in (tuple(global_size) + (1,1,1))[:3])
    if wait: st = time.perf_counter()
    # pass plain ints only: they cross the Python->JS boundary without allocating proxies
    self.dev.bridge.dispatch(self.pid, gx, gy, gz, len(bufs), *bufs, *(int(v) for v in vals))
    if wait:
      self.dev.synchronize()
      return time.perf_counter() - st
    return None

  def __del__(self):
    try: self.dev.bridge.releaseProgram(self.pid)
    except Exception: pass

class WebGpuAllocator(LRUAllocator):
  """GPU buffers live in the JS bridge and are referenced from Python by integer handles."""
  def _alloc(self, size:int, options:BufferSpec) -> int:
    # WebGPU buffers have to be 4-byte aligned
    try: return self.dev.bridge.createBuffer(round_up(size, 4))
    except Exception as e: raise MemoryError(str(e)) from None
  def _free(self, opaque:int, options:BufferSpec): self.dev.bridge.destroyBuffer(opaque)
  def _copyin(self, dest:int, src:memoryview): self.dev.bridge.writeBuffer(dest, src)
  def _copyout(self, dest:memoryview, src:int):
    data = _await(self.dev.bridge.readBuffer(src, dest.nbytes))
    data.assign_to(dest.cast("B") if dest.format != "B" else dest)

class WebGpuDevice(Compiled):
  def __init__(self, device:str):
    self.bridge = get_bridge()
    info = self.bridge.info().to_py()
    self.info, self.features = info, list(info.get("features", []))
    arch = "shader-f16" if "shader-f16" in self.features else None
    allocator = WebGpuAllocator(self)
    if NEW_DEVICE_API:
      super().__init__(device, allocator, [WGSLRenderer], WebGPUProgram, **({"arch": arch} if arch and "arch" in _COMPILED_PARAMS else {}))
    else:  # tinygrad 0.12
      from tinygrad.device import CompilerSet, CompilerPair, Compiler
      super().__init__(device, allocator, CompilerSet([CompilerPair(WGSLRenderer, Compiler)]), functools.partial(WebGPUProgram, self))

  def compile(self, name:str, src:str) -> int: return _await(self.bridge.compile(name, src))
  def synchronize(self): _await(self.bridge.sync())
