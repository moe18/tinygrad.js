"""tinygrad.js: run upstream tinygrad in the browser (Pyodide) on the browser's WebGPU.

Usage inside Pyodide (the TypeScript runtime does this for you)::

    import tinygrad_js
    tinygrad_js.install(bridge, default_device="WEBGPU")   # preferably before `import tinygrad`
    from tinygrad import Tensor
"""
from __future__ import annotations
import os, sys

__all__ = ["install", "install_cuda", "install_torch", "stats", "device_info", "applied_patches", "run_training", "TrainingResult"]

from .training import run_training, TrainingResult  # imports tinygrad lazily, at call time

# Browser constraints, applied through tinygrad's own configuration knobs:
# PARALLEL=0: tinygrad >= 0.13 compiles kernels in a multiprocessing pool; Pyodide has no processes.
BROWSER_ENV = {"PARALLEL": "0"}

def _set_knob(key:str, value:str) -> None:
  os.environ[key] = value
  helpers = sys.modules.get("tinygrad.helpers")
  var = getattr(helpers, key, None) if helpers is not None else None
  if var is not None and hasattr(var, "value"): var.value = int(value) if value.isdigit() else value  # already imported

def install(bridge=None, default_device:str|None="WEBGPU") -> None:
  """Register the browser devices with tinygrad and (optionally) choose the default device.

  * ``bridge``: the JavaScript WebGPU bridge object (``src/bridge.ts``). ``None`` means no WEBGPU device.
  * ``default_device``: ``"WEBGPU"``, ``"CPU"``/``"PYTHON"`` (tinygrad's pure-Python interpreter) or ``None``.
  """
  for key, value in BROWSER_ENV.items():
    if key not in os.environ: _set_knob(key, value)
  if default_device is not None:
    _set_knob("DEV", default_device)
    dev_mod = sys.modules.get("tinygrad.device")
    if dev_mod is not None: dev_mod.Device.__dict__.pop("DEFAULT", None)  # tinygrad 0.12 caches DEFAULT
  # tinygrad resolves devices with importlib.import_module(f"tinygrad.runtime.ops_{name}"), which returns the entries
  # placed in sys.modules here. This overrides upstream's native modules without modifying tinygrad's files.
  if bridge is not None:
    from . import webgpu
    webgpu.set_bridge(bridge)
    sys.modules["tinygrad.runtime.ops_webgpu"] = webgpu
  from . import cpu  # upstream's CPU device needs a native compiler; in the browser "CPU" is interpreted (cpu.py)
  sys.modules["tinygrad.runtime.ops_cpu"] = cpu
  from . import patches
  patches.apply_all()

def install_cuda(host) -> None:
  """Make ``tinygrad.cuda`` (CUDA C/C++ kernels on tensors, python/tinygrad_js/cuda.py) importable, backed by the JS host."""
  from . import cuda
  cuda.set_host(host)
  sys.modules["tinygrad.cuda"] = cuda
  tg = sys.modules.get("tinygrad")
  if tg is not None: setattr(tg, "cuda", cuda)

def install_torch(force:bool=False) -> bool:
  """Make ``import torch`` resolve to tinygrad.js's PyTorch-compatible shim (``tinygrad_js/_shims/torch``), which implements the
  common part of the PyTorch API on top of tinygrad. It is not PyTorch; see docs/TORCH.md.

  Returns True if the shim is (now) importable as ``torch``. Does nothing and returns False if a real ``torch`` package is
  installed, unless ``force`` is set. Raises ``RuntimeError`` if tinygrad is too old for the shim (it needs tinygrad >= 0.14).
  """
  import importlib.util
  shim_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_shims")
  if shim_dir in sys.path: return True
  loaded = sys.modules.get("torch")
  if loaded is not None and not getattr(loaded, "__tinygrad_js_shim__", False) and not force: return False
  if not force and importlib.util.find_spec("torch") is not None: return False
  if (reason := torch_shim_unavailable_reason()) is not None: raise RuntimeError(reason)
  if force:
    for name in [n for n in sys.modules if n == "torch" or n.startswith("torch.")]: del sys.modules[name]
  sys.path.insert(0, shim_dir)
  importlib.invalidate_caches()
  return True

def torch_shim_unavailable_reason() -> str|None:
  """None if the torch shim can run on the installed tinygrad, otherwise why not."""
  from tinygrad.uop.ops import UOp
  if all(hasattr(UOp, a) for a in ("has_buffer_identity", "is_virtual")): return None
  import importlib.metadata
  return f"the tinygrad.js torch shim needs tinygrad >= 0.14 (found {importlib.metadata.version('tinygrad')})"

def applied_patches() -> list[str]:
  from . import patches
  return list(patches.applied)

def stats() -> dict:
  """Counters from the JS bridge (kernels compiled/dispatched, bytes moved, GPU memory) and tinygrad."""
  from . import webgpu
  out:dict = {}
  if webgpu._bridge is not None: out.update(webgpu._bridge.stats().to_py())
  try:
    from tinygrad.helpers import GlobalCounters
    out["tinygrad_kernel_count"] = GlobalCounters.kernel_count
    out["tinygrad_mem_used"] = GlobalCounters.mem_used
  except Exception: pass
  return out

def device_info() -> dict:
  from . import webgpu
  return webgpu._bridge.info().to_py() if webgpu._bridge is not None else {}
