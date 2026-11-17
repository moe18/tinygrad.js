"""The (only) behavioural changes tinygrad.js makes to upstream tinygrad, applied at runtime by ``install()``.

Each patch is small, guarded by a check that the upstream code still looks the way the patch expects, and listed in
``applied`` (reported to JavaScript as a diagnostic). See docs/COMPATIBILITY.md.
"""
from __future__ import annotations
import inspect

applied: list[str] = []

def realize_deviceless_on_default() -> None:
  """tinygrad >= 0.13 realizes tensors whose graph has no device (e.g. built only from ``Tensor.arange`` and Python
  scalars) on the hard-coded ``"CPU"`` device in ``Tensor._buffer``. In the browser that device is tinygrad's Python
  interpreter, so instead realize them on ``Device.DEFAULT`` (WEBGPU) and only then copy to the host, like any other
  tensor. Numerics are unchanged; only where the kernel runs differs.

  Upstream candidate: let ``Tensor._buffer`` clone device-less tensors to ``Device.DEFAULT``, or make the host device
  configurable.
  """
  from tinygrad.tensor import Tensor
  from tinygrad.device import Device
  orig = Tensor._buffer
  if getattr(orig, "_tinygrad_js", False): return
  try: src = inspect.getsource(orig)
  except (OSError, TypeError): return
  if 'clone("CPU")' not in src: return  # not needed for this tinygrad version (e.g. 0.12 has no device-less tensors)
  def _buffer(self):
    if self.uop.device is None and Device.DEFAULT != "CPU": return orig(self.clone(Device.DEFAULT))
    return orig(self)
  _buffer._tinygrad_js = True  # type: ignore[attr-defined]
  _buffer.__doc__, _buffer.__wrapped__ = orig.__doc__, orig  # type: ignore[attr-defined]
  Tensor._buffer = _buffer
  applied.append("Tensor._buffer: device-less tensors are realized on Device.DEFAULT instead of CPU")

def apply_all() -> list[str]:
  realize_deviceless_on_default()
  return applied
