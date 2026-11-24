# Shared helpers for the PyTorch test programs. Each program is plain PyTorch code that runs unmodified under real PyTorch
# (reference, tests/reference/generate_torch.py) and under the tinygrad.js torch shim (natively and in the browser).
# The programs avoid framework random numbers wherever values are compared: the shim's random streams differ from PyTorch's.
import json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def _device_name():
  if getattr(torch, "__tinygrad_js_shim__", False):
    from tinygrad import Device
    return Device.DEFAULT
  return "torch"

def emit(**kw):
  def num(x):  # JSON has no NaN/Infinity
    return x if not isinstance(x, float) or math.isfinite(x) else ("nan" if math.isnan(x) else "inf" if x > 0 else "-inf")
  def conv(v):
    if isinstance(v, torch.Tensor): v = v.detach().cpu().numpy()
    if isinstance(v, np.ndarray):
      # dtype is reported as a kind: the shim narrows 64-bit types to 32 bits on WebGPU
      kind = {"f": "float", "i": "int", "u": "int", "b": "bool"}[v.dtype.kind]
      return {"shape": list(v.shape), "dtype": kind, "data": [num(x) for x in v.astype(np.float64).ravel().tolist()]}
    if isinstance(v, float): return num(v)
    if isinstance(v, torch.Size): return list(v)
    if isinstance(v, (list, tuple)): return [conv(x) for x in v]
    if isinstance(v, dict): return {str(k): conv(x) for k, x in v.items()}
    if isinstance(v, (np.floating, np.integer)): return num(v.item())
    if isinstance(v, np.bool_): return bool(v)
    return v
  print("RESULT_JSON:" + json.dumps({"device": _device_name(), **{k: conv(v) for k, v in kw.items()}}))

def det(*shape, seed=0, scale=1.0):
  """Deterministic pseudo-random float32 tensor that does not depend on the framework's RNG."""
  rng = np.random.RandomState(seed)
  return torch.tensor((rng.standard_normal(shape) * scale).astype(np.float32))

def det_int(*shape, high, seed=0):
  return torch.tensor(np.random.RandomState(seed).randint(0, high, size=shape).astype(np.int64))

def det_init(model, seed=100, scale=0.2):
  """Overwrite every parameter with deterministic values (in registration order)."""
  with torch.no_grad():
    for i, p in enumerate(model.parameters()):
      p.copy_(det(*p.shape, seed=seed + i, scale=scale))
  return model
