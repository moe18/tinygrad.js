"""torch.save / torch.load in PyTorch's zip checkpoint format, so files are interchangeable with real PyTorch."""
from __future__ import annotations
import collections, io, os, pickle, zipfile
import numpy as np
from ._tensor import Tensor, Size, device, dtype
from . import _utils, storage as _storage

class _StorageRef:
  def __init__(self, key, array): self.key, self.array = key, array

class _Pickler(pickle.Pickler):
  def __init__(self, f):
    super().__init__(f, protocol=2)
    self.storages: list[_StorageRef] = []
  def persistent_id(self, obj):
    if isinstance(obj, _StorageRef): return ("storage", _storage.BY_NUMPY[obj.array.dtype], obj.key, "cpu", obj.array.size)
    return None
  def reducer_override(self, obj):
    if isinstance(obj, Tensor):
      from .nn.parameter import Parameter
      # a Parameter is rebuilt from a plain tensor, which is pickled by the branch below
      if isinstance(obj, Parameter): return (_utils._rebuild_parameter, (obj.detach(), obj.requires_grad, collections.OrderedDict()))
      a = np.array(obj._numpy(), order="C")   # np.array keeps 0-d shapes (np.ascontiguousarray would not)
      ref = _StorageRef(str(len(self.storages)), a.reshape(-1))
      self.storages.append(ref)
      strides = tuple(s // a.itemsize for s in a.strides)
      return (_utils._rebuild_tensor_v2, (ref, 0, tuple(a.shape), strides, obj.requires_grad, collections.OrderedDict()))
    return NotImplemented

def save(obj, f, pickle_module=None, pickle_protocol=2, _use_new_zipfile_serialization=True):
  """Save ``obj`` (state dicts, tensors, nested containers, modules) as a PyTorch zip checkpoint."""
  buf = io.BytesIO()
  p = _Pickler(buf)
  p.dump(obj)
  name = "archive"
  out = f if hasattr(f, "write") else open(os.fspath(f), "wb")
  try:
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
      z.writestr(f"{name}/data.pkl", buf.getvalue())
      z.writestr(f"{name}/byteorder", "little")
      for ref in p.storages: z.writestr(f"{name}/data/{ref.key}", ref.array.astype(ref.array.dtype.newbyteorder("<"), copy=False).tobytes())
      z.writestr(f"{name}/version", "3\n")
  finally:
    if out is not f: out.close()

_SAFE = {
  ("collections", "OrderedDict"): collections.OrderedDict, ("builtins", "set"): set, ("builtins", "frozenset"): frozenset,
  ("builtins", "dict"): dict, ("builtins", "list"): list, ("builtins", "tuple"): tuple, ("builtins", "slice"): slice,
  ("builtins", "complex"): complex, ("builtins", "bytearray"): bytearray,
  ("torch._utils", "_rebuild_tensor_v2"): _utils._rebuild_tensor_v2, ("torch._utils", "_rebuild_tensor"): _utils._rebuild_tensor_v2,
  ("torch._utils", "_rebuild_parameter"): _utils._rebuild_parameter,
  ("torch._utils", "_rebuild_parameter_with_state"): _utils._rebuild_parameter_with_state,
  ("torch._utils", "_rebuild_from_numpy"): _utils._rebuild_from_numpy,
  ("torch", "Size"): Size, ("torch", "device"): device,
}

class _Unpickler(pickle.Unpickler):
  def __init__(self, f, zf, prefix, weights_only):
    super().__init__(f)
    self.zf, self.prefix, self.weights_only, self.cache = zf, prefix, weights_only, {}
  def find_class(self, module, name):
    if (module, name) in _SAFE: return _SAFE[(module, name)]
    if module == "torch" and name.endswith("Storage") and hasattr(_storage, name): return getattr(_storage, name)
    if module == "torch":
      import torch
      if isinstance(getattr(torch, name, None), dtype): return getattr(torch, name)
    if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name in ("scalar", "_reconstruct"): return getattr(np.core.multiarray if hasattr(np, "core") else np._core.multiarray, name)
    if module == "numpy" and name in ("dtype", "ndarray"): return getattr(np, name)
    if not self.weights_only: return super().find_class(module, name)
    raise pickle.UnpicklingError(f"torch.load(weights_only=True): unsupported global {module}.{name}. "
                                 "Pass weights_only=False to unpickle arbitrary objects from a file you trust.")
  def persistent_load(self, pid):
    kind, stype, key, _location, numel = pid[:5]
    if kind != "storage": raise pickle.UnpicklingError(f"unknown persistent id {kind!r}")
    if key not in self.cache:
      raw = self.zf.read(f"{self.prefix}/data/{key}")
      self.cache[key] = stype(np.frombuffer(raw, dtype=np.dtype(stype._np).newbyteorder("<")).astype(stype._np, copy=False))
    return self.cache[key]

def load(f, map_location=None, pickle_module=None, *, weights_only=True, mmap=None, **kwargs):
  """Load a PyTorch zip checkpoint (written by real PyTorch >= 1.6 or by this shim). ``map_location`` is ignored."""
  data = f.read() if hasattr(f, "read") else open(os.fspath(f), "rb").read()
  if not zipfile.is_zipfile(io.BytesIO(data)):
    raise RuntimeError("torch.load: only PyTorch's zip checkpoint format (PyTorch >= 1.6, the default) is supported by the tinygrad.js shim")
  with zipfile.ZipFile(io.BytesIO(data)) as zf:
    pkl = next((n for n in zf.namelist() if n.endswith("/data.pkl")), None)
    if pkl is None: raise RuntimeError("torch.load: not a PyTorch checkpoint (no data.pkl)")
    return _Unpickler(io.BytesIO(zf.read(pkl)), zf, pkl[:-len("/data.pkl")], weights_only).load()
