"""Typed storage classes, only as far as torch.save / torch.load need them."""
from __future__ import annotations
import numpy as np

class _Storage:
  _np = np.float32
  def __init__(self, array=None): self._array = array
  def __len__(self): return len(self._array)
  def tolist(self): return self._array.tolist()

def _make(name, npd):
  cls = type(name, (_Storage,), {"_np": npd})
  cls.__module__ = "torch"   # PyTorch pickles storages as torch.FloatStorage etc.
  return cls
FloatStorage, DoubleStorage, HalfStorage = _make("FloatStorage", np.float32), _make("DoubleStorage", np.float64), _make("HalfStorage", np.float16)
LongStorage, IntStorage, ShortStorage = _make("LongStorage", np.int64), _make("IntStorage", np.int32), _make("ShortStorage", np.int16)
CharStorage, ByteStorage, BoolStorage = _make("CharStorage", np.int8), _make("ByteStorage", np.uint8), _make("BoolStorage", np.bool_)
BY_NUMPY = {np.dtype(c._np): c for c in (FloatStorage, DoubleStorage, HalfStorage, LongStorage, IntStorage, ShortStorage, CharStorage,
                                         ByteStorage, BoolStorage)}
