"""Functions that PyTorch checkpoints refer to by name (``torch._utils._rebuild_tensor_v2`` ...)."""
from __future__ import annotations
import numpy as np

def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad=False, backward_hooks=None, metadata=None):
  from ._tensor import tensor
  a = storage._array
  size, stride = tuple(size), tuple(stride)
  if len(size) == 0: arr = a[storage_offset:storage_offset + 1].reshape(())
  else: arr = np.lib.stride_tricks.as_strided(a[storage_offset:], shape=size, strides=tuple(s * a.itemsize for s in stride))
  return tensor(np.array(arr), requires_grad=requires_grad)   # np.array copies and keeps 0-d shapes
_rebuild_tensor = _rebuild_tensor_v2

def _rebuild_parameter(data, requires_grad, backward_hooks):
  from .nn.parameter import Parameter
  return Parameter(data, requires_grad)

def _rebuild_parameter_with_state(data, requires_grad, backward_hooks, state):
  p = _rebuild_parameter(data, requires_grad, backward_hooks)
  if isinstance(state, dict): p.__dict__.update(state)
  return p

def _rebuild_from_numpy(array, requires_grad=False):
  """Used by plain ``pickle`` / ``copy`` of a shim tensor (torch.save writes PyTorch's own format instead)."""
  from ._tensor import tensor
  return tensor(array, requires_grad=requires_grad)
