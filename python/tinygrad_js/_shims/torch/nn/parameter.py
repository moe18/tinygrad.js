from __future__ import annotations
from .._tensor import Tensor, _unique
from tinygrad import Tensor as _TG

class Parameter(Tensor):
  """A leaf tensor that requires grad and is registered by ``nn.Module`` when assigned as an attribute."""
  def __new__(cls, data=None, requires_grad=True):
    w = Tensor.__new__(cls)
    t = _TG.zeros(0) if data is None else data._tt
    Tensor._init(w, _unique(t.detach()), False, True)
    w._persist = True
    if requires_grad: w.requires_grad = True
    return w
  def __init__(self, data=None, requires_grad=True): pass
  def __repr__(self): return "Parameter containing:\n" + Tensor.__repr__(self)
  __str__ = __repr__
  def __reduce__(self):
    from .._utils import _rebuild_parameter
    return (_rebuild_parameter, (self.detach(), self.requires_grad, None))

class Buffer(Tensor):
  def __new__(cls, data=None, persistent=True):
    w = Tensor.__new__(cls)
    Tensor._init(w, _unique(data._tt.detach()), False, True)
    w._persist, w.persistent = True, persistent
    return w
  def __init__(self, data=None, persistent=True): pass

class UninitializedParameter(Parameter): pass
