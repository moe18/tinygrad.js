"""torch.nn.utils: gradient clipping and parameter/vector helpers."""
from __future__ import annotations
from ..._tensor import Tensor, _mk, no_grad
from . import rnn

def _params(parameters): return [parameters] if isinstance(parameters, Tensor) else list(parameters)

def clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None):
  """Scales gradients so their global norm is at most ``max_norm``. Returns the norm before clipping (as a tensor)."""
  grads = [p.grad for p in _params(parameters) if p.grad is not None]
  if not grads: return _mk(Tensor([0.0])._t.reshape(()))
  if norm_type == float("inf"):
    total = grads[0].abs().max()
    for g in grads[1:]: total = total.maximum(g.abs().max())
  else:
    total = sum((g.abs() ** norm_type).sum() for g in grads) ** (1.0 / norm_type)
  coef = (max_norm / (total + 1e-6)).clamp(max=1.0)
  with no_grad():
    for g in grads: g._inplace(g * coef)
  return total
clip_grad_norm = clip_grad_norm_
def get_total_norm(tensors, norm_type=2.0, **k): return sum((g.abs() ** norm_type).sum() for g in tensors) ** (1.0 / norm_type)
def clip_grad_value_(parameters, clip_value, foreach=None):
  with no_grad():
    for p in _params(parameters):
      if p.grad is not None: p.grad._inplace(p.grad.clamp(-clip_value, clip_value))
def parameters_to_vector(parameters):
  from ... import cat
  return cat([p.reshape(-1) for p in parameters])
def vector_to_parameters(vec, parameters):
  i = 0
  with no_grad():
    for p in parameters:
      n = p.numel()
      p.copy_(vec[i:i + n].reshape(p.shape)); i += n
