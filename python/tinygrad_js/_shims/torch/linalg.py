"""torch.linalg: norms and a few helpers. Decompositions and solvers are not implemented."""
from __future__ import annotations
from ._tensor import _wrap

def vector_norm(x, ord=2, dim=None, keepdim=False, dtype=None): return x.norm(ord, dim, keepdim)
def matrix_norm(x, ord="fro", dim=(-2, -1), keepdim=False):
  if ord == "fro": return x.norm("fro", dim, keepdim)
  raise NotImplementedError(f"torch.linalg.matrix_norm(ord={ord!r}) is not supported by the tinygrad.js torch shim")
def norm(x, ord=None, dim=None, keepdim=False):
  return x.norm(2 if ord is None else ord, dim, keepdim)
def vecdot(x, y, dim=-1): return (x * y).sum(dim)
def matmul(a, b): return a @ b
def multi_dot(tensors):
  out = tensors[0]
  for t in tensors[1:]: out = out @ t
  return out
def cross(a, b, dim=-1):
  from . import cross as c
  return c(a, b, dim)
def qr(x, mode="reduced"): return _wrap(x._t.qr(), x)
def svd(x, full_matrices=True): return _wrap(x._t.svd(full_matrices), x)
def __getattr__(name):
  raise AttributeError(f"torch.linalg.{name} is not implemented by the tinygrad.js torch shim")
