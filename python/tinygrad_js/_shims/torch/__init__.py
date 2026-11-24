"""A PyTorch-compatible API on top of tinygrad, for running PyTorch-style code in the browser with tinygrad.js.

This is **not PyTorch**. It implements the commonly used part of the ``torch`` API (tensors, autograd, ``torch.nn``,
``torch.optim``, ``torch.utils.data``, ``torch.save``/``torch.load``) by translating it to tinygrad, which runs it on WebGPU.
See docs/TORCH.md in the tinygrad.js repository for what is covered and where behaviour differs.
"""
from __future__ import annotations
import math as _math, builtins as _b
import numpy as _np
from tinygrad import Tensor as _TG, dtypes as _td

__version__ = "0.1.0+tinygrad.js"
__tinygrad_js_shim__ = True

from ._tensor import (Tensor, Size, dtype, device, finfo, iinfo, Generator, default_generator,
  float32, float, float64, double, float16, half, int64, long, int32, int, int16, short, int8, uint8, bool,
  no_grad, enable_grad, set_grad_enabled, inference_mode, is_grad_enabled, get_default_dtype, set_default_dtype, set_printoptions,
  manual_seed, seed, tensor, as_tensor, asarray, from_numpy, is_tensor, numel, zeros, ones, empty, full, zeros_like, ones_like,
  empty_like, full_like, rand, randn, rand_like, randn_like, randint, randint_like, randperm, normal, bernoulli, multinomial, arange,
  linspace, logspace, eye, scalar_tensor, _mk, _wrap, _unwrap, _coerce, _flush, _long, _shape_args, _tg_dtype, _minmax)

def _typed(name, dt):
  """Legacy typed constructors: torch.FloatTensor([1, 2]), torch.LongTensor(3) and x.type(torch.FloatTensor)."""
  def __new__(cls, *args):
    if len(args) == 1 and not isinstance(args[0], _b.int): return tensor(args[0], dtype=dt)
    return zeros(*args, dtype=dt)
  return type(name, (Tensor,), {"__new__": __new__, "_dtype": dt})
FloatTensor, DoubleTensor, HalfTensor = _typed("FloatTensor", float32), _typed("DoubleTensor", float64), _typed("HalfTensor", float16)
LongTensor, IntTensor, ShortTensor = _typed("LongTensor", int64), _typed("IntTensor", int32), _typed("ShortTensor", int16)
ByteTensor, BoolTensor = _typed("ByteTensor", uint8), _typed("BoolTensor", bool)

pi, e, inf, nan = _math.pi, _math.e, _math.inf, _math.nan
newaxis = None

def _t(x) -> Tensor: return x if isinstance(x, Tensor) else _mk(_coerce(x))

# ---- functions that are Tensor methods with the tensor as first argument
def _method(name):
  def fn(input, *args, out=None, **kwargs): return getattr(_t(input), name)(*args, **kwargs)
  fn.__name__ = fn.__qualname__ = name
  return fn
for _n in ("""abs absolute neg negative exp log log2 log10 log1p expm1 sqrt rsqrt sin cos tan tanh sigmoid relu floor ceil round trunc
  sign reciprocal square erf asin acos atan sinh cosh asinh acosh atanh exp2 isnan isinf isfinite frac nan_to_num logical_not
  add sub subtract mul multiply div divide true_divide floor_divide remainder fmod pow matmul mm bmm mv dot inner outer maximum
  minimum lerp addcmul addcdiv atan2 eq ne lt le gt ge logical_and logical_or logical_xor bitwise_and bitwise_or bitwise_xor
  bitwise_not isclose allclose equal clamp clip clamp_min clamp_max masked_fill softmax log_softmax sum mean prod std var amax amin
  argmax argmin all any logsumexp norm cumsum cumprod count_nonzero topk sort argsort median nonzero argwhere unique bincount
  reshape flatten ravel unflatten squeeze unsqueeze transpose swapaxes swapdims t permute movedim broadcast_to repeat_interleave
  tile flip roll chunk split unbind select narrow tril triu diag diagonal gather index_select take masked_select scatter scatter_add
  scatter_reduce index_add detach clone numel""").split():
  globals()[_n] = _method(_n)

def max(input, dim=None, keepdim=False, **k): return _t(input).max(k.get("other", dim), keepdim)
def min(input, dim=None, keepdim=False, **k): return _t(input).min(k.get("other", dim), keepdim)
def where(condition, input=None, other=None):
  if input is None: return condition.nonzero(as_tuple=True)
  if not isinstance(input, Tensor) and not isinstance(other, Tensor): return _wrap(_unwrap(condition).where(input, other))
  return _t(input).where(condition, other) if isinstance(input, Tensor) else _wrap(_unwrap(condition).where(input, other._t), other)
def cat(tensors, dim=0, *, axis=None, out=None):
  ts = [_t(x) for x in tensors]
  dt = _np.result_type(*[_np.dtype(x.dtype.name) for x in ts]).name if len({x.dtype for x in ts}) > 1 else None
  tt = [x._t if dt is None else x._t.cast(_tg_dtype(dt)) for x in ts]
  return _wrap(tt[0].cat(*tt[1:], dim=dim if axis is None else axis), *ts)
concat = concatenate = cat
def stack(tensors, dim=0, *, out=None):
  ts = [_t(x) for x in tensors]
  return _wrap(ts[0]._t.stack(*[x._t for x in ts[1:]], dim=dim), *ts)
def hstack(tensors): return cat([x.reshape(-1) if x.ndim == 0 else x for x in tensors], 0 if tensors[0].ndim <= 1 else 1)
def vstack(tensors): return cat([x.reshape(1, -1) if x.ndim <= 1 else x for x in tensors], 0)
def einsum(equation, *operands):
  ops = operands[0] if len(operands) == 1 and isinstance(operands[0], (list, tuple)) else operands
  return _wrap(_TG.einsum(equation, *[o._t for o in ops]), *ops)
def meshgrid(*tensors, indexing="ij"):
  ts = tensors[0] if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)) else tensors
  return tuple(_wrap(list(ts[0]._t.meshgrid(*[x._t for x in ts[1:]], indexing=indexing)), *ts))
def cdist(x1, x2, p=2.0): return (x1.unsqueeze(-2) - x2.unsqueeze(-3)).norm(p, dim=-1)
def tensordot(a, b, dims=2):
  if isinstance(dims, _b.int): da, db = list(range(a.ndim - dims, a.ndim)), list(range(dims))
  else: da, db = list(dims[0]), list(dims[1])
  ra, rb = [d for d in range(a.ndim) if d not in da], [d for d in range(b.ndim) if d not in db]
  k = _math.prod(a.shape[d] for d in da)
  out = a.permute(*ra, *da).reshape(-1, k) @ b.permute(*db, *rb).reshape(k, -1)
  return out.reshape(*[a.shape[d] for d in ra], *[b.shape[d] for d in rb])
def broadcast_tensors(*ts):
  shape = _np.broadcast_shapes(*[x.shape for x in ts])
  return tuple(x.expand(*shape) for x in ts)
def broadcast_shapes(*shapes): return Size(_np.broadcast_shapes(*shapes))
def result_type(*xs): return globals()[_np.result_type(*[_np.dtype(x.dtype.name) if isinstance(x, Tensor) else x for x in xs]).name]
def can_cast(a, b): return _b.bool(_np.can_cast(_np.dtype(a.name), _np.dtype(b.name)))
def is_floating_point(x): return x.is_floating_point()
def is_nonzero(x): return _b.bool(x.item() != 0)
def trace(x): return x.diagonal().sum()
def cumulative_trapezoid(y, dx=1.0): return ((y[..., 1:] + y[..., :-1]) * dx / 2).cumsum(-1)
def cross(a, b, dim=-1):
  a, b = a.movedim(dim, -1), b.movedim(dim, -1)
  r = stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1], a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
             a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], -1)
  return r.movedim(-1, dim)
def one_hot(x, num_classes=-1): return nn.functional.one_hot(x, num_classes)
def dropout(input, p=0.5, train=True): return nn.functional.dropout(input, p, train)
def conv2d(*a, **k): return nn.functional.conv2d(*a, **k)
def flatten_parameters(*a, **k): pass

# ---- runtime / configuration no-ops that PyTorch scripts commonly call
def set_float32_matmul_precision(p): pass
def use_deterministic_algorithms(mode, **k): pass
def set_num_threads(n): pass
def get_num_threads(): return 1
def compile(model=None, **k):
  """No-op: returns the model unchanged. Use tinygrad_js.run_training for a compiled (TinyJit) training loop."""
  return model if model is not None else (lambda m: m)
def is_inference_mode_enabled(): return not is_grad_enabled()
def get_default_device(): return _tensor._current_device()
def set_default_device(d): pass
def autocast(*a, **k): return amp.autocast(*a, **k)

from . import _tensor, _utils, autograd, nn, optim, utils, cuda, backends, linalg, random, amp, testing, distributions, jit
from .serialization import save, load
from .nn import functional
from .nn.parameter import Parameter
from .autograd import grad as _grad  # noqa: F401
from .storage import (FloatStorage, DoubleStorage, HalfStorage, LongStorage, IntStorage, ShortStorage, CharStorage, ByteStorage,
                      BoolStorage)
mps = backends.mps
