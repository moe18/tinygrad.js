"""torch.nn.init. Values are drawn with tinygrad's random number generator, so they differ from PyTorch's for the same seed,
but the distributions are the same."""
from __future__ import annotations
import math
import numpy as np
from tinygrad import Tensor as _TG
from .._tensor import no_grad, _coerce
from .. import _tensor

def calculate_gain(nonlinearity, param=None):
  if nonlinearity in ("linear", "conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d", "conv_transpose3d", "sigmoid"): return 1.0
  if nonlinearity == "tanh": return 5.0 / 3
  if nonlinearity == "relu": return math.sqrt(2.0)
  if nonlinearity == "leaky_relu": return math.sqrt(2.0 / (1 + (0.01 if param is None else param) ** 2))
  if nonlinearity == "selu": return 3.0 / 4
  raise ValueError(f"Unsupported nonlinearity {nonlinearity}")

def _calculate_fan_in_and_fan_out(tensor):
  if tensor.dim() < 2: raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
  receptive = math.prod(tensor.shape[2:])
  return tensor.shape[1] * receptive, tensor.shape[0] * receptive
def _calculate_correct_fan(tensor, mode):
  fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
  return fan_in if mode == "fan_in" else fan_out

def uniform_(tensor, a=0.0, b=1.0, generator=None):
  with no_grad(): return tensor.uniform_(a, b)
def normal_(tensor, mean=0.0, std=1.0, generator=None):
  with no_grad(): return tensor.normal_(mean, std)
def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0, generator=None):
  # inverse-CDF sampling on the host (tinygrad has no erfinv): exact truncated normal
  from statistics import NormalDist
  nd = NormalDist()
  lo, hi = nd.cdf((a - mean) / std), nd.cdf((b - mean) / std)
  u = _tensor._host_rng.uniform(lo, hi, size=tuple(tensor.shape))
  vals = np.vectorize(nd.inv_cdf, otypes=[np.float64])(np.clip(u, 1e-12, 1 - 1e-12)) * std + mean if u.size else u
  with no_grad(): return tensor._overwrite(_coerce(np.clip(vals, a, b).astype(np.float32)), fresh=True)
def constant_(tensor, val):
  with no_grad(): return tensor.fill_(val)
def ones_(tensor): return constant_(tensor, 1.0)
def zeros_(tensor): return constant_(tensor, 0.0)
def eye_(tensor):
  with no_grad(): return tensor._overwrite(_TG.eye(*tensor.shape).contiguous(), fresh=False)
def xavier_uniform_(tensor, gain=1.0, generator=None):
  fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
  bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
  return uniform_(tensor, -bound, bound)
def xavier_normal_(tensor, gain=1.0, generator=None):
  fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
  return normal_(tensor, 0.0, gain * math.sqrt(2.0 / (fan_in + fan_out)))
def kaiming_uniform_(tensor, a=0, mode="fan_in", nonlinearity="leaky_relu", generator=None):
  bound = calculate_gain(nonlinearity, a) * math.sqrt(3.0 / _calculate_correct_fan(tensor, mode))
  return uniform_(tensor, -bound, bound)
def kaiming_normal_(tensor, a=0, mode="fan_in", nonlinearity="leaky_relu", generator=None):
  return normal_(tensor, 0.0, calculate_gain(nonlinearity, a) / math.sqrt(_calculate_correct_fan(tensor, mode)))
def orthogonal_(tensor, gain=1, generator=None):
  rows, cols = tensor.shape[0], tensor.numel() // tensor.shape[0]
  flat = _tensor._host_rng.standard_normal((rows, cols))
  if rows < cols: flat = flat.T
  q, r = np.linalg.qr(flat)
  q = q * np.sign(np.diag(r))
  if rows < cols: q = q.T
  with no_grad(): return tensor._overwrite(_coerce((gain * q).reshape(tuple(tensor.shape)).astype(np.float32)), fresh=True)
uniform, normal, constant, xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal = (uniform_, normal_, constant_, xavier_uniform_,
                                                                                            xavier_normal_, kaiming_uniform_, kaiming_normal_)
