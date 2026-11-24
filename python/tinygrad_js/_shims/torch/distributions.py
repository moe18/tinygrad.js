"""torch.distributions: the handful of distributions used by introductory RL / VAE code."""
from __future__ import annotations
import math
from ._tensor import Tensor, _mk, _coerce, _long
from tinygrad import Tensor as _TG

def _t(x): return x if isinstance(x, Tensor) else _mk(_coerce(float(x)))

class Distribution:
  def sample(self, sample_shape=()): raise NotImplementedError
  def rsample(self, sample_shape=()): raise NotImplementedError(f"{type(self).__name__} has no reparameterized sample")
  def log_prob(self, value): raise NotImplementedError

class Normal(Distribution):
  def __init__(self, loc, scale, validate_args=None): self.loc, self.scale = _t(loc), _t(scale)
  @property
  def mean(self): return self.loc
  @property
  def stddev(self): return self.scale
  @property
  def variance(self): return self.scale ** 2
  def rsample(self, sample_shape=()):
    shape = tuple(sample_shape) + tuple((self.loc + self.scale).shape)
    return self.loc + self.scale * _mk(_TG.randn(*shape))
  def sample(self, sample_shape=()): return self.rsample(sample_shape).detach()
  def log_prob(self, value): return -((value - self.loc) ** 2) / (2 * self.scale ** 2) - self.scale.log() - math.log(math.sqrt(2 * math.pi))
  def entropy(self): return 0.5 + 0.5 * math.log(2 * math.pi) + self.scale.log()

class Uniform(Distribution):
  def __init__(self, low, high, validate_args=None): self.low, self.high = _t(low), _t(high)
  def rsample(self, sample_shape=()):
    shape = tuple(sample_shape) + tuple((self.low + self.high).shape)
    return self.low + (self.high - self.low) * _mk(_TG.rand(*shape))
  def sample(self, sample_shape=()): return self.rsample(sample_shape).detach()
  def log_prob(self, value): return -(self.high - self.low).log() + 0 * value
  def entropy(self): return (self.high - self.low).log()

class Bernoulli(Distribution):
  def __init__(self, probs=None, logits=None, validate_args=None):
    self.probs = _t(probs) if probs is not None else _t(logits).sigmoid()
  def sample(self, sample_shape=()):
    shape = tuple(sample_shape) + tuple(self.probs.shape)
    return _mk((_TG.rand(*shape) < self.probs.detach()._t).cast(self.probs._t.dtype))
  def log_prob(self, value): return value * self.probs.clamp(min=1e-7).log() + (1 - value) * (1 - self.probs).clamp(min=1e-7).log()
  def entropy(self): return -(self.probs * self.probs.clamp(min=1e-7).log() + (1 - self.probs) * (1 - self.probs).clamp(min=1e-7).log())

class Categorical(Distribution):
  def __init__(self, probs=None, logits=None, validate_args=None):
    if probs is not None:
      self.probs = _t(probs) / _t(probs).sum(-1, keepdim=True)
      self.logits = self.probs.clamp(min=1e-38).log()
    else:
      self.logits = _t(logits).log_softmax(-1)
      self.probs = self.logits.exp()
  def sample(self, sample_shape=()):
    n = math.prod(sample_shape) if len(tuple(sample_shape)) else 1
    p = self.probs.detach()
    flat = p.reshape(-1, p.shape[-1])
    s = _mk(flat._t.multinomial(n, True).cast(_long()))            # (batch, n)
    s = s.permute(1, 0).reshape(*tuple(sample_shape), *p.shape[:-1])
    return s
  def log_prob(self, value):
    v = value.long().unsqueeze(-1)
    lg = self.logits.expand(*v.shape[:-1], self.logits.shape[-1]) if self.logits.shape[:-1] != v.shape[:-1] else self.logits
    return lg.gather(-1, v).squeeze(-1)
  def entropy(self): return -(self.probs * self.logits).sum(-1)

def kl_divergence(p, q):
  if isinstance(p, Normal) and isinstance(q, Normal):
    vr = (p.scale / q.scale) ** 2
    return 0.5 * (vr + ((p.loc - q.loc) / q.scale) ** 2 - 1 - vr.log())
  raise NotImplementedError("kl_divergence is only implemented for Normal/Normal in the tinygrad.js torch shim")
