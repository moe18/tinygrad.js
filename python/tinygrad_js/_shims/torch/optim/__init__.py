"""torch.optim: SGD, Adam, AdamW, RMSprop, Adagrad with PyTorch's update rules, parameter groups and state.

Parameters and optimizer state are updated in place in their GPU buffers and the learning rate is read from a buffer, so a
training step compiles to the same kernels every time and can be replayed by TinyJit (``tinygrad_js.run_training``).
"""
from __future__ import annotations
import math
import numpy as np
from tinygrad import Tensor as _TG, dtypes as _td
from .._tensor import Tensor, _flush, _storage, _mk, no_grad
from . import lr_scheduler

class _required: pass
required = _required()

def _buf(value) -> _TG:
  """A realized scalar float32 buffer, so that changing the value does not change (and recompile) the kernels reading it."""
  return _TG(np.array([value], dtype=np.float32)).realize()

class _Group(dict):
  """A parameter group. Assigning ``group['lr']`` updates the GPU buffer the update kernels read the learning rate from."""
  def __setitem__(self, key, value):
    old = self.get(key)
    super().__setitem__(key, value)
    if key == "lr" and "_lr_buf" in self and float(old) != float(value):
      self["_lr_buf"].assign(_TG(np.array([float(value)], dtype=np.float32))).realize()

class Optimizer:
  def __init__(self, params, defaults):
    self.defaults = defaults
    self.param_groups: list[_Group] = []
    self.state: dict = {}
    self._t = None
    params = list(params)
    if len(params) == 0: raise ValueError("optimizer got an empty parameter list")
    if not isinstance(params[0], dict): params = [{"params": params}]
    for g in params: self.add_param_group(g)

  def add_param_group(self, group):
    ps = group["params"]
    ps = [ps] if isinstance(ps, Tensor) else list(ps)
    for p in ps:
      if not isinstance(p, Tensor): raise TypeError("optimizer can only optimize Tensors, but one of the params is " + type(p).__name__)
      if not p.is_leaf: raise ValueError("can't optimize a non-leaf Tensor")
    g = _Group({**self.defaults, **{k: v for k, v in group.items() if k != "params"}, "params": ps})
    if isinstance(g.get("lr"), _required): raise ValueError("parameter group didn't specify a value of required optimization parameter lr")
    if g["lr"] < 0: raise ValueError(f"Invalid learning rate: {g['lr']}")
    self.param_groups.append(g)

  def zero_grad(self, set_to_none=True):
    for g in self.param_groups:
      for p in g["params"]:
        if p.grad is not None: p.grad = None if set_to_none else _mk(p.grad._t.zeros_like())

  def _state(self, p, names) -> dict:
    """Per-parameter state tensors (zeros, realized buffers), created on first use."""
    st = self.state.setdefault(p, {})
    new = [n for n in names if n not in st]
    for n in new: st[n] = _TG.zeros(*p.shape, dtype=p._t.dtype).contiguous()
    if new: _TG.realize(*[st[n] for n in new])
    return st

  def step(self, closure=None):
    loss = None
    if closure is not None:
      from .._tensor import enable_grad
      with enable_grad(): loss = closure()
    self._pending = []
    with no_grad():
      for g in self.param_groups:
        if "_lr_buf" not in g: dict.__setitem__(g, "_lr_buf", _buf(g["lr"]))
        lr = g["_lr_buf"].reshape(())
        for p in g["params"]:
          if p.grad is None: continue
          self._update(p, p.grad._t.detach(), lr, g)
      # one sync point: parameters, optimizer state and every tensor still waiting to be computed (loss, predictions, ...)
      _flush(*self._pending)
    self._pending = []
    return loss

  def _put(self, state:_TG, value:_TG) -> None:
    """Write a new value into an optimizer state buffer (realized by step() together with the parameter updates)."""
    state.assign(value)
    self._pending.append(state)

  def _update(self, p:Tensor, grad:_TG, lr:_TG, group:dict): raise NotImplementedError

  def state_dict(self):
    from .. import _mk
    index = {id(p): i for i, p in enumerate(p for g in self.param_groups for p in g["params"])}
    groups = [{**{k: v for k, v in g.items() if k not in ("params", "_lr_buf")}, "params": [index[id(p)] for p in g["params"]]}
              for g in self.param_groups]
    state = {index[id(p)]: {k: _mk(v + 0) for k, v in st.items()} for p, st in self.state.items() if id(p) in index}
    return {"state": state, "param_groups": groups}
  def load_state_dict(self, sd):
    order = [p for g in self.param_groups for p in g["params"]]
    for g, saved in zip(self.param_groups, sd["param_groups"]):
      for k, v in saved.items():
        if k != "params": g[k] = v
    for i, st in sd["state"].items():
      self.state[order[int(i)]] = {k: _storage((v._t if isinstance(v, Tensor) else _TG(np.asarray(v))).contiguous()) for k, v in st.items()}
  def __repr__(self):
    s = type(self).__name__ + " ("
    for i, g in enumerate(self.param_groups):
      s += f"\nParameter Group {i}\n" + "\n".join(f"    {k}: {v}" for k, v in sorted(g.items()) if k not in ("params", "_lr_buf"))
    return s + "\n)"

class SGD(Optimizer):
  def __init__(self, params, lr=1e-3, momentum=0, dampening=0, weight_decay=0, nesterov=False, *, maximize=False, foreach=None,
               differentiable=False, fused=None):
    if nesterov and (momentum <= 0 or dampening != 0): raise ValueError("Nesterov momentum requires a momentum and zero dampening")
    super().__init__(params, dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov, maximize=maximize))
  def _update(self, p, g, lr, group):
    if group["maximize"]: g = -g
    if group["weight_decay"]: g = g + group["weight_decay"] * p._t
    mu, damp = group["momentum"], group["dampening"]
    if mu:
      st = self._state(p, ["momentum_buffer"])
      scale = 1
      if damp:
        # PyTorch sets the buffer to the plain gradient on the first step and applies dampening only afterwards
        if "_seen" not in st: st["_seen"] = _buf(0.0)
        scale = 1 - damp * st["_seen"].reshape(())
        self._put(st["_seen"], st["_seen"] * 0 + 1)
      new = st["momentum_buffer"] * mu + g * scale
      self._put(st["momentum_buffer"], new)
      g = g + new * mu if group["nesterov"] else new
    p._assign(p._t - lr * g)

class Adam(Optimizer):
  _decoupled = False
  def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, *, foreach=None, maximize=False,
               capturable=False, differentiable=False, fused=None, decoupled_weight_decay=False):
    super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, maximize=maximize))
    self._decoupled = self._decoupled or decoupled_weight_decay
  def _update(self, p, g, lr, group):
    b1, b2 = group["betas"]
    if group["maximize"]: g = -g
    st = self._state(p, ["exp_avg", "exp_avg_sq"] + (["max_exp_avg_sq"] if group["amsgrad"] else []))
    if "b1_t" not in st: st["b1_t"], st["b2_t"] = _buf(1.0), _buf(1.0)
    self._put(st["b1_t"], st["b1_t"] * b1); self._put(st["b2_t"], st["b2_t"] * b2)
    b1_t, b2_t = st["b1_t"].reshape(()), st["b2_t"].reshape(())
    x = p._t
    if group["weight_decay"]:
      if self._decoupled: x = x - lr * group["weight_decay"] * x
      else: g = g + group["weight_decay"] * x
    m = st["exp_avg"] * b1 + g * (1 - b1)
    v = st["exp_avg_sq"] * b2 + (g * g) * (1 - b2)
    self._put(st["exp_avg"], m); self._put(st["exp_avg_sq"], v)
    if group["amsgrad"]:
      vmax = st["max_exp_avg_sq"].maximum(v)
      self._put(st["max_exp_avg_sq"], vmax)
      v = vmax
    denom = v.sqrt() / (1 - b2_t).sqrt() + group["eps"]
    p._assign(x - lr / (1 - b1_t) * m / denom)

class AdamW(Adam):
  _decoupled = True
  def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-2, amsgrad=False, **kw):
    kw.pop("decoupled_weight_decay", None)
    super().__init__(params, lr, betas, eps, weight_decay, amsgrad, **kw)

class RMSprop(Optimizer):
  def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False, **kw):
    super().__init__(params, dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum, centered=centered))
  def _update(self, p, g, lr, group):
    a = group["alpha"]
    if group["weight_decay"]: g = g + group["weight_decay"] * p._t
    st = self._state(p, ["square_avg"] + (["grad_avg"] if group["centered"] else []) + (["momentum_buffer"] if group["momentum"] else []))
    sq = st["square_avg"] * a + (g * g) * (1 - a)
    self._put(st["square_avg"], sq)
    if group["centered"]:
      ga = st["grad_avg"] * a + g * (1 - a)
      self._put(st["grad_avg"], ga)
      avg = (sq - ga * ga).sqrt() + group["eps"]
    else: avg = sq.sqrt() + group["eps"]
    if group["momentum"]:
      buf = st["momentum_buffer"] * group["momentum"] + g / avg
      self._put(st["momentum_buffer"], buf)
      p._assign(p._t - lr * buf)
    else: p._assign(p._t - lr * g / avg)

class Adagrad(Optimizer):
  def __init__(self, params, lr=1e-2, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10, **kw):
    if lr_decay: raise NotImplementedError("Adagrad lr_decay is not supported by the tinygrad.js torch shim")
    super().__init__(params, dict(lr=lr, weight_decay=weight_decay, eps=eps, initial_accumulator_value=initial_accumulator_value))
  def _update(self, p, g, lr, group):
    if group["weight_decay"]: g = g + group["weight_decay"] * p._t
    new = p not in self.state
    st = self._state(p, ["sum"])
    acc = st["sum"] + g * g + (group["initial_accumulator_value"] if new else 0)
    self._put(st["sum"], acc)
    p._assign(p._t - lr * g / (acc.sqrt() + group["eps"]))

class Adadelta(Optimizer):
  def __init__(self, params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0, **kw):
    super().__init__(params, dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay))
  def _update(self, p, g, lr, group):
    rho, eps = group["rho"], group["eps"]
    if group["weight_decay"]: g = g + group["weight_decay"] * p._t
    st = self._state(p, ["square_avg", "acc_delta"])
    sq = st["square_avg"] * rho + (g * g) * (1 - rho)
    delta = (st["acc_delta"] + eps).sqrt() / (sq + eps).sqrt() * g
    self._put(st["square_avg"], sq)
    self._put(st["acc_delta"], st["acc_delta"] * rho + (delta * delta) * (1 - rho))
    p._assign(p._t - lr * delta)
