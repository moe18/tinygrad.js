"""torch.Tensor on top of tinygrad: dtypes, devices, grad mode, the Tensor wrapper and the functions that create tensors.

How PyTorch's eager, mutable tensors are mapped onto tinygrad's lazy, functional ones:

* A ``Tensor`` wraps one tinygrad Tensor (``_t``). Operations build tinygrad's lazy graph.
* Autograd state (``requires_grad``, ``is_leaf``, ``grad``) lives on the wrapper. ``backward()`` asks tinygrad for the gradients of
  the leaves that are reachable from the result, and accumulates them into ``.grad`` like PyTorch does.
* Values are computed at *sync points*: whenever a value is read (``item()``, ``numpy()``, ``print``) or storage is mutated
  (``optimizer.step()``, in-place updates of leaves). At a sync point every tensor that is still alive and not yet computed is
  computed too, in the same tinygrad schedule. That gives eager semantics (a tensor computed before a parameter update keeps its
  pre-update value) without evaluating every operation separately.
* In-place operations on leaves that require grad (parameters) write into the same GPU buffer, so optimizers work under TinyJit.
  In-place operations on other tensors rebind the wrapper to a new value; views of such a tensor do not see the change.
"""
from __future__ import annotations
import functools, math, weakref
from collections import namedtuple
import numpy as np
from tinygrad import Tensor as TG, Device, dtypes as _td

try: from tinygrad.uop.ops import Ops as _Ops
except Exception: _Ops = None  # pragma: no cover

# ---------------------------------------------------------------- dtypes

class dtype:
  """A torch dtype. ``tg`` is the tinygrad dtype it maps to on a device that supports it."""
  def __init__(self, name:str, tg, aliases=()):
    self.name, self.tg, self.aliases = name, tg, aliases
    self.is_floating_point = _td.is_float(tg)
    self.is_signed = not _td.is_unsigned(tg) and tg != _td.bool
    self.is_complex = False
    self.itemsize = tg.itemsize
  def __repr__(self): return f"torch.{self.name}"
  # Two dtypes are equal when they are stored the same way on the current device. WebGPU has no 64-bit types, so there
  # `tensor.dtype == torch.long` holds for the int32 tensors that stand in for int64 (and likewise float64 / float32).
  def __eq__(self, other): return isinstance(other, dtype) and _narrow(self.tg) == _narrow(other.tg)
  def __ne__(self, other): return not self == other
  def __hash__(self): return hash(_narrow(self.tg))
  def __reduce__(self): return self.name  # pickles as the global torch.<name>, like PyTorch
  def to_real(self): return self

float32 = dtype("float32", _td.float32); float = float32
float64 = dtype("float64", _td.float64); double = float64
float16 = dtype("float16", _td.float16); half = float16
int64 = dtype("int64", _td.int64); long = int64
int32 = dtype("int32", _td.int32); int = int32
int16 = dtype("int16", _td.int16); short = int16
int8 = dtype("int8", _td.int8)
uint8 = dtype("uint8", _td.uint8)
bool = dtype("bool", _td.bool)
_ALL_DTYPES = [float32, float64, float16, int64, int32, int16, int8, uint8, bool]
_BY_TG = {d.tg: d for d in _ALL_DTYPES}
_BY_TG.update({_td.uint16: int32, _td.uint32: int64, _td.uint64: int64})

import builtins as _b

@functools.lru_cache(None)
def _supported(dev:str) -> frozenset|None:
  try: return frozenset(Device[dev].renderer.supported_dtypes())
  except Exception:
    return frozenset({_td.bool, _td.int8, _td.uint8, _td.int16, _td.uint16, _td.int32, _td.uint32, _td.float32}) if dev == "WEBGPU" else None

def _narrow(tg):
  """The dtype actually used on the default device: WebGPU has no 64-bit types, so float64 -> float32 and int64 -> int32."""
  sup = _supported(Device.DEFAULT)
  if sup is None or tg in sup: return tg
  if _td.is_float(tg): return _td.float32
  if tg == _td.bool: return tg
  return _td.int32

def _tg_dtype(dt):
  """torch dtype / Python type / numpy dtype / None -> (narrowed) tinygrad dtype or None."""
  if dt is None: return None
  if isinstance(dt, dtype): return _narrow(dt.tg)
  if dt is _b.float: return _td.float32
  if dt is _b.int: return _narrow(_td.int64)
  if dt is _b.bool: return _td.bool
  if isinstance(dt, str):
    for d in _ALL_DTYPES:
      if d.name == dt or f"torch.{d.name}" == dt: return _narrow(d.tg)
  try:
    name = np.dtype(dt).name
    for d in _ALL_DTYPES:
      if d.name == name: return _narrow(d.tg)
  except Exception: pass
  raise TypeError(f"unsupported dtype: {dt!r}")

def _torch_dtype(tg) -> dtype:
  try: return _BY_TG[tg]
  except KeyError: return _BY_TG[_td.float32 if _td.is_float(tg) else _td.int32]

_default_dtype = float32
def get_default_dtype(): return _default_dtype
def set_default_dtype(d):
  global _default_dtype
  _default_dtype = d
def _default_float(): return _narrow(_default_dtype.tg)
def _long(): return _narrow(_td.int64)

class finfo:
  def __init__(self, dt=None):
    f = np.finfo(np.dtype((dt or float32).name))
    self.eps, self.max, self.min, self.tiny, self.bits = _b.float(f.eps), _b.float(f.max), _b.float(f.min), _b.float(f.tiny), f.bits
    self.smallest_normal, self.resolution, self.dtype = self.tiny, _b.float(f.resolution), (dt or float32).name
class iinfo:
  def __init__(self, dt):
    i = np.iinfo(np.dtype(dt.name)); self.max, self.min, self.bits, self.dtype = _b.int(i.max), _b.int(i.min), i.bits, dt.name

# ---------------------------------------------------------------- devices

class device:
  """``torch.device``. Every device name is accepted and maps to tinygrad's default device (WebGPU in the browser)."""
  def __init__(self, type="cpu", index=None):
    if isinstance(type, device): type, index = type.type, type.index
    if ":" in type: type, i = type.split(":", 1); index = _b.int(i)
    self.type, self.index = type, index
  def __repr__(self): return f"device(type='{self.type}'" + (f", index={self.index})" if self.index is not None else ")")
  def __str__(self): return self.type + (f":{self.index}" if self.index is not None else "")
  def __eq__(self, o): return isinstance(o, (device, str)) and str(device(o) if isinstance(o, str) else o) == str(self)
  def __hash__(self): return hash(str(self))

def _current_device() -> device: return device("webgpu" if Device.DEFAULT == "WEBGPU" else "cpu")

# ---------------------------------------------------------------- grad mode

_grad_enabled = True
def is_grad_enabled(): return _grad_enabled

class set_grad_enabled:
  """Context manager / decorator / plain call, like torch.set_grad_enabled."""
  def __init__(self, mode:bool):
    global _grad_enabled
    self.mode, self.prev = _b.bool(mode), _grad_enabled
    if type(self) is set_grad_enabled: _grad_enabled = self.mode
  def __enter__(self):
    global _grad_enabled
    if type(self) is not set_grad_enabled: self.prev = _grad_enabled
    _grad_enabled = self.mode
  def __exit__(self, *exc):
    global _grad_enabled
    _grad_enabled = self.prev
  def __call__(self, fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
      with type(self)(self.mode) if type(self) is set_grad_enabled else type(self)(): return fn(*a, **k)
    return wrapper

class no_grad(set_grad_enabled):
  def __new__(cls, fn=None):
    self = object.__new__(cls)
    if callable(fn): self.__init__(); return self(fn)  # used as @torch.no_grad without parentheses
    return self
  def __init__(self, fn=None): self.mode, self.prev = False, _grad_enabled
class enable_grad(no_grad):
  def __init__(self, fn=None): self.mode, self.prev = True, _grad_enabled
class inference_mode(no_grad):
  def __init__(self, mode=True): self.mode, self.prev = not mode if isinstance(mode, _b.bool) else False, _grad_enabled

# ---------------------------------------------------------------- sync points

class _IdWeakSet:
  """A weak set keyed by object identity (``Tensor.__eq__`` is elementwise, so weakref.WeakSet cannot hold tensors)."""
  def __init__(self): self._d: dict[int, weakref.ref] = {}
  def add(self, o):
    k = id(o)
    if k not in self._d: self._d[k] = weakref.ref(o, lambda _, k=k, d=self._d: d.pop(k, None))
  def discard(self, o): self._d.pop(id(o), None)
  def __contains__(self, o): return id(o) in self._d
  def __iter__(self): return iter([o for r in list(self._d.values()) if (o := r()) is not None])
  def __len__(self): return len(self._d)

_live = _IdWeakSet()          # wrappers whose value is not computed yet (or that keep a graph for backward)
_grad_targets = _IdWeakSet()  # leaves that require grad, and tensors with retain_grad()

def _flush(*extra:TG) -> None:
  """Compute every live tensor that has not been computed yet (plus ``extra``) in one tinygrad schedule.

  tinygrad's ``realize`` replaces the realized graph node by a buffer in *every* tensor that contains it, and gradients do not flow
  through buffers. So a value is always computed through a fresh CONTIGUOUS node, which leaves the node below it (and with it any
  graph that backward() may still need) untouched."""
  outs, done = list(extra), []
  for w in _live:
    if w._rg and not w.is_leaf and not w._freed:
      # the graph may still be needed for backward(): compute a detached copy and keep the graph
      if w._val is None:
        w._val = w._tt.detach().contiguous()
        outs.append(w._val)
    else:
      if w._val is not None: w._tt, w._val = w._val, None
      elif not w._lazy and not w._tt.uop.has_buffer_identity():
        if not w._tt.uop.is_virtual: w._tt = w._tt.contiguous()
        # a device-less expression (built only from arange/linspace/Python numbers) has no storage and would be recomputed, and
        # grow, with every use: give it a buffer unless it is a small constant
        elif not _is_small(w._tt.uop): w._tt = _unique(w._tt)
      outs.append(w._tt); done.append(w)
  if outs: outs[0].realize(*outs[1:])
  for w in done:
    if w._lazy:
      if not w._tt.uop.has_buffer_identity(): w._tt = _storage(w._tt)
      w._lazy = False
    _live.discard(w)

def _is_small(uop, limit=12) -> bool:
  """True if the graph below ``uop`` has at most ``limit`` nodes."""
  seen, stack = set(), [uop]
  while stack:
    u = stack.pop()
    if u in seen: continue
    seen.add(u)
    if len(seen) > limit: return False
    stack.extend(u.src)
  return True

def _unique(t:TG) -> TG:
  """A lazy copy with an identity of its own. tinygrad shares graph nodes for equal expressions (and all constants), but every
  leaf that requires grad must be a distinct gradient target."""
  return t.clone(Device.DEFAULT)

def _storage(t:TG) -> TG:
  """A realized tinygrad tensor that owns a buffer (so it can be assigned to)."""
  if t.uop.has_buffer_identity(): return t
  if t.uop.is_virtual: t = _unique(t)
  t.realize()
  return t if t.uop.has_buffer_identity() else _unique(t).realize()

def _copy_into(pairs) -> None:
  """Write tinygrad tensors into wrappers (``[(dst, src), ...]``) with a single sync point; used by load_state_dict."""
  need_flush = False
  for dst, src in pairs:
    if src.dtype != dst._tt.dtype: src = src.cast(dst._tt.dtype)
    if ((dst._rg and dst.is_leaf) or dst._persist) and dst._tt.uop.has_buffer_identity():
      dst._assign(src.detach()); need_flush = True
    else: dst._overwrite(src)
  if need_flush: _flush()

def _diff_slice(root) -> set:
  """UOps reachable from ``root`` without crossing a DETACH: the part of the graph backward() differentiates."""
  seen, stack = set(), [root]
  while stack:
    u = stack.pop()
    if u in seen: continue
    seen.add(u)
    if _Ops is not None and u.op is _Ops.DETACH: continue
    stack.extend(u.src)
  return seen

# ---------------------------------------------------------------- helpers

class Size(tuple):
  def __repr__(self): return f"torch.Size({list(self)})"
  def numel(self): return math.prod(self)
  def __getitem__(self, i):
    r = tuple.__getitem__(self, i)
    return Size(r) if isinstance(i, slice) else r

class _GradFn:
  def __init__(self, name="TinygradBackward"): self._name = name
  def name(self): return self._name
  def __repr__(self): return f"<{self._name}>"
  next_functions = ()

def _is_seq(x): return isinstance(x, (list, tuple))
def _shape_args(args) -> tuple:
  if len(args) == 1 and _is_seq(args[0]): return tuple(_b.int(a) for a in args[0])
  return tuple(_b.int(a) for a in args)
def _dim(dim):
  if isinstance(dim, list): return tuple(dim)
  return dim

def _mk(t:TG, requires_grad=False, leaf=True) -> "Tensor":
  w = Tensor.__new__(Tensor)
  w._init(t, requires_grad, leaf)
  return w

def _unwrap(x):
  """Tensor wrapper -> tinygrad tensor; Python numbers and numpy scalars pass through."""
  if isinstance(x, Tensor): return x._t
  if isinstance(x, np.ndarray): return TG(x, dtype=_narrow(_td.from_np(x.dtype) if hasattr(_td, "from_np") else _np_to_tg(x.dtype)))
  if isinstance(x, np.generic): return x.item()
  return x

def _np_to_tg(npd):
  name = np.dtype(npd).name
  for d in _ALL_DTYPES:
    if d.name == name: return d.tg
  return {"uint16": _td.uint16, "uint32": _td.uint32, "uint64": _td.uint64}.get(name, _td.float32)

def _wrap(t, *srcs) -> "Tensor":
  """Wrap the tinygrad result of an operation on ``srcs`` and propagate requires_grad like PyTorch."""
  if not isinstance(t, TG):
    if isinstance(t, tuple): return tuple(_wrap(x, *srcs) for x in t)
    if isinstance(t, list): return [_wrap(x, *srcs) for x in t]
    return t
  any_rg = any(isinstance(s, Tensor) and s.requires_grad for s in srcs)
  rg = any_rg and _grad_enabled and _td.is_float(t.dtype)
  if any_rg and not rg: t = t.detach()
  return _mk(t, rg, not rg)

def _coerce(data, dt=None) -> TG:
  """Python data / numpy array / Tensor -> tinygrad tensor with PyTorch's dtype inference."""
  tg = _tg_dtype(dt)
  if isinstance(data, Tensor): return data._t if tg is None or tg == data._t.dtype else data._t.cast(tg)
  if isinstance(data, TG): return data if tg is None else data.cast(tg)
  if isinstance(data, (np.ndarray, np.generic)):
    a = np.asarray(data)
    want = tg or _narrow(_np_to_tg(a.dtype))
    if a.ndim == 0: return TG(a.item(), dtype=want)   # (np.ascontiguousarray would turn a 0-d array into shape (1,))
    return TG(np.ascontiguousarray(a), dtype=want)
  if _is_seq(data) and any(isinstance(x, Tensor) for x in _flatten(data)):
    data = _nest_to_py(data)
  if isinstance(data, (list, tuple)):
    a = np.array(data)
    if a.dtype == object: raise ValueError("expected a sequence of equal-length sequences")
    if tg is None: tg = _td.bool if a.dtype == np.bool_ else _long() if a.dtype.kind in "iu" else _default_float()
    return TG(np.ascontiguousarray(a.astype(np.float64 if a.dtype.kind == "f" else a.dtype)), dtype=tg)
  if isinstance(data, _b.bool): return TG(data, dtype=tg or _td.bool)
  if isinstance(data, _b.int): return TG(data, dtype=tg or _long())
  if isinstance(data, _b.float): return TG(data, dtype=tg or _default_float())
  raise TypeError(f"can't convert {type(data).__name__} to a torch tensor")

def _flatten(x):
  if _is_seq(x):
    for y in x: yield from _flatten(y)
  else: yield x
def _nest_to_py(x):
  if _is_seq(x): return [_nest_to_py(y) for y in x]
  return x.tolist() if isinstance(x, Tensor) else x

def _index(idx):
  """Convert a PyTorch index expression into one tinygrad understands. Boolean masks become integer index tensors."""
  def one(i):
    if isinstance(i, Tensor):
      if i._t.dtype == _td.bool: return tuple(TG(a.astype(np.int32)) for a in np.nonzero(i.numpy()))
      return i._t
    if isinstance(i, np.ndarray): return one(_mk(_coerce(i)))
    if isinstance(i, np.integer): return _b.int(i)
    if isinstance(i, list) and len(i) and all(isinstance(x, (_b.bool, np.bool_)) for x in i): return one(_mk(_coerce(i)))
    return i
  if isinstance(idx, tuple):
    out = []
    for i in idx:
      r = one(i)
      out.extend(r) if isinstance(r, tuple) and isinstance(i, (Tensor, np.ndarray, list)) else out.append(r)
    return tuple(out)
  r = one(idx)
  return r if not (isinstance(r, tuple) and isinstance(idx, (Tensor, np.ndarray, list))) else (r if len(r) > 1 else r[0])

_minmax = namedtuple("minmax", ["values", "indices"])
_PRINT = {"precision": 4, "threshold": 1000, "edgeitems": 3, "linewidth": 80, "sci_mode": None}
def set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=None):
  for k, v in (("precision", precision), ("threshold", threshold), ("edgeitems", edgeitems), ("linewidth", linewidth), ("sci_mode", sci_mode)):
    if v is not None: _PRINT[k] = v

def _format(a:np.ndarray, indent:int) -> str:
  if a.dtype.kind == "f":
    finite = a[np.isfinite(a)]
    if a.size and finite.size and np.all(finite == np.round(finite)) and (not finite.size or np.abs(finite).max() < 1e8):
      fmt = lambda v: (f"{v:.0f}." if np.isfinite(v) else str(v))
    elif _PRINT["sci_mode"] or (_PRINT["sci_mode"] is None and finite.size and
                                 (np.abs(finite).max() >= 1e8 or 0 < np.abs(finite[finite != 0]).min(initial=1.0) < 1e-4)):
      fmt = lambda v: f"{v:.{_PRINT['precision']}e}"
    else: fmt = lambda v: f"{v:.{_PRINT['precision']}f}"
    strs = np.vectorize(fmt, otypes=[object])(a) if a.size else a.astype(object)
  elif a.dtype.kind == "b": strs = np.vectorize(lambda v: "True" if v else "False", otypes=[object])(a) if a.size else a.astype(object)
  else: strs = a.astype(str).astype(object)
  width = max((len(s) for s in strs.ravel()), default=0)
  def rec(s, depth):
    if not isinstance(s, np.ndarray) or s.ndim == 0:
      v = s if isinstance(s, str) else str(s.item())
      return v.rjust(width) if a.ndim else v
    edge = _PRINT["edgeitems"]
    cut = a.size > _PRINT["threshold"] and s.shape[0] > 2 * edge
    items = [rec(x, depth + 1) for x in (list(s[:edge]) + list(s[-edge:]) if cut else s)]
    if cut: items.insert(edge, "...")
    if s.ndim == 1:
      per_line = max(1, (_PRINT["linewidth"] - indent - depth) // (width + 2))
      lines = [", ".join(items[i:i + per_line]) for i in range(0, len(items), per_line)]
      return "[" + (",\n" + " " * (indent + depth + 1)).join(lines) + "]"
    return "[" + ("," + "\n" * (s.ndim - 1) + " " * (indent + depth + 1)).join(items) + "]"
  return rec(strs, 0)

# ---------------------------------------------------------------- Tensor

class Tensor:
  __array_priority__ = 1000

  def __init__(self, *args, **kwargs):
    # torch.Tensor(2, 3) -> uninitialized float tensor; torch.Tensor([1, 2]) -> float tensor from data
    if len(args) == 1 and not isinstance(args[0], _b.int): t = _coerce(args[0], float32)
    else: t = TG.zeros(*_shape_args(args), dtype=_default_float())
    self._init(t, False, True)

  def _init(self, t:TG, requires_grad=False, leaf=True):
    self._tt, self._val, self._freed, self._base, self._persist, self._lazy = t, None, False, None, False, False
    self.is_leaf, self._rg, self.grad, self._retain = leaf, False, None, False
    if requires_grad: self.requires_grad = True
    _live.add(self)

  @property
  def _t(self) -> TG:
    """The tinygrad tensor, for use in operations. A leaf that requires grad gets its buffer before it enters a graph:
    tinygrad may rewrite reshapes of a tensor that is realized later, and the leaf would no longer be found by backward()."""
    if self._lazy: _flush()
    return self._tt
  @_t.setter
  def _t(self, t:TG): self._tt = t

  def _set(self, t:TG, rg=None, leaf=None):
    """Rebind this wrapper to a new value (and, optionally, new autograd flags)."""
    self._tt, self._val, self._freed = t, None, False
    if rg is not None:
      self._rg, self.is_leaf = rg, (not rg if leaf is None else leaf)
      if rg and self.is_leaf: _grad_targets.add(self)
      elif not self._retain: _grad_targets.discard(self)
    self._lazy = self._rg and self.is_leaf and not t.uop.has_buffer_identity()
    _live.add(self)

  # ---- autograd state
  @property
  def requires_grad(self): return self._rg
  @requires_grad.setter
  def requires_grad(self, v):
    v = _b.bool(v)
    if v and not _td.is_float(self._tt.dtype): raise RuntimeError("only Tensors of floating point dtype can require gradients")
    if not v and not self.is_leaf and self._rg:
      raise RuntimeError("you can only change requires_grad flags of leaf variables. Use .detach() to get a tensor without grad.")
    self._rg = v
    # constants are shared graph nodes in tinygrad: a leaf that requires grad needs storage of its own
    if v and self.is_leaf and self._tt.uop.is_virtual: self._tt = _unique(self._tt)
    if v and self.is_leaf:
      _grad_targets.add(self)
      self._lazy = not self._tt.uop.has_buffer_identity()
      if self._lazy: _live.add(self)
    elif not v: _grad_targets.discard(self)
  def requires_grad_(self, requires_grad=True): self.requires_grad = requires_grad; return self
  def retain_grad(self):
    if not self.is_leaf: self._retain = True; _grad_targets.add(self)
  @property
  def retains_grad(self): return self._retain
  @property
  def grad_fn(self): return _GradFn() if self._rg and not self.is_leaf else None
  def register_hook(self, hook): raise NotImplementedError("Tensor.register_hook is not supported by the tinygrad.js torch shim")

  def backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None):
    if not self._rg: raise RuntimeError("element 0 of tensors does not require grad and does not have a grad_fn")
    if self._freed:
      raise RuntimeError("Trying to backward through the graph a second time (or directly access saved tensors after they have "
                         "already been freed). Specify retain_graph=True if you need to backward through the graph a second time.")
    if gradient is None and self._t.numel() != 1: raise RuntimeError("grad can be implicitly created only for scalar outputs")
    root = self._t if gradient is not None or self._t.shape == () else self._t.reshape(())
    reach = _diff_slice(root.uop)
    if inputs is not None: targets = [w for w in ([inputs] if isinstance(inputs, Tensor) else inputs) if w._t.uop in reach]
    else: targets = [w for w in _grad_targets if w._tt.uop in reach]
    if targets:
      grads = root.gradient(*[w._t for w in targets], gradient=None if gradient is None else _unwrap(gradient))
      for w, g in zip(targets, grads):
        if g.dtype != w._t.dtype: g = g.cast(w._t.dtype)
        if not create_graph: g = g.detach()
        if w.grad is None: w.grad = _mk(g, create_graph, not create_graph)
        else: w.grad = _mk(w.grad._t + g, create_graph or w.grad.requires_grad, not (create_graph or w.grad.requires_grad))
    if not (retain_graph if retain_graph is not None else create_graph):
      for w in _live:
        if w._rg and not w.is_leaf and w._tt.uop in reach: w._freed = True

  def detach(self): return _mk(self._t.detach() if self._rg else self._t)
  def detach_(self):
    self._set(self._t.detach(), False, True); return self
  @property
  def data(self):
    w = _mk(self._t.detach() if self._rg else self._t)
    w._base = self
    return w
  @data.setter
  def data(self, value):
    value = _mk(_coerce(value))
    with no_grad():
      if value.shape == self.shape: self._inplace(value)
      else: self._set(value._t.detach())
  def clone(self): return _wrap(self._t + 0 if not self._t.uop.has_buffer_identity() else self._t.clone(), self)

  # ---- in-place machinery
  def _inplace(self, new) -> "Tensor":
    """Make this tensor hold the value of ``new`` (same shape after broadcasting, cast to this dtype)."""
    nt = new._t if isinstance(new, Tensor) else _coerce(new)
    new_rg = isinstance(new, Tensor) and new.requires_grad and _grad_enabled
    if nt.dtype != self._t.dtype: nt = nt.cast(self._t.dtype)
    if nt.shape != self._t.shape: nt = nt.expand(self._t.shape)
    if self._base is not None:
      with no_grad(): self._base._inplace(_mk(nt.detach()))
      self._set(self._base._t.detach() if self._base._rg else self._base._t)
    elif self._rg and self.is_leaf:
      if _grad_enabled: raise RuntimeError("a leaf Variable that requires grad is being used in an in-place operation.")
      self._assign(nt.detach())
      _flush()
    elif self._persist: self._assign(nt.detach())   # module buffers (e.g. BatchNorm statistics): written at the next sync point
    else: self._set(nt, new_rg, not new_rg)
    return self

  def _assign(self, nt:TG) -> TG:
    """Write ``nt`` into this tensor's buffer (lazily; the caller flushes). Keeps the buffer identity, which TinyJit needs."""
    self._tt = _storage(self._tt)
    self._val, self._lazy = None, False
    self._tt.assign(nt)
    _live.add(self)
    return self._tt

  def _overwrite(self, nt:TG, fresh=False) -> "Tensor":
    """In-place operations that replace the whole value without reading it (fill_, normal_, copy_ ...).
    ``fresh``: ``nt`` is a new random/constant tensor that nothing else refers to."""
    if self._base is not None or (((self._rg and self.is_leaf) or self._persist) and self._tt.uop.has_buffer_identity()):
      return self._inplace(_mk(nt))
    if nt.dtype != self._tt.dtype: nt = nt.cast(self._tt.dtype)
    if nt.shape != self._tt.shape: nt = nt.expand(self._tt.shape)
    # no storage yet (e.g. a parameter being initialized): just replace the lazy initial value
    if self._rg or self._persist: nt = nt.detach() if fresh else _unique(nt.detach())
    self._set(nt)
    return self

  # ---- reading values
  def _value(self) -> TG:
    _flush()
    t = self._val if self._val is not None else self._tt
    if not t.uop.has_buffer_identity() and not t.uop.is_virtual: t = t.contiguous().realize()
    return t
  def numpy(self):
    if self._rg: raise RuntimeError("Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.")
    return self._numpy()
  def _numpy(self) -> np.ndarray: return np.array(self._value().numpy())
  def __array__(self, dtype=None, copy=None):
    a = self._numpy()
    return a if dtype is None else a.astype(dtype)
  def item(self):
    if self._t.numel() != 1: raise RuntimeError(f"a Tensor with {self._t.numel()} elements cannot be converted to Scalar")
    return self._value().item()
  def tolist(self): return self._numpy().tolist()
  def __float__(self): return _b.float(self.item())
  def __int__(self): return _b.int(self.item())
  def __index__(self): return _b.int(self.item())
  def __bool__(self):
    if self._t.numel() != 1: raise RuntimeError("Boolean value of Tensor with more than one value is ambiguous")
    return _b.bool(self.item())
  def __len__(self):
    if not self._t.shape: raise TypeError("len() of a 0-d tensor")
    return self._t.shape[0]
  def __iter__(self):
    if not self._t.shape: raise TypeError("iteration over a 0-d tensor")
    return iter([self[i] for i in range(self._t.shape[0])])
  def __hash__(self): return id(self)
  def __contains__(self, x): return _b.bool((self == x).any().item())
  def __format__(self, spec): return format(self.item(), spec) if self._t.numel() == 1 and self._t.ndim == 0 else format(repr(self), spec)
  def __repr__(self):
    a = self._numpy()
    body = "tensor(" + _format(a, 7)
    dt = self.dtype
    if a.size == 0 and a.ndim > 1: body += f", size={tuple(a.shape)}"
    if dt not in (float32, int64, bool) or (a.size == 0 and dt is not float32): body += f", dtype={dt!r}"
    if self._rg: body += ", grad_fn=<TinygradBackward>" if not self.is_leaf else ", requires_grad=True"
    return body + ")"
  __str__ = __repr__
  def __reduce__(self):
    from ._utils import _rebuild_from_numpy
    return (_rebuild_from_numpy, (self._numpy(), self._rg))
  def __deepcopy__(self, memo):
    w = type(self).__new__(type(self))
    Tensor._init(w, _unique(self._t.detach()), False, True)
    if self._rg: w.requires_grad = True
    w._persist = self._persist
    memo[id(self)] = w
    return w

  # ---- properties
  @property
  def shape(self): return Size(self._tt.shape)
  def size(self, dim=None): return Size(self._tt.shape) if dim is None else self._tt.shape[dim]
  def dim(self): return self._tt.ndim
  ndimension = dim
  @property
  def ndim(self): return self._tt.ndim
  def numel(self): return self._tt.numel()
  nelement = numel
  @property
  def dtype(self): return _torch_dtype(self._tt.dtype)
  @property
  def device(self): return _current_device()
  @property
  def is_cuda(self): return False
  def get_device(self): return -1
  def element_size(self): return self._tt.dtype.itemsize
  @property
  def itemsize(self): return self._tt.dtype.itemsize
  @property
  def nbytes(self): return self._tt.numel() * self._tt.dtype.itemsize
  def is_floating_point(self): return _td.is_float(self._tt.dtype)
  def is_complex(self): return False
  def is_contiguous(self, *a, **k): return True
  @property
  def T(self): return self.permute(*reversed(range(self._tt.ndim)))
  @property
  def mT(self): return self.transpose(-2, -1)
  @property
  def H(self): return self.mT
  @property
  def real(self): return self
  @property
  def layout(self): return "torch.strided"

  # ---- dtype / device conversion
  def to(self, *args, **kwargs):
    dt = kwargs.get("dtype")
    for a in args:
      if isinstance(a, dtype) or a in (_b.float, _b.int, _b.bool): dt = a
      elif isinstance(a, Tensor): dt = a.dtype
    return self if dt is None or _tg_dtype(dt) == self._t.dtype else _wrap(self._t.cast(_tg_dtype(dt)), self)
  def type(self, dt=None, **kw):
    if dt is None: return f"torch.{ {'float32': 'Float', 'float64': 'Double', 'int64': 'Long', 'int32': 'Int', 'bool': 'Bool'}.get(self.dtype.name, self.dtype.name) }Tensor"
    return self.to(getattr(dt, "_dtype", dt))   # dt may be a legacy class such as torch.FloatTensor
  def type_as(self, other): return self.to(other.dtype)
  def float(self): return self.to(float32)
  def double(self): return self.to(float64)
  def half(self): return self.to(float16)
  def long(self): return self.to(int64)
  def int(self): return self.to(int32)
  def short(self): return self.to(int16)
  def byte(self): return self.to(uint8)
  def bool(self): return self.to(bool)
  def cpu(self, *a, **k): return self
  def cuda(self, *a, **k): return self
  def contiguous(self, *a, **k): return self
  def pin_memory(self, *a, **k): return self
  def share_memory_(self): return self

  # ---- indexing
  def __getitem__(self, idx):
    return _wrap(self._t[_index(idx)], self, *(i for i in (idx if isinstance(idx, tuple) else (idx,)) if isinstance(i, Tensor)))
  def __setitem__(self, idx, value):
    v = value._t if isinstance(value, Tensor) else value
    if isinstance(v, (np.ndarray, list, tuple)): v = _coerce(v)
    if isinstance(v, TG) and v.dtype != self._t.dtype: v = v.cast(self._t.dtype)
    if isinstance(idx, Tensor) and idx._t.dtype == _td.bool and idx._t.shape == self._t.shape[:idx._t.ndim] and \
       (not isinstance(v, TG) or v.ndim == 0 or v.shape == self._t.shape):
      mask = idx._t.reshape(idx._t.shape + (1,) * (self._t.ndim - idx._t.ndim))
      new = mask.where(v, self._t)
    else:
      if not isinstance(v, TG): v = TG(v, dtype=self._t.dtype)
      new = self._t._getitem(_index(idx), v)
    rg = _grad_enabled and _td.is_float(new.dtype) and ((isinstance(value, Tensor) and value.requires_grad) or (self._rg and not self.is_leaf))
    self._inplace(_mk(new, rg, not rg))

  # ---- arithmetic
  def _bin(self, name, other, reverse=False):
    o = _unwrap(other)
    if o is NotImplemented or isinstance(other, (str, type(None))): return NotImplemented
    return _wrap(getattr(self._t, name)(o, reverse) if reverse else getattr(self._t, name)(o), self, other)
  def add(self, other, *, alpha=1): return self._bin("add", other if alpha == 1 else other * alpha)
  def sub(self, other, *, alpha=1): return self._bin("sub", other if alpha == 1 else other * alpha)
  subtract = sub
  def mul(self, other): return self._bin("mul", other)
  multiply = mul
  def div(self, other, *, rounding_mode=None):
    if rounding_mode is None: return self._bin("div", other)
    o = _unwrap(other)
    q = self._t.div(o)
    q = q.trunc() if rounding_mode == "trunc" else q.floor()
    isint = not _td.is_float(self._t.dtype) and not (isinstance(o, _b.float) or (isinstance(o, TG) and _td.is_float(o.dtype)))
    return _wrap(q.cast(self._t.dtype) if isint else q, self, other)
  divide = true_divide = div
  def floor_divide(self, other): return self.div(other, rounding_mode="floor")
  def remainder(self, other):
    o = _unwrap(other)
    return _wrap(self._t - (self._t.div(o)).floor().cast(self._t.dtype) * o if not _td.is_float(self._t.dtype) else self._t - (self._t / o).floor() * o, self, other)
  def fmod(self, other):
    o = _unwrap(other)
    return _wrap(self._t - (self._t / o).trunc().cast(self._t.dtype) * o, self, other)
  def pow(self, other): return self._bin("pow", other)
  def matmul(self, other): return _wrap(self._t.matmul(other._t), self, other)
  def mm(self, other): return self.matmul(other)
  bmm = mm
  def mv(self, other): return self.matmul(other)
  def dot(self, other): return _wrap((self._t * other._t).sum(), self, other)
  inner = dot
  def outer(self, other): return _wrap(self._t.reshape(-1, 1) * other._t.reshape(1, -1), self, other)
  def maximum(self, other): return self._bin("maximum", other)
  def minimum(self, other): return self._bin("minimum", other)
  def lerp(self, end, weight): return self + (end - self) * weight
  def addcmul(self, t1, t2, *, value=1): return self + t1 * t2 * value
  def addcdiv(self, t1, t2, *, value=1): return self + t1 / t2 * value
  def atan2(self, other):
    y, x = self._t, _unwrap(other)
    if not isinstance(x, TG): x = TG(x, dtype=y.dtype)
    base = (y / x).atan()
    r = (x > 0).where(base, (x < 0).where((y >= 0).where(base + math.pi, base - math.pi), (y > 0).where(math.pi / 2, (y < 0).where(-math.pi / 2, 0.0))))
    return _wrap(r, self, other)

  __add__ = lambda self, o: self._bin("add", o)
  __radd__ = lambda self, o: self._bin("add", o, True)
  __sub__ = lambda self, o: self._bin("sub", o)
  __rsub__ = lambda self, o: self._bin("sub", o, True)
  __mul__ = lambda self, o: self._bin("mul", o)
  __rmul__ = lambda self, o: self._bin("mul", o, True)
  __truediv__ = lambda self, o: self._bin("div", o)
  __rtruediv__ = lambda self, o: self._bin("div", o, True)
  __floordiv__ = lambda self, o: self.floor_divide(o)
  __rfloordiv__ = lambda self, o: _mk(_coerce(o)).floor_divide(self) if not isinstance(o, Tensor) else o.floor_divide(self)
  __mod__ = lambda self, o: self.remainder(o)
  __pow__ = lambda self, o: self._bin("pow", o)
  __rpow__ = lambda self, o: self._bin("pow", o, True)
  __matmul__ = lambda self, o: self.matmul(o)
  __rmatmul__ = lambda self, o: _mk(_coerce(o)).matmul(self)
  __neg__ = lambda self: _wrap(-self._t, self)
  __pos__ = lambda self: self
  __abs__ = lambda self: self.abs()
  __invert__ = lambda self: _wrap(self._t.logical_not() if self._t.dtype == _td.bool else self._t.bitwise_not(), self)
  __and__ = lambda self, o: self._bin("bitwise_and", o)
  __rand__ = __and__
  __or__ = lambda self, o: self._bin("bitwise_or", o)
  __ror__ = __or__
  __xor__ = lambda self, o: self._bin("bitwise_xor", o)
  __rxor__ = __xor__
  __lshift__ = lambda self, o: self._bin("lshift", o)
  __rshift__ = lambda self, o: self._bin("rshift", o)
  __eq__ = lambda self, o: self._bin("eq", o)
  __ne__ = lambda self, o: self._bin("ne", o)
  __lt__ = lambda self, o: self._bin("__lt__", o)
  __le__ = lambda self, o: self._bin("__le__", o)
  __gt__ = lambda self, o: self._bin("__gt__", o)
  __ge__ = lambda self, o: self._bin("__ge__", o)
  eq, ne, lt, le, gt, ge = __eq__, __ne__, __lt__, __le__, __gt__, __ge__
  greater, greater_equal, less, less_equal, not_equal = __gt__, __ge__, __lt__, __le__, __ne__
  def __iadd__(self, o): return self._inplace(self + o)
  def __isub__(self, o): return self._inplace(self - o)
  def __imul__(self, o): return self._inplace(self * o)
  def __itruediv__(self, o): return self._inplace(self / o)
  def __ipow__(self, o): return self._inplace(self ** o)
  def __iand__(self, o): return self._inplace(self & o)
  def __ior__(self, o): return self._inplace(self | o)
  def logical_and(self, o): return _wrap((self._t != 0) & (_unwrap(o) != 0), self, o)
  def logical_or(self, o): return _wrap((self._t != 0) | (_unwrap(o) != 0), self, o)
  def logical_xor(self, o): return _wrap((self._t != 0) ^ (_unwrap(o) != 0), self, o)
  def logical_not(self): return _wrap(self._t == 0, self)
  bitwise_and, bitwise_or, bitwise_xor, bitwise_not = __and__, __or__, __xor__, __invert__
  def isclose(self, other, rtol=1e-05, atol=1e-08, equal_nan=False): return _wrap(self._t.isclose(_unwrap(other), rtol, atol, equal_nan), self)
  def allclose(self, other, rtol=1e-05, atol=1e-08, equal_nan=False): return _b.bool(self.isclose(other, rtol, atol, equal_nan).all().item())
  def equal(self, other): return self.shape == other.shape and _b.bool((self == other).all().item())

  # ---- elementwise functions with torch-specific signatures
  def clamp(self, min=None, max=None): return _wrap(self._t.clamp(_unwrap(min), _unwrap(max)), self, min, max)
  clip = clamp
  def clamp_min(self, min): return self.clamp(min=min)
  def clamp_max(self, max): return self.clamp(max=max)
  def log1p(self): return _wrap((self._t + 1).log(), self)
  def expm1(self): return _wrap(self._t.exp() - 1, self)
  def sigmoid(self): return _wrap(self._t.sigmoid(), self)
  def frac(self): return _wrap(self._t - self._t.trunc(), self)
  def nan_to_num(self, nan=0.0, posinf=None, neginf=None):
    f = np.finfo(np.float32)
    t = self._t.isnan().where(nan, self._t)
    t = (t == math.inf).where(f.max if posinf is None else posinf, t)
    return _wrap((t == -math.inf).where(f.min if neginf is None else neginf, t), self)
  def where(self, condition, other): return _wrap(_unwrap(condition).where(self._t, _unwrap(other)), self, other)
  def masked_fill(self, mask, value): return _wrap(_unwrap(mask).where(_unwrap(value), self._t), self, value)
  def softmax(self, dim=-1, dtype=None): return _wrap(self._t.softmax(dim), self)
  def log_softmax(self, dim=-1, dtype=None): return _wrap(self._t.log_softmax(dim), self)
  def relu(self): return _wrap(self._t.relu(), self)

  # ---- reductions
  def _red(self, name, dim=None, keepdim=False, **kw): return _wrap(getattr(self._t, name)(_dim(dim), keepdim, **kw), self)
  def sum(self, dim=None, keepdim=False, dtype=None, axis=None):
    t = self._t.sum(_dim(dim if axis is None else axis), keepdim)
    if dtype is not None: t = t.cast(_tg_dtype(dtype))
    elif not _td.is_float(self._t.dtype): t = t.cast(_long())
    return _wrap(t, self)
  def mean(self, dim=None, keepdim=False, dtype=None, axis=None):
    if not _td.is_float(self._t.dtype) and dtype is None: raise RuntimeError(f"mean(): could not infer output dtype. Input dtype must be either a floating point or complex dtype. Got: {self.dtype.name.capitalize()}")
    return _wrap((self._t if dtype is None else self._t.cast(_tg_dtype(dtype))).mean(_dim(dim if axis is None else axis), keepdim), self)
  def prod(self, dim=None, keepdim=False, dtype=None):
    t = self._t.prod(_dim(dim), keepdim)
    return _wrap(t if _td.is_float(self._t.dtype) else t.cast(_long()), self)
  def std(self, dim=None, unbiased=None, keepdim=False, *, correction=1):
    if isinstance(dim, _b.bool): dim, unbiased = None, dim
    return _wrap(self._t.std(_dim(dim), keepdim, correction if unbiased is None else _b.int(unbiased)), self)
  def var(self, dim=None, unbiased=None, keepdim=False, *, correction=1):
    if isinstance(dim, _b.bool): dim, unbiased = None, dim
    return _wrap(self._t.var(_dim(dim), keepdim, correction if unbiased is None else _b.int(unbiased)), self)
  def _minmax(self, name, arg, dim, keepdim):
    if isinstance(dim, Tensor): return self.maximum(dim) if name == "max" else self.minimum(dim)
    if dim is None: return _wrap(getattr(self._t, name)(), self)
    return _minmax(_wrap(getattr(self._t, name)(dim, keepdim), self), _mk(getattr(self._t, arg)(dim, keepdim).cast(_long())))
  def max(self, dim=None, keepdim=False): return self._minmax("max", "argmax", dim, keepdim)
  def min(self, dim=None, keepdim=False): return self._minmax("min", "argmin", dim, keepdim)
  def amax(self, dim=None, keepdim=False): return self._red("max", dim, keepdim)
  def amin(self, dim=None, keepdim=False): return self._red("min", dim, keepdim)
  def argmax(self, dim=None, keepdim=False): return _mk(self._t.argmax(dim, keepdim).cast(_long()))
  def argmin(self, dim=None, keepdim=False): return _mk(self._t.argmin(dim, keepdim).cast(_long()))
  def all(self, dim=None, keepdim=False): return _mk(self._t.all(_dim(dim), keepdim))
  def any(self, dim=None, keepdim=False): return _mk(self._t.any(_dim(dim), keepdim))
  def logsumexp(self, dim, keepdim=False): return self._red("logsumexp", dim, keepdim)
  def norm(self, p="fro", dim=None, keepdim=False):
    if p in ("fro", 2, 2.0): return _wrap(self._t.square().sum(_dim(dim), keepdim).sqrt(), self)
    if p in (1, 1.0): return _wrap(self._t.abs().sum(_dim(dim), keepdim), self)
    if p in (math.inf, "inf"): return _wrap(self._t.abs().max(_dim(dim), keepdim), self)
    return _wrap(self._t.abs().pow(p).sum(_dim(dim), keepdim).pow(1.0 / p), self)
  def cumsum(self, dim): return _wrap(self._t.cumsum(dim), self)
  def cumprod(self, dim): return _wrap(self._t.cumprod(dim), self)
  def count_nonzero(self, dim=None): return _mk((self._t != 0).sum(_dim(dim)).cast(_long()))
  def topk(self, k, dim=-1, largest=True, sorted=True):
    v, i = self._t.topk(k, dim, largest)
    return _minmax(_wrap(v, self), _mk(i.cast(_long())))
  def sort(self, dim=-1, descending=False, stable=False):
    v, i = self._t.sort(dim, descending)
    return _minmax(_wrap(v, self), _mk(i.cast(_long())))
  def argsort(self, dim=-1, descending=False, stable=False): return _mk(self._t.argsort(dim, descending).cast(_long()))
  def median(self, dim=None, keepdim=False):
    if dim is None:
      v = self._t.flatten().sort()[0]
      return _wrap(v[(v.shape[0] - 1) // 2], self)
    v, i = self._t.sort(dim)
    k = (self._t.shape[dim] - 1) // 2
    sel = [slice(None)] * self._t.ndim
    sel[dim] = slice(k, k + 1) if keepdim else k
    return _minmax(_wrap(v[tuple(sel)], self), _mk(i[tuple(sel)].cast(_long())))
  def nonzero(self, as_tuple=False):
    idx = np.nonzero(self._numpy())
    if as_tuple: return tuple(_mk(TG(i.astype(np.int64), dtype=_long())) for i in idx)
    return _mk(TG(np.ascontiguousarray(np.stack(idx, axis=1).astype(np.int64)), dtype=_long()))
  def argwhere(self): return self.nonzero()
  def unique(self, sorted=True, return_inverse=False, return_counts=False, dim=None):
    r = np.unique(self._numpy(), return_inverse=return_inverse, return_counts=return_counts, axis=dim)
    if not isinstance(r, tuple): return _mk(_coerce(r))
    return tuple(_mk(_coerce(x.reshape(self.shape) if n == 1 and return_inverse and dim is None else x)) for n, x in enumerate(r))
  def bincount(self, weights=None, minlength=0):
    return _mk(_coerce(np.bincount(self._numpy(), None if weights is None else weights._numpy(), minlength)))
  def multinomial(self, num_samples, replacement=False, *, generator=None):
    return _mk(self._t.multinomial(num_samples, replacement).cast(_long()))

  # ---- shape manipulation
  def view(self, *shape):
    if len(shape) == 1 and isinstance(shape[0], dtype): return self.to(shape[0])
    return _wrap(self._t.reshape(_shape_args(shape)), self)
  reshape = view
  def view_as(self, other): return self.view(other.shape)
  reshape_as = view_as
  def flatten(self, start_dim=0, end_dim=-1): return _wrap(self._t.flatten(start_dim, end_dim), self)
  ravel = flatten
  def unflatten(self, dim, sizes): return _wrap(self._t.unflatten(dim, tuple(sizes)), self)
  def squeeze(self, dim=None):
    if isinstance(dim, (tuple, list)):
      t = self._t
      for d in sorted((d % t.ndim for d in dim), reverse=True): t = t.squeeze(d) if t.shape[d] == 1 else t
      return _wrap(t, self)
    if dim is not None and self._t.ndim and self._t.shape[dim] != 1: return self
    return _wrap(self._t.squeeze(dim), self)
  def unsqueeze(self, dim): return _wrap(self._t.unsqueeze(dim), self)
  def transpose(self, dim0, dim1): return _wrap(self._t.transpose(dim0, dim1), self)
  swapaxes = swapdims = transpose
  def t(self): return self if self._t.ndim < 2 else self.transpose(0, 1)
  def permute(self, *dims): return _wrap(self._t.permute(_shape_args(dims)), self)
  def movedim(self, source, destination):
    order = [d for d in range(self._t.ndim) if d != source % self._t.ndim]
    order.insert(destination % self._t.ndim, source % self._t.ndim)
    return self.permute(*order)
  def expand(self, *sizes): return _wrap(self._t.expand(_shape_args(sizes)), self)
  def expand_as(self, other): return self.expand(*other.shape)
  def broadcast_to(self, shape): return self.expand(*shape)
  def repeat(self, *sizes): return _wrap(self._t.repeat(_shape_args(sizes)), self)
  def repeat_interleave(self, repeats, dim=None): return _wrap(self._t.repeat_interleave(repeats, dim), self)
  def tile(self, *dims):
    dims = _shape_args(dims)
    return self.repeat(*((1,) * (self._t.ndim - len(dims)) + dims))
  def flip(self, *dims): return _wrap(self._t.flip(_shape_args(dims)), self)
  def roll(self, shifts, dims=None):
    if dims is None: return _wrap(self._t.flatten().roll(shifts, 0).reshape(self._t.shape), self)
    return _wrap(self._t.roll(shifts, dims), self)
  def chunk(self, chunks, dim=0): return tuple(_wrap(self._t.chunk(chunks, dim), self))
  def split(self, split_size, dim=0): return tuple(_wrap(self._t.split(split_size, dim), self))
  def unbind(self, dim=0): return tuple(self.select(dim, i) for i in range(self._t.shape[dim]))
  def select(self, dim, index): return self[(slice(None),) * (dim % self._t.ndim) + (index,)]
  def narrow(self, dim, start, length): return self[(slice(None),) * (dim % self._t.ndim) + (slice(start, start + length),)]
  def tril(self, diagonal=0): return _wrap(self._t.tril(diagonal), self)
  def triu(self, diagonal=0): return _wrap(self._t.triu(diagonal), self)
  def diag(self, diagonal=0):
    if self._t.ndim == 1 and diagonal == 0: return _wrap(self._t.diag(), self)
    if self._t.ndim == 2: return _wrap(self._t.diagonal(diagonal), self)
    raise NotImplementedError("diag with a nonzero diagonal on a 1-D tensor")
  def diagonal(self, offset=0, dim1=0, dim2=1): return _wrap(self._t.diagonal(offset, dim1, dim2), self)
  def gather(self, dim, index): return _wrap(self._t.gather(dim, index._t), self)
  def index_select(self, dim, index): return self[(slice(None),) * (dim % self._t.ndim) + (index,)]
  def take(self, index): return self.flatten()[index]
  def masked_select(self, mask): return self.expand(*np.broadcast_shapes(self.shape, mask.shape))[mask.expand(*np.broadcast_shapes(self.shape, mask.shape))]
  def scatter(self, dim, index, src=None, reduce=None, *, value=None):
    s = _unwrap(src if src is not None else value)
    return _wrap(self._t.scatter(dim, index._t, s, reduce), self, src)
  def scatter_add(self, dim, index, src): return _wrap(self._t.scatter_reduce(dim, index._t, src._t, "sum"), self, src)
  def scatter_reduce(self, dim, index, src, reduce, *, include_self=True):
    return _wrap(self._t.scatter_reduce(dim, index._t, src._t, reduce, include_self), self, src)
  def index_add(self, dim, index, source):
    shape = [1] * self._t.ndim; shape[dim] = -1
    idx = index._t.reshape(shape).expand(source._t.shape)
    return _wrap(self._t.scatter_reduce(dim, idx, source._t, "sum"), self, source)
  def index_fill(self, dim, index, value):
    onehot = (TG.arange(self._t.shape[dim]).reshape(-1, 1) == index._t.reshape(1, -1)).any(1)
    shape = [1] * self._t.ndim; shape[dim] = -1
    return _wrap(onehot.reshape(shape).where(value, self._t), self)
  def unfold(self, dimension, size, step): return _wrap(self._t.unfold(dimension, size, step), self)

  # ---- new_* and fills
  def new_tensor(self, data, dtype=None, **k): return _mk(_coerce(data, dtype or (self.dtype if not isinstance(data, Tensor) else None)))
  def new_zeros(self, *size, dtype=None, **k): return _mk(TG.zeros(*_shape_args(size), dtype=_tg_dtype(dtype) or self._t.dtype))
  def new_ones(self, *size, dtype=None, **k): return _mk(TG.ones(*_shape_args(size), dtype=_tg_dtype(dtype) or self._t.dtype))
  def new_full(self, size, fill_value, dtype=None, **k): return _mk(TG.full(tuple(size), fill_value, dtype=_tg_dtype(dtype) or self._t.dtype))
  new_empty = new_zeros
  def fill_(self, value): return self._overwrite(TG.full(self._tt.shape, _unwrap(value) if not isinstance(value, Tensor) else value.item(), dtype=self._tt.dtype), fresh=True)
  def zero_(self): return self._overwrite(TG.zeros(*self._tt.shape, dtype=self._tt.dtype), fresh=True)
  def copy_(self, src, non_blocking=False): return self._overwrite(_coerce(src).detach())
  def normal_(self, mean=0.0, std=1.0, *, generator=None): return self._overwrite(TG.randn(*self._tt.shape) * std + mean, fresh=True)
  def uniform_(self, a=0.0, b=1.0, *, generator=None): return self._overwrite(TG.rand(*self._tt.shape) * (b - a) + a, fresh=True)
  def bernoulli_(self, p=0.5, *, generator=None): return self._overwrite((TG.rand(*self._tt.shape) < _unwrap(p)).cast(self._tt.dtype), fresh=True)
  def random_(self, from_=0, to=None, *, generator=None):
    if to is None: from_, to = 0, from_
    return self._overwrite(TG.randint(*self._tt.shape, low=from_, high=to).cast(self._tt.dtype), fresh=True)
  def exponential_(self, lambd=1.0, *, generator=None): return self._overwrite(-(1 - TG.rand(*self._tt.shape)).log() / lambd, fresh=True)
  def bernoulli(self, *, generator=None): return _mk((TG.rand(*self._t.shape) < self._t).cast(self._t.dtype))
  def apply_(self, fn): return self._overwrite(_coerce(np.vectorize(fn)(self._numpy())))
  def masked_fill_(self, mask, value): return self._inplace(self.masked_fill(mask, value))
  def clamp_(self, min=None, max=None): return self._inplace(self.clamp(min, max))
  clip_ = clamp_
  def add_(self, other, *, alpha=1): return self._inplace(self.add(other, alpha=alpha))
  def sub_(self, other, *, alpha=1): return self._inplace(self.sub(other, alpha=alpha))
  def div_(self, other, *, rounding_mode=None): return self._inplace(self.div(other, rounding_mode=rounding_mode))
  def addcmul_(self, t1, t2, *, value=1): return self._inplace(self.addcmul(t1, t2, value=value))
  def addcdiv_(self, t1, t2, *, value=1): return self._inplace(self.addcdiv(t1, t2, value=value))
  def lerp_(self, end, weight): return self._inplace(self.lerp(end, weight))
  def squeeze_(self, dim=None): return self._reshape_inplace(self.squeeze(dim))
  def unsqueeze_(self, dim): return self._reshape_inplace(self.unsqueeze(dim))
  def t_(self): return self._reshape_inplace(self.t())
  def transpose_(self, d0, d1): return self._reshape_inplace(self.transpose(d0, d1))
  def resize_(self, *sizes): return self._reshape_inplace(self.new_zeros(*sizes))
  def _reshape_inplace(self, new):
    if self._rg and self.is_leaf: self._set(new._t.detach(), True, True)
    else: self._set(new._t, new._rg, new.is_leaf)
    return self

  def __getattr__(self, name):
    # Anything not defined above falls through to the tinygrad Tensor method of the same name. PyTorch's `dim`/`keepdim`
    # keyword names are translated, and Tensor arguments and results are unwrapped/wrapped.
    if name.startswith("_"): raise AttributeError(name)
    if name.endswith("_") and hasattr(type(self), name[:-1]):
      return lambda *a, **k: self._inplace(getattr(self, name[:-1])(*a, **k))
    fn = getattr(TG, name, None)
    if fn is None or not callable(fn):
      if name.endswith("_") and callable(getattr(TG, name[:-1], None)): return lambda *a, **k: self._inplace(getattr(self, name[:-1])(*a, **k))
      raise AttributeError(f"'Tensor' object has no attribute '{name}' (not implemented by the tinygrad.js torch shim)")
    def method(*args, **kwargs):
      if "dim" in kwargs and name not in _KEEPS_DIM_KW: kwargs["axis"] = _dim(kwargs.pop("dim"))
      r = fn(self._t, *[_unwrap(a) for a in args], **{k: _unwrap(v) for k, v in kwargs.items()})
      return _wrap(r, self, *args, *kwargs.values())
    return method

_KEEPS_DIM_KW = {"cat", "stack", "split", "chunk", "squeeze", "unsqueeze", "gather", "scatter", "topk", "sort", "argsort", "unflatten",
                 "repeat_interleave", "unfold"}

def _add_unary(name, tg_name=None, differentiable=True):
  tg_name = tg_name or name
  def method(self): return _wrap(getattr(self._t, tg_name)(), self) if differentiable else _mk(getattr(self._t, tg_name)())
  method.__name__ = name
  setattr(Tensor, name, method)
for _n in ("abs exp log log2 log10 sqrt rsqrt sin cos tan tanh floor ceil trunc sign reciprocal square erf asin acos atan sinh cosh "
           "exp2 neg asinh acosh atanh").split():
  _add_unary(_n)
for _n in ("isnan", "isinf", "isfinite"): _add_unary(_n, differentiable=False)
Tensor.round = lambda self, decimals=0: _wrap(self._t.round() if decimals == 0 else (self._t * 10 ** decimals).round() / 10 ** decimals, self)
Tensor.absolute, Tensor.negative, Tensor.arcsin, Tensor.arccos, Tensor.arctan = Tensor.abs, Tensor.neg, Tensor.asin, Tensor.acos, Tensor.atan
Tensor.sgn, Tensor.fix = Tensor.sign, Tensor.trunc
Tensor.__round__ = lambda self, n=None: round(self.item(), n)

# in-place variants (`x.relu_()`, `x.mul_(2)`, ...) for every method above that has none yet
for _n in ("abs exp log log2 log10 sqrt rsqrt sin cos tan tanh floor ceil trunc sign reciprocal square erf neg round sigmoid relu mul "
           "pow remainder fmod floor_divide true_divide divide multiply subtract maximum minimum log1p expm1 nan_to_num tril triu "
           "cumsum scatter scatter_add index_add index_fill logical_not logical_and logical_or frac clamp_min clamp_max "
           "float double half long int short bool").split():
  if not hasattr(Tensor, _n + "_"):
    setattr(Tensor, _n + "_", (lambda n: lambda self, *a, **k: self._inplace(getattr(self, n)(*a, **k)))(_n))

# ---------------------------------------------------------------- creation functions

_host_rng = np.random.default_rng()
class Generator:
  def __init__(self, device="cpu"): self.device, self._seed = device, None
  def manual_seed(self, seed): self._seed = seed; manual_seed(seed); return self
  def seed(self): return self._seed
  def initial_seed(self): return self._seed
default_generator = Generator()

def manual_seed(seed):
  """Seeds tinygrad's random number generator. The random streams differ from PyTorch's for the same seed."""
  global _host_rng
  TG.manual_seed(_b.int(seed))
  _host_rng = np.random.default_rng(_b.int(seed))
  default_generator._seed = seed
  return default_generator
seed = lambda: manual_seed(np.random.SeedSequence().entropy % (2 ** 31))

def _new(t:TG, requires_grad=False) -> Tensor: return _mk(t, requires_grad, True)

def tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False) -> Tensor:
  t = _coerce(data, dtype)
  if isinstance(data, Tensor): t = _unique(t.detach())
  return _new(t, requires_grad)
def as_tensor(data, dtype=None, device=None): return data.to(dtype) if isinstance(data, Tensor) else tensor(data, dtype)
asarray = as_tensor
def from_numpy(a): return _new(_coerce(a))
def is_tensor(x): return isinstance(x, Tensor)
def numel(x): return x.numel()

def _float_dtype(dt): return _tg_dtype(dt) or _default_float()
def zeros(*size, dtype=None, device=None, requires_grad=False, out=None, layout=None): return _new(TG.zeros(*_shape_args(size), dtype=_float_dtype(dtype)), requires_grad)
def ones(*size, dtype=None, device=None, requires_grad=False, out=None, layout=None): return _new(TG.ones(*_shape_args(size), dtype=_float_dtype(dtype)), requires_grad)
empty = zeros
def full(size, fill_value, dtype=None, device=None, requires_grad=False):
  dt = _tg_dtype(dtype) or (_td.bool if isinstance(fill_value, _b.bool) else _long() if isinstance(fill_value, _b.int) else _default_float())
  return _new(TG.full(tuple(size) if _is_seq(size) else (size,), _unwrap(fill_value) if not isinstance(fill_value, Tensor) else fill_value.item(), dtype=dt), requires_grad)
def zeros_like(x, dtype=None, device=None, requires_grad=False): return _new(TG.zeros(*x.shape, dtype=_tg_dtype(dtype) or x._t.dtype), requires_grad)
def ones_like(x, dtype=None, device=None, requires_grad=False): return _new(TG.ones(*x.shape, dtype=_tg_dtype(dtype) or x._t.dtype), requires_grad)
empty_like = zeros_like
def full_like(x, fill_value, dtype=None, device=None, requires_grad=False): return _new(TG.full(tuple(x.shape), fill_value, dtype=_tg_dtype(dtype) or x._t.dtype), requires_grad)
def rand(*size, dtype=None, device=None, requires_grad=False, generator=None): return _new(TG.rand(*_shape_args(size), dtype=_float_dtype(dtype)), requires_grad)
def randn(*size, dtype=None, device=None, requires_grad=False, generator=None): return _new(TG.randn(*_shape_args(size), dtype=_float_dtype(dtype)), requires_grad)
def rand_like(x, dtype=None, device=None, requires_grad=False): return rand(*x.shape, dtype=dtype or x.dtype, requires_grad=requires_grad)
def randn_like(x, dtype=None, device=None, requires_grad=False): return randn(*x.shape, dtype=dtype or x.dtype, requires_grad=requires_grad)
def randint(low, high=None, size=None, dtype=None, device=None, requires_grad=False, generator=None):
  if size is None: low, high, size = 0, low, high
  return _new(TG.randint(*_shape_args((size,)), low=low, high=high).cast(_tg_dtype(dtype) or _long()))
def randint_like(x, low, high=None, **k): return randint(low, high, tuple(x.shape), dtype=x.dtype) if high is not None else randint(low, tuple(x.shape), dtype=x.dtype)
def randperm(n, dtype=None, device=None, generator=None, requires_grad=False): return _new(TG(_host_rng.permutation(n).astype(np.int64), dtype=_tg_dtype(dtype) or _long()))
def normal(mean=0.0, std=1.0, size=None, generator=None, **k):
  if size is None: size = (mean.shape if isinstance(mean, Tensor) else std.shape)
  return _new(TG.randn(*size) * _unwrap(std) + _unwrap(mean))
def bernoulli(x, generator=None): return x.bernoulli()
def multinomial(x, num_samples, replacement=False, generator=None): return x.multinomial(num_samples, replacement)
def arange(start, end=None, step=1, dtype=None, device=None, requires_grad=False):
  if end is None: start, end = 0, start
  start, end, step = (v.item() if isinstance(v, Tensor) else v for v in (start, end, step))
  isf = any(isinstance(v, _b.float) for v in (start, end, step))
  return _new(TG.arange(start, end, step, dtype=_tg_dtype(dtype) or (_default_float() if isf else _long())), requires_grad)
def linspace(start, end, steps, dtype=None, device=None, requires_grad=False): return _new(TG.linspace(start, end, steps, dtype=_float_dtype(dtype)), requires_grad)
def logspace(start, end, steps, base=10.0, dtype=None, **k): return _new(TG(base, dtype=_float_dtype(dtype)).pow(TG.linspace(start, end, steps, dtype=_float_dtype(dtype))))
def eye(n, m=None, dtype=None, device=None, requires_grad=False): return _new(TG.eye(n, m, dtype=_float_dtype(dtype)), requires_grad)
def scalar_tensor(v, dtype=None, **k): return tensor(v, dtype=dtype)
