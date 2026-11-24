"""torch.nn modules: the Module system and the common layers, with PyTorch's parameter names, shapes and default initialization."""
from __future__ import annotations
import math, itertools
from collections import OrderedDict, namedtuple
from .. import _tensor
from .._tensor import Tensor, _mk, _coerce, _copy_into, _unique, _tg_dtype, no_grad, zeros, ones, empty, tensor
from .parameter import Parameter, Buffer
from . import functional as F, init

_IncompatibleKeys = namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"])
class _IncompatibleKeys(_IncompatibleKeys):
  def __repr__(self): return "<All keys matched successfully>" if not self.missing_keys and not self.unexpected_keys else super().__repr__()

class _Handle:
  def __init__(self, d, k): self.d, self.k = d, k
  def remove(self): self.d.pop(self.k, None)
  def __enter__(self): return self
  def __exit__(self, *a): self.remove()

def _addindent(s, n):
  lines = s.split("\n")
  return lines[0] if len(lines) == 1 else lines[0] + "\n" + "\n".join(" " * n + l for l in lines[1:])

class Module:
  training: bool = True
  _hook_ids = itertools.count()

  def __init__(self, *args, **kwargs):
    d = self.__dict__
    d["training"] = True
    d["_parameters"], d["_buffers"], d["_modules"] = OrderedDict(), OrderedDict(), OrderedDict()
    d["_non_persistent_buffers_set"] = set()
    d["_forward_hooks"], d["_forward_pre_hooks"] = OrderedDict(), OrderedDict()

  def forward(self, *args, **kwargs):
    raise NotImplementedError(f'Module [{type(self).__name__}] is missing the required "forward" function')

  def __call__(self, *args, **kwargs):
    for hook in list(self._forward_pre_hooks.values()):
      r = hook(self, args)
      if r is not None: args = r if isinstance(r, tuple) else (r,)
    out = self.forward(*args, **kwargs)
    for hook in list(self._forward_hooks.values()):
      r = hook(self, args, out)
      if r is not None: out = r
    return out
  def register_forward_hook(self, hook, **k):
    i = next(Module._hook_ids); self._forward_hooks[i] = hook
    return _Handle(self._forward_hooks, i)
  def register_forward_pre_hook(self, hook, **k):
    i = next(Module._hook_ids); self._forward_pre_hooks[i] = hook
    return _Handle(self._forward_pre_hooks, i)

  # ---- attribute registration
  def __setattr__(self, name, value):
    d = self.__dict__
    if isinstance(value, (Parameter, Module)) and "_parameters" not in d:
      raise AttributeError(f"cannot assign {'parameters' if isinstance(value, Parameter) else 'module'} before Module.__init__() call")
    if "_parameters" not in d: return object.__setattr__(self, name, value)
    params, buffers, modules = d["_parameters"], d["_buffers"], d["_modules"]
    if isinstance(value, Parameter):
      for other in (d, buffers, modules): other.pop(name, None)
      params[name] = value
    elif isinstance(value, Module):
      for other in (d, params, buffers): other.pop(name, None)
      modules[name] = value
    elif isinstance(value, Buffer):
      for other in (d, params, modules): other.pop(name, None)
      buffers[name] = value
      if not value.persistent: d["_non_persistent_buffers_set"].add(name)
    elif name in params:
      if value is not None: raise TypeError(f"cannot assign '{type(value).__name__}' as parameter '{name}' (torch.nn.Parameter or None expected)")
      params[name] = None
    elif name in buffers:
      if value is not None and not isinstance(value, Tensor): raise TypeError(f"cannot assign '{type(value).__name__}' as buffer '{name}' (torch.Tensor or None expected)")
      if value is not None: value._persist = True
      buffers[name] = value
    elif name in modules:
      if value is not None: raise TypeError(f"cannot assign '{type(value).__name__}' as child module '{name}' (torch.nn.Module or None expected)")
      modules[name] = None
    else: object.__setattr__(self, name, value)
  def __getattr__(self, name):
    d = self.__dict__
    for store in ("_parameters", "_buffers", "_modules"):
      if store in d and name in d[store]: return d[store][name]
    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
  def __delattr__(self, name):
    for store in (self._parameters, self._buffers, self._modules):
      if name in store: del store[name]; return
    object.__delattr__(self, name)
  def __dir__(self): return list(object.__dir__(self)) + list(self._parameters) + list(self._buffers) + list(self._modules)

  def register_parameter(self, name, param):
    if param is None: self._parameters[name] = None
    else: setattr(self, name, param)
  def register_buffer(self, name, tensor, persistent=True):
    if "_buffers" not in self.__dict__: raise AttributeError("cannot assign buffer before Module.__init__() call")
    if tensor is not None and not isinstance(tensor, Tensor): raise TypeError(f"cannot assign '{type(tensor).__name__}' object to buffer '{name}' (torch Tensor or None required)")
    self.__dict__.pop(name, None)
    if tensor is not None: tensor._persist = True
    self._buffers[name] = tensor
    if persistent: self._non_persistent_buffers_set.discard(name)
    else: self._non_persistent_buffers_set.add(name)
  def add_module(self, name, module): self._modules[name] = module
  register_module = add_module

  # ---- traversal
  def named_modules(self, memo=None, prefix="", remove_duplicate=True):
    memo = set() if memo is None else memo
    if id(self) in memo and remove_duplicate: return
    memo.add(id(self))
    yield prefix, self
    for name, m in self._modules.items():
      if m is not None: yield from m.named_modules(memo, prefix + ("." if prefix else "") + name, remove_duplicate)
  def modules(self): return (m for _, m in self.named_modules())
  def named_children(self): return ((n, m) for n, m in self._modules.items() if m is not None)
  def children(self): return (m for _, m in self.named_children())
  def _named_members(self, store, prefix, recurse):
    seen = set()
    for mprefix, m in (self.named_modules(prefix=prefix) if recurse else [(prefix, self)]):
      for name, v in getattr(m, store).items():
        if v is None or id(v) in seen: continue
        seen.add(id(v))
        yield mprefix + ("." if mprefix else "") + name, v
  def named_parameters(self, prefix="", recurse=True, remove_duplicate=True): return self._named_members("_parameters", prefix, recurse)
  def parameters(self, recurse=True): return (p for _, p in self.named_parameters(recurse=recurse))
  def named_buffers(self, prefix="", recurse=True, remove_duplicate=True): return self._named_members("_buffers", prefix, recurse)
  def buffers(self, recurse=True): return (b for _, b in self.named_buffers(recurse=recurse))
  def get_submodule(self, target):
    m = self
    for part in (target.split(".") if target else []): m = getattr(m, part)
    return m
  def get_parameter(self, target):
    mod, _, name = target.rpartition(".")
    return self.get_submodule(mod)._parameters[name]
  def get_buffer(self, target):
    mod, _, name = target.rpartition(".")
    return self.get_submodule(mod)._buffers[name]
  def apply(self, fn):
    for m in self.children(): m.apply(fn)
    fn(self)
    return self

  # ---- state
  def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
    out = OrderedDict() if destination is None else destination
    for name, p in self._parameters.items():
      if p is not None: out[prefix + name] = p if keep_vars else p.data
    for name, b in self._buffers.items():
      if b is not None and name not in self._non_persistent_buffers_set: out[prefix + name] = b if keep_vars else b.data
    for name, m in self._modules.items():
      if m is not None: m.state_dict(destination=out, prefix=prefix + name + ".", keep_vars=keep_vars)
    return out
  def load_state_dict(self, state_dict, strict=True, assign=False):
    own = self.state_dict(keep_vars=True)
    missing = [k for k in own if k not in state_dict]
    unexpected = [k for k in state_dict if k not in own]
    errors, pairs = [], []
    for k, dst in own.items():
      if k not in state_dict: continue
      src = state_dict[k] if isinstance(state_dict[k], Tensor) else _mk(_coerce(state_dict[k]))
      if tuple(src.shape) != tuple(dst.shape):
        errors.append(f"size mismatch for {k}: copying a param with shape {src.shape} from checkpoint, the shape in current model is {dst.shape}.")
      else: pairs.append((dst, src._t))
    if strict and unexpected: errors.insert(0, "Unexpected key(s) in state_dict: " + ", ".join(f'"{k}"' for k in unexpected) + ". ")
    if strict and missing: errors.insert(0, "Missing key(s) in state_dict: " + ", ".join(f'"{k}"' for k in missing) + ". ")
    if errors: raise RuntimeError(f"Error(s) in loading state_dict for {type(self).__name__}:\n\t" + "\n\t".join(errors))
    with no_grad(): _copy_into(pairs)
    return _IncompatibleKeys(missing, unexpected)

  def train(self, mode=True):
    for m in self.modules(): m.__dict__["training"] = mode
    return self
  def eval(self): return self.train(False)
  def requires_grad_(self, requires_grad=True):
    for p in self.parameters(): p.requires_grad_(requires_grad)
    return self
  def zero_grad(self, set_to_none=True):
    for p in self.parameters():
      if p.grad is not None: p.grad = None if set_to_none else _mk(p.grad._t.zeros_like())
  def _cast(self, dt, floats_only=True):
    tg = _tg_dtype(dt)
    with no_grad():
      for t in itertools.chain(self.parameters(), self.buffers()):
        if t.is_floating_point() and t._t.dtype != tg: t._set(_unique(t._t.detach().cast(tg)))
    return self
  def to(self, *args, **kwargs):
    dt = kwargs.get("dtype")
    for a in args:
      if isinstance(a, _tensor.dtype): dt = a
    return self if dt is None else self._cast(dt)
  def float(self): return self._cast(_tensor.float32)
  def double(self): return self._cast(_tensor.float64)
  def half(self): return self._cast(_tensor.float16)
  def cpu(self): return self
  def cuda(self, device=None): return self
  def share_memory(self): return self
  def compile(self, *a, **k): pass

  # ---- repr
  def extra_repr(self): return ""
  def _get_name(self): return type(self).__name__
  def __repr__(self):
    extra = self.extra_repr()
    lines = extra.split("\n") if extra else []
    children = [f"({name}): {_addindent(repr(m), 2)}" for name, m in self._modules.items()]
    if not children and len(lines) <= 1: return f"{self._get_name()}({extra})"
    return f"{self._get_name()}(\n  " + "\n  ".join(lines + children) + "\n)"

# ---------------------------------------------------------------- containers

class Sequential(Module):
  def __init__(self, *args):
    super().__init__()
    if len(args) == 1 and isinstance(args[0], OrderedDict):
      for k, m in args[0].items(): self.add_module(k, m)
    else:
      for i, m in enumerate(args): self.add_module(str(i), m)
  def forward(self, input):
    for m in self._modules.values(): input = m(input)
    return input
  def __len__(self): return len(self._modules)
  def __iter__(self): return iter(self._modules.values())
  def __getitem__(self, idx):
    if isinstance(idx, slice): return Sequential(OrderedDict(list(self._modules.items())[idx]))
    return list(self._modules.values())[idx]
  def __setitem__(self, idx, module): self._modules[list(self._modules.keys())[idx]] = module
  def __delitem__(self, idx):
    for k in (list(self._modules.keys())[idx] if isinstance(idx, slice) else [list(self._modules.keys())[idx]]): del self._modules[k]
  def append(self, module): self.add_module(str(len(self)), module); return self
  def extend(self, modules):
    for m in modules: self.append(m)
    return self
  def insert(self, index, module):
    mods = list(self._modules.values()); mods.insert(index, module)
    self._modules.clear()
    for i, m in enumerate(mods): self.add_module(str(i), m)
    return self

class ModuleList(Module):
  def __init__(self, modules=None):
    super().__init__()
    if modules is not None: self.extend(modules)
  def _reindex(self, mods):
    self._modules.clear()
    for i, m in enumerate(mods): self._modules[str(i)] = m
  def append(self, module): self._modules[str(len(self))] = module; return self
  def extend(self, modules):
    for m in modules: self.append(m)
    return self
  def insert(self, index, module):
    mods = list(self._modules.values()); mods.insert(index, module); self._reindex(mods)
  def pop(self, idx=-1):
    mods = list(self._modules.values()); m = mods.pop(idx); self._reindex(mods); return m
  def __len__(self): return len(self._modules)
  def __iter__(self): return iter(self._modules.values())
  def __getitem__(self, idx):
    if isinstance(idx, slice): return ModuleList(list(self._modules.values())[idx])
    return list(self._modules.values())[idx]
  def __setitem__(self, idx, module): self._modules[str(range(len(self))[idx])] = module
  def __delitem__(self, idx):
    mods = list(self._modules.values()); del mods[idx]; self._reindex(mods)
  def __iadd__(self, modules): return self.extend(modules)
  def __add__(self, other): return ModuleList(list(self) + list(other))
  def forward(self, *a, **k): raise NotImplementedError("ModuleList is not callable; iterate over it")
  def __repr__(self):
    reprs = [repr(m) for m in self]
    if not reprs: return self._get_name() + "()"
    lines, start = [], 0
    for i in range(1, len(reprs) + 1):
      if i == len(reprs) or reprs[i] != reprs[start]:
        n = i - start
        lines.append((f"({start}): " if n == 1 else f"({start}-{i - 1}): {n} x ") + _addindent(reprs[start], 2))
        start = i
    return self._get_name() + "(\n  " + "\n  ".join(lines) + "\n)"

class ModuleDict(Module):
  def __init__(self, modules=None):
    super().__init__()
    if modules is not None: self.update(modules)
  def __getitem__(self, key): return self._modules[key]
  def __setitem__(self, key, module): self.add_module(key, module)
  def __delitem__(self, key): del self._modules[key]
  def __len__(self): return len(self._modules)
  def __iter__(self): return iter(self._modules)
  def __contains__(self, key): return key in self._modules
  def keys(self): return self._modules.keys()
  def items(self): return self._modules.items()
  def values(self): return self._modules.values()
  def pop(self, key): return self._modules.pop(key)
  def clear(self): self._modules.clear()
  def update(self, modules):
    for k, m in (modules.items() if hasattr(modules, "items") else modules): self[k] = m

class ParameterList(Module):
  def __init__(self, values=None):
    super().__init__()
    for v in (values or []): self.append(v)
  def append(self, value):
    self._parameters[str(len(self._parameters))] = value if isinstance(value, Parameter) else Parameter(value)
    return self
  def extend(self, values):
    for v in values: self.append(v)
    return self
  def __len__(self): return len(self._parameters)
  def __iter__(self): return iter(self._parameters.values())
  def __getitem__(self, idx): return list(self._parameters.values())[idx]
  def __setitem__(self, idx, value): self._parameters[str(range(len(self))[idx])] = value if isinstance(value, Parameter) else Parameter(value)
  def extra_repr(self):
    return "\n".join(f"    ({k}): Parameter containing: [{p.dtype} of size {'x'.join(str(d) for d in p.shape)}]" for k, p in self._parameters.items())
  def __repr__(self): return f"{self._get_name()}(\n{self.extra_repr()}\n)" if len(self) else f"{self._get_name()}()"

class ParameterDict(Module):
  def __init__(self, parameters=None):
    super().__init__()
    for k, v in (parameters or {}).items(): self[k] = v
  def __getitem__(self, key): return self._parameters[key]
  def __setitem__(self, key, value): self._parameters[key] = value if isinstance(value, Parameter) else Parameter(value)
  def __len__(self): return len(self._parameters)
  def __iter__(self): return iter(self._parameters)
  def __contains__(self, key): return key in self._parameters
  def keys(self): return self._parameters.keys()
  def items(self): return self._parameters.items()
  def values(self): return self._parameters.values()

# ---------------------------------------------------------------- simple layers

class Identity(Module):
  def __init__(self, *a, **k): super().__init__()
  def forward(self, input): return input

class Linear(Module):
  def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
    super().__init__()
    self.in_features, self.out_features = in_features, out_features
    self.weight = Parameter(empty(out_features, in_features))
    self.bias = Parameter(empty(out_features)) if bias else None
    self.reset_parameters()
  def reset_parameters(self):
    init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
      bound = 1 / math.sqrt(self.in_features) if self.in_features > 0 else 0
      init.uniform_(self.bias, -bound, bound)
  def forward(self, input): return F.linear(input, self.weight, self.bias)
  def extra_repr(self): return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

class Bilinear(Module):
  def __init__(self, in1_features, in2_features, out_features, bias=True, device=None, dtype=None):
    super().__init__()
    self.in1_features, self.in2_features, self.out_features = in1_features, in2_features, out_features
    bound = 1 / math.sqrt(in1_features)
    self.weight = Parameter(empty(out_features, in1_features, in2_features).uniform_(-bound, bound))
    self.bias = Parameter(empty(out_features).uniform_(-bound, bound)) if bias else None
  def forward(self, input1, input2): return F.bilinear(input1, input2, self.weight, self.bias)

class Flatten(Module):
  def __init__(self, start_dim=1, end_dim=-1):
    super().__init__(); self.start_dim, self.end_dim = start_dim, end_dim
  def forward(self, input): return input.flatten(self.start_dim, self.end_dim)
  def extra_repr(self): return f"start_dim={self.start_dim}, end_dim={self.end_dim}"
class Unflatten(Module):
  def __init__(self, dim, unflattened_size):
    super().__init__(); self.dim, self.unflattened_size = dim, tuple(unflattened_size)
  def forward(self, input): return input.unflatten(self.dim, self.unflattened_size)
  def extra_repr(self): return f"dim={self.dim}, unflattened_size={self.unflattened_size}"

class Embedding(Module):
  def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False,
               _weight=None, _freeze=False, device=None, dtype=None):
    super().__init__()
    self.num_embeddings, self.embedding_dim, self.padding_idx = num_embeddings, embedding_dim, padding_idx
    if _weight is None:
      self.weight = Parameter(empty(num_embeddings, embedding_dim), requires_grad=not _freeze)
      init.normal_(self.weight)
      if padding_idx is not None:
        with no_grad(): self.weight[padding_idx] = 0.0
    else: self.weight = Parameter(_weight, requires_grad=not _freeze)
  def forward(self, input): return F.embedding(input, self.weight, self.padding_idx)
  @classmethod
  def from_pretrained(cls, embeddings, freeze=True, padding_idx=None, **k):
    return cls(embeddings.shape[0], embeddings.shape[1], padding_idx, _weight=embeddings, _freeze=freeze)
  def extra_repr(self): return f"{self.num_embeddings}, {self.embedding_dim}" + (f", padding_idx={self.padding_idx}" if self.padding_idx is not None else "")

class _Dropout(Module):
  _fn = staticmethod(F.dropout)
  def __init__(self, p=0.5, inplace=False):
    super().__init__()
    if p < 0 or p > 1: raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
    self.p, self.inplace = p, inplace
  def forward(self, input): return self._fn(input, self.p, self.training, self.inplace)
  def extra_repr(self): return f"p={self.p}, inplace={self.inplace}"
class Dropout(_Dropout): pass
class Dropout1d(_Dropout): _fn = staticmethod(F.dropout1d)
class Dropout2d(_Dropout): _fn = staticmethod(F.dropout2d)
class Dropout3d(_Dropout): _fn = staticmethod(F.dropout3d)
class AlphaDropout(_Dropout): pass

# ---------------------------------------------------------------- convolution and pooling

def _ntuple(x, n): return tuple(x) if isinstance(x, (tuple, list)) else (x,) * n

class _ConvNd(Module):
  _nd, _transposed = 2, False
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros",
               device=None, dtype=None, output_padding=0):
    super().__init__()
    if padding_mode != "zeros": raise NotImplementedError(f"padding_mode={padding_mode!r} is not supported by the tinygrad.js torch shim")
    nd = self._nd
    self.in_channels, self.out_channels, self.groups, self.padding_mode = in_channels, out_channels, groups, padding_mode
    self.kernel_size, self.stride, self.dilation = _ntuple(kernel_size, nd), _ntuple(stride, nd), _ntuple(dilation, nd)
    self.padding = padding if isinstance(padding, str) else _ntuple(padding, nd)
    self.output_padding = _ntuple(output_padding, nd)
    shape = (in_channels, out_channels // groups) if self._transposed else (out_channels, in_channels // groups)
    self.weight = Parameter(empty(*shape, *self.kernel_size))
    self.bias = Parameter(empty(out_channels)) if bias else None
    self.reset_parameters()
  def reset_parameters(self):
    init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
      fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
      bound = 1 / math.sqrt(fan_in) if fan_in else 0
      init.uniform_(self.bias, -bound, bound)
  def forward(self, input): return F._conv(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self._nd)
  def extra_repr(self):
    s = f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}"
    if self.padding != (0,) * self._nd: s += f", padding={self.padding}"
    if self.dilation != (1,) * self._nd: s += f", dilation={self.dilation}"
    if self.groups != 1: s += f", groups={self.groups}"
    if self.bias is None: s += ", bias=False"
    return s
class Conv1d(_ConvNd): _nd = 1
class Conv2d(_ConvNd): _nd = 2
class Conv3d(_ConvNd): _nd = 3
class _ConvTransposeNd(_ConvNd):
  _transposed = True
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1,
               padding_mode="zeros", device=None, dtype=None):
    super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, output_padding=output_padding)
  def forward(self, input):
    return F._conv_transpose(input, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation, self._nd)
class ConvTranspose1d(_ConvTransposeNd): _nd = 1
class ConvTranspose2d(_ConvTransposeNd): _nd = 2

class _Pool(Module):
  _fn, _nd = None, 2
  def __init__(self, kernel_size, stride=None, padding=0, **kw):
    super().__init__()
    self.kernel_size, self.stride, self.padding, self.kw = kernel_size, stride if stride is not None else kernel_size, padding, kw
    for k, v in kw.items(): setattr(self, k, v)
  def forward(self, input): return type(self)._fn(input, self.kernel_size, self.stride, self.padding, **self.kw)
  def extra_repr(self): return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"
class _MaxPool(_Pool):
  def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
    super().__init__(kernel_size, stride, padding, dilation=dilation, ceil_mode=ceil_mode, return_indices=return_indices)
  def extra_repr(self): return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}, ceil_mode={self.ceil_mode}"
class MaxPool1d(_MaxPool): _fn = staticmethod(F.max_pool1d)
class MaxPool2d(_MaxPool): _fn = staticmethod(F.max_pool2d)
class MaxPool3d(_MaxPool): _fn = staticmethod(F.max_pool3d)
class AvgPool1d(_Pool): _fn = staticmethod(F.avg_pool1d)
class AvgPool2d(_Pool): _fn = staticmethod(F.avg_pool2d)
class AvgPool3d(_Pool): _fn = staticmethod(F.avg_pool3d)
class _Adaptive(Module):
  _fn = None
  def __init__(self, output_size, return_indices=False):
    super().__init__(); self.output_size = output_size
  def forward(self, input): return type(self)._fn(input, self.output_size)
  def extra_repr(self): return f"output_size={self.output_size}"
class AdaptiveAvgPool1d(_Adaptive): _fn = staticmethod(F.adaptive_avg_pool1d)
class AdaptiveAvgPool2d(_Adaptive): _fn = staticmethod(F.adaptive_avg_pool2d)
class AdaptiveMaxPool1d(_Adaptive): _fn = staticmethod(F.adaptive_max_pool1d)
class AdaptiveMaxPool2d(_Adaptive): _fn = staticmethod(F.adaptive_max_pool2d)

class Upsample(Module):
  def __init__(self, size=None, scale_factor=None, mode="nearest", align_corners=None):
    super().__init__(); self.size, self.scale_factor, self.mode, self.align_corners = size, scale_factor, mode, align_corners
  def forward(self, input): return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)
class PixelShuffle(Module):
  def __init__(self, upscale_factor):
    super().__init__(); self.upscale_factor = upscale_factor
  def forward(self, input): return F.pixel_shuffle(input, self.upscale_factor)
class ZeroPad2d(Module):
  def __init__(self, padding):
    super().__init__(); self.padding = _ntuple(padding, 4)
  def forward(self, input): return F.pad(input, self.padding)
class ConstantPad1d(Module):
  _n = 2
  def __init__(self, padding, value):
    super().__init__(); self.padding, self.value = _ntuple(padding, self._n), value
  def forward(self, input): return F.pad(input, self.padding, "constant", self.value)
class ConstantPad2d(ConstantPad1d): _n = 4
class ReflectionPad2d(Module):
  def __init__(self, padding):
    super().__init__(); self.padding = _ntuple(padding, 4)
  def forward(self, input): return F.pad(input, self.padding, "reflect")
class ReplicationPad2d(ReflectionPad2d):
  def forward(self, input): return F.pad(input, self.padding, "replicate")

# ---------------------------------------------------------------- normalization

class _BatchNorm(Module):
  def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
    super().__init__()
    self.num_features, self.eps, self.momentum, self.affine, self.track_running_stats = num_features, eps, momentum, affine, track_running_stats
    self.weight = Parameter(ones(num_features)) if affine else None
    self.bias = Parameter(zeros(num_features)) if affine else None
    if track_running_stats:
      self.register_buffer("running_mean", zeros(num_features))
      self.register_buffer("running_var", ones(num_features))
      self.register_buffer("num_batches_tracked", tensor(0, dtype=_tensor.int64))
    else:
      self.register_buffer("running_mean", None); self.register_buffer("running_var", None); self.register_buffer("num_batches_tracked", None)
  def reset_running_stats(self):
    if self.track_running_stats:
      with no_grad(): self.running_mean.zero_(); self.running_var.fill_(1.0); self.num_batches_tracked.zero_()
  def reset_parameters(self):
    self.reset_running_stats()
    if self.affine: init.ones_(self.weight); init.zeros_(self.bias)
  def forward(self, input):
    training = self.training or self.running_mean is None
    momentum = self.momentum
    if self.training and self.track_running_stats:
      with no_grad(): self.num_batches_tracked._inplace(self.num_batches_tracked + 1)
      if momentum is None: momentum = 1.0 / float(self.num_batches_tracked.item())   # cumulative moving average
    return F.batch_norm(input, self.running_mean if not self.training or self.track_running_stats else None,
                        self.running_var if not self.training or self.track_running_stats else None,
                        self.weight, self.bias, training, 0.0 if momentum is None else momentum, self.eps)
  def extra_repr(self): return f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine}, track_running_stats={self.track_running_stats}"
class BatchNorm1d(_BatchNorm): pass
class BatchNorm2d(_BatchNorm): pass
class BatchNorm3d(_BatchNorm): pass

class LayerNorm(Module):
  def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True, bias=True, device=None, dtype=None):
    super().__init__()
    self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    self.eps, self.elementwise_affine = eps, elementwise_affine
    self.weight = Parameter(ones(*self.normalized_shape)) if elementwise_affine else None
    self.bias = Parameter(zeros(*self.normalized_shape)) if elementwise_affine and bias else None
  def forward(self, input): return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
  def extra_repr(self): return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"
class GroupNorm(Module):
  def __init__(self, num_groups, num_channels, eps=1e-05, affine=True, device=None, dtype=None):
    super().__init__()
    self.num_groups, self.num_channels, self.eps, self.affine = num_groups, num_channels, eps, affine
    self.weight = Parameter(ones(num_channels)) if affine else None
    self.bias = Parameter(zeros(num_channels)) if affine else None
  def forward(self, input): return F.group_norm(input, self.num_groups, self.weight, self.bias, self.eps)
  def extra_repr(self): return f"{self.num_groups}, {self.num_channels}, eps={self.eps}, affine={self.affine}"
class _InstanceNorm(Module):
  def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False, device=None, dtype=None):
    super().__init__()
    self.num_features, self.eps, self.affine = num_features, eps, affine
    self.weight = Parameter(ones(num_features)) if affine else None
    self.bias = Parameter(zeros(num_features)) if affine else None
  def forward(self, input): return F.instance_norm(input, weight=self.weight, bias=self.bias, eps=self.eps)
class InstanceNorm1d(_InstanceNorm): pass
class InstanceNorm2d(_InstanceNorm): pass
class RMSNorm(Module):
  def __init__(self, normalized_shape, eps=None, elementwise_affine=True, device=None, dtype=None):
    super().__init__()
    self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    self.eps = eps
    self.weight = Parameter(ones(*self.normalized_shape)) if elementwise_affine else None
  def forward(self, input):
    dims = tuple(range(-len(self.normalized_shape), 0))
    out = input * ((input ** 2).mean(dims, keepdim=True) + (self.eps if self.eps is not None else 1.1920929e-07)).rsqrt()
    return out if self.weight is None else out * self.weight

# ---------------------------------------------------------------- activations

def _activation(name, fn, params=(), inplace=False):
  """Build an activation module class: ``params`` are (name, default) pairs of constructor arguments passed on to ``fn``."""
  def __init__(self, *args, **kwargs):
    Module.__init__(self)
    spec = list(params) + ([("inplace", False)] if inplace else [])
    for (k, default), v in itertools.zip_longest(spec, args[:len(spec)], fillvalue=None):
      setattr(self, k, kwargs.get(k, default if v is None else v))
  def forward(self, input): return fn(input, **{k: getattr(self, k) for k, _ in list(params) + ([("inplace", False)] if inplace else [])})
  def extra_repr(self): return ", ".join(f"{k}={getattr(self, k)!r}" for k, _ in params) + (", inplace=True" if inplace and self.inplace else "")
  return type(name, (Module,), {"__init__": __init__, "forward": forward, "extra_repr": extra_repr})

ReLU = _activation("ReLU", F.relu, inplace=True)
ReLU6 = _activation("ReLU6", F.relu6, inplace=True)
LeakyReLU = _activation("LeakyReLU", F.leaky_relu, [("negative_slope", 0.01)], inplace=True)
ELU = _activation("ELU", F.elu, [("alpha", 1.0)], inplace=True)
SELU = _activation("SELU", F.selu, inplace=True)
CELU = _activation("CELU", F.celu, [("alpha", 1.0)], inplace=True)
GELU = _activation("GELU", F.gelu, [("approximate", "none")])
SiLU = _activation("SiLU", F.silu, inplace=True)
Mish = _activation("Mish", F.mish, inplace=True)
Sigmoid = _activation("Sigmoid", F.sigmoid)
Tanh = _activation("Tanh", F.tanh)
Hardtanh = _activation("Hardtanh", F.hardtanh, [("min_val", -1.0), ("max_val", 1.0)], inplace=True)
Hardsigmoid = _activation("Hardsigmoid", F.hardsigmoid, inplace=True)
Hardswish = _activation("Hardswish", F.hardswish, inplace=True)
Softplus = _activation("Softplus", F.softplus, [("beta", 1.0), ("threshold", 20.0)])
Softsign = _activation("Softsign", F.softsign)
LogSigmoid = _activation("LogSigmoid", F.logsigmoid)
Threshold = _activation("Threshold", F.threshold, [("threshold", None), ("value", None)], inplace=True)
Softmax = _activation("Softmax", F.softmax, [("dim", None)])
LogSoftmax = _activation("LogSoftmax", F.log_softmax, [("dim", None)])
Softmin = _activation("Softmin", F.softmin, [("dim", None)])
class Softmax2d(Module):
  def forward(self, input): return input.softmax(-3)
class PReLU(Module):
  def __init__(self, num_parameters=1, init=0.25, device=None, dtype=None):
    super().__init__(); self.num_parameters = num_parameters
    self.weight = Parameter(empty(num_parameters).fill_(init))
  def forward(self, input): return F.prelu(input, self.weight)

# ---------------------------------------------------------------- losses

class _Loss(Module):
  def __init__(self, size_average=None, reduce=None, reduction="mean"):
    super().__init__(); self.reduction = reduction
def _loss(name, fn, params=()):
  def __init__(self, *args, **kwargs):
    Module.__init__(self)
    self.reduction = kwargs.pop("reduction", "mean")
    for k in ("size_average", "reduce"): kwargs.pop(k, None)
    for (k, default), v in itertools.zip_longest(params, args[:len(params)], fillvalue=None): setattr(self, k, kwargs.get(k, default if v is None else v))
    for k, _ in params:
      if isinstance(getattr(self, k), Tensor): self.register_buffer(k, self.__dict__.pop(k))
  def forward(self, *inputs): return fn(*inputs, reduction=self.reduction, **{k: getattr(self, k) for k, _ in params})
  return type(name, (_Loss,), {"__init__": __init__, "forward": forward})

MSELoss = _loss("MSELoss", F.mse_loss)
L1Loss = _loss("L1Loss", F.l1_loss)
SmoothL1Loss = _loss("SmoothL1Loss", F.smooth_l1_loss, [("beta", 1.0)])
HuberLoss = _loss("HuberLoss", F.huber_loss, [("delta", 1.0)])
BCELoss = _loss("BCELoss", F.binary_cross_entropy, [("weight", None)])
BCEWithLogitsLoss = _loss("BCEWithLogitsLoss", F.binary_cross_entropy_with_logits, [("weight", None), ("pos_weight", None)])
NLLLoss = _loss("NLLLoss", F.nll_loss, [("weight", None), ("ignore_index", -100)])
KLDivLoss = _loss("KLDivLoss", F.kl_div, [("log_target", False)])
PoissonNLLLoss = _loss("PoissonNLLLoss", F.poisson_nll_loss, [("log_input", True), ("full", False), ("eps", 1e-08)])
MarginRankingLoss = _loss("MarginRankingLoss", F.margin_ranking_loss, [("margin", 0.0)])
HingeEmbeddingLoss = _loss("HingeEmbeddingLoss", F.hinge_embedding_loss, [("margin", 1.0)])
TripletMarginLoss = _loss("TripletMarginLoss", F.triplet_margin_loss, [("margin", 1.0), ("p", 2.0), ("eps", 1e-06), ("swap", False)])
CosineEmbeddingLoss = _loss("CosineEmbeddingLoss", F.cosine_embedding_loss, [("margin", 0.0)])
SoftMarginLoss = _loss("SoftMarginLoss", F.soft_margin_loss)
GaussianNLLLoss = _loss("GaussianNLLLoss", F.gaussian_nll_loss, [("full", False), ("eps", 1e-06)])
class CrossEntropyLoss(_Loss):
  def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction="mean", label_smoothing=0.0):
    super().__init__(reduction=reduction)
    self.register_buffer("weight", weight)
    self.ignore_index, self.label_smoothing = ignore_index, label_smoothing
  def forward(self, input, target):
    return F.cross_entropy(input, target, self.weight, ignore_index=self.ignore_index, reduction=self.reduction, label_smoothing=self.label_smoothing)
class CosineSimilarity(Module):
  def __init__(self, dim=1, eps=1e-08):
    super().__init__(); self.dim, self.eps = dim, eps
  def forward(self, x1, x2): return F.cosine_similarity(x1, x2, self.dim, self.eps)
class PairwiseDistance(Module):
  def __init__(self, p=2.0, eps=1e-06, keepdim=False):
    super().__init__(); self.p, self.eps, self.keepdim = p, eps, keepdim
  def forward(self, x1, x2): return F.pairwise_distance(x1, x2, self.p, self.eps, self.keepdim)

# ---------------------------------------------------------------- recurrent layers

class _RNNCellBase(Module):
  _gates = 1
  def __init__(self, input_size, hidden_size, bias=True, device=None, dtype=None, **kw):
    super().__init__()
    self.input_size, self.hidden_size, self.bias = input_size, hidden_size, bias
    g, k = self._gates * hidden_size, 1 / math.sqrt(hidden_size)
    self.weight_ih = Parameter(empty(g, input_size).uniform_(-k, k))
    self.weight_hh = Parameter(empty(g, hidden_size).uniform_(-k, k))
    self.bias_ih = Parameter(empty(g).uniform_(-k, k)) if bias else None
    self.bias_hh = Parameter(empty(g).uniform_(-k, k)) if bias else None
  def _lin(self, x, h): return F.linear(x, self.weight_ih, self.bias_ih), F.linear(h, self.weight_hh, self.bias_hh)
  def extra_repr(self): return f"{self.input_size}, {self.hidden_size}"
class RNNCell(_RNNCellBase):
  def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh", device=None, dtype=None):
    super().__init__(input_size, hidden_size, bias); self.nonlinearity = nonlinearity
  def forward(self, input, hx=None):
    hx = zeros(*input.shape[:-1], self.hidden_size) if hx is None else hx
    a, b = self._lin(input, hx)
    return (a + b).tanh() if self.nonlinearity == "tanh" else (a + b).relu()
class LSTMCell(_RNNCellBase):
  _gates = 4
  def forward(self, input, hx=None):
    h, c = (zeros(*input.shape[:-1], self.hidden_size),) * 2 if hx is None else hx
    a, b = self._lin(input, h)
    i, f, g, o = (a + b).chunk(4, -1)
    c = f.sigmoid() * c + i.sigmoid() * g.tanh()
    return o.sigmoid() * c.tanh(), c
class GRUCell(_RNNCellBase):
  _gates = 3
  def forward(self, input, hx=None):
    hx = zeros(*input.shape[:-1], self.hidden_size) if hx is None else hx
    a, b = self._lin(input, hx)
    ir, iz, in_ = a.chunk(3, -1)
    hr, hz, hn = b.chunk(3, -1)
    r, z = (ir + hr).sigmoid(), (iz + hz).sigmoid()
    n = (in_ + r * hn).tanh()
    return (1 - z) * n + z * hx

class _RNNBase(Module):
  _gates, _mode = 1, "RNN"
  def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False,
               nonlinearity="tanh", proj_size=0, device=None, dtype=None):
    super().__init__()
    if proj_size: raise NotImplementedError("proj_size is not supported by the tinygrad.js torch shim")
    self.input_size, self.hidden_size, self.num_layers, self.bias = input_size, hidden_size, num_layers, bias
    self.batch_first, self.dropout, self.bidirectional, self.nonlinearity = batch_first, dropout, bidirectional, nonlinearity
    g, k = self._gates * hidden_size, 1 / math.sqrt(hidden_size)
    for layer in range(num_layers):
      for d in range(2 if bidirectional else 1):
        sfx = f"_l{layer}" + ("_reverse" if d else "")
        in_size = input_size if layer == 0 else hidden_size * (2 if bidirectional else 1)
        setattr(self, "weight_ih" + sfx, Parameter(empty(g, in_size).uniform_(-k, k)))
        setattr(self, "weight_hh" + sfx, Parameter(empty(g, hidden_size).uniform_(-k, k)))
        if bias:
          setattr(self, "bias_ih" + sfx, Parameter(empty(g).uniform_(-k, k)))
          setattr(self, "bias_hh" + sfx, Parameter(empty(g).uniform_(-k, k)))
  def flatten_parameters(self): pass
  def _step(self, x, state, sfx):
    a = F.linear(x, getattr(self, "weight_ih" + sfx), getattr(self, "bias_ih" + sfx) if self.bias else None)
    b = F.linear(state[0], getattr(self, "weight_hh" + sfx), getattr(self, "bias_hh" + sfx) if self.bias else None)
    if self._mode == "LSTM":
      i, f, g, o = (a + b).chunk(4, -1)
      c = f.sigmoid() * state[1] + i.sigmoid() * g.tanh()
      return (o.sigmoid() * c.tanh(), c)
    if self._mode == "GRU":
      ir, iz, in_ = a.chunk(3, -1)
      hr, hz, hn = b.chunk(3, -1)
      r, z = (ir + hr).sigmoid(), (iz + hz).sigmoid()
      return ((1 - z) * (in_ + r * hn).tanh() + z * state[0],)
    return ((a + b).tanh() if self.nonlinearity == "tanh" else (a + b).relu(),)
  def forward(self, input, hx=None):
    from .. import stack, cat
    unbatched = input.ndim == 2
    x = input.unsqueeze(1) if unbatched else (input.transpose(0, 1) if self.batch_first else input)   # (T, N, C)
    t, n = x.shape[0], x.shape[1]
    dirs = 2 if self.bidirectional else 1
    lstm = self._mode == "LSTM"
    if hx is None: h0 = zeros(self.num_layers * dirs, n, self.hidden_size); c0 = zeros(self.num_layers * dirs, n, self.hidden_size) if lstm else None
    else:
      h0, c0 = hx if lstm else (hx, None)
      if unbatched: h0, c0 = h0.unsqueeze(1), (c0.unsqueeze(1) if lstm else None)
    hn, cn = [], []
    for layer in range(self.num_layers):
      outs = []
      for d in range(dirs):
        idx, sfx = layer * dirs + d, f"_l{layer}" + ("_reverse" if d else "")
        state = (h0[idx], c0[idx]) if lstm else (h0[idx],)
        seq = []
        for step in (range(t - 1, -1, -1) if d else range(t)):
          state = self._step(x[step], state, sfx)
          seq.append(state[0])
        if d: seq.reverse()
        outs.append(stack(seq))
        hn.append(state[0])
        if lstm: cn.append(state[1])
      x = outs[0] if dirs == 1 else cat(outs, -1)
      if self.dropout and self.training and layer < self.num_layers - 1: x = F.dropout(x, self.dropout, True)
    out = x.squeeze(1) if unbatched else (x.transpose(0, 1) if self.batch_first else x)
    h, c = stack(hn), (stack(cn) if lstm else None)
    if unbatched: h, c = h.squeeze(1), (c.squeeze(1) if lstm else None)
    return out, ((h, c) if lstm else h)
  def extra_repr(self): return f"{self.input_size}, {self.hidden_size}" + (f", num_layers={self.num_layers}" if self.num_layers != 1 else "") + \
    (", batch_first=True" if self.batch_first else "") + (", bidirectional=True" if self.bidirectional else "")
class RNN(_RNNBase): pass
class LSTM(_RNNBase): _gates, _mode = 4, "LSTM"
class GRU(_RNNBase): _gates, _mode = 3, "GRU"

# ---------------------------------------------------------------- attention and transformers

class MultiheadAttention(Module):
  def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None,
               batch_first=False, device=None, dtype=None):
    super().__init__()
    if add_bias_kv or add_zero_attn or (kdim not in (None, embed_dim)) or (vdim not in (None, embed_dim)):
      raise NotImplementedError("add_bias_kv, add_zero_attn, kdim and vdim are not supported by the tinygrad.js torch shim")
    self.embed_dim, self.num_heads, self.dropout, self.batch_first, self.head_dim = embed_dim, num_heads, dropout, batch_first, embed_dim // num_heads
    self.in_proj_weight = Parameter(empty(3 * embed_dim, embed_dim))
    init.xavier_uniform_(self.in_proj_weight)
    self.in_proj_bias = Parameter(zeros(3 * embed_dim)) if bias else None
    self.out_proj = Linear(embed_dim, embed_dim, bias=bias)
    if bias: init.zeros_(self.out_proj.bias)
  def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False):
    unbatched = query.ndim == 2
    if unbatched: query, key, value = query.unsqueeze(1), key.unsqueeze(1), value.unsqueeze(1)
    elif self.batch_first: query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
    l, n, e = query.shape
    s, h, d = key.shape[0], self.num_heads, self.head_dim
    wq, wk, wv = self.in_proj_weight.chunk(3, 0)
    bq, bk, bv = self.in_proj_bias.chunk(3, 0) if self.in_proj_bias is not None else (None, None, None)
    q = F.linear(query, wq, bq).reshape(l, n, h, d).permute(1, 2, 0, 3)     # (N, H, L, D)
    k = F.linear(key, wk, bk).reshape(s, n, h, d).permute(1, 2, 0, 3)
    v = F.linear(value, wv, bv).reshape(s, n, h, d).permute(1, 2, 0, 3)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
    def apply_mask(scores, mask):   # bool: True = not allowed; float: added
      return scores.masked_fill(mask, -math.inf) if mask.dtype is _tensor.bool else scores + mask
    if attn_mask is not None: apply_to = attn_mask if attn_mask.ndim == 2 else attn_mask.reshape(n, h, l, s); scores = apply_mask(scores, apply_to)
    if key_padding_mask is not None: scores = apply_mask(scores, key_padding_mask.reshape(n, 1, 1, s))
    attn = scores.softmax(-1)
    if self.dropout and self.training: attn = F.dropout(attn, self.dropout, True)
    out = self.out_proj((attn @ v).permute(2, 0, 1, 3).reshape(l, n, e))
    weights = (attn.mean(1) if average_attn_weights else attn) if need_weights else None
    if unbatched: out, weights = out.squeeze(1), (weights.squeeze(0) if weights is not None else None)
    elif self.batch_first: out = out.transpose(0, 1)
    return out, weights

def _act(a): return {"relu": F.relu, "gelu": F.gelu}[a] if isinstance(a, str) else a

class TransformerEncoderLayer(Module):
  def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", layer_norm_eps=1e-05, batch_first=False,
               norm_first=False, bias=True, device=None, dtype=None):
    super().__init__()
    self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias, batch_first=batch_first)
    self.linear1, self.dropout, self.linear2 = Linear(d_model, dim_feedforward, bias=bias), Dropout(dropout), Linear(dim_feedforward, d_model, bias=bias)
    self.norm_first = norm_first
    self.norm1, self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias), LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
    self.dropout1, self.dropout2 = Dropout(dropout), Dropout(dropout)
    self.activation = _act(activation)
  def _sa(self, x, mask, kpm): return self.dropout1(self.self_attn(x, x, x, attn_mask=mask, key_padding_mask=kpm, need_weights=False)[0])
  def _ff(self, x): return self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))
  def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
    x = src
    if self.norm_first:
      x = x + self._sa(self.norm1(x), src_mask, src_key_padding_mask)
      return x + self._ff(self.norm2(x))
    x = self.norm1(x + self._sa(x, src_mask, src_key_padding_mask))
    return self.norm2(x + self._ff(x))

class TransformerDecoderLayer(Module):
  def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", layer_norm_eps=1e-05, batch_first=False,
               norm_first=False, bias=True, device=None, dtype=None):
    super().__init__()
    self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias, batch_first=batch_first)
    self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias, batch_first=batch_first)
    self.linear1, self.dropout, self.linear2 = Linear(d_model, dim_feedforward, bias=bias), Dropout(dropout), Linear(dim_feedforward, d_model, bias=bias)
    self.norm_first = norm_first
    self.norm1, self.norm2, self.norm3 = (LayerNorm(d_model, eps=layer_norm_eps, bias=bias) for _ in range(3))
    self.dropout1, self.dropout2, self.dropout3 = Dropout(dropout), Dropout(dropout), Dropout(dropout)
    self.activation = _act(activation)
  def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, **k):
    sa = lambda x: self.dropout1(self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, need_weights=False)[0])
    ca = lambda x: self.dropout2(self.multihead_attn(x, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask, need_weights=False)[0])
    ff = lambda x: self.dropout3(self.linear2(self.dropout(self.activation(self.linear1(x)))))
    x = tgt
    if self.norm_first:
      x = x + sa(self.norm1(x)); x = x + ca(self.norm2(x))
      return x + ff(self.norm3(x))
    x = self.norm1(x + sa(x)); x = self.norm2(x + ca(x))
    return self.norm3(x + ff(x))

def _clones(layer, n):
  import copy
  return ModuleList([copy.deepcopy(layer) for _ in range(n)])
class TransformerEncoder(Module):
  def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True):
    super().__init__()
    self.layers, self.num_layers, self.norm = _clones(encoder_layer, num_layers), num_layers, norm
  def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=None):
    for layer in self.layers: src = layer(src, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
    return src if self.norm is None else self.norm(src)
class TransformerDecoder(Module):
  def __init__(self, decoder_layer, num_layers, norm=None):
    super().__init__()
    self.layers, self.num_layers, self.norm = _clones(decoder_layer, num_layers), num_layers, norm
  def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, **k):
    for layer in self.layers: tgt = layer(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
    return tgt if self.norm is None else self.norm(tgt)
class Transformer(Module):
  def __init__(self, *a, **k): raise NotImplementedError("nn.Transformer is not provided by the tinygrad.js torch shim; compose TransformerEncoder / TransformerDecoder")
  @staticmethod
  def generate_square_subsequent_mask(sz, device=None, dtype=None):
    from .. import full
    return full((sz, sz), -math.inf).triu(1)
