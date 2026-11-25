"""tinygrad.cuda: run CUDA C/C++ kernels on tinygrad tensors, in the browser.

    from tinygrad import Tensor
    from tinygrad.cuda import CUDAKernel
    kernel = CUDAKernel('''
    __global__ void vector_add(float* a, float* b, float* c, int n) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n) c[i] = a[i] + b[i];
    }''')
    a, b, c = Tensor.rand(1024), Tensor.rand(1024), Tensor.zeros(1024)
    kernel.launch(grid=(4, 1, 1), block=(256, 1, 1), args=[a, b, c, 1024])   # `await kernel.launch(...)` also works
    print(c.numpy())

The kernel is compiled by tinygrad.js's CUDA compiler (WGSL for WEBGPU tensors, WebAssembly for CPU tensors) and inserted
into tinygrad's graph as a precompiled program (``Tensor.custom_kernel``): it is scheduled lazily with tinygrad's own
kernels, respects their data dependencies, is captured by TinyJit, and on WEBGPU it reads and writes the tensors' GPU
buffers directly (no copies). Pointer arguments the kernel writes are updated in place, with the same semantics as
``Tensor.assign``. CPU-device tensors are copied into the CPU backend's memory for the launch (an explicit transfer).

Custom gradients: subclass ``CUDAFunction`` (forward/backward, like ``torch.autograd.Function``).
Needs tinygrad >= 0.14 (``Tensor.custom_kernel``).
"""
from __future__ import annotations
import json, threading
from typing import Any, Sequence

__all__ = ["CUDAKernel", "CUDAFunction", "CUDAError", "device_info", "synchronize", "library", "serialize_tensors", "deserialize_tensors"]

MARKER = b"TGJSCUDA1:"
_host = None          # the JS TinygradCudaHost (src/cuda/tinygrad-host.ts)
_state = threading.local()

class CUDAError(RuntimeError):
  """A CUDA compile or launch error (the message holds the compiler diagnostics / cudaError name)."""

def set_host(host) -> None:
  global _host
  _host = host

def _js():
  if _host is None: raise CUDAError("tinygrad.cuda is not available in this runtime (no tinygrad.js CUDA host)")
  return _host

def _await(p):
  from .webgpu import _await as aw
  try: return aw(p)
  except RuntimeError as e: raise CUDAError(str(e).removeprefix("WebGPU: ")) from None

def _call(fn, *args):
  try: return fn(*args)
  except Exception as e:  # JsException
    msg = getattr(e, "message", None) or str(e)
    raise CUDAError(msg.removeprefix("Error: ")) from None

def _dim3(v, what: str) -> tuple[int, int, int]:
  if isinstance(v, int): v = (v,)
  t = tuple(int(x) for x in v) + (1, 1, 1)
  t = t[:3]
  if any(x < 1 for x in t): raise CUDAError(f"{what} dimensions must be positive, got {v}")
  return t  # type: ignore[return-value]

def _need_custom_kernel():
  from tinygrad import Tensor
  if not hasattr(Tensor, "custom_kernel"):
    import importlib.metadata
    raise CUDAError(f"tinygrad.cuda tensor integration needs tinygrad >= 0.14 (found {importlib.metadata.version('tinygrad')})")

def _device_kind(dev: str) -> str:
  d = dev.split(":")[0].upper()
  if d == "WEBGPU": return "webgpu"
  if d in ("PYTHON", "CPU"): return "cpu"
  raise CUDAError(f"tinygrad.cuda: tensors on device {dev} are not supported (WEBGPU, CPU and PYTHON are)")

class _Done:
  """Returned by launch(): `await` it (or ignore it). Holds the tensors the kernel wrote."""
  def __init__(self, outputs): self.outputs = outputs
  def __await__(self):
    return self
    yield  # pragma: no cover  (the yield makes this a generator: awaiting it completes immediately with self)
  def __repr__(self): return f"<CUDA launch: {len(self.outputs)} output tensor(s)>"

class CUDAKernel:
  """A CUDA C/C++ source with one or more __global__ kernels.

  * ``CUDAKernel(src)`` uses the only kernel; ``CUDAKernel(src, "name")`` or ``kernel["name<256>"]`` selects one
    (template kernels need their arguments).
  * ``options``: ``{"mode": "optimized" | "direct", "defines": {...}, "boundsCheck": bool, "fp64": "error" | "demote"}``.
  """
  def __init__(self, source: str, name: str | None = None, options: dict | None = None, _mid: int | None = None):
    self.source = source
    self.options = dict(options or {})
    self._mid = _mid if _mid is not None else _call(_js().compile, source, json.dumps(self.options))
    self.kernels = json.loads(_call(_js().signatures, self._mid))
    if name is None:
      if len(self.kernels) != 1:
        names = ", ".join(k["name"] for k in self.kernels) or "none"
        raise CUDAError(f"the source defines {len(self.kernels)} kernels ({names}): pass the kernel name")
      name = self.kernels[0]["name"]
    self.name = name
    base = name.split("<")[0].strip()
    sig = next((k for k in self.kernels if k["name"] == base), None)
    if sig is None: raise CUDAError(f"no kernel named {base!r}")
    if sig["template"] and "<" not in name: raise CUDAError(f"kernel {base!r} is a template: use {base}<...>")
    self.signature = sig

  def __getitem__(self, name: str) -> CUDAKernel: return CUDAKernel(self.source, name, self.options, _mid=self._mid)
  def __repr__(self): return f"CUDAKernel({self.name}({', '.join(p['type'] + ' ' + p['name'] for p in self.signature['params'])}))"

  # ---------------------------------------------------------------- launching
  def __call__(self, *args, grid, block, shared_mem: int = 0, lazy: bool = False): return self.launch(grid, block, list(args), shared_mem, lazy=lazy)

  def launch(self, grid, block, args: Sequence[Any], shared_mem: int = 0, *, lazy: bool | None = None) -> _Done:
    """Launch on tensors. Pointer parameters take Tensors, others Python numbers (or dicts / lists for struct and vector
    parameters). Tensors the kernel writes are updated in place (like ``Tensor.assign``). By default the kernel is
    realized immediately (CUDA semantics); ``lazy=True`` only records it in tinygrad's graph (it runs when its outputs are
    needed). Inside CUDAFunction.backward, launches are lazy."""
    _need_custom_kernel()
    from tinygrad import Tensor
    from tinygrad.uop.ops import UOp, Ops, KernelInfo, ProgramInfo
    grid, block = _dim3(grid, "grid"), _dim3(block, "block")
    params = self.signature["params"]
    if len(args) != len(params):
      raise CUDAError(f"{self.name} takes {len(params)} arguments ({', '.join(p['type'] + ' ' + p['name'] for p in params)}), got {len(args)}")
    tensors: list[Tensor] = []
    slot_of: list[int] = []
    scalars: list[Any] = []
    for p, a in zip(params, args):
      if p["pointer"]:
        if not isinstance(a, Tensor): raise CUDAError(f"argument {p['name']} ({p['type']}) must be a Tensor, got {type(a).__name__}")
        # the same tensor passed twice shares one buffer (the kernel sees aliased pointers)
        idx = next((i for i, t in enumerate(tensors) if t is a), None)
        if idx is None: idx = len(tensors); tensors.append(a)
        slot_of.append(idx)
      else:
        if isinstance(a, Tensor): raise CUDAError(f"argument {p['name']} ({p['type']}) is a scalar parameter; pass a Python number (use .item() on a Tensor)")
        scalars.append(_jsonable(a))
    if not tensors: raise CUDAError("tinygrad.cuda launches need at least one Tensor argument (it decides the device)")
    from tinygrad import Device
    for t in tensors:
      # device-less tensors (built from Tensor.arange / constants in tinygrad >= 0.13) live on the default device
      if t.device is None: t.uop = t.clone(Device.DEFAULT).uop
    devices = {t.device if isinstance(t.device, str) else t.device[0] for t in tensors}
    if len(devices) != 1: raise CUDAError(f"all tensor arguments must be on one device, got {sorted(devices)}")
    dev = devices.pop()
    kind = _device_kind(dev)
    desc = json.loads(_call(_js().describe, self._mid, self.name, json.dumps(block), shared_mem, kind))
    ptr_params = [p for p in desc["params"] if p["kind"] == "ptr"]
    written_slots = sorted({slot_of[i] for i, p in enumerate(ptr_params) if p["written"]})
    read_slots = sorted({slot_of[i] for i, p in enumerate(ptr_params) if p["read"]})
    lib = MARKER + json.dumps({"module": self._mid, "kernel": self.name, "grid": grid, "block": block, "smem": shared_mem, "args": scalars,
                               "slots": slot_of, "written": [p["written"] for p in ptr_params], "device": kind}, separators=(",", ":")).encode()
    name = "cuda_" + "".join(c if c.isalnum() else "_" for c in desc["kernel"])
    info = ProgramInfo(name=name, global_size=grid, local_size=block, vars=(), globals=tuple(slot_of),
                       outs=tuple(written_slots), ins=tuple(read_slots))
    def fxn(*placeholders):
      sink = UOp.sink(*placeholders, arg=KernelInfo(name=name))
      return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(placeholders)), UOp(Ops.SOURCE, arg=self.source), UOp(Ops.BINARY, arg=lib)), arg=info)
    _install_cpu_runtime(dev)
    # views of other tensors' buffers go in as contiguous copies; written ones are stored back with assign
    # kernels see flat buffers: pass 1-D sources (a reshape of a buffer is still the same buffer, no copy)
    srcs = [(t.contiguous() if _is_view(t) else t).reshape(-1) for t in tensors]
    outs = Tensor.custom_kernel(*srcs, fxn=fxn)
    written: list[Tensor] = []
    for i in written_slots:
      t, o = tensors[i], outs[i].reshape(tensors[i].shape)
      if _is_view(t): t.assign(o)
      else: t.uop = o.uop     # a whole buffer (in place), or a lazy value materialized for the kernel
      written.append(t)
    if lazy is None: lazy = getattr(_state, "lazy", False)
    if not lazy and written: Tensor.realize(*written)
    return _Done(written)

  def inspect(self, block, what: str = "code", device: str = "webgpu", shared_mem: int = 0) -> str:
    """The kernel's IR (``what="ir"``) or generated code (WGSL for device "webgpu", a WebAssembly summary for "cpu")."""
    return _await(_call(_js().inspect, self._mid, self.name, json.dumps(_dim3(block, "block")), shared_mem, device, what))

def _jsonable(a):
  if isinstance(a, bool): return a
  if isinstance(a, (int, float)): return a
  if isinstance(a, dict): return {k: _jsonable(v) for k, v in a.items()}
  if isinstance(a, (list, tuple)): return [_jsonable(v) for v in a]
  try:
    import numpy as np
    if isinstance(a, np.generic): return a.item()
  except ImportError: pass
  raise CUDAError(f"unsupported scalar argument of type {type(a).__name__}")

def _is_view(t) -> bool:
  """a slice / permute / expand ... of another tensor's buffer; a pure reshape of a whole buffer is the buffer itself"""
  u = t.uop
  base = u.base
  if base is u or base.op.name not in ("BUFFER", "AFTER", "PARAM", "MSTACK", "MSELECT"): return False
  while u.op.name == "RESHAPE": u = u.src[0]
  return u is not base

# ---------------------------------------------------------------- the device programs
def webgpu_program_call(prg, bufs, global_size, local_size, wait):
  """WebGPUProgram for a CUDA kernel: dispatch on the tensors' buffers through the shared WebGPU command stream."""
  meta = prg.cuda_meta
  handles = [int(bufs[i]) for i in range(len(bufs))]
  _await(_call(_js().launchWebGPU, meta["module"], meta["kernel"], json.dumps(meta["grid"]), json.dumps(meta["block"]), meta["smem"],
               handles, json.dumps(meta["args"])))
  if wait:
    import time
    st = time.perf_counter()
    prg.dev.synchronize()
    return time.perf_counter() - st
  return None

def parse_lib(lib: bytes) -> dict | None:
  if isinstance(lib, (bytes, bytearray)) and lib.startswith(MARKER): return json.loads(bytes(lib[len(MARKER):]).decode())
  return None

class CudaCpuProgram:
  """Program object for CUDA kernels on tinygrad's CPU / PYTHON devices (runs the WebAssembly backend)."""
  def __init__(self, dev, meta: dict): self.dev, self.meta = dev, meta
  def __call__(self, *bufs, global_size=(1, 1, 1), local_size=(1, 1, 1), vals=(), wait=False, **kw):
    import time
    m = self.meta
    st = time.perf_counter()
    outs = _await(_call(_js().launchCPU, m["module"], m["kernel"], json.dumps(m["grid"]), json.dumps(m["block"]), m["smem"],
                        list(bufs), json.dumps(m["written"]), json.dumps(m["args"])))
    for b, o in zip([b for b, w in zip(bufs, m["written"]) if w], outs):
      mv = b if isinstance(b, memoryview) else memoryview(b)
      mv.cast("B")[:] = o.to_bytes() if hasattr(o, "to_bytes") else bytes(o)
    return time.perf_counter() - st if wait else None

def _install_cpu_runtime(dev: str) -> None:
  """wrap the CPU/PYTHON device's program class so CUDA programs run on the WebAssembly backend"""
  if _device_kind(dev) != "cpu": return
  from tinygrad import Device
  d = Device[dev]
  if getattr(d, "_tgjs_cuda", False): return
  orig = d.runtime_t
  def runtime_t(device, obj):
    meta = parse_lib(obj.lib)
    if meta is not None: return CudaCpuProgram(device, meta)
    return orig(device, obj)
  d.runtime_t = runtime_t
  d._tgjs_cuda = True

# ---------------------------------------------------------------- custom autograd
class _Ctx:
  def __init__(self): self.saved_tensors: tuple = ()
  def save_for_backward(self, *ts): self.saved_tensors = ts

class CUDAFunction:
  """A differentiable operation with hand-written forward and backward passes (usually CUDA kernels):

      class Square(CUDAFunction):
        @staticmethod
        def forward(ctx, x):
          y = Tensor.empty(*x.shape)
          square_kernel.launch(grid, block, [x, y, x.numel()])
          ctx.save_for_backward(x)
          return y
        @staticmethod
        def backward(ctx, gy):
          (x,) = ctx.saved_tensors
          gx = Tensor.empty(*x.shape)
          square_bwd.launch(grid, block, [x, gy, gx, x.numel()])
          return gx
      y = Square.apply(x); y.sum().backward()

  forward receives detached inputs; backward receives the output gradient(s) and returns one gradient (or None) per
  Tensor input. With several outputs, backward is called once per output that receives a gradient, with zero gradients
  for the others (exact, because backward is linear in its output gradients). Launches inside backward are lazy.
  """
  @staticmethod
  def forward(ctx, *args): raise NotImplementedError
  @staticmethod
  def backward(ctx, *grads): raise NotImplementedError

  @classmethod
  def apply(cls, *args):
    _need_custom_kernel()
    from tinygrad import Tensor
    ctx = _Ctx()
    detached = [a.detach() if isinstance(a, Tensor) else a for a in args]
    out = cls.forward(ctx, *detached)
    single = isinstance(out, Tensor)
    outs = (out,) if single else tuple(out)
    inputs = [a for a in args if isinstance(a, Tensor)]
    input_pos = [i for i, a in enumerate(args) if isinstance(a, Tensor)]
    hooked = []
    for oi, o in enumerate(outs):
      def grad_fxn(grad, call, oi=oi, o=o):
        g = Tensor(grad)
        gs = [g if j == oi else Tensor.zeros(*x.shape, dtype=x.dtype, device=x.device) for j, x in enumerate(outs)]
        prev = getattr(_state, "lazy", False)
        _state.lazy = True
        try: res = cls.backward(ctx, *gs)
        finally: _state.lazy = prev
        res = (res,) if isinstance(res, Tensor) or res is None else tuple(res)
        if len(res) == len(args) and len(args) != len(inputs): res = tuple(res[i] for i in input_pos)
        if len(res) != len(inputs): raise CUDAError(f"{cls.__name__}.backward returned {len(res)} gradients for {len(inputs)} tensor inputs")
        grads = []
        for r, x in zip(res, inputs):
          if r is None: grads.append(None); continue
          if tuple(r.shape) != tuple(x.shape): raise CUDAError(f"{cls.__name__}.backward: gradient of shape {r.shape} for an input of shape {x.shape}")
          grads.append(r.cast(x.dtype).uop)
        return (None, *grads)
      h = Tensor.custom_kernel(o.detach().reshape(-1), *[x.reshape(-1) for x in inputs], fxn=_noop_program(o.device), grad_fxn=_flat_grads(grad_fxn, o, inputs))[0].reshape(o.shape)
      hooked.append(h)
    return hooked[0] if single else tuple(hooked)

def _flat_grads(gf, out, inputs):
  """the hook CALL works on flat tensors: reshape the incoming gradient and the returned ones"""
  def wrapped(grad, call):
    res = gf(grad.reshape(out.shape), call)
    return tuple(r if r is None else r.reshape((x.numel(),)) for r, x in zip(res, (out, *inputs)))
  return wrapped

_NOOP = "__global__ void tgjs_autograd_hook() {}"
_noop_kernel: CUDAKernel | None = None

def _noop_program(device):
  """an empty kernel that only carries the gradient function in tinygrad's graph"""
  global _noop_kernel
  from tinygrad.uop.ops import UOp, Ops, KernelInfo, ProgramInfo
  if _noop_kernel is None: _noop_kernel = CUDAKernel(_NOOP)
  dev = device if isinstance(device, str) else device[0]
  kind = _device_kind(dev)
  _install_cpu_runtime(dev)
  lib = MARKER + json.dumps({"module": _noop_kernel._mid, "kernel": "tgjs_autograd_hook", "grid": [1, 1, 1], "block": [1, 1, 1], "smem": 0,
                             "args": [], "slots": [], "written": [], "device": kind}).encode()
  info = ProgramInfo(name="cuda_autograd_hook", global_size=(1, 1, 1), local_size=(1, 1, 1), vars=(), globals=(), outs=(), ins=())
  def fxn(*placeholders):
    return UOp(Ops.PROGRAM, src=(UOp.sink(*placeholders, arg=KernelInfo(name="cuda_autograd_hook")), UOp(Ops.LINEAR, src=tuple(placeholders)),
                                 UOp(Ops.SOURCE, arg=_NOOP), UOp(Ops.BINARY, arg=lib)), arg=info)
  return fxn

# ---------------------------------------------------------------- utilities
def library(name: str) -> str:
  """CUDA source of a tinygrad.js kernel library group: "matmul", "softmax", "norms", "attention", "rope", "optimizers", ..."""
  return _call(_js().library, name)

def device_info(device: str | None = None) -> dict:
  """capabilities of the CUDA backend serving tensors on `device` (default: Device.DEFAULT)"""
  from tinygrad import Device
  return json.loads(_call(_js().info, _device_kind(device or Device.DEFAULT)))

def synchronize(device: str | None = None) -> str:
  """wait for CUDA work; returns (and prints) device printf output; raises CUDAError for device-side errors"""
  from tinygrad import Device
  out = _await(_call(_js().synchronize, _device_kind(device or Device.DEFAULT)))
  if out: print(out, end="")
  return out

_MAGIC = 0x544a4754
_NP = {"float32": "<f4", "float16": "<f2", "float64": "<f8", "int8": "i1", "uint8": "u1", "int16": "<i2", "uint16": "<u2", "int32": "<i4",
       "uint32": "<u4", "int64": "<i8", "uint64": "<u8", "bool": "?"}

def serialize_tensors(tensors: dict, meta: dict | None = None) -> bytes:
  """Same container format as the JS serializeTensors (parameters, gradients, optimizer state, ...)."""
  import numpy as np, struct
  entries, blobs, off = [], [], 0
  for name, t in tensors.items():
    a = t.numpy() if hasattr(t, "numpy") else np.asarray(t)
    dt = next((k for k, v in _NP.items() if np.dtype(v) == a.dtype.newbyteorder("<") or np.dtype(v) == a.dtype), None)
    if dt is None: raise CUDAError(f"tensor {name}: unsupported dtype {a.dtype}")
    b = np.ascontiguousarray(a).astype(_NP[dt], copy=False).tobytes()
    entries.append({"name": name, "dtype": dt, "shape": list(a.shape), "offset": off, "bytes": len(b)})
    pad = (-len(b)) % 8
    blobs.append(b + b"\0" * pad); off += len(b) + pad
  header = json.dumps({"meta": meta or {}, "tensors": entries}).encode()
  hpad = (-len(header)) % 8
  return struct.pack("<III4x", _MAGIC, 1, len(header)) + header + b"\0" * hpad + b"".join(blobs)

def deserialize_tensors(data: bytes):
  """-> (meta, {name: numpy array})"""
  import numpy as np, struct
  magic, ver, hl = struct.unpack_from("<III", data, 0)
  if magic != _MAGIC: raise CUDAError("not a tinygrad.js tensor container")
  if ver != 1: raise CUDAError(f"unsupported container version {ver}")
  header = json.loads(data[16:16 + hl])
  base = 16 + hl + ((-hl) % 8)
  out = {}
  for e in header["tensors"]:
    out[e["name"]] = np.frombuffer(data, dtype=_NP[e["dtype"]], count=int(np.prod(e["shape"])) if e["shape"] else 1, offset=base + e["offset"]).reshape(e["shape"]).copy()
  return header["meta"], out
