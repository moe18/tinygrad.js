"""torch.jit: TorchScript does not exist here. script/trace return their argument unchanged so decorated code still runs."""
def script(obj=None, *a, **k): return obj if obj is not None else (lambda o: o)
def trace(func, *a, **k): return func
def ignore(fn=None, **k): return fn if fn is not None else (lambda f: f)
unused = export = ignore
def is_scripting(): return False
def is_tracing(): return False
class ScriptModule:
  def __init_subclass__(cls, **kw): raise NotImplementedError("TorchScript is not supported by the tinygrad.js torch shim")
def save(*a, **k): raise NotImplementedError("TorchScript is not supported by the tinygrad.js torch shim")
load = save
