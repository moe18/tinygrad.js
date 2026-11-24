"""torch.backends: flags that scripts set (cudnn.benchmark = True ...) are accepted and ignored."""
class _Flags:
  def __init__(self, **kw): self.__dict__.update(kw)
  def is_available(self): return False
  def is_built(self): return False
cudnn = _Flags(benchmark=False, deterministic=False, enabled=False, allow_tf32=False)
cuda = _Flags(matmul=_Flags(allow_tf32=False))
mps = _Flags()
mkldnn = _Flags(enabled=False)
openmp = _Flags()
