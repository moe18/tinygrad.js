"""torch.cuda: there is no CUDA in the browser. is_available() is False; scripts that select "cuda if available" get "cpu",
and tensors run on tinygrad's default device (WebGPU) whatever device name is used."""
def is_available(): return False
def device_count(): return 0
def manual_seed(seed):
  from .. import manual_seed as ms
  ms(seed)
manual_seed_all = manual_seed
def empty_cache(): pass
def synchronize(device=None): pass
def current_device(): raise RuntimeError("CUDA is not available in the tinygrad.js torch shim")
def get_device_name(device=None): raise RuntimeError("CUDA is not available in the tinygrad.js torch shim")
def is_bf16_supported(*a, **k): return False
def memory_allocated(device=None): return 0
def max_memory_allocated(device=None): return 0
def reset_peak_memory_stats(device=None): pass
class amp:
  @staticmethod
  def autocast(*a, **k):
    from ..amp import autocast
    return autocast(*a, **k)
  @staticmethod
  def GradScaler(*a, **k):
    from ..amp import GradScaler
    return GradScaler(*a, **k)
