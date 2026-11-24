"""torch.amp: mixed precision is not used; autocast is a no-op and GradScaler passes calls straight through."""
class autocast:
  def __init__(self, *a, **k): pass
  def __enter__(self): return self
  def __exit__(self, *a): return False
  def __call__(self, fn): return fn
class GradScaler:
  def __init__(self, *a, **k): pass
  def scale(self, loss): return loss
  def step(self, optimizer, *a, **k): return optimizer.step(*a, **k)
  def update(self): pass
  def unscale_(self, optimizer): pass
  def get_scale(self): return 1.0
  def state_dict(self): return {}
  def load_state_dict(self, sd): pass
  def is_enabled(self): return False
