"""torch.optim.lr_scheduler: closed-form learning-rate schedules."""
from __future__ import annotations
import math
from bisect import bisect_right

class LRScheduler:
  def __init__(self, optimizer, last_epoch=-1):
    self.optimizer = optimizer
    for g in optimizer.param_groups: g.setdefault("initial_lr", g["lr"])
    self.base_lrs = [g["initial_lr"] for g in optimizer.param_groups]
    self.last_epoch = last_epoch
    self._step_count = 0
    self.step()
  def _lr(self, base_lr, t): raise NotImplementedError
  def get_lr(self): return [self._lr(b, self.last_epoch) for b in self.base_lrs]
  def get_last_lr(self): return self._last_lr
  def step(self, epoch=None):
    self._step_count += 1
    self.last_epoch = self.last_epoch + 1 if epoch is None else epoch
    lrs = self.get_lr()
    for g, lr in zip(self.optimizer.param_groups, lrs): g["lr"] = lr
    self._last_lr = lrs
  def state_dict(self): return {k: v for k, v in self.__dict__.items() if k != "optimizer" and not callable(v)}
  def load_state_dict(self, sd): self.__dict__.update(sd)
_LRScheduler = LRScheduler

class LambdaLR(LRScheduler):
  def __init__(self, optimizer, lr_lambda, last_epoch=-1):
    self.lr_lambdas = list(lr_lambda) if isinstance(lr_lambda, (list, tuple)) else [lr_lambda] * len(optimizer.param_groups)
    super().__init__(optimizer, last_epoch)
  def get_lr(self): return [b * f(self.last_epoch) for b, f in zip(self.base_lrs, self.lr_lambdas)]
class StepLR(LRScheduler):
  def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
    self.step_size, self.gamma = step_size, gamma
    super().__init__(optimizer, last_epoch)
  def _lr(self, b, t): return b * self.gamma ** (t // self.step_size)
class MultiStepLR(LRScheduler):
  def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
    self.milestones, self.gamma = sorted(milestones), gamma
    super().__init__(optimizer, last_epoch)
  def _lr(self, b, t): return b * self.gamma ** bisect_right(self.milestones, t)
class ExponentialLR(LRScheduler):
  def __init__(self, optimizer, gamma, last_epoch=-1):
    self.gamma = gamma
    super().__init__(optimizer, last_epoch)
  def _lr(self, b, t): return b * self.gamma ** t
class ConstantLR(LRScheduler):
  def __init__(self, optimizer, factor=1.0 / 3, total_iters=5, last_epoch=-1):
    self.factor, self.total_iters = factor, total_iters
    super().__init__(optimizer, last_epoch)
  def _lr(self, b, t): return b * (self.factor if t < self.total_iters else 1.0)
class LinearLR(LRScheduler):
  def __init__(self, optimizer, start_factor=1.0 / 3, end_factor=1.0, total_iters=5, last_epoch=-1):
    self.start_factor, self.end_factor, self.total_iters = start_factor, end_factor, total_iters
    super().__init__(optimizer, last_epoch)
  def _lr(self, b, t): return b * (self.start_factor + (self.end_factor - self.start_factor) * min(t, self.total_iters) / self.total_iters)
class PolynomialLR(LRScheduler):
  def __init__(self, optimizer, total_iters=5, power=1.0, last_epoch=-1):
    self.total_iters, self.power = total_iters, power
    super().__init__(optimizer, last_epoch)
  def _lr(self, b, t): return b * (1.0 - min(t, self.total_iters) / self.total_iters) ** self.power
class CosineAnnealingLR(LRScheduler):
  def __init__(self, optimizer, T_max, eta_min=0.0, last_epoch=-1):
    self.T_max, self.eta_min = T_max, eta_min
    super().__init__(optimizer, last_epoch)
  def _lr(self, b, t): return self.eta_min + (b - self.eta_min) * (1 + math.cos(math.pi * t / self.T_max)) / 2
class CosineAnnealingWarmRestarts(LRScheduler):
  def __init__(self, optimizer, T_0, T_mult=1, eta_min=0.0, last_epoch=-1):
    self.T_0, self.T_mult, self.eta_min = T_0, T_mult, eta_min
    super().__init__(optimizer, last_epoch)
  def _lr(self, b, t):
    t_i, t_cur = self.T_0, t
    while t_cur >= t_i: t_cur -= t_i; t_i *= self.T_mult
    return self.eta_min + (b - self.eta_min) * (1 + math.cos(math.pi * t_cur / t_i)) / 2
class OneCycleLR(LRScheduler):
  """Learning-rate part of the one-cycle policy. Momentum cycling (``cycle_momentum``) is not applied."""
  def __init__(self, optimizer, max_lr, total_steps=None, epochs=None, steps_per_epoch=None, pct_start=0.3, anneal_strategy="cos",
               cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=1e4, three_phase=False, last_epoch=-1):
    self.total_steps = total_steps if total_steps is not None else epochs * steps_per_epoch
    self.max_lrs = list(max_lr) if isinstance(max_lr, (list, tuple)) else [max_lr] * len(optimizer.param_groups)
    self.pct_start, self.anneal, self.div_factor, self.final_div_factor = pct_start, anneal_strategy, div_factor, final_div_factor
    super().__init__(optimizer, last_epoch)
  def _anneal(self, start, end, pct):
    return end + (start - end) / 2.0 * (math.cos(math.pi * pct) + 1) if self.anneal == "cos" else (end - start) * pct + start
  def get_lr(self):
    t = self.last_epoch
    if t > self.total_steps: raise ValueError(f"Tried to step {t} times. The specified number of total steps is {self.total_steps}")
    up_end = float(self.pct_start * self.total_steps) - 1
    out = []
    for mx in self.max_lrs:
      init, final = mx / self.div_factor, mx / self.div_factor / self.final_div_factor
      if t <= up_end: out.append(self._anneal(init, mx, t / up_end if up_end else 1.0))
      else: out.append(self._anneal(mx, final, (t - up_end) / (self.total_steps - 1 - up_end)))
    return out
class SequentialLR(LRScheduler):
  def __init__(self, optimizer, schedulers, milestones, last_epoch=-1):
    self.optimizer, self._schedulers, self._milestones, self.last_epoch = optimizer, schedulers, milestones, last_epoch + 1
    for g in optimizer.param_groups: g["lr"] = g["initial_lr"]
    self._schedulers[0].last_epoch = -1; self._schedulers[0].step()
    self._last_lr = schedulers[0].get_last_lr()
  def step(self):
    self.last_epoch += 1
    idx = bisect_right(self._milestones, self.last_epoch)
    if idx > 0 and self._milestones[idx - 1] == self.last_epoch: self._schedulers[idx].step(0)
    else: self._schedulers[idx].step()
    self._last_lr = self._schedulers[idx].get_last_lr()
class ReduceLROnPlateau:
  def __init__(self, optimizer, mode="min", factor=0.1, patience=10, threshold=1e-4, threshold_mode="rel", cooldown=0, min_lr=0, eps=1e-08, verbose=False):
    self.optimizer, self.mode, self.factor, self.patience, self.threshold, self.threshold_mode = optimizer, mode, factor, patience, threshold, threshold_mode
    self.cooldown, self.eps, self.cooldown_counter, self.num_bad_epochs, self.last_epoch = cooldown, eps, 0, 0, 0
    self.min_lrs = list(min_lr) if isinstance(min_lr, (list, tuple)) else [min_lr] * len(optimizer.param_groups)
    self.best = math.inf if mode == "min" else -math.inf
    self._last_lr = [g["lr"] for g in optimizer.param_groups]
  def _better(self, a):
    if self.mode == "min": return a < (self.best * (1 - self.threshold) if self.threshold_mode == "rel" else self.best - self.threshold)
    return a > (self.best * (1 + self.threshold) if self.threshold_mode == "rel" else self.best + self.threshold)
  def step(self, metrics, epoch=None):
    current = float(metrics)
    self.last_epoch += 1
    if self._better(current): self.best, self.num_bad_epochs = current, 0
    else: self.num_bad_epochs += 1
    if self.cooldown_counter > 0: self.cooldown_counter -= 1; self.num_bad_epochs = 0
    if self.num_bad_epochs > self.patience:
      for g, mn in zip(self.optimizer.param_groups, self.min_lrs):
        new = max(g["lr"] * self.factor, mn)
        if g["lr"] - new > self.eps: g["lr"] = new
      self.cooldown_counter, self.num_bad_epochs = self.cooldown, 0
    self._last_lr = [g["lr"] for g in self.optimizer.param_groups]
  def get_last_lr(self): return self._last_lr
