"""Training-loop helper: plain warm-up steps, then a TinyJit loop, with a fallback to plain execution.

Learners write an ordinary step function; the harness gets TinyJit's speed without the learner having to know its
rules. Works on tinygrad 0.12 and 0.14, in the browser and natively.

    from tinygrad_js import run_training

    def train_step(x, y):
      opt.zero_grad()
      loss = model(x).sparse_categorical_crossentropy(y)
      loss.backward()
      opt.step()
      return loss

    result = run_training(train_step, steps=200, args=(X, Y))
    print(result.summary())

What it does, per step index ``i`` in ``range(steps)``:

* ``i < warmup``: runs ``step`` normally (eager tinygrad) and records the loss of every step. Grade correctness on
  these: they cannot be affected by TinyJit.
* ``i >= warmup``: runs ``TinyJit(step)``. TinyJit's own first call runs normally, its second records the kernels,
  and later calls replay them without running the Python body. Losses are read every ``record_every`` steps and on
  the last step (each read is a GPU->CPU sync, so reading every step costs speed).
* Before each TinyJit call the helper checks the arguments against those of the first TinyJit call: Tensor shape,
  dtype and device, and the value of every non-Tensor argument. TinyJit itself would silently replay with the recorded
  values (Python numbers are frozen; on tinygrad 0.14 a changed shape even returns a wrong-shaped result), so on any
  change the helper stops using TinyJit instead.
* If the jitted call raises (e.g. on tinygrad >= 0.13, ``.item()``/``.numpy()`` inside the step raises ``JitError``
  while recording; tinygrad 0.12 allows it and replays correctly), the error is recorded in
  ``jit_fallback_reason`` and that step and all remaining steps run without TinyJit. The step that raised may already
  have been partly applied (e.g. parameters updated) before the error.

Not detected: Python values the step computes *inside* itself from Python state, e.g. a learning-rate schedule driven by
a global counter, or ``if epoch > 5:`` on a global. TinyJit freezes whatever they were while recording. Keep step
functions free of such logic (pass changing values as Tensors), and grade correctness on the warm-up steps.
"""
from __future__ import annotations
import contextlib, time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

__all__ = ["run_training", "TrainingResult"]

@dataclass
class TrainingResult:
  steps: int
  """Number of steps run."""
  losses: list[tuple[int, float]] = field(default_factory=list)
  """(step index, loss) for every warm-up step, every ``record_every``-th step and the last step."""
  jit_used: bool = False
  """True if every step after warm-up ran through TinyJit."""
  jit_fallback_reason: str | None = None
  """Why TinyJit was abandoned (``"ExceptionType: message"``), or None."""
  jit_fallback_step: int | None = None
  seconds: float = 0.0
  ms_per_step_plain: float | None = None
  """Average wall time of the steps that ran without TinyJit (including loss reads)."""
  ms_per_step_jit: float | None = None
  """Average wall time of the TinyJit steps, including TinyJit's two recording calls and loss reads."""

  @property
  def initial_loss(self) -> float | None: return self.losses[0][1] if self.losses else None
  @property
  def final_loss(self) -> float | None: return self.losses[-1][1] if self.losses else None
  @property
  def warmup_losses(self) -> list[float]:
    """Losses of the plain warm-up steps, in order (never affected by TinyJit)."""
    return [v for i, v in self.losses if i < self._warmup]
  _warmup: int = field(default=0, repr=False)

  def summary(self) -> str:
    mode = "TinyJit" if self.jit_used else ("plain (TinyJit fell back: " + self.jit_fallback_reason + ")" if self.jit_fallback_reason
                                             else "plain")
    parts = [f"{self.steps} steps in {self.seconds:.2f}s, {mode}"]
    if self.initial_loss is not None: parts.append(f"loss {self.initial_loss:.4f} -> {self.final_loss:.4f}")
    if self.ms_per_step_plain is not None: parts.append(f"{self.ms_per_step_plain:.1f} ms/step plain")
    if self.ms_per_step_jit is not None: parts.append(f"{self.ms_per_step_jit:.1f} ms/step jit")
    return ", ".join(parts)

def _training_context():
  from tinygrad import Tensor
  if hasattr(Tensor, "train"): return Tensor.train()       # tinygrad <= 0.12
  from tinygrad.helpers import Context
  return Context(TRAINING=1)                                 # tinygrad >= 0.13

def run_training(step:Callable[..., Any], steps:int, *, args:Sequence[Any]|Callable[[int], Sequence[Any]]=(), warmup:int=2,
                 jit:bool=True, train:bool=True, record_every:int=10, print_every:int=0) -> TrainingResult:
  """Run ``step(*args)`` ``steps`` times; see the module docstring.

  * ``step``: a function doing one optimization step and returning the loss Tensor (or None).
  * ``args``: Tensors passed to every call, or a function ``i -> args`` returning per-step Tensors (e.g. a minibatch).
    Under TinyJit, per-step args must keep the same shapes and dtypes.
  * ``warmup``: plain steps before TinyJit takes over (losses of all of them are recorded).
  * ``jit``: set False to run everything plainly (same results, slower).
  * ``train``: run steps in training mode (``Tensor.train()`` / ``Context(TRAINING=1)``), needed by optimizers on 0.12.
  * ``record_every``: read the loss every N steps (0: only warm-up and the last step).
  * ``print_every``: print ``step i  loss x`` every N steps (0: never).
  """
  from tinygrad import Tensor, TinyJit, Device
  if steps < 0 or warmup < 0: raise ValueError("steps and warmup must be >= 0")

  def get_args(i:int) -> tuple:
    a = args(i) if callable(args) else args
    a = tuple(a) if isinstance(a, (tuple, list)) else (a,)
    if (ts := [x for x in a if isinstance(x, Tensor)]): Tensor.realize(*ts)
    return a

  def plain(*a):
    out = step(*a)
    return out.realize() if isinstance(out, Tensor) else out

  def call(fn, i:int, a:tuple|None=None):
    with (_training_context() if train else contextlib.nullcontext()):
      return fn(*(get_args(i) if a is None else a))

  def sync():
    with contextlib.suppress(Exception): Device[Device.DEFAULT].synchronize()

  def signature(a:tuple) -> tuple:
    return tuple(("tensor", x.shape, x.dtype, x.device) if isinstance(x, Tensor) else
                 ("value", type(x), x) if isinstance(x, (int, float, bool, str, type(None))) else ("object", id(x)) for x in a)

  def describe_change(old:tuple, new:tuple) -> str:
    if len(old) != len(new): return f"the number of arguments changed ({len(old)} -> {len(new)})"
    for n, (o, w) in enumerate(zip(old, new)):
      if o != w:
        if o[0] == "tensor" and w[0] == "tensor": return f"argument {n} changed from shape {o[1]} {o[2]} to shape {w[1]} {w[2]}"
        if o[0] == "value": return f"argument {n} changed from {o[2]!r} to {repr(w[2]) if w[0] == 'value' else 'another object'}"
        return f"argument {n} is a different object"
    return "arguments changed"

  res = TrainingResult(steps=steps, _warmup=min(warmup, steps))
  jitted = TinyJit(plain) if jit else None
  jit_sig: tuple|None = None
  t_plain = t_jit = 0.0
  n_plain = n_jit = 0
  start = time.perf_counter()
  for i in range(steps):
    t0 = time.perf_counter()
    used_jit = False
    if jitted is not None and i >= warmup:
      a = get_args(i)
      if jit_sig is None: jit_sig = signature(a)
      if (sig := signature(a)) != jit_sig:
        res.jit_fallback_reason = f"TinyJit would replay stale inputs: {describe_change(jit_sig, sig)}"
        res.jit_fallback_step, jitted = i, None
        out = call(plain, i, a)
      else:
        try:
          out, used_jit = call(jitted, i, a), True
        except Exception as e:  # KeyboardInterrupt (runtime.interrupt()) is not caught
          res.jit_fallback_reason, res.jit_fallback_step, jitted = f"{type(e).__name__}: {e}".strip(), i, None
          if n_jit: sync()
          t0 = time.perf_counter()
          out = call(plain, i, a)
    else:
      out = call(plain, i)
    want_print = bool(print_every) and i % print_every == 0
    if isinstance(out, Tensor) and (i < warmup or i == steps - 1 or want_print or (record_every and i % record_every == 0)):
      value = out.item()
      res.losses.append((i, value))
      if want_print or (print_every and i == steps - 1): print(f"step {i:4d}  loss {value:.4f}")
    elif want_print: print(f"step {i:4d}")
    if used_jit: t_jit += time.perf_counter() - t0; n_jit += 1
    else: t_plain += time.perf_counter() - t0; n_plain += 1
    if i == warmup - 1: sync()  # do not bill warm-up GPU work to the first TinyJit step
  sync()
  res.seconds = time.perf_counter() - start
  res.jit_used = jit and res.jit_fallback_reason is None and n_jit == max(0, steps - warmup)
  res.ms_per_step_plain = t_plain / n_plain * 1e3 if n_plain else None
  res.ms_per_step_jit = t_jit / n_jit * 1e3 if n_jit else None
  return res
