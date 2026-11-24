# tinygrad_js.run_training with steps written against the torch shim: TinyJit must give the same training as plain execution.
import json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from tinygrad_js import run_training

rng = np.random.RandomState(0)
X = torch.tensor(rng.standard_normal((256, 2)).astype(np.float32))
Y = ((X * X).sum(dim=1, keepdim=True) < 1.0).float()

def make(kind):
  model = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 1))
  with torch.no_grad():
    for i, p in enumerate(model.parameters()):
      p.copy_(torch.tensor((np.random.RandomState(10 + i).standard_normal(tuple(p.shape)) * 0.3).astype(np.float32)))
  opt = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9) if kind == "sgd" else torch.optim.Adam(model.parameters(), lr=0.02)
  def step(x, y):
    opt.zero_grad()
    loss = F.binary_cross_entropy_with_logits(model(x), y)
    loss.backward()
    opt.step()
    return loss
  return model, opt, step

out = {}
for kind in ("sgd", "adam", "sched"):
  res = {}
  for mode, jit in (("plain", False), ("jit", True)):
    model, opt, step = make(kind)
    if kind == "sched":
      # a learning-rate schedule driven from outside the step: changing group["lr"] updates the buffer the jitted kernels read
      losses = []
      for chunk in range(4):
        for g in opt.param_groups: g["lr"] = 0.02 * (0.5 ** chunk)
        r = run_training(step, 10, args=(X, Y), jit=jit, record_every=1, warmup=2 if chunk == 0 else 0)
        losses += [v for _, v in r.losses]
        res[mode + "_jit"] = r.jit_used
    else:
      r = run_training(step, 40, args=(X, Y), jit=jit, record_every=1)
      losses = [v for _, v in r.losses]
      res[mode + "_jit"] = r.jit_used
    res[mode + "_losses"] = losses
    res[mode + "_params"] = [float(p.detach().abs().sum()) for p in model.parameters()]
    if mode == "jit": res.update(jit_used=r.jit_used, reason=r.jit_fallback_reason, ms_jit=r.ms_per_step_jit, summary=r.summary())
    else: res["ms_plain"] = r.ms_per_step_plain
  out[kind] = res
print("RESULT_JSON:" + json.dumps(out))
