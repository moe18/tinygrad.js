# Exercises tinygrad_js.run_training. Runs natively (PYTHONPATH=python) and in the browser.
from tinygrad import nn
from tinygrad.nn.optim import Adam
from tinygrad_js import run_training

def setup(seed=3):
  Tensor.manual_seed(seed)
  model_l1, model_l2 = nn.Linear(2, 32), nn.Linear(32, 1)
  model = lambda x: model_l2(model_l1(x).relu())
  opt = Adam([model_l1.weight, model_l1.bias, model_l2.weight, model_l2.bias], lr=0.03)
  X = (Tensor.rand(256, 2) * 2 - 1).realize()
  Y = ((X * X).sum(axis=1, keepdim=True) < 0.36).float().realize()
  def step(x, y):
    opt.zero_grad()
    loss = (model(x).sigmoid() - y).square().mean()
    loss.backward()
    opt.step()
    return loss
  return step, X, Y, opt, model

out = {}
# 1. the same training with and without TinyJit gives the same losses
step, X, Y, *_ = setup()
plain = run_training(step, 60, args=(X, Y), jit=False, record_every=1)
step, X, Y, *_ = setup()
jitted = run_training(step, 60, args=(X, Y), record_every=1)
out["plain_losses"] = [v for _, v in plain.losses]
out["jit_losses"] = [v for _, v in jitted.losses]
out["jit_used"], out["plain_jit_used"] = jitted.jit_used, plain.jit_used
out["ms_plain"], out["ms_jit"] = jitted.ms_per_step_plain, jitted.ms_per_step_jit
out["summary"] = jitted.summary()
out["warmup_losses"] = jitted.warmup_losses

# 2. .item() inside the step: TinyJit raises while recording -> fallback, training still completes
step, X, Y, *_ = setup()
def chatty_step(x, y):
  loss = step(x, y)
  if loss.item() < 0: print("never")
  return loss
r = run_training(chatty_step, 30, args=(X, Y))
out["item_fallback"] = [r.jit_used, r.jit_fallback_reason, r.jit_fallback_step, r.initial_loss > r.final_loss]

# 3. a Python number argument that changes -> detected (TinyJit would freeze it)
step, X, Y, opt, model = setup()
def scaled_step(x, y, scale):
  opt.zero_grad()
  loss = (model(x).sigmoid() - y).square().mean() * scale
  loss.backward(); opt.step()
  return loss
r = run_training(scaled_step, 12, args=lambda i: (X, Y, 1.0 if i < 6 else 0.5))
out["scalar_fallback"] = [r.jit_used, r.jit_fallback_reason, r.jit_fallback_step]

# 4. changing input shape -> detected (tinygrad 0.14 TinyJit would return a wrong-shaped result)
step, X, Y, *_ = setup()
r = run_training(step, 12, args=lambda i: (X[:128], Y[:128]) if i < 8 else (X, Y))
out["shape_fallback"] = [r.jit_used, r.jit_fallback_reason, r.jit_fallback_step]

# 5. per-step minibatches of a fixed shape stay on TinyJit
step, X, Y, *_ = setup()
def batch(i):
  idx = Tensor.randint(64, high=256)
  return X[idx], Y[idx]
r = run_training(step, 40, args=batch, print_every=20)
out["minibatch"] = [r.jit_used, r.jit_fallback_reason, r.final_loss < r.initial_loss]
emit(**out)
