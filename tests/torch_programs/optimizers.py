# Every optimizer and learning-rate schedule against PyTorch's update rules, on a small deterministic regression problem.
out = {}
X, Y = det(32, 4, seed=1), det(32, 2, seed=2)
def make_model(): return det_init(nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2)))
def train(make_opt, steps=12):
  model = make_model()
  opt = make_opt(model)
  losses = []
  for _ in range(steps):
    opt.zero_grad()
    loss = F.mse_loss(model(X), Y)
    loss.backward()
    opt.step()
    losses.append(loss.item())
  return [losses, [p.detach().clone() for p in model.parameters()]]
configs = {
  "sgd": lambda m: torch.optim.SGD(m.parameters(), lr=0.1),
  "sgd_momentum": lambda m: torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9),
  "sgd_nesterov": lambda m: torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9, nesterov=True),
  "sgd_wd": lambda m: torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.8, weight_decay=0.1),
  "sgd_dampening": lambda m: torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9, dampening=0.3),
  "adam": lambda m: torch.optim.Adam(m.parameters(), lr=0.01),
  "adam_wd": lambda m: torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=0.1, betas=(0.8, 0.95)),
  "adam_amsgrad": lambda m: torch.optim.Adam(m.parameters(), lr=0.01, amsgrad=True),
  "adamw": lambda m: torch.optim.AdamW(m.parameters(), lr=0.01, weight_decay=0.1),
  "rmsprop": lambda m: torch.optim.RMSprop(m.parameters(), lr=0.01),
  "rmsprop_momentum": lambda m: torch.optim.RMSprop(m.parameters(), lr=0.01, momentum=0.9, alpha=0.9),
  "rmsprop_centered": lambda m: torch.optim.RMSprop(m.parameters(), lr=0.01, centered=True),
  "adagrad": lambda m: torch.optim.Adagrad(m.parameters(), lr=0.1),
  "adadelta": lambda m: torch.optim.Adadelta(m.parameters(), lr=1.0),
  "param_groups": lambda m: torch.optim.AdamW([{"params": m[0].parameters(), "weight_decay": 0.2}, {"params": m[2].parameters(), "lr": 0.05, "weight_decay": 0.0}], lr=0.01),
}
for name, mk in configs.items(): out[name] = train(mk)

# learning-rate schedulers: the schedule itself, and its effect on training
def schedule(make_sched, steps=14):
  model = make_model()
  opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
  sched = make_sched(opt)
  lrs, losses = [], []
  for _ in range(steps):
    opt.zero_grad()
    loss = F.mse_loss(model(X), Y)
    loss.backward()
    opt.step()
    sched.step()
    lrs.append(sched.get_last_lr()[0]); losses.append(loss.item())
  return [lrs, losses, opt.param_groups[0]["lr"]]
S = torch.optim.lr_scheduler
scheds = {
  "step": lambda o: S.StepLR(o, step_size=4, gamma=0.5), "multistep": lambda o: S.MultiStepLR(o, milestones=[3, 9], gamma=0.1),
  "exponential": lambda o: S.ExponentialLR(o, gamma=0.9), "cosine": lambda o: S.CosineAnnealingLR(o, T_max=10, eta_min=0.001),
  "lambda": lambda o: S.LambdaLR(o, lr_lambda=lambda e: 0.95 ** e), "linear": lambda o: S.LinearLR(o, start_factor=0.2, total_iters=6),
  "constant": lambda o: S.ConstantLR(o, factor=0.5, total_iters=4), "onecycle": lambda o: S.OneCycleLR(o, max_lr=0.3, total_steps=14, cycle_momentum=False),
  "warm_restarts": lambda o: S.CosineAnnealingWarmRestarts(o, T_0=5), "polynomial": lambda o: S.PolynomialLR(o, total_iters=10, power=2.0),
}
for name, mk in scheds.items(): out["sched_" + name] = schedule(mk)
# ReduceLROnPlateau and manual learning-rate changes
model = make_model(); opt = torch.optim.SGD(model.parameters(), lr=0.1)
plateau = S.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=1)
lrs = []
for metric in [1.0, 0.9, 0.95, 0.96, 0.97, 0.5, 0.6, 0.7, 0.8]:
  plateau.step(metric); lrs.append(opt.param_groups[0]["lr"])
out["plateau"] = lrs
model = make_model(); opt = torch.optim.Adam(model.parameters(), lr=0.01)
losses = []
for it in range(10):
  for group in opt.param_groups: group["lr"] = 0.01 * (it + 1) / 10 if it < 5 else 0.005
  opt.zero_grad(); loss = F.mse_loss(model(X), Y); loss.backward()
  norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
  opt.step(); losses.append([loss.item(), float(norm)])
out["manual_lr_and_clip"] = [losses, [p.detach().clone() for p in model.parameters()]]
# clip_grad_value_, zero_grad(set_to_none=False), skipping parameters without gradients, closure
model = make_model(); opt = torch.optim.SGD(model.parameters(), lr=0.1)
F.mse_loss(model(X), Y).backward()
torch.nn.utils.clip_grad_value_(model.parameters(), 0.01)
out["clip_value"] = [p.grad.clone() for p in model.parameters()]
opt.zero_grad(set_to_none=False)
out["zero_grad_keep"] = [p.grad.clone() for p in model.parameters()]
opt.zero_grad()
out["zero_grad_none"] = [p.grad is None for p in model.parameters()]
model[2].weight.requires_grad_(False)
before = model[2].weight.detach().clone()
loss = F.mse_loss(model(X), Y); loss.backward(); opt.step()
out["frozen_param_unchanged"] = [torch.equal(before, model[2].weight.detach()), model[2].weight.grad is None, model[0].weight.grad is not None]
def closure():
  opt.zero_grad(); l = F.mse_loss(model(X), Y); l.backward(); return l
out["closure"] = opt.step(closure).item()
out["optimizer_state"] = [sorted(k for k in torch.optim.Adam(make_model().parameters()).state_dict().keys()), len(opt.param_groups), opt.defaults["lr"]]
emit(**out)
