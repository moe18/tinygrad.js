from tinygrad import nn
from tinygrad.nn.optim import SGD, Adam
Tensor.manual_seed(42)
class MLP:
  def __init__(self): self.l1, self.l2 = nn.Linear(2, 16), nn.Linear(16, 1)
  def __call__(self, x): return self.l2(self.l1(x).relu())
# synthetic dataset: label = 1 when the point is inside the circle of radius 0.6
X = Tensor.rand(256, 2) * 2 - 1
Y = ((X * X).sum(axis=1, keepdim=True) < 0.36).float()
out = {}
for name, make_opt in (("sgd", lambda p: SGD(p, lr=0.5, momentum=0.9)), ("adam", lambda p: Adam(p, lr=0.03))):
  Tensor.manual_seed(7)
  model = MLP()
  opt = make_opt(nn.state.get_parameters(model))
  losses = []
  with training_context():
    for step in range(60):
      opt.zero_grad()
      loss = (model(X).sigmoid() - Y).square().mean()
      loss.backward()
      opt.step()
      losses.append(loss.item())
  out[name + "_losses"] = losses
  out[name + "_acc"] = ((model(X).sigmoid() > 0.5).float() == Y).float().mean().item()
emit(**out)
