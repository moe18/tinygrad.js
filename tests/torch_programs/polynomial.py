# "Learning PyTorch with Examples": fit y = sin(x) with a third-order polynomial, five ways.
# (Initial coefficients are fixed numbers instead of torch.randn so that results can be compared across frameworks.)
out = {}
dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.linspace(-math.pi, math.pi, 200, device=device, dtype=dtype)
y = torch.sin(x)

# 1. tensors with manual gradients
a, b, c, d = (torch.tensor(v, device=device, dtype=dtype) for v in (0.3, -0.2, 0.1, 0.05))
learning_rate = 1e-5
losses = []
for t in range(60):
  y_pred = a + b * x + c * x ** 2 + d * x ** 3
  loss = (y_pred - y).pow(2).sum().item()
  losses.append(loss)
  grad_y_pred = 2.0 * (y_pred - y)
  grad_a = grad_y_pred.sum()
  grad_b = (grad_y_pred * x).sum()
  grad_c = (grad_y_pred * x ** 2).sum()
  grad_d = (grad_y_pred * x ** 3).sum()
  a -= learning_rate * grad_a
  b -= learning_rate * grad_b
  c -= learning_rate * grad_c
  d -= learning_rate * grad_d
out["manual"] = [losses, a, b, c, d]

# 2. autograd
a, b, c, d = (torch.tensor(v, device=device, dtype=dtype, requires_grad=True) for v in (0.3, -0.2, 0.1, 0.05))
losses = []
for t in range(60):
  y_pred = a + b * x + c * x ** 2 + d * x ** 3
  loss = (y_pred - y).pow(2).sum()
  if t % 10 == 9: losses.append(loss.item())
  loss.backward()
  with torch.no_grad():
    a -= learning_rate * a.grad
    b -= learning_rate * b.grad
    c -= learning_rate * c.grad
    d -= learning_rate * d.grad
    a.grad = None
    b.grad = None
    c.grad = None
    d.grad = None
out["autograd"] = [losses, a.item(), b.item(), c.item(), d.item()]
out["autograd_str"] = f"Result: y = {a.item():.3f} + {b.item():.3f} x + {c.item():.3f} x^2 + {d.item():.3f} x^3"

# 3. nn.Sequential, updating parameters by hand
p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)
model = torch.nn.Sequential(torch.nn.Linear(3, 1), torch.nn.Flatten(0, 1))
with torch.no_grad():
  model[0].weight.copy_(torch.tensor([[-0.2, 0.1, 0.05]])); model[0].bias.fill_(0.3)
loss_fn = torch.nn.MSELoss(reduction="sum")
learning_rate = 1e-5
losses = []
for t in range(60):
  y_pred = model(xx)
  loss = loss_fn(y_pred, y)
  if t % 10 == 9: losses.append(loss.item())
  model.zero_grad()
  loss.backward()
  with torch.no_grad():
    for param in model.parameters():
      param -= learning_rate * param.grad
linear_layer = model[0]
out["nn"] = [losses, linear_layer.bias.item(), linear_layer.weight[:, 0].item(), linear_layer.weight[:, 1].item(), linear_layer.weight[:, 2].item()]

# 4. optim
model = torch.nn.Sequential(torch.nn.Linear(3, 1), torch.nn.Flatten(0, 1))
with torch.no_grad():
  model[0].weight.copy_(torch.tensor([[-0.2, 0.1, 0.05]])); model[0].bias.fill_(0.3)
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
losses = []
for t in range(60):
  y_pred = model(xx)
  loss = loss_fn(y_pred, y)
  if t % 10 == 9: losses.append(loss.item())
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
out["optim"] = [losses, model[0].bias, model[0].weight]

# 5. custom nn.Module
class Polynomial3(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.a = torch.nn.Parameter(torch.tensor(0.3))
    self.b = torch.nn.Parameter(torch.tensor(-0.2))
    self.c = torch.nn.Parameter(torch.tensor(0.1))
    self.d = torch.nn.Parameter(torch.tensor(0.05))
  def forward(self, x):
    return self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3
  def string(self):
    return f"y = {self.a.item():.4f} + {self.b.item():.4f} x + {self.c.item():.4f} x^2 + {self.d.item():.4f} x^3"
model = Polynomial3()
criterion = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
losses = []
for t in range(60):
  y_pred = model(x)
  loss = criterion(y_pred, y)
  if t % 10 == 9: losses.append(loss.item())
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
out["module"] = [losses, model.string(), [n for n, _ in model.named_parameters()]]
emit(**out)
