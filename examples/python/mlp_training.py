# Train a small MLP on a synthetic 2-D classification problem (is a point inside a ring?),
# using tinygrad's standard nn.Linear, Adam and TinyJit APIs.
# Works with tinygrad 0.12 (Tensor.train()) and 0.13+ (Context(TRAINING=1)).
import time
from tinygrad import Tensor, TinyJit, nn, Device
from tinygrad.helpers import Context
from tinygrad.nn.optim import Adam

training = Tensor.train if hasattr(Tensor, "train") else (lambda: Context(TRAINING=1))
Tensor.manual_seed(0)

class MLP:
  def __init__(self):
    self.l1, self.l2, self.l3 = nn.Linear(2, 32), nn.Linear(32, 32), nn.Linear(32, 1)
  def __call__(self, x: Tensor) -> Tensor:
    return self.l3(self.l2(self.l1(x).relu()).relu())

# synthetic dataset: label 1 for points in the ring 0.3 < r < 0.7
X = Tensor.rand(512, 2) * 2 - 1
r = (X * X).sum(axis=1, keepdim=True).sqrt()
Y = ((r > 0.3) * (r < 0.7)).float().realize()

model = MLP()
opt = Adam(nn.state.get_parameters(model), lr=0.02)

@TinyJit
def train_step() -> Tensor:
  with training():
    opt.zero_grad()
    loss = model(X).binary_crossentropy_logits(Y)
    loss.backward()
    opt.step()
    return loss.realize()

st = time.perf_counter()
initial = None
for step in range(150):
  loss = train_step()
  if step % 30 == 0 or step == 149:
    value = loss.item()
    initial = value if initial is None else initial
    print(f"step {step:3d}  loss {value:.4f}")
acc = ((model(X) > 0).float() == Y).float().mean().item()
print(f"device {Device.DEFAULT}, {time.perf_counter() - st:.2f}s")
print(f"final loss {value:.4f} (initial {initial:.4f}), accuracy {acc:.3f}")
