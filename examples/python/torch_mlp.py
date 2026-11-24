# Plain PyTorch code: an MLP classifier with nn.Module, DataLoader, Adam and a train/eval loop.
# In tinygrad.js, `import torch` is a PyTorch-compatible layer on top of tinygrad (docs/TORCH.md), so this file runs unmodified in
# the browser on WebGPU. It also runs unmodified with real PyTorch (`python examples/python/torch_mlp.py`).
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# synthetic dataset: is a 2-D point inside the ring 0.4 < r < 0.8?
X = torch.rand(512, 2) * 2 - 1
r = (X ** 2).sum(dim=1).sqrt()
y = ((r > 0.4) & (r < 0.8)).long()
loader = DataLoader(TensorDataset(X, y), batch_size=128, shuffle=True)

class MLP(nn.Module):
  def __init__(self, hidden=64):
    super().__init__()
    self.net = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))
  def forward(self, x):
    return self.net(x)

model = MLP().to(device)
print(model)
print("parameters:", sum(p.numel() for p in model.parameters()))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

def accuracy():
  model.eval()
  with torch.no_grad():
    return (model(X.to(device)).argmax(dim=1) == y.to(device)).float().mean().item()

for epoch in range(30):
  model.train()
  total = 0.0
  for xb, yb in loader:
    xb, yb = xb.to(device), yb.to(device)
    optimizer.zero_grad()
    loss = loss_fn(model(xb), yb)
    loss.backward()
    optimizer.step()
    total += loss.item()
  scheduler.step()
  if epoch % 5 == 4:
    print(f"epoch {epoch + 1:2d}  loss {total / len(loader):.4f}  accuracy {accuracy():.3f}  lr {scheduler.get_last_lr()[0]:.4f}")

torch.save(model.state_dict(), "ring_mlp.pt")
restored = MLP()
restored.load_state_dict(torch.load("ring_mlp.pt"))
restored.eval()
print("restored model agrees:", torch.allclose(restored(X), model(X)))
print(f"final accuracy {accuracy():.3f} on {X.device}")
