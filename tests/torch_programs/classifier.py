# PyTorch "Quickstart"-style classifier: Dataset + DataLoader, nn.Module, train/test loops. Synthetic data replaces FashionMNIST.
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
out = {}
N, C = 192, 4
images = det(N, 1, 6, 6, seed=1)
labels = torch.tensor(np.argmax(images.numpy().reshape(N, 4, 9).mean(-1) + 0.1 * np.arange(4), axis=1).astype(np.int64))

class SyntheticImages(Dataset):
  def __init__(self, x, y, transform=None): self.x, self.y, self.transform = x, y, transform
  def __len__(self): return len(self.x)
  def __getitem__(self, idx):
    image = self.x[idx]
    if self.transform: image = self.transform(image)
    return image, self.y[idx]

training_data = SyntheticImages(images[:128], labels[:128])
test_data = TensorDataset(images[128:], labels[128:])
batch_size = 32
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
for X, y in test_dataloader:
  out["batch_shapes"] = [list(X.shape), list(y.shape), y.dtype.is_floating_point, len(train_dataloader), len(test_dataloader), len(train_dataloader.dataset)]
  break
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

class NeuralNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.flatten = nn.Flatten()
    self.linear_relu_stack = nn.Sequential(nn.Linear(6 * 6, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, C))
  def forward(self, x):
    x = self.flatten(x)
    logits = self.linear_relu_stack(x)
    return logits

model = det_init(NeuralNetwork()).to(device)
out["repr"] = repr(model)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

def train(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  model.train()
  log = []
  for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if batch % 2 == 0:
      loss, current = loss.item(), (batch + 1) * len(X)
      log.append([loss, current, size])
  return log

def test(dataloader, model, loss_fn):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  model.eval()
  test_loss, correct = 0, 0
  with torch.no_grad():
    for X, y in dataloader:
      X, y = X.to(device), y.to(device)
      pred = model(X)
      test_loss += loss_fn(pred, y).item()
      correct += (pred.argmax(1) == y).type(torch.float).sum().item()
  test_loss /= num_batches
  correct /= size
  return [test_loss, correct]

history = []
for t in range(6):
  history.append([train(train_dataloader, model, loss_fn, optimizer), test(test_dataloader, model, loss_fn)])
out["history"] = history
out["improved"] = history[-1][1][0] < history[0][1][0]

# predictions with the trained model, read after the weights were updated
model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
  pred = model(x.unsqueeze(0))
  out["prediction"] = [pred, int(pred[0].argmax(0)), int(y), F.softmax(pred, dim=1).sum().item()]
# values computed before an optimizer step keep their pre-step value when read afterwards
model.train()
X, y = next(iter(train_dataloader))
pred = model(X); loss = loss_fn(pred, y)
optimizer.zero_grad(); loss.backward(); optimizer.step()
acc_before = (pred.argmax(1) == y).float().mean().item()
out["read_after_step"] = [loss.item(), pred.detach(), acc_before, loss_fn(model(X), y).item() < loss.item()]

# shuffling DataLoaders, drop_last, custom collate, random_split, iterating twice
loader = DataLoader(TensorDataset(torch.arange(10).float().view(10, 1), torch.arange(10)), batch_size=4, shuffle=True, drop_last=True)
seen = sorted(int(v) for xb, yb in loader for v in yb)
epoch2 = sorted(int(v) for xb, yb in loader for v in yb)
pairs_ok = all(float(xb[i, 0]) == float(yb[i]) for xb, yb in loader for i in range(len(yb)))
a, b = random_split(TensorDataset(torch.arange(10)), [7, 3])
collated = next(iter(DataLoader([{"x": torch.ones(2) * i, "n": i} for i in range(4)], batch_size=4)))
custom = next(iter(DataLoader(list(range(6)), batch_size=3, collate_fn=lambda items: sum(items))))
out["loader"] = [len(seen), len(set(seen)), len(epoch2), len(loader), pairs_ok, len(a), len(b), sorted(int(a[i][0]) for i in range(7)) != list(range(7)) or True,
                 collated["x"], collated["n"], custom, [list(xb.shape) for xb, in DataLoader(TensorDataset(torch.zeros(10, 3)), batch_size=4)],
                 len(DataLoader(TensorDataset(torch.zeros(10, 3)), batch_size=4, drop_last=True))]
emit(**out)
