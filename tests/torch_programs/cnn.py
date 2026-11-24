# The convolutional network from the PyTorch "Training a Classifier" (CIFAR-10) tutorial, on synthetic 3x32x32 images.
out = {}
class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)
  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

net = det_init(Net(), scale=0.08)
inputs_all, labels_all = det(16, 3, 32, 32, seed=1), det_int(16, high=10, seed=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
running_loss, losses = 0.0, []
for epoch in range(3):
  for i in range(0, 16, 8):
    inputs, labels = inputs_all[i:i + 8], labels_all[i:i + 8]
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    losses.append(loss.item())
out["losses"] = losses
out["running_loss"] = running_loss
correct, total = 0, 0
with torch.no_grad():
  outputs = net(inputs_all)
  _, predicted = torch.max(outputs, 1)
  total += labels_all.size(0)
  correct += (predicted == labels_all).sum().item()
out["eval"] = [outputs, predicted, total, correct]
out["params"] = [sum(p.numel() for p in net.parameters()), [n for n, _ in net.named_parameters()], [list(p.shape) for p in net.parameters()]]

# a small ResNet-style block with BatchNorm, trained for a few steps (train-mode statistics, then eval)
class Block(nn.Module):
  def __init__(self, c):
    super().__init__()
    self.conv1, self.bn1 = nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c)
    self.conv2, self.bn2 = nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c)
  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out += x
    return F.relu(out)
class TinyResNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.stem = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(inplace=True))
    self.layers = nn.ModuleList([Block(8), Block(8)])
    self.head = nn.Linear(8, 5)
  def forward(self, x):
    x = self.stem(x)
    for layer in self.layers: x = layer(x)
    return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))
model = det_init(TinyResNet(), scale=0.15)
xs, ys = det(8, 3, 8, 8, seed=5), det_int(8, high=5, seed=6)
opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
model.train()
res_losses = []
for step in range(5):
  opt.zero_grad()
  loss = F.cross_entropy(model(xs), ys)
  loss.backward()
  opt.step()
  res_losses.append(loss.item())
model.eval()
with torch.no_grad(): eval_logits = model(xs)
out["resnet"] = [res_losses, eval_logits, model.stem[1].running_mean.clone(), model.layers[1].bn2.running_var.clone(), int(model.stem[1].num_batches_tracked),
                 len(model.state_dict()), [k for k in model.state_dict().keys()][:8]]
emit(**out)
