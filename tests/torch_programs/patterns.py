# Training patterns from common tutorials: GAN-style alternating updates, target networks and EMA, policy gradients,
# fine-tuning with frozen layers, hooks, and the nn.Module introspection API.
import copy
from collections import OrderedDict
out = {}

# ---- GAN-style training (DCGAN tutorial structure): D and G steps share the generator output
netG = det_init(nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 6), nn.Tanh()), seed=10)
netD = det_init(nn.Sequential(nn.Linear(6, 16), nn.LeakyReLU(0.2), nn.Linear(16, 1), nn.Sigmoid()), seed=20)
criterion = nn.BCELoss()
optimizerD = torch.optim.Adam(netD.parameters(), lr=0.01, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=0.01, betas=(0.5, 0.999))
real_label, fake_label = 1.0, 0.0
log = []
for i in range(4):
  real = det(8, 6, seed=30 + i) * 0.5 + 0.5
  noise = det(8, 4, seed=40 + i)
  netD.zero_grad()
  label = torch.full((8,), real_label, dtype=torch.float)
  output = netD(real).view(-1)
  errD_real = criterion(output, label)
  errD_real.backward()
  D_x = output.mean().item()
  fake = netG(noise)
  label.fill_(fake_label)
  output = netD(fake.detach()).view(-1)
  errD_fake = criterion(output, label)
  errD_fake.backward()
  D_G_z1 = output.mean().item()
  errD = errD_real + errD_fake
  optimizerD.step()
  netG.zero_grad()
  label.fill_(real_label)
  output = netD(fake).view(-1)       # the generator graph is used again after D was updated
  errG = criterion(output, label)
  errG.backward()
  D_G_z2 = output.mean().item()
  optimizerG.step()
  log.append([errD.item(), errG.item(), D_x, D_G_z1, D_G_z2])
out["gan"] = [log, netG[0].weight.detach().clone(), netD[2].bias.detach().clone(), netG[0].weight.grad is not None]

# ---- DQN-style target network: deepcopy, load_state_dict, soft updates, and TD targets under no_grad
policy = det_init(nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 2)), seed=50)
target = copy.deepcopy(policy)
target2 = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 2)); target2.load_state_dict(policy.state_dict())
opt = torch.optim.AdamW(policy.parameters(), lr=0.01, amsgrad=True)
dq = []
for i in range(4):
  state, next_state = det(8, 3, seed=60 + i), det(8, 3, seed=70 + i)
  action = det_int(8, 1, high=2, seed=80 + i); reward = det(8, seed=90 + i)
  q = policy(state).gather(1, action)
  with torch.no_grad(): next_q = target(next_state).max(1).values
  expected = reward + 0.9 * next_q
  loss = nn.SmoothL1Loss()(q, expected.unsqueeze(1))
  opt.zero_grad(); loss.backward()
  torch.nn.utils.clip_grad_value_(policy.parameters(), 100)
  opt.step()
  tsd, psd = target.state_dict(), policy.state_dict()
  for key in psd: tsd[key] = psd[key] * 0.1 + tsd[key] * 0.9
  target.load_state_dict(tsd)
  with torch.no_grad():   # EMA written with in-place operations on .data
    for pt, pp in zip(target2.parameters(), policy.parameters()): pt.data.mul_(0.9).add_(pp.data, alpha=0.1)
  dq.append(loss.item())
out["dqn"] = [dq, target[0].weight.detach().clone(), target2[0].weight.detach().clone(), policy[2].bias.detach().clone(),
              torch.allclose(target[0].weight, target2[0].weight, atol=1e-6), target[0].weight is not policy[0].weight]

# ---- REINFORCE-style policy gradient with torch.distributions (actions are fixed so results are comparable)
from torch.distributions import Categorical, Normal
pol = det_init(nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 3)), seed=110)
popt = torch.optim.Adam(pol.parameters(), lr=0.02)
pg = []
for ep in range(3):
  states = det(6, 4, seed=120 + ep)
  dist = Categorical(logits=pol(states))
  actions = det_int(6, high=3, seed=130 + ep)
  returns = torch.tensor([1.0, 0.5, -0.2, 0.8, 0.1, -1.0])
  returns = (returns - returns.mean()) / (returns.std() + 1e-8)
  loss = -(dist.log_prob(actions) * returns).sum() - 0.01 * dist.entropy().sum()
  popt.zero_grad(); loss.backward(); popt.step()
  pg.append(loss.item())
sample = Categorical(probs=torch.tensor([0.0, 1.0, 0.0])).sample()
nd = Normal(torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.5]))
out["policy_gradient"] = [pg, pol[2].weight.detach().clone(), int(sample), list(Categorical(logits=torch.zeros(5, 3)).sample().shape), nd.log_prob(torch.tensor([0.5, 0.5])),
                          nd.entropy(), list(nd.sample((4,)).shape), Categorical(probs=torch.tensor([0.2, 0.3, 0.5])).entropy(), nd.rsample().requires_grad]

# ---- fine-tuning: freeze the backbone, replace the head, optimize only what requires grad
backbone = det_init(nn.Sequential(OrderedDict([("fc1", nn.Linear(5, 8)), ("act", nn.ReLU()), ("fc2", nn.Linear(8, 4))])), seed=140)
for param in backbone.parameters(): param.requires_grad = False
backbone.fc2 = det_init(nn.Linear(8, 2), seed=150)
trainable = [p for p in backbone.parameters() if p.requires_grad]
fopt = torch.optim.SGD(filter(lambda p: p.requires_grad, backbone.parameters()), lr=0.1, momentum=0.9)
frozen_before = backbone.fc1.weight.clone()
xs, ys = det(10, 5, seed=160), det_int(10, high=2, seed=161)
ft = []
for _ in range(4):
  fopt.zero_grad(); l = F.cross_entropy(backbone(xs), ys); l.backward(); fopt.step(); ft.append(l.item())
out["finetune"] = [ft, len(trainable), torch.equal(frozen_before, backbone.fc1.weight), backbone.fc1.weight.grad is None, backbone.fc2.weight.detach().clone(),
                   [n for n, p in backbone.named_parameters() if p.requires_grad], sum(p.numel() for p in backbone.parameters() if p.requires_grad)]

# ---- hooks, module introspection, containers
class Inner(nn.Module):
  def __init__(self):
    super().__init__()
    self.lin = nn.Linear(2, 2)
    self.gain = nn.Parameter(torch.ones(1))
    self.register_buffer("count", torch.zeros(1))
  def forward(self, x): return self.lin(x) * self.gain
  def extra_repr(self): return "custom=True"
class Outer(nn.Module):
  def __init__(self):
    super().__init__()
    self.blocks = nn.ModuleList([Inner(), Inner()])
    self.heads = nn.ModuleDict({"a": nn.Linear(2, 1), "b": nn.Linear(2, 3, bias=False)})
    self.extra = nn.ParameterList([nn.Parameter(torch.zeros(2)), nn.Parameter(torch.ones(3))])
    self.drop = nn.Dropout(0.1)
    self.act = nn.Sequential(nn.ReLU(), nn.Sequential(nn.Tanh()))
    self.plain_list = [nn.Linear(2, 2)]     # not registered
  def forward(self, x, head="a"):
    for b in self.blocks: x = b(x)
    return self.heads[head](self.act(x))
net = Outer()
acts = {}
handle = net.blocks[0].register_forward_hook(lambda mod, inp, outp: acts.__setitem__("shape", (list(inp[0].shape), list(outp.shape))))
net(torch.zeros(3, 2)); handle.remove(); acts2 = dict(acts); acts.clear(); net(torch.zeros(5, 2))
counted = []
net.apply(lambda mod: counted.append(type(mod).__name__))
net.eval()
modes_eval = [net.training, net.drop.training, net.blocks[1].lin.training]
net.train()
out["modules"] = [repr(net), [n for n, _ in net.named_parameters()], [n for n, _ in net.named_buffers()], [n for n, _ in net.named_children()],
                  [n for n, _ in net.named_modules()], list(net.state_dict().keys()), len(list(net.parameters())), len(list(net.modules())), len(list(net.children())),
                  acts2, acts, counted, modes_eval, net.training, len(net.blocks), list(net.heads.keys()), "a" in net.heads, isinstance(net.blocks[0], Inner),
                  list(net(torch.zeros(4, 2), head="b").shape), hasattr(net, "blocks"), hasattr(net, "nope"), net.get_submodule("blocks.1.lin").in_features,
                  repr(nn.Sequential(nn.Conv2d(1, 2, 3, padding=1), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(8, 2), nn.Dropout(0.5),
                                     nn.Embedding(3, 2), nn.GELU(), nn.LeakyReLU(0.1), nn.Softmax(dim=1), nn.Identity())),
                  [type(m).__name__ for m in nn.Sequential(nn.ReLU(), nn.Tanh(), nn.Sigmoid())[1:]], len(nn.Sequential(nn.ReLU(), nn.Tanh())),
                  repr(nn.CrossEntropyLoss()), repr(nn.MSELoss()), repr(nn.Linear(2, 3, bias=False)), type(net.blocks[0].gain).__name__, isinstance(net.blocks[0].gain, torch.Tensor),
                  net.blocks[0].gain.requires_grad, net.blocks[0].count.requires_grad, repr(nn.Parameter(torch.tensor([1.0, 2.0])))]
try:
  class Bad(nn.Module):
    def __init__(self): self.lin = nn.Linear(1, 1)
  Bad(); out["missing_super_init"] = "no error"
except AttributeError: out["missing_super_init"] = "AttributeError"
emit(**out)
