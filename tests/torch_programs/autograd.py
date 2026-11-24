out = {}
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()
out["flags"] = [x.requires_grad, x.is_leaf, y.requires_grad, y.is_leaf, x.grad is None, y.grad_fn is not None, x.grad_fn is None,
                (x * 2).detach().requires_grad, torch.ones(2).requires_grad]
y.backward()
out["grad1"] = x.grad.clone()
# gradients accumulate until zeroed
z = (x * 3).sum(); z.backward()
out["grad_accumulated"] = x.grad.clone()
x.grad.zero_()
(x.exp()).sum().backward()
out["grad_after_zero"] = x.grad
x.grad = None
# reading a value before backward keeps the graph
w = torch.tensor([[0.5, -1.0], [2.0, 0.25]], requires_grad=True)
b = torch.tensor([0.1, -0.2], requires_grad=True)
inp = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
h = torch.tanh(inp @ w + b)
loss = (h ** 2).mean()
out["loss_before_backward"] = loss.item()
out["h_before_backward"] = h
loss.backward()
out["w_grad"], out["b_grad"] = w.grad, b.grad
try: loss.backward(); out["second_backward"] = "no error"
except RuntimeError as e: out["second_backward"] = "RuntimeError"
# retain_graph allows a second backward, and grads accumulate
v = torch.tensor([2.0, 3.0], requires_grad=True)
l2 = (v ** 3).sum()
l2.backward(retain_graph=True); l2.backward()
out["retain_graph"] = v.grad.clone()
# no_grad and detach
with torch.no_grad():
  ng = v * 2
out["no_grad"] = [ng.requires_grad, torch.is_grad_enabled(), ng]
@torch.no_grad()
def helper(t): return t * 3
out["no_grad_decorator"] = [helper(v).requires_grad, torch.is_grad_enabled()]
with torch.set_grad_enabled(False): out["set_grad_enabled"] = (v * 2).requires_grad
v.grad = None
dl = (v * v.detach()).sum(); dl.backward()
out["detach"] = v.grad
# torch.autograd.grad, higher-order derivatives
t = torch.tensor(1.5, requires_grad=True)
f = t ** 3 + 2 * t
(g1,) = torch.autograd.grad(f, t, create_graph=True)
(g2,) = torch.autograd.grad(g1, t)
out["higher_order"] = [g1, g2, t.grad is None]
p, q = torch.tensor([1.0, 2.0], requires_grad=True), torch.tensor([3.0, 4.0], requires_grad=True)
gp, gq = torch.autograd.grad((p * q).sum() + (p ** 2).sum(), [p, q])
out["grad_multi"] = [gp, gq]
unused = torch.autograd.grad((p * 2).sum(), [p, q], allow_unused=True)
out["grad_unused"] = [unused[0], unused[1] is None]
# non-scalar backward with an explicit gradient; scalar backward on shape (1,)
r = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
(r * r).backward(torch.tensor([1.0, 0.1, 0.01]))
out["vector_backward"] = r.grad.clone()
r.grad = None
(r.sum().view(1) * 2).backward()
out["shape1_backward"] = r.grad
try: (r * 2).backward(); out["nonscalar_error"] = "no error"
except RuntimeError: out["nonscalar_error"] = "RuntimeError"
# leaves: in-place updates need no_grad, parameters not in the graph keep grad None
leaf = torch.tensor([1.0, 2.0], requires_grad=True)
other = torch.tensor([5.0], requires_grad=True)
try: leaf += 1; out["leaf_inplace"] = "no error"
except RuntimeError: out["leaf_inplace"] = "RuntimeError"
(leaf * 2).sum().backward()
out["unused_leaf_grad_is_none"] = other.grad is None
with torch.no_grad():
  leaf -= 0.5 * leaf.grad
  leaf.grad.zero_()
out["manual_update"] = [leaf.detach().clone(), leaf.grad.clone(), leaf.requires_grad, leaf.is_leaf]
(leaf ** 2).sum().backward()
out["grad_after_update"] = leaf.grad
# gradients through common operations
a = torch.tensor([[0.3, -1.2, 2.0], [1.5, 0.7, -0.4]], requires_grad=True)
ops = {
  "relu": lambda u: u.relu().sum(), "sigmoid": lambda u: torch.sigmoid(u).sum(), "softmax": lambda u: (F.softmax(u, dim=1) * torch.tensor([1.0, 2.0, 3.0])).sum(),
  "log_softmax": lambda u: F.log_softmax(u, dim=1)[:, 0].sum(), "max": lambda u: u.max(dim=1).values.sum(), "mean": lambda u: u.mean(),
  "index": lambda u: u[0, 1] * 3 + u[1].sum(), "matmul": lambda u: (u @ u.T).sum(), "abs": lambda u: u.abs().sum(), "pow": lambda u: (u ** 3).sum(),
  "div": lambda u: (1 / (u ** 2 + 1)).sum(), "cat": lambda u: torch.cat([u, u * 2], dim=0).sum(), "where": lambda u: torch.where(u > 0, u * 2, u * 3).sum(),
  "clamp": lambda u: u.clamp(-1, 1).sum(), "norm": lambda u: u.norm(), "std": lambda u: u.std(), "gather": lambda u: u.gather(1, torch.tensor([[2], [0]])).sum(),
  "transpose": lambda u: (u.T * torch.arange(6.0).view(3, 2)).sum(), "exp_log": lambda u: (u.exp() + 1).log().sum(), "sum_dim": lambda u: (u.sum(0) ** 2).sum(),
  "masked_fill": lambda u: u.masked_fill(u < 0, 0.0).sum(), "setitem": lambda u: _setitem(u), "var": lambda u: u.var(dim=1).sum(),
}
def _setitem(u):
  buf = torch.zeros(2, 3)
  buf[0] = u[1] * 2
  buf[1, 1:] = u[0, :2]
  return (buf ** 2).sum()
grads = {}
for name, fn in ops.items():
  a.grad = None
  fn(a).backward()
  grads[name] = a.grad.clone()
out["op_grads"] = grads
# .data shares storage and does not track gradients
dd = torch.tensor([1.0, 2.0], requires_grad=True)
dd.data.mul_(2)
dd.data[0] = 10.0
out["data"] = [dd, dd.data.requires_grad, dd.requires_grad]
# requires_grad_ and freezing
fr = torch.ones(2); fr.requires_grad_()
(fr * 3).sum().backward()
out["requires_grad_"] = fr.grad
fr.requires_grad_(False)
out["frozen"] = [(fr * 3).requires_grad, fr.requires_grad]
# two leaves created from equal constants are independent
z1, z2 = torch.zeros(2, requires_grad=True), torch.zeros(2, requires_grad=True)
s1, s2 = torch.tensor(1.0, requires_grad=True), torch.tensor(1.0, requires_grad=True)
(z1 * 2 + z2 * 5 + s1 * 3 + s2 * 7).sum().backward()
out["independent_leaves"] = [z1.grad, z2.grad, s1.grad, s2.grad]
emit(**out)
