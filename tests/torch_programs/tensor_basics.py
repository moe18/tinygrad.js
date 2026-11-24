out = {}
a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = torch.tensor([10.0, 20.0, 30.0])
i = torch.tensor([[1, 2, 3], [4, 5, 6]])
out["dtypes"] = [str(a.dtype), str(torch.tensor([1.5]).dtype), str(torch.tensor([True]).dtype), str((i / 2).dtype), str((i * 1.5).dtype),
                 str(torch.zeros(2).dtype), str(a.bool().dtype), str((a > 2).dtype), str(a.float().dtype)]
out["int_kinds"] = [torch.tensor([1, 2]).dtype.is_floating_point, i.dtype.is_floating_point, (i + i).is_floating_point(), a.argmax().is_floating_point()]
out["shape"] = [list(a.shape), a.size(0), a.size(-1), a.dim(), a.ndim, a.numel(), len(a), list(a.T.shape), list(torch.Size([2, 3]))]
out["arith"] = [a + b, a - b, a * b, a / b, a ** 2, 2 ** a, -a, a % 2, a // 2, 1 - a, 1 / a, a @ a.T, a.matmul(b), torch.matmul(a, b),
                i + 1, i * 2, i // 2, i % 2, i / 2, a + i, a.add(b, alpha=2), a.sub(1), a.mul(b), a.div(2), torch.div(i, 2, rounding_mode="floor")]
out["unary"] = [a.exp(), a.log(), a.sqrt(), a.sin(), a.cos(), a.tanh(), torch.sigmoid(a), a.abs(), (-a).relu(), a.reciprocal(), a.square(),
                torch.exp(a), torch.log(a), torch.sqrt(a), (a / 4).floor(), (a / 4).ceil(), (a / 4).round(), torch.sign(a - 3), a.rsqrt(),
                torch.log1p(a), torch.expm1(a / 10), torch.erf(a / 4), a.neg(), torch.clamp(a, 2, 5), a.clamp(min=3), a.clamp(max=3), a.clip(2.5, 4.5)]
out["reduce"] = [a.sum(), a.sum(0), a.sum(dim=1, keepdim=True), a.mean(), a.mean(1), a.prod(), a.prod(1), a.max(), a.min(), a.max(1).values,
                 a.max(dim=1).indices, a.min(0)[0], a.min(0)[1], a.argmax(), a.argmax(dim=1), a.argmin(dim=0), a.std(), a.var(), a.std(0),
                 a.var(1, keepdim=True), a.norm(), a.norm(dim=1), a.amax(1), a.amin(0), a.cumsum(1), a.cumsum(dim=0), a.logsumexp(1),
                 (a > 2).all(), (a > 2).any(), (a > 2).any(1), (a > 2).sum(), torch.sum(a, dim=(0, 1)), i.sum(), i.sum(1), torch.mean(a, 0),
                 torch.max(a, 1)[0], torch.max(a), torch.maximum(a, b / 10), torch.minimum(a, b / 10), a.median(), torch.std(a, dim=1, unbiased=False)]
out["compare"] = [a > 2, a >= 2, a < 2, a <= 2, a == 2, a != 2, torch.eq(a, 2.0), a.gt(3), (a > 2) & (a < 5), (a < 2) | (a > 5), ~(a > 2),
                  torch.where(a > 3, a, torch.zeros_like(a)), torch.where(a > 3, 1.0, 0.0), a.masked_fill(a > 3, -1.0), torch.isnan(a / 0 - a / 0),
                  torch.allclose(a, a + 1e-9), torch.equal(a, a.clone()), a.isfinite(), torch.logical_and(a > 1, a < 4)]
out["index"] = [a[0], a[1, 2], a[:, 1], a[0, 1:], a[-1], a[:, -1], a[..., 0], a[None].shape, a[:, None, :].shape, a[1:, :2], a[::1, ::2],
                a[torch.tensor([1, 0])], a[:, torch.tensor([2, 0])], a[torch.tensor([0, 1]), torch.tensor([2, 0])], a[a > 3], a[[1, 0]],
                a[0][1], b[torch.tensor([True, False, True])], a[torch.tensor([True, False])], a.flatten()[2:5], a[1, [0, 2]]]
out["shapes"] = [a.view(3, 2), a.view(-1), a.reshape(1, 6), a.reshape(-1, 2), a.flatten(), a.unsqueeze(0).shape, a.unsqueeze(-1).shape,
                 a.unsqueeze(1).squeeze(1).shape, a.T, a.t(), a.transpose(0, 1), a.permute(1, 0), a.expand(2, 2, 3).shape, b.expand(2, 3),
                 b.repeat(2), b.repeat(2, 1), a.repeat(1, 2), torch.cat([a, a]), torch.cat([a, a], dim=1), torch.stack([b, b]),
                 torch.stack([b, b], dim=1), a.chunk(3, dim=1)[1], a.split(2, dim=1)[0], a.split([1, 2], dim=1)[1], torch.flatten(a, 0),
                 a.flip(0), a.flip(1), a.roll(1, 1), a.tril(), a.triu(), torch.tril(torch.ones(3, 3)), a.contiguous().view(6),
                 a.unbind(0)[1], b.view(1, 3).squeeze(), a.narrow(1, 1, 2), a.movedim(0, 1), a.view(2, 3, 1).squeeze(-1), a.unflatten(1, (3, 1)).shape,
                 torch.squeeze(a.view(1, 2, 3, 1)).shape, a.swapaxes(0, 1), a.view_as(a), a.repeat_interleave(2, dim=0)]
out["create"] = [torch.zeros(2, 3), torch.ones(2), torch.full((2, 2), 7.0), torch.arange(5), torch.arange(1, 10, 3), torch.arange(0, 1, 0.25),
                 torch.linspace(0, 1, 5), torch.eye(3), torch.zeros_like(a), torch.ones_like(i), torch.full_like(a, 2.0), torch.empty(2, 2).shape,
                 torch.tensor(3.5), torch.tensor([[1, 2], [3, 4]]), torch.from_numpy(np.arange(4, dtype=np.float32)), torch.tensor(np.ones((2, 2), dtype=np.float32)),
                 torch.arange(6).view(2, 3), torch.arange(6.0).reshape(3, 2), torch.zeros((2, 3)).shape, torch.ones([2, 1]).shape,
                 torch.as_tensor([1.0, 2.0]), torch.rand(3, 4).shape, torch.randn(2, 5).shape, torch.randint(0, 5, (3,)).shape, torch.randperm(5).sort().values,
                 torch.arange(3).float(), torch.tensor([1, 2]).to(torch.float32), a.long(), a.int(), a.to(torch.int32), i.float(), i.double().float()]
s = torch.tensor([3.0, 1.0, 2.0, 5.0, 4.0])
out["sort"] = [s.sort().values, s.sort().indices, s.sort(descending=True)[0], torch.sort(s)[1], s.argsort(), s.topk(2).values, s.topk(2).indices,
               torch.topk(s, 3, largest=False)[0], a.sort(dim=1, descending=True).values, a.topk(2, dim=1).indices]
idx = torch.tensor([[0, 0], [1, 0]])
out["gather"] = [a.gather(1, idx), torch.gather(a, 1, idx), torch.zeros(2, 3).scatter(1, idx, 1.0), torch.zeros(2, 3).scatter(1, idx, a[:, :2]),
                 torch.zeros(3).scatter_add(0, torch.tensor([0, 0, 2]), torch.tensor([1.0, 2.0, 3.0])), a.index_select(1, torch.tensor([2, 0])),
                 F.one_hot(torch.tensor([0, 2, 1]), 3), F.one_hot(torch.tensor([0, 2, 1])), a.masked_select(a > 4), torch.nonzero(a > 4), (a > 4).nonzero(as_tuple=True)[1]]
out["einsum"] = [torch.einsum("ij,j->i", a, b), torch.einsum("ij,kj->ik", a, a), torch.einsum("ii->", torch.eye(3)), torch.outer(b, b), torch.dot(b, b),
                 torch.bmm(a.view(1, 2, 3), a.T.reshape(1, 3, 2)), torch.mm(a, a.T), torch.trace(torch.eye(3) * 2), torch.cdist(a, a), torch.tensordot(a, a.T, dims=1),
                 torch.meshgrid(torch.arange(2), torch.arange(3), indexing="ij")[1], torch.cross(torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]), dim=0)]
# in-place operations and aliasing through the same object
c = a.clone(); c += 1; c *= 2; c -= b; c /= 2
d = torch.zeros(2, 3); d[0] = 1.0; d[1, 2] = 5.0; d[:, 0] = torch.tensor([7.0, 8.0]); d[d > 6] = -1.0
e = torch.arange(6.0); e[1:3] = 0; e[-1] += 10; e[torch.tensor([0, 3])] = 9.0
f = torch.ones(3); f.add_(2).mul_(3); f.sub_(1); f.div_(2); f.clamp_(0, 3.5)
g = torch.zeros(4); g.fill_(2.0); h = torch.empty(2, 2).zero_(); k = torch.zeros(3).copy_(b)
m = torch.zeros(2, 2)
for r in range(2):
  for col in range(2): m[r, col] = r * 2 + col
n = torch.ones(2, 3); n.masked_fill_(a > 3, 0.0); o = a.clone(); o[0, :] = o[1, :]; p = a.clone(); p.t_(); q = a.clone().exp_().log_()
out["inplace"] = [c, d, e, f, g, h, k, m, n, o, p, q]
out["python"] = [a[0, 0].item(), float(a.sum()), int(i.max()), bool((a > 0).all()), a.tolist(), i[0].tolist(), a.numpy().sum().item(), [float(x) for x in b],
                 [r.sum().item() for r in a], 3.0 in b, len(b), "%.2f" % a.mean(), f"{a.max():.1f}", f"{a.sum().item():.3f}", a.mean().item() > 3,
                 b.numpy().tolist(), np.asarray(a).shape == (2, 3), isinstance(a, torch.Tensor), torch.is_tensor(a), a.is_floating_point()]
out["repr"] = [repr(a), repr(b), str(i), repr(torch.tensor(3.5)), repr(torch.tensor([1.5, 2.25])), repr(a > 2), repr(torch.zeros(2, 2)), repr(a.shape),
               repr(torch.tensor([0.1234567, 12.5])), repr(torch.arange(3)), repr(torch.tensor(7)), repr(torch.tensor([])), repr(torch.tensor([1e-5, 1.0])),
               repr(torch.tensor([[1.5], [-2.0]])), repr(torch.tensor([1.0, 2.0], requires_grad=True))]
emit(**out)
