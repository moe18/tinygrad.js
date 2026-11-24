# Short learner-style snippets: the kind of PyTorch code written in tutorials and exercises (linear regression from scratch,
# positional encodings, a VAE loss, a character LSTM, k-means, and one-line API uses). Each result must equal PyTorch's.
R = {}
def chk(name, fn):
  try: R[name] = fn()
  except Exception as e: R[name] = f"ERROR {type(e).__name__}: {str(e)[:150]}"

# 1. linear regression from scratch with w.grad.zero_()
def linreg():
  X = det(100, 3, seed=1); true_w = torch.tensor([2.0, -3.4, 1.0]); yv = X @ true_w + 4.2
  w = torch.zeros(3, requires_grad=True); b = torch.zeros(1, requires_grad=True)
  for epoch in range(30):
    l = ((X @ w + b - yv) ** 2 / 2).mean()
    l.backward()
    with torch.no_grad():
      w -= 0.3 * w.grad; b -= 0.3 * b.grad
      w.grad.zero_(); b.grad.zero_()
  return [round(v, 2) for v in w.tolist()] + [round(b.item(), 2)]
chk("linreg", linreg)

# 2. positional encoding with strided slice assignment
def posenc():
  d_model, max_len = 8, 6
  pe = torch.zeros(max_len, d_model)
  position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
  div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
  pe[:, 0::2] = torch.sin(position * div_term)
  pe[:, 1::2] = torch.cos(position * div_term)
  pe = pe.unsqueeze(0).transpose(0, 1)
  return [list(pe.shape), round(pe.sum().item(), 3), round(pe[3, 0, 1].item(), 4)]
chk("posenc", posenc)

# 3. VAE loss with reparameterization
def vae():
  enc = nn.Linear(6, 4); dec = nn.Linear(2, 6)
  with torch.no_grad():
    for i, p in enumerate(list(enc.parameters()) + list(dec.parameters())): p.copy_(det(*p.shape, seed=20 + i) * 0.3)
  x = torch.sigmoid(det(5, 6, seed=3))
  h = enc(x); mu, logvar = h[:, :2], h[:, 2:]
  std = torch.exp(0.5 * logvar); eps = torch.ones_like(std) * 0.1
  z = mu + eps * std
  recon = torch.sigmoid(dec(z))
  BCE = F.binary_cross_entropy(recon, x, reduction="sum")
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  loss = BCE + KLD
  loss.backward()
  return [round(loss.item(), 4), round(enc.weight.grad.abs().sum().item(), 4)]
chk("vae", vae)

# 4. char-level LSTM step with hidden state detach
def charlstm():
  torch.manual_seed(0)
  emb, lstm, head = nn.Embedding(12, 8), nn.LSTM(8, 16, batch_first=True), nn.Linear(16, 12)
  params = list(emb.parameters()) + list(lstm.parameters()) + list(head.parameters())
  opt = torch.optim.Adam(params, lr=0.05)
  seq = torch.tensor([[i % 12 for i in range(j, j + 9)] for j in range(4)])
  hidden = None; first = last = None
  for it in range(15):
    out, hidden = lstm(emb(seq[:, :-1]), hidden)
    hidden = tuple(h.detach() for h in hidden)
    loss = F.cross_entropy(head(out).reshape(-1, 12), seq[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 5.0); opt.step()
    first = first if first is not None else loss.item(); last = loss.item()
  return last < first * 0.5
chk("charlstm", charlstm)

# 5. k-means with argmin and index-based updates
def kmeans():
  pts = torch.cat([det(20, 2, seed=5) * 0.2 + 2, det(20, 2, seed=6) * 0.2 - 2])
  cent = pts[[0, 39]].clone()
  for _ in range(5):
    d = torch.cdist(pts, cent); assign = d.argmin(dim=1)
    cent = torch.stack([pts[assign == k].mean(dim=0) for k in range(2)])
  return [[round(v, 1) for v in c] for c in cent.tolist()], assign.bincount().tolist()
chk("kmeans", kmeans)

# 6. misc API probes
chk("eye_index", lambda: torch.eye(3)[torch.tensor([2, 0])].tolist())
chk("inference_mode", lambda: [torch.inference_mode()(lambda: (torch.ones(2, requires_grad=True) * 2).requires_grad)(), torch.is_grad_enabled()])
chk("generator", lambda: list(torch.randn(3, generator=torch.Generator().manual_seed(0)).shape))
chk("np_roundtrip", lambda: torch.from_numpy(np.arange(6).reshape(2, 3)).float().mean(dim=0).detach().cpu().numpy().tolist())
chk("acc", lambda: torch.sum(torch.argmax(torch.tensor([[0.1, 0.9], [0.8, 0.2]]), dim=1) == torch.tensor([1, 1])).item())
chk("triu_mask", lambda: torch.triu(torch.ones(3, 3), diagonal=1).bool().tolist())
chk("masked_fill_", lambda: torch.zeros(2, 2).masked_fill_(torch.eye(2).bool(), float("-inf")).exp().tolist())
chk("seq_star", lambda: len(nn.Sequential(*[nn.Linear(2, 2) for _ in range(3)])))
chk("view_size0", lambda: list(det(4, 2, 3).view(det(4, 2, 3).size(0), -1).shape))
chk("xavier_apply", lambda: (lambda m: (m.apply(lambda mod: nn.init.xavier_uniform_(mod.weight) if isinstance(mod, nn.Linear) else None), float(m[0].weight.abs().max()) < 1.5)[1])(nn.Sequential(nn.Linear(4, 4), nn.ReLU())))
chk("max_tuple", lambda: [v.tolist() for v in torch.max(torch.tensor([[1.0, 5.0], [7.0, 2.0]]), dim=1)])
chk("zeros_device", lambda: str(torch.zeros(2, device=torch.device("cpu")).sum().item()))
chk("loss_list", lambda: round(torch.stack([torch.tensor(1.0), torch.tensor(2.0)]).mean().item(), 2))
chk("clone_detach", lambda: torch.tensor([1.0], requires_grad=True).clone().detach().requires_grad)
chk("int_div", lambda: [(torch.tensor(7) // 2).item(), (torch.tensor(7) % 3).item(), (torch.tensor(7) / 2).item(), torch.tensor([1, 2]).float().dtype == torch.float32])
chk("where_cond", lambda: torch.where(torch.tensor([1.0, -1.0]) > 0)[0].tolist())
chk("tensor_bool_index_assign", lambda: (lambda t: (t.__setitem__(t < 0, 0.0), t.tolist())[1])(torch.tensor([1.0, -2.0, 3.0])))
chk("std_mean_normalize", lambda: [round(v, 3) for v in ((lambda t: (t - t.mean(dim=0)) / t.std(dim=0))(torch.tensor([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]]))).flatten().tolist()])
chk("softmax_T", lambda: [round(v, 3) for v in F.softmax(torch.tensor([1.0, 2.0, 3.0]) / 0.5, dim=-1).tolist()])
chk("embedding_bag_like", lambda: nn.Embedding(5, 2)(torch.tensor([[0, 1], [2, 3]])).mean(dim=1).shape == torch.Size([2, 2]))
chk("param_count", lambda: sum(p.numel() for p in nn.Conv2d(3, 8, 3).parameters() if p.requires_grad))
chk("mse_item_fmt", lambda: f"{F.mse_loss(torch.tensor([1.0, 2.0]), torch.tensor([1.5, 2.5])):.4f}")
chk("no_grad_numpy", lambda: (torch.ones(2, requires_grad=True) * 2).detach().numpy().tolist())
chk("numpy_requires_grad_error", lambda: (torch.ones(2, requires_grad=True) * 2).numpy().tolist())
chk("chained_cmp", lambda: ((torch.arange(6) > 1) & (torch.arange(6) < 4)).sum().item())
chk("scatter_onehot", lambda: torch.zeros(2, 3).scatter_(1, torch.tensor([[2], [0]]), 1.0).tolist())
chk("log_softmax_nll", lambda: round(F.nll_loss(F.log_softmax(torch.tensor([[1.0, 2.0, 0.5]]), dim=1), torch.tensor([1])).item(), 4))
chk("cosine_sched_print", lambda: [round(g, 5) for g in (lambda o, s: [(o.step(), s.step(), s.get_last_lr()[0])[2] for _ in range(3)])(*(lambda o: (o, torch.optim.lr_scheduler.CosineAnnealingLR(o, 3)))(torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)))])
chk("module_dict_iter", lambda: sorted(k for k, v in nn.ModuleDict({"a": nn.ReLU(), "b": nn.Tanh()}).items()))
chk("register_buffer_to", lambda: (lambda m: (m.register_buffer("mask", torch.ones(2)), m.to("cpu"), list(m.state_dict().keys()))[2])(nn.Linear(1, 1)))
chk("tensor_iter_unpack", lambda: [int(a + b) for a, b in torch.tensor([[1, 2], [3, 4]])])
chk("argsort_desc", lambda: torch.argsort(torch.tensor([0.3, 0.9, 0.1]), descending=True).tolist())
chk("multinomial_shape", lambda: list(torch.multinomial(torch.tensor([0.1, 0.9]), 4, replacement=True).shape))
chk("randint_range", lambda: bool(((torch.randint(2, 5, (50,)) >= 2) & (torch.randint(2, 5, (50,)) < 5)).all()))
chk("dropout_train_eval", lambda: (lambda d: (d.eval(), d(torch.ones(4)).tolist())[1])(nn.Dropout(0.9)))
chk("grad_none_after_zero", lambda: (lambda m, o: (m(torch.ones(1, 2)).sum().backward(), o.zero_grad(), m.weight.grad is None)[2])(*(lambda m: (m, torch.optim.SGD(m.parameters(), lr=0.1)))(nn.Linear(2, 1))))
chk("einsum_attn", lambda: list(torch.einsum("bhqd,bhkd->bhqk", det(2, 2, 3, 4), det(2, 2, 3, 4, seed=1)).shape))
chk("pad_sequence", lambda: nn.utils.rnn.pad_sequence([torch.ones(2), torch.ones(3)], batch_first=True).tolist())
chk("flatten_start", lambda: list(torch.flatten(det(2, 3, 4), start_dim=1).shape))
chk("type_float_tensor", lambda: torch.tensor([1, 2]).type(torch.FloatTensor).dtype == torch.float32)
chk("item_int_type", lambda: type(torch.tensor([3]).sum().item()).__name__)
chk("dataset_len_idx", lambda: (lambda ds: [len(ds), ds[1][0].tolist(), int(ds[1][1])])(torch.utils.data.TensorDataset(torch.arange(6.0).view(3, 2), torch.tensor([0, 1, 0]))))
chk("no_bias_conv_bn_fuse_shapes", lambda: [list(p.shape) for p in nn.Sequential(nn.Conv2d(1, 2, 3, bias=False), nn.BatchNorm2d(2)).parameters()])
chk("set_default_printoptions", lambda: (torch.set_printoptions(precision=2), str(torch.tensor([1.23456])), torch.set_printoptions(precision=4))[1])
emit(**R)
