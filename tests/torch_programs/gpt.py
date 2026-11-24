# A nanoGPT-style character transformer (model code follows karpathy/nanoGPT's model.py), trained for a few steps.
import inspect
from dataclasses import dataclass
out = {}

class LayerNorm(nn.Module):
  """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
  def __init__(self, ndim, bias):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(ndim))
    self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
  def forward(self, input):
    return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
  def __init__(self, config, flash):
    super().__init__()
    assert config.n_embd % config.n_head == 0
    self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
    self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
    self.attn_dropout = nn.Dropout(config.dropout)
    self.resid_dropout = nn.Dropout(config.dropout)
    self.n_head = config.n_head
    self.n_embd = config.n_embd
    self.dropout = config.dropout
    self.flash = flash and hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    if not self.flash:
      self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
  def forward(self, x):
    B, T, C = x.size()
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    if self.flash:
      y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
    else:
      att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
      att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
      att = F.softmax(att, dim=-1)
      att = self.attn_dropout(att)
      y = att @ v
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    y = self.resid_dropout(self.c_proj(y))
    return y

class MLP(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
    self.gelu = nn.GELU()
    self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
    self.dropout = nn.Dropout(config.dropout)
  def forward(self, x):
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.c_proj(x)
    x = self.dropout(x)
    return x

class Block(nn.Module):
  def __init__(self, config, flash):
    super().__init__()
    self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
    self.attn = CausalSelfAttention(config, flash)
    self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
    self.mlp = MLP(config)
  def forward(self, x):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x

@dataclass
class GPTConfig:
  block_size: int = 16
  vocab_size: int = 24
  n_layer: int = 2
  n_head: int = 2
  n_embd: int = 32
  dropout: float = 0.0
  bias: bool = True

class GPT(nn.Module):
  def __init__(self, config, flash=True):
    super().__init__()
    self.config = config
    self.transformer = nn.ModuleDict(dict(
      wte = nn.Embedding(config.vocab_size, config.n_embd),
      wpe = nn.Embedding(config.block_size, config.n_embd),
      drop = nn.Dropout(config.dropout),
      h = nn.ModuleList([Block(config, flash) for _ in range(config.n_layer)]),
      ln_f = LayerNorm(config.n_embd, bias=config.bias),
    ))
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    self.transformer.wte.weight = self.lm_head.weight   # weight tying
    self.apply(self._init_weights)
    for pn, p in self.named_parameters():
      if pn.endswith('c_proj.weight'):
        torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
  def get_num_params(self, non_embedding=True):
    n_params = sum(p.numel() for p in self.parameters())
    if non_embedding: n_params -= self.transformer.wpe.weight.numel()
    return n_params
  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None: torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
  def forward(self, idx, targets=None):
    device = idx.device
    b, t = idx.size()
    assert t <= self.config.block_size
    pos = torch.arange(0, t, dtype=torch.long, device=device)
    tok_emb = self.transformer.wte(idx)
    pos_emb = self.transformer.wpe(pos)
    x = self.transformer.drop(tok_emb + pos_emb)
    for block in self.transformer.h: x = block(x)
    x = self.transformer.ln_f(x)
    if targets is not None:
      logits = self.lm_head(x)
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    else:
      logits = self.lm_head(x[:, [-1], :])
      loss = None
    return logits, loss
  def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
    param_dict = {pn: p for pn, p in self.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [{'params': decay_params, 'weight_decay': weight_decay}, {'params': nodecay_params, 'weight_decay': 0.0}]
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    return optimizer, len(decay_params), len(nodecay_params)
  @torch.no_grad()
  def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, greedy=False):
    for _ in range(max_new_tokens):
      idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
      logits, _ = self(idx_cond)
      logits = logits[:, -1, :] / temperature
      if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.argmax(probs, dim=-1, keepdim=True) if greedy else torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx

# data: a periodic character stream, batches cut at fixed offsets
text = ("the quick brown fox jumps over the lazy dog. " * 8)
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
config = GPTConfig(vocab_size=len(chars))
def get_batch(it, batch_size=8):
  ix = [(it * 7 + j * 13) % (len(data) - config.block_size - 1) for j in range(batch_size)]
  x = torch.stack([data[i:i + config.block_size] for i in ix])
  y = torch.stack([data[i + 1:i + 1 + config.block_size] for i in ix])
  return x, y
def get_lr(it, warmup_iters=3, lr_decay_iters=12, learning_rate=3e-3, min_lr=3e-4):
  if it < warmup_iters: return learning_rate * (it + 1) / (warmup_iters + 1)
  if it > lr_decay_iters: return min_lr
  decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
  coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
  return min_lr + coeff * (learning_rate - min_lr)

results = {}
for name, flash in (("flash", True), ("manual", False)):
  model = GPT(config, flash=flash)
  # same deterministic weights in both frameworks (the tied embedding/lm_head weight is one parameter)
  det_init(model, scale=0.05)
  optimizer, n_decay, n_nodecay = model.configure_optimizers(0.1, 3e-3, (0.9, 0.95), "cpu")
  model.train()
  losses = []
  for it in range(12):
    lr = get_lr(it)
    for param_group in optimizer.param_groups: param_group['lr'] = lr
    X, Y = get_batch(it)
    logits, loss = model(X, Y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    losses.append(loss.item())
  model.eval()
  start = data[:4].view(1, 4)
  generated = model.generate(start, max_new_tokens=6, top_k=5, greedy=True)
  sampled = model.generate(start, max_new_tokens=5, temperature=0.8, top_k=10)
  logits_eval, _ = model(data[:8].view(1, 8))
  results[name] = [losses, generated, logits_eval, [n_decay, n_nodecay, model.get_num_params(), len(list(model.parameters())), len(model.state_dict())],
                   list(sampled.shape), bool((sampled >= 0).all() and (sampled < config.vocab_size).all()), model.transformer.wte.weight is model.lm_head.weight,
                   losses[-1] < losses[0]]
out.update(results)
out["state_dict_keys"] = list(GPT(config, flash=False).state_dict().keys())
emit(**out)
