# Forward results of the nn layers and losses with deterministic parameters, compared with PyTorch.
out = {}
x2 = det(4, 6, seed=1); img = det(2, 3, 8, 8, seed=2); seq = det(2, 5, 8, seed=3)
def layer(m, *inputs, seed=50):
  det_init(m, seed=seed)
  m.eval()
  return m(*inputs)
out["linear"] = [layer(nn.Linear(6, 3), x2), layer(nn.Linear(6, 3, bias=False), x2), layer(nn.Bilinear(6, 6, 2), x2, x2 * 0.5), nn.Identity()(x2), nn.Flatten()(img).shape,
                 nn.Flatten(0, 1)(img).shape, nn.Unflatten(1, (2, 3))(x2).shape]
out["conv"] = [layer(nn.Conv2d(3, 4, 3), img), layer(nn.Conv2d(3, 4, 3, padding=1), img), layer(nn.Conv2d(3, 4, 3, stride=2, padding=1), img),
               layer(nn.Conv2d(3, 6, 3, groups=3), img), layer(nn.Conv2d(3, 4, 3, dilation=2), img), layer(nn.Conv2d(3, 4, (1, 3), bias=False), img),
               layer(nn.Conv2d(3, 4, 3, padding="same"), img), layer(nn.Conv1d(5, 4, 3, padding=1), seq), layer(nn.Conv1d(5, 4, 2, stride=2), seq),
               layer(nn.ConvTranspose2d(3, 2, 2, stride=2), img), layer(nn.ConvTranspose2d(3, 2, 3, stride=2, padding=1, output_padding=1), img),
               F.conv2d(img, det(2, 3, 3, 3, seed=7), padding=1)]
out["pool"] = [nn.MaxPool2d(2)(img), nn.MaxPool2d(3, stride=2, padding=1)(img), nn.AvgPool2d(2)(img), nn.AvgPool2d(3, stride=1, padding=1)(img),
               nn.AdaptiveAvgPool2d(1)(img), nn.AdaptiveAvgPool2d((4, 4))(img), nn.AdaptiveAvgPool2d((1, 1))(img).flatten(1), F.max_pool2d(img, 2, 2),
               nn.MaxPool1d(2)(seq), nn.AvgPool1d(2)(seq), F.adaptive_avg_pool2d(img, 2), nn.AdaptiveMaxPool2d(1)(img), F.avg_pool2d(img, 4),
               F.interpolate(img, scale_factor=2, mode="nearest"), F.interpolate(img, size=(4, 4), mode="bilinear", align_corners=False), nn.Upsample(scale_factor=2)(img).shape,
               F.pad(img, (1, 2, 0, 1)), F.pad(x2, (1, 1), value=3.0), nn.ZeroPad2d(1)(img).shape, nn.PixelShuffle(1)(img).shape]
# normalization: training mode updates running statistics, eval mode uses them
bn = det_init(nn.BatchNorm2d(3)); bn.train()
tr1 = bn(img); tr2 = bn(img * 2 + 1)
stats = [bn.running_mean.clone(), bn.running_var.clone(), int(bn.num_batches_tracked)]
bn.eval(); ev = bn(img)
bn1 = det_init(nn.BatchNorm1d(6)); bn1.train(); b1 = bn1(x2); bn1.eval(); b1e = bn1(x2)
bn_nt = nn.BatchNorm1d(6, affine=False, track_running_stats=False)
out["batchnorm"] = [tr1, tr2, stats, ev, b1, b1e, bn1.running_mean.clone(), bn1.running_var.clone(), bn_nt(x2), sorted(bn.state_dict().keys())]
out["norm"] = [layer(nn.LayerNorm(8), seq), layer(nn.LayerNorm([5, 8]), seq), layer(nn.LayerNorm(8, elementwise_affine=False), seq), layer(nn.GroupNorm(3, 3), img),
               layer(nn.GroupNorm(1, 3), img), F.layer_norm(seq, (8,)), F.normalize(x2, dim=1), F.normalize(x2, p=1, dim=0), F.cosine_similarity(x2, x2.flip(0), dim=1),
               nn.InstanceNorm2d(3)(img), F.batch_norm(x2, torch.zeros(6), torch.ones(6)), F.pairwise_distance(x2, x2.flip(0))]
emb = det_init(nn.Embedding(10, 4))
ids = torch.tensor([[1, 2, 9], [0, 0, 3]])
out["embedding"] = [emb(ids), emb(torch.tensor(5)), F.embedding(ids, emb.weight), nn.Embedding.from_pretrained(det(5, 3, seed=9))(torch.tensor([4, 0])),
                    nn.Embedding(6, 2, padding_idx=0).weight[0], list(nn.Embedding(7, 3).weight.shape)]
acts = [nn.ReLU(), nn.ReLU6(), nn.LeakyReLU(), nn.LeakyReLU(0.2), nn.ELU(), nn.SELU(), nn.GELU(), nn.GELU(approximate="tanh"), nn.SiLU(), nn.Mish(), nn.Sigmoid(), nn.Tanh(),
        nn.Hardtanh(), nn.Hardsigmoid(), nn.Hardswish(), nn.Softplus(), nn.Softplus(beta=2.0), nn.Softsign(), nn.LogSigmoid(), nn.Softmax(dim=1), nn.LogSoftmax(dim=-1),
        nn.Softmin(dim=1), nn.CELU(), nn.Threshold(0.1, 20.0), nn.PReLU()]
out["activations"] = [a(x2 * 2) for a in acts]
out["functional_acts"] = [F.relu(x2), F.gelu(x2), F.silu(x2), F.softmax(x2, dim=-1), F.log_softmax(x2, dim=1), F.leaky_relu(x2, 0.1), F.elu(x2), F.softplus(x2),
                          F.sigmoid(x2), F.tanh(x2), torch.relu(x2), torch.softmax(x2, 1), x2.softmax(0), F.hardtanh(x2, -0.5, 0.5), F.relu6(x2 * 4), F.logsigmoid(x2), F.mish(x2)]
out["dropout"] = [nn.Dropout(0.5).eval()(x2), F.dropout(x2, 0.5, training=False), nn.Dropout(0.0)(x2), nn.Dropout(0.5)(torch.ones(1000)).mean().item() > 0.5,
                  sorted(set(nn.Dropout(0.5)(torch.ones(1000)).tolist())), nn.Dropout2d(0.5).eval()(img).shape, nn.Dropout(0.25)(torch.ones(4000)).sum().item() > 3000]
logits = det(5, 4, seed=11); target = torch.tensor([0, 3, 1, 1, 2]); probs = torch.softmax(det(5, 4, seed=12), 1)
wts = torch.tensor([1.0, 2.0, 0.5, 1.5]); seg_logits = det(2, 4, 3, 3, seed=13); seg_target = det_int(2, 3, 3, high=4, seed=14)
yb = (det(5, 4, seed=15) > 0).float(); pb = torch.sigmoid(logits)
out["losses"] = {
  "ce": F.cross_entropy(logits, target), "ce_module": nn.CrossEntropyLoss()(logits, target), "ce_sum": F.cross_entropy(logits, target, reduction="sum"),
  "ce_none": F.cross_entropy(logits, target, reduction="none"), "ce_weight": F.cross_entropy(logits, target, weight=wts),
  "ce_ignore": F.cross_entropy(logits, torch.tensor([0, -100, 1, -100, 2])), "ce_ignore_idx": F.cross_entropy(logits, torch.tensor([0, -1, 1, 1, -1]), ignore_index=-1),
  "ce_smooth": F.cross_entropy(logits, target, label_smoothing=0.1), "ce_smooth_weight": nn.CrossEntropyLoss(weight=wts, label_smoothing=0.2)(logits, target),
  "ce_probs": F.cross_entropy(logits, probs), "ce_nd": F.cross_entropy(seg_logits, seg_target), "ce_nd_none": F.cross_entropy(seg_logits, seg_target, reduction="none"),
  "ce_unbatched": F.cross_entropy(logits[0], target[0]), "nll": F.nll_loss(F.log_softmax(logits, 1), target), "nll_module": nn.NLLLoss(weight=wts)(F.log_softmax(logits, 1), target),
  "mse": F.mse_loss(logits, probs), "mse_sum": nn.MSELoss(reduction="sum")(logits, probs), "mse_none": F.mse_loss(logits, probs, reduction="none"),
  "l1": F.l1_loss(logits, probs), "l1_module": nn.L1Loss()(logits, probs), "smooth_l1": F.smooth_l1_loss(logits, probs), "smooth_l1_beta": nn.SmoothL1Loss(beta=0.5)(logits, probs),
  "huber": F.huber_loss(logits, probs, delta=0.7), "huber_module": nn.HuberLoss()(logits, probs), "bce": F.binary_cross_entropy(pb, yb), "bce_module": nn.BCELoss()(pb, yb),
  "bce_logits": F.binary_cross_entropy_with_logits(logits, yb), "bce_logits_module": nn.BCEWithLogitsLoss()(logits, yb),
  "bce_logits_pos": nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0, 2.0, 3.0, 0.5]))(logits, yb), "bce_weight": F.binary_cross_entropy(pb, yb, weight=wts),
  "kl": F.kl_div(F.log_softmax(logits, 1), probs, reduction="batchmean"), "kl_module": nn.KLDivLoss(reduction="sum")(F.log_softmax(logits, 1), probs),
  "margin": nn.MarginRankingLoss(margin=0.5)(logits[:, 0], logits[:, 1], torch.tensor([1.0, -1.0, 1.0, 1.0, -1.0])), "cosine_emb": nn.CosineEmbeddingLoss()(logits, probs, torch.tensor([1, -1, 1, -1, 1])),
  "triplet": nn.TripletMarginLoss()(logits, probs, logits.flip(0)), "poisson": nn.PoissonNLLLoss()(logits, yb), "gaussian": nn.GaussianNLLLoss()(logits, probs, torch.ones(5, 4) * 0.5),
}
# gradients of the main losses with respect to the logits
lg = logits.clone().requires_grad_(True)
grads = {}
for name, fn in {"ce": lambda z: F.cross_entropy(z, target), "ce_smooth": lambda z: F.cross_entropy(z, target, label_smoothing=0.1, weight=wts),
                 "bce_logits": lambda z: F.binary_cross_entropy_with_logits(z, yb), "mse": lambda z: F.mse_loss(z, probs), "huber": lambda z: F.huber_loss(z, probs),
                 "nll": lambda z: F.nll_loss(F.log_softmax(z, 1), target, ignore_index=1)}.items():
  lg.grad = None; fn(lg).backward(); grads[name] = lg.grad.clone()
out["loss_grads"] = grads
# attention and transformer layers (dropout disabled)
q = det(2, 4, 8, seed=20)
mha = det_init(nn.MultiheadAttention(8, 2, batch_first=True)); mha.eval()
attn_out, attn_w = mha(q, q, q)
causal = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
attn_masked, _ = mha(q, q, q, attn_mask=causal, need_weights=False)
pad_mask = torch.tensor([[False, False, False, True], [False, False, True, True]])
attn_pad, w_pad = mha(q, q, q, key_padding_mask=pad_mask)
mha_sf = det_init(nn.MultiheadAttention(8, 4)); mha_sf.eval()
enc_layer = det_init(nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0, batch_first=True)); enc_layer.eval()
enc_pre = det_init(nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0, batch_first=True, norm_first=True, activation="gelu")); enc_pre.eval()
encoder = det_init(nn.TransformerEncoder(nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0, batch_first=True), num_layers=2, enable_nested_tensor=False)); encoder.eval()
dec_layer = det_init(nn.TransformerDecoderLayer(8, 2, dim_feedforward=16, dropout=0.0, batch_first=True)); dec_layer.eval()
qh = q.view(2, 4, 2, 4).transpose(1, 2)
out["attention"] = [attn_out, attn_w, attn_masked, attn_pad, w_pad, mha_sf(q.transpose(0, 1), q.transpose(0, 1), q.transpose(0, 1))[0], enc_layer(q), enc_pre(q),
                    enc_layer(q, src_mask=causal), encoder(q), encoder(q, src_key_padding_mask=pad_mask), dec_layer(q, q * 0.5),
                    F.scaled_dot_product_attention(qh, qh, qh), F.scaled_dot_product_attention(qh, qh, qh, is_causal=True),
                    F.scaled_dot_product_attention(qh, qh, qh, attn_mask=~causal), sorted(mha.state_dict().keys()), sorted(enc_layer.state_dict().keys()),
                    nn.Transformer.generate_square_subsequent_mask(3)]
# recurrent layers
rs = det(3, 5, 4, seed=30)
def rnn(m, x, hx=None):
  det_init(m, seed=60)
  o, h = m(x) if hx is None else m(x, hx)
  return [o, list(h) if isinstance(h, tuple) else h]
out["recurrent"] = [rnn(nn.RNN(4, 6, batch_first=True), rs), rnn(nn.RNN(4, 6), rs), rnn(nn.RNN(4, 6, nonlinearity="relu", bias=False, batch_first=True), rs),
                    rnn(nn.GRU(4, 6, batch_first=True), rs), rnn(nn.GRU(4, 6, num_layers=2, batch_first=True), rs), rnn(nn.LSTM(4, 6, batch_first=True), rs),
                    rnn(nn.LSTM(4, 6, num_layers=2, bidirectional=True, batch_first=True), rs), rnn(nn.GRU(4, 6, bidirectional=True), rs),
                    rnn(nn.LSTM(4, 6, batch_first=True), rs, (det(1, 3, 6, seed=31), det(1, 3, 6, seed=32))), rnn(nn.GRU(4, 6), rs[0]),
                    det_init(nn.LSTMCell(4, 6))(rs[:, 0])[0], det_init(nn.GRUCell(4, 6))(rs[:, 0]), det_init(nn.RNNCell(4, 6))(rs[:, 0]),
                    sorted(nn.LSTM(4, 6, num_layers=2, bidirectional=True).state_dict().keys()), [list(p.shape) for p in nn.GRU(4, 6).parameters()]]
# default initialization: PyTorch's distributions (bounds and spread), checked statistically
lin = nn.Linear(400, 300); conv = nn.Conv2d(16, 32, 3); e2 = nn.Embedding(200, 64)
bound = 1 / math.sqrt(400)
out["init"] = [lin.weight.min().item() >= -bound - 1e-6, lin.weight.max().item() <= bound + 1e-6, abs(lin.weight.std().item() - bound / math.sqrt(3)) < 0.003,
               abs(lin.weight.mean().item()) < 0.002, lin.bias.abs().max().item() <= bound + 1e-6, abs(conv.weight.std().item() - (1 / math.sqrt(144)) / math.sqrt(3)) < 0.003,
               abs(e2.weight.std().item() - 1.0) < 0.03, abs(e2.weight.mean().item()) < 0.03, nn.LayerNorm(4).weight.tolist(), nn.BatchNorm1d(3).bias.tolist(),
               abs(nn.init.xavier_uniform_(torch.empty(200, 300)).std().item() - math.sqrt(2 / 500)) < 0.003,
               abs(nn.init.kaiming_normal_(torch.empty(300, 200), nonlinearity="relu").std().item() - math.sqrt(2 / 200)) < 0.004,
               abs(nn.init.normal_(torch.empty(300, 300), mean=1.0, std=0.02).mean().item() - 1.0) < 0.001, nn.init.constant_(torch.empty(2, 2), 0.3).tolist(),
               nn.init.trunc_normal_(torch.empty(1000), std=0.5, a=-1.0, b=1.0).abs().max().item() <= 1.0, nn.init.zeros_(torch.empty(3)).tolist(),
               torch.allclose(nn.init.orthogonal_(torch.empty(4, 4)) @ nn.init.orthogonal_(torch.empty(4, 4)).T * 0 + torch.eye(4), torch.eye(4)),
               nn.init.calculate_gain("relu"), nn.init.calculate_gain("leaky_relu", 0.2), nn.init.calculate_gain("tanh")]
emit(**out)
