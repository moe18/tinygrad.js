"""torch.nn.functional on top of tinygrad."""
from __future__ import annotations
import math, builtins as _b
from tinygrad import Tensor as _TG, dtypes as _td
from .._tensor import _mk, _wrap, _long, no_grad

def _pair(x, n=2): return tuple(x) if isinstance(x, (tuple, list)) else (x,) * n
def _maybe_inplace(input, out, inplace): return input._inplace(out) if inplace else out
def _reduce(loss, reduction):
  if reduction == "mean": return loss.mean()
  if reduction == "sum": return loss.sum()
  if reduction == "none": return loss
  raise ValueError(f"{reduction} is not a valid value for reduction")

# ---- activations
def relu(input, inplace=False): return _maybe_inplace(input, _wrap(input._t.relu(), input), inplace)
def relu6(input, inplace=False): return _maybe_inplace(input, input.clamp(0, 6), inplace)
def leaky_relu(input, negative_slope=0.01, inplace=False): return _maybe_inplace(input, _wrap(input._t.leaky_relu(negative_slope), input), inplace)
def elu(input, alpha=1.0, inplace=False): return _maybe_inplace(input, _wrap(input._t.elu(alpha), input), inplace)
def selu(input, inplace=False): return _maybe_inplace(input, _wrap(input._t.selu(), input), inplace)
def celu(input, alpha=1.0, inplace=False): return _maybe_inplace(input, _wrap(input._t.celu(alpha), input), inplace)
def gelu(input, approximate="none"):
  if approximate == "tanh": return _wrap(input._t.gelu(), input)
  return _wrap(0.5 * input._t * (1 + (input._t / math.sqrt(2.0)).erf()), input)
def silu(input, inplace=False): return _maybe_inplace(input, _wrap(input._t * input._t.sigmoid(), input), inplace)
swish = silu
def mish(input, inplace=False): return _maybe_inplace(input, _wrap(input._t.mish(), input), inplace)
def sigmoid(input): return input.sigmoid()
def tanh(input): return input.tanh()
def hardtanh(input, min_val=-1.0, max_val=1.0, inplace=False): return _maybe_inplace(input, input.clamp(min_val, max_val), inplace)
def hardsigmoid(input, inplace=False): return _maybe_inplace(input, (input / 6 + 0.5).clamp(0, 1), inplace)
def hardswish(input, inplace=False): return _maybe_inplace(input, input * (input + 3).clamp(0, 6) / 6, inplace)
def softplus(input, beta=1.0, threshold=20.0):
  x = input._t
  return _wrap((x * beta > threshold).where(x, (1 + (x * beta).clip(max_=threshold).exp()).log() / beta), input)
def softsign(input): return input / (1 + input.abs())
def logsigmoid(input): return -softplus(-input)
def threshold(input, threshold, value, inplace=False): return _maybe_inplace(input, _wrap((input._t > threshold).where(input._t, value), input), inplace)
def prelu(input, weight):
  w = weight if weight.numel() == 1 or input.ndim < 2 else weight.reshape(1, -1, *([1] * (input.ndim - 2)))
  return input.clamp(min=0) + w * input.clamp(max=0)
def _softmax_dim(ndim): return 0 if ndim in (0, 1, 3) else 1
def softmax(input, dim=None, _stacklevel=3, dtype=None): return input.softmax(_softmax_dim(input.ndim) if dim is None else dim)
def log_softmax(input, dim=None, _stacklevel=3, dtype=None): return input.log_softmax(_softmax_dim(input.ndim) if dim is None else dim)
def softmin(input, dim=None): return softmax(-input, dim)
def gumbel_softmax(logits, tau=1.0, hard=False, eps=1e-10, dim=-1):
  g = -(-(_mk(_TG.rand(*logits.shape)).clamp(min=1e-20)).log()).log()
  y = ((logits + g) / tau).softmax(dim)
  if not hard: return y
  onehot = one_hot(y.argmax(dim), logits.shape[dim]).to(y.dtype).movedim(-1, dim)
  return onehot - y.detach() + y

# ---- linear / dropout / embedding
def linear(input, weight, bias=None):
  return _wrap(input._t.linear(weight._t.transpose() if weight._t.ndim == 2 else weight._t, None if bias is None else bias._t), input, weight, bias)
def bilinear(input1, input2, weight, bias=None):
  out = (input1.unsqueeze(-2).unsqueeze(-1) * weight * input2.unsqueeze(-2).unsqueeze(-2)).sum((-1, -2))
  return out if bias is None else out + bias
def dropout(input, p=0.5, training=True, inplace=False):
  if not training or p == 0: return input
  if p >= 1: return _maybe_inplace(input, input * 0, inplace)
  mask = (_TG.rand(*input.shape) >= p).cast(input._t.dtype) / (1 - p)
  return _maybe_inplace(input, _wrap(input._t * mask, input), inplace)
def _channel_dropout(input, p, training, inplace, nd):
  if not training or p == 0: return input
  mask = (_TG.rand(*input.shape[:input.ndim - nd], *([1] * nd)) >= p).cast(input._t.dtype) / (1 - p)
  return _maybe_inplace(input, _wrap(input._t * mask, input), inplace)
def dropout1d(input, p=0.5, training=True, inplace=False): return _channel_dropout(input, p, training, inplace, 1)
def dropout2d(input, p=0.5, training=True, inplace=False): return _channel_dropout(input, p, training, inplace, 2)
def dropout3d(input, p=0.5, training=True, inplace=False): return _channel_dropout(input, p, training, inplace, 3)
alpha_dropout = feature_alpha_dropout = dropout
def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
  return _wrap(weight._t[input._t], weight)
def one_hot(tensor, num_classes=-1):
  if num_classes < 0: num_classes = _b.int(tensor.max().item()) + 1
  return _mk(tensor._t.one_hot(num_classes).cast(_long()))

# ---- convolution / pooling
def _conv_padding(padding, kernel, dilation, nd):
  if padding == "valid": return 0
  if padding == "same":
    total = [d * (k - 1) for d, k in zip(_pair(dilation, nd), kernel)]
    if all(t % 2 == 0 for t in total): return tuple(t // 2 for t in total)
    return tuple(x for t in reversed(total) for x in (t // 2, t - t // 2))  # flat (left, right, top, bottom ...) form
  return tuple(padding) if isinstance(padding, (tuple, list)) else padding
def _conv(input, weight, bias, stride, padding, dilation, groups, nd):
  unbatched = input.ndim == nd + 1
  x = input._t.unsqueeze(0) if unbatched else input._t
  out = x.conv2d(weight._t, None if bias is None else bias._t, groups=groups, stride=stride, dilation=dilation,
                 padding=_conv_padding(padding, weight.shape[2:], dilation, nd))
  return _wrap(out.squeeze(0) if unbatched else out, input, weight, bias)
def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1): return _conv(input, weight, bias, stride, padding, dilation, groups, 1)
def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1): return _conv(input, weight, bias, stride, padding, dilation, groups, 2)
def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1): return _conv(input, weight, bias, stride, padding, dilation, groups, 3)
def _conv_transpose(input, weight, bias, stride, padding, output_padding, groups, dilation, nd):
  unbatched = input.ndim == nd + 1
  x = input._t.unsqueeze(0) if unbatched else input._t
  out = x.conv_transpose2d(weight._t, None if bias is None else bias._t, groups=groups, stride=stride, dilation=dilation,
                           padding=tuple(padding) if isinstance(padding, (tuple, list)) else padding, output_padding=output_padding)
  return _wrap(out.squeeze(0) if unbatched else out, input, weight, bias)
def conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
  return _conv_transpose(input, weight, bias, stride, padding, output_padding, groups, dilation, 1)
def conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
  return _conv_transpose(input, weight, bias, stride, padding, output_padding, groups, dilation, 2)

def _pool(input, name, nd, kernel_size, stride, padding, **kw):
  unbatched = input.ndim == nd + 1
  x = input._t.unsqueeze(0) if unbatched else input._t
  out = getattr(x, name)(_pair(kernel_size, nd), stride if stride is None else _pair(stride, nd), padding=_pair(padding, nd) if isinstance(padding, (tuple, list)) else padding, **kw)
  if isinstance(out, tuple):
    return _wrap(out[0].squeeze(0) if unbatched else out[0], input), _mk((out[1].squeeze(0) if unbatched else out[1]).cast(_long()))
  return _wrap(out.squeeze(0) if unbatched else out, input)
def max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
  return _pool(input, "max_pool2d", 1, kernel_size, stride, padding, dilation=dilation, ceil_mode=ceil_mode, return_indices=return_indices)
def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
  return _pool(input, "max_pool2d", 2, kernel_size, stride, padding, dilation=dilation, ceil_mode=ceil_mode, return_indices=return_indices)
def max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
  return _pool(input, "max_pool2d", 3, kernel_size, stride, padding, dilation=dilation, ceil_mode=ceil_mode, return_indices=return_indices)
def avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
  return _pool(input, "avg_pool2d", 1, kernel_size, stride, padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad)
def avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
  return _pool(input, "avg_pool2d", 2, kernel_size, stride, padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad)
def avg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
  return _pool(input, "avg_pool2d", 3, kernel_size, stride, padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad)
def _adaptive(input, output_size, nd, kind):
  out = tuple(input.shape[-nd + i] if o is None else o for i, o in enumerate(_pair(output_size, nd)))
  insz = tuple(input.shape[-nd:])
  if all(o == 1 for o in out):
    dims = tuple(range(-nd, 0))
    return input.mean(dims, keepdim=True) if kind == "avg" else input.amax(dims, keepdim=True)
  if out == insz: return input
  if any(i % o for i, o in zip(insz, out)):
    raise NotImplementedError(f"adaptive_{kind}_pool: input size {insz} must be a multiple of output size {out} in the tinygrad.js torch shim")
  k = tuple(i // o for i, o in zip(insz, out))
  return _pool(input, "avg_pool2d" if kind == "avg" else "max_pool2d", nd, k, k, 0)
def adaptive_avg_pool1d(input, output_size): return _adaptive(input, output_size, 1, "avg")
def adaptive_avg_pool2d(input, output_size): return _adaptive(input, output_size, 2, "avg")
def adaptive_max_pool1d(input, output_size, return_indices=False): return _adaptive(input, output_size, 1, "max")
def adaptive_max_pool2d(input, output_size, return_indices=False): return _adaptive(input, output_size, 2, "max")

def pad(input, pad, mode="constant", value=None): return _wrap(input._t.pad(tuple(pad), mode=mode, value=0.0 if value is None else value), input)
def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None, recompute_scale_factor=None, antialias=False):
  nd = input.ndim - 2
  if size is None:
    sf = _pair(scale_factor, nd)
    size = tuple(_b.int(math.floor(s * f)) for s, f in zip(input.shape[2:], sf))
  mode = {"bilinear": "linear", "trilinear": "linear", "bicubic": "linear"}.get(mode, mode)
  return _wrap(input._t.interpolate(_pair(size, nd), mode=mode, align_corners=_b.bool(align_corners)), input)
upsample = interpolate
def pixel_shuffle(input, r):
  n, c, h, w = input.shape[-4:] if input.ndim >= 4 else (1, *input.shape)
  x = input.reshape(-1, c // (r * r), r, r, h, w).permute(0, 1, 4, 2, 5, 3).reshape(-1, c // (r * r), h * r, w * r)
  return x if input.ndim >= 4 else x.squeeze(0)
def unfold(input, kernel_size, dilation=1, padding=0, stride=1):
  k, d, p, s = _pair(kernel_size), _pair(dilation), _pair(padding), _pair(stride)
  x = _wrap(input._t.pad((p[1], p[1], p[0], p[0])), input) if any(p) else input
  if d != (1, 1): raise NotImplementedError("F.unfold with dilation is not supported by the tinygrad.js torch shim")
  n, c = x.shape[0], x.shape[1]
  patches = _wrap(x._t.unfold(2, k[0], s[0]).unfold(3, k[1], s[1]), x)   # (N, C, oh, ow, kh, kw)
  return patches.permute(0, 1, 4, 5, 2, 3).reshape(n, c * k[0] * k[1], -1)

# ---- normalization
def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05):
  axes = tuple(i for i in range(input.ndim) if i != 1)
  shape = [1, -1] + [1] * (input.ndim - 2)
  if training or running_mean is None:
    mean = input.mean(axes)
    var = ((input - mean.reshape(shape)) ** 2).mean(axes)
    if training and running_mean is not None:
      n = input.numel() // input.shape[1]
      with no_grad():
        m = momentum
        running_mean._inplace(running_mean * (1 - m) + mean.detach() * m)
        running_var._inplace(running_var * (1 - m) + var.detach() * (m * n / _b.max(n - 1, 1)))
  else: mean, var = running_mean, running_var
  out = (input - mean.reshape(shape)) / (var.reshape(shape) + eps).sqrt()
  if weight is not None: out = out * weight.reshape(shape)
  if bias is not None: out = out + bias.reshape(shape)
  return out
def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
  n = 1 if isinstance(normalized_shape, _b.int) else len(normalized_shape)
  out = _wrap(input._t.layernorm(axis=tuple(range(-n, 0)), eps=eps), input)
  if weight is not None: out = out * weight
  if bias is not None: out = out + bias
  return out
def group_norm(input, num_groups, weight=None, bias=None, eps=1e-05):
  shape = [1, -1] + [1] * (input.ndim - 2)
  x = input._t.reshape(input.shape[0], num_groups, -1).layernorm(axis=-1, eps=eps).reshape(tuple(input.shape))
  out = _wrap(x, input)
  if weight is not None: out = out * weight.reshape(shape)
  if bias is not None: out = out + bias.reshape(shape)
  return out
def instance_norm(input, running_mean=None, running_var=None, weight=None, bias=None, use_input_stats=True, momentum=0.1, eps=1e-05):
  shape = [1, -1] + [1] * (input.ndim - 2)
  out = _wrap(input._t.layernorm(axis=tuple(range(2, input.ndim)), eps=eps), input)
  if weight is not None: out = out * weight.reshape(shape)
  if bias is not None: out = out + bias.reshape(shape)
  return out
def normalize(input, p=2.0, dim=1, eps=1e-12): return input / input.norm(p, dim, keepdim=True).clamp(min=eps)
def cosine_similarity(x1, x2, dim=1, eps=1e-08):
  return (x1 * x2).sum(dim) / (x1.norm(2, dim).clamp(min=eps) * x2.norm(2, dim).clamp(min=eps))
def pairwise_distance(x1, x2, p=2.0, eps=1e-06, keepdim=False): return (x1 - x2 + eps).norm(p, -1, keepdim)

# ---- losses
def _nll(logp, target, weight, ignore_index, reduction, label_smoothing=0.0):
  """Negative log likelihood of class-index targets. ``logp``: wrapper of log-probabilities with the class dim at 1."""
  unbatched = logp.ndim == 1
  if unbatched: logp, target = logp.unsqueeze(0), target.unsqueeze(0)
  c = logp.shape[1]
  cshape = [1, c] + [1] * (logp.ndim - 2)
  t = target._t
  onehot = (t.unsqueeze(1) == _TG.arange(c).reshape(cshape))                    # (N, C, ...)
  valid = (t != ignore_index)
  lp = logp._t
  w = None if weight is None else weight._t.reshape(cshape)
  wt = valid.cast(lp.dtype) if w is None else (onehot.where(w, 0)).sum(1) * valid   # per-element weight
  nll = -(onehot.where(lp, 0)).sum(1) * wt
  if label_smoothing > 0:
    smooth = -((lp if w is None else lp * w).sum(1)) / c * valid
    loss = nll * (1 - label_smoothing) + smooth * label_smoothing
  else: loss = nll
  if reduction == "mean": loss = loss.sum() / wt.sum()
  elif reduction == "sum": loss = loss.sum()
  elif unbatched: loss = loss.squeeze(0)
  return _wrap(loss, logp)
def nll_loss(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction="mean"):
  return _nll(input, target, weight, ignore_index, reduction)
def cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction="mean", label_smoothing=0.0):
  dim = 0 if input.ndim == 1 else 1
  logp = input.log_softmax(dim)
  if target.is_floating_point():   # class probabilities
    c = input.shape[dim]
    y = target * (1 - label_smoothing) + label_smoothing / c if label_smoothing > 0 else target
    if weight is not None: y = y * weight.reshape([1, c] + [1] * (input.ndim - 2) if input.ndim > 1 else [c])
    loss = -(logp * y).sum(dim)
    return loss.sum() / (input.shape[0] if input.ndim > 1 else 1) if reduction == "mean" else _reduce(loss, reduction)
  return _nll(logp, target, weight, ignore_index, reduction, label_smoothing)
def mse_loss(input, target, size_average=None, reduce=None, reduction="mean", weight=None): return _reduce((input - target) ** 2, reduction)
def l1_loss(input, target, size_average=None, reduce=None, reduction="mean"): return _reduce((input - target).abs(), reduction)
def smooth_l1_loss(input, target, size_average=None, reduce=None, reduction="mean", beta=1.0):
  d = (input - target).abs()
  if beta == 0: return _reduce(d, reduction)
  return _reduce(_wrap((d._t < beta).where(0.5 * d._t * d._t / beta, d._t - 0.5 * beta), d), reduction)
def huber_loss(input, target, reduction="mean", delta=1.0, weight=None):
  d = (input - target).abs()
  return _reduce(_wrap((d._t <= delta).where(0.5 * d._t * d._t, delta * (d._t - 0.5 * delta)), d), reduction)
def binary_cross_entropy(input, target, weight=None, size_average=None, reduce=None, reduction="mean"):
  loss = -(target * input.log().clamp(min=-100) + (1 - target) * (1 - input).log().clamp(min=-100))
  return _reduce(loss if weight is None else loss * weight, reduction)
def binary_cross_entropy_with_logits(input, target, weight=None, size_average=None, reduce=None, reduction="mean", pos_weight=None):
  log1pexp = _wrap((1 + (-input._t.abs()).exp()).log(), input)   # log(1 + exp(-|x|))
  if pos_weight is None: loss = input.clamp(min=0) - input * target + log1pexp
  else: loss = (1 - target) * input + (1 + (pos_weight - 1) * target) * (log1pexp + (-input).clamp(min=0))
  return _reduce(loss if weight is None else loss * weight, reduction)
def kl_div(input, target, size_average=None, reduce=None, reduction="mean", log_target=False):
  loss = target.exp() * (target - input) if log_target else _wrap((target._t > 0).where(target._t * (target._t.clip(min_=1e-38).log() - input._t), 0), input, target)
  if reduction == "batchmean": return loss.sum() / input.shape[0]
  return _reduce(loss, reduction)
def poisson_nll_loss(input, target, log_input=True, full=False, size_average=None, eps=1e-08, reduce=None, reduction="mean"):
  return _reduce(input.exp() - target * input if log_input else input - target * (input + eps).log(), reduction)
def margin_ranking_loss(input1, input2, target, margin=0.0, size_average=None, reduce=None, reduction="mean"):
  return _reduce((-target * (input1 - input2) + margin).clamp(min=0), reduction)
def hinge_embedding_loss(input, target, margin=1.0, size_average=None, reduce=None, reduction="mean"):
  return _reduce(_wrap((target._t == 1).where(input._t, (margin - input._t).relu()), input), reduction)
def triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2.0, eps=1e-06, swap=False, size_average=None, reduce=None, reduction="mean"):
  return _reduce((pairwise_distance(anchor, positive, p, eps) - pairwise_distance(anchor, negative, p, eps) + margin).clamp(min=0), reduction)
def cosine_embedding_loss(input1, input2, target, margin=0.0, size_average=None, reduce=None, reduction="mean"):
  cos = cosine_similarity(input1, input2, dim=-1)
  return _reduce(_wrap((target._t == 1).where(1 - cos._t, (cos._t - margin).relu()), cos), reduction)
def soft_margin_loss(input, target, size_average=None, reduce=None, reduction="mean"): return _reduce((1 + (-target * input).exp()).log(), reduction)
def gaussian_nll_loss(input, target, var, full=False, eps=1e-06, reduction="mean"):
  var = var.clamp(min=eps)
  loss = 0.5 * (var.log() + (input - target) ** 2 / var)
  return _reduce(loss + 0.5 * math.log(2 * math.pi) if full else loss, reduction)

# ---- attention
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
  l, s = query.shape[-2], key.shape[-2]
  scores = (query @ key.transpose(-2, -1)) * (1 / math.sqrt(query.shape[-1]) if scale is None else scale)
  if is_causal: scores = _wrap(_TG.ones(l, s, dtype=_td.bool).tril().where(scores._t, -math.inf), scores)
  if attn_mask is not None:
    scores = _wrap(attn_mask._t.where(scores._t, -math.inf), scores) if attn_mask._t.dtype == _td.bool else scores + attn_mask
  attn = scores.softmax(-1)
  if dropout_p > 0: attn = dropout(attn, dropout_p, True)
  return attn @ value

def multi_head_attention_forward(*a, **k):
  raise NotImplementedError("use nn.MultiheadAttention; F.multi_head_attention_forward is not provided by the tinygrad.js torch shim")
