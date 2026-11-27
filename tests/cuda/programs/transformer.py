# One pre-norm attention block (RMSNorm -> QKV projection -> RoPE -> causal attention -> output projection + residual)
# computed twice: with CUDA kernels (rmsnorm, rope, flash attention) and with tinygrad ops. Outputs must agree.
import json, math
import numpy as np
from tinygrad import Tensor
from tinygrad.cuda import CUDAKernel, library
flash = CUDAKernel(library("attention"), "flash_attention_fwd<32, 32, 16>")
rope = CUDAKernel(library("rope"), "rope_apply")
rms = CUDAKernel(library("norms"), "rmsnorm_fwd<64>")
B, H, T, D = 2, 4, 40, 16
C = H * D
rng = np.random.default_rng(2)
x_np = rng.standard_normal((B, T, C)).astype(np.float32)
Wq, Wk, Wv, Wo = [Tensor((rng.standard_normal((C, C)) / math.sqrt(C)).astype(np.float32)) for _ in range(4)]
g = Tensor(rng.uniform(0.5, 1.5, C).astype(np.float32))

def heads(t): return t.reshape(B, T, H, D).permute(0, 2, 1, 3).contiguous()
def rope_ref(t, pos0=0, base=10000.0):
  i = np.arange(D // 2); ang = (pos0 + np.arange(T))[:, None] * base ** (-2.0 * i / D)
  c, s = Tensor(np.cos(ang).astype(np.float32)), Tensor(np.sin(ang).astype(np.float32))
  a, b = t[..., 0::2], t[..., 1::2]
  return Tensor.stack(a * c - b * s, a * s + b * c, dim=-1).reshape(B, H, T, D)

def block_tinygrad(x):
  h = x * (x.square().mean(-1, keepdim=True) + 1e-6).rsqrt() * g
  q, k, v = heads(h @ Wq), heads(h @ Wk), heads(h @ Wv)
  o = rope_ref(q).scaled_dot_product_attention(rope_ref(k), v, is_causal=True)
  return x + o.permute(0, 2, 1, 3).reshape(B, T, C) @ Wo

def block_cuda(x):
  h = Tensor.empty(B * T, C, device=x.device); rr = Tensor.empty(B * T, device=x.device)
  rms.launch(B * T, 64, [x.reshape(B * T, C).contiguous(), g, h, rr, C, 1e-6])
  h = h.reshape(B, T, C)
  q, k, v = heads(h @ Wq), heads(h @ Wk), heads(h @ Wv)
  n = B * H * T * D // 2
  rope.launch((n + 127) // 128, 128, [q, B, H, T, D, 0, 10000.0])
  rope.launch((n + 127) // 128, 128, [k, B, H, T, D, 0, 10000.0])
  o, lse = Tensor.empty(B, H, T, D, device=x.device), Tensor.empty(B, H, T, device=x.device)
  flash.launch(((T + 31) // 32, B * H), 32, [q, k, v, o, lse, B, H, H, T, T, D, 1.0 / math.sqrt(D), 1])
  return x + o.permute(0, 2, 1, 3).reshape(B, T, C) @ Wo, (q, k, v, o)

x = Tensor(x_np)
ref = block_tinygrad(x).numpy()
out, (q, k, v, o) = block_cuda(x)
attn_ref = q.scaled_dot_product_attention(k, v, is_causal=True).numpy()
attn_err = float(np.abs(o.numpy() - attn_ref).max())
block_err = float(np.abs(out.numpy() - ref).max() / (np.abs(ref).max()))
print("RESULT_JSON:" + json.dumps({"max_attn_err": attn_err, "max_block_err": block_err, "device": x.device}))
