# A differentiable CUDA layer: forward and backward are CUDA kernels, autograd and optimizers are tinygrad's.
import numpy as np
from tinygrad import Tensor, nn
from tinygrad.helpers import Context
from tinygrad.cuda import CUDAKernel, CUDAFunction

k = CUDAKernel("""
// y = x * sigmoid(x)  (SiLU)
__global__ void silu_fwd(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { float s = 1.0f / (1.0f + __expf(-x[i])); y[i] = x[i] * s; }
}
// dy/dx = s * (1 + x * (1 - s))
__global__ void silu_bwd(const float* x, const float* gy, float* gx, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { float s = 1.0f / (1.0f + __expf(-x[i])); gx[i] = gy[i] * s * (1.0f + x[i] * (1.0f - s)); }
}
""", "silu_fwd")
k_bwd = k["silu_bwd"]

class SiLU(CUDAFunction):
  @staticmethod
  def forward(ctx, x):
    y = Tensor.empty(*x.shape, device=x.device)
    k.launch((x.numel() + 255) // 256, 256, [x.contiguous(), y, x.numel()])
    ctx.save_for_backward(x)
    return y
  @staticmethod
  def backward(ctx, gy):
    (x,) = ctx.saved_tensors
    gx = Tensor.empty(*x.shape, device=x.device)
    k_bwd.launch((x.numel() + 255) // 256, 256, [x.contiguous(), gy.contiguous(), gx, x.numel()])
    return gx

x = Tensor(np.random.default_rng(0).standard_normal((4, 8)).astype(np.float32))  # tinygrad >= 0.14 tracks gradients by default
SiLU.apply(x).sum().backward()
g_cuda = x.grad.numpy()
x.grad = None
x.silu().sum().backward()
print("max gradient difference vs tinygrad's silu:", abs(g_cuda - x.grad.numpy()).max())

# train a tiny model that uses the CUDA activation
l1, l2 = nn.Linear(2, 16), nn.Linear(16, 1)
opt = nn.optim.Adam([l1.weight, l1.bias, l2.weight, l2.bias], lr=0.05)
X = Tensor.randn(128, 2); Y = (X[:, 0:1] * X[:, 1:2] > 0).float()
with Context(TRAINING=1):
  for step in range(60):
    opt.zero_grad()
    loss = ((l2(SiLU.apply(l1(X))).sigmoid() - Y) ** 2).mean()
    loss.backward(); opt.step()
    if step % 20 == 0 or step == 59: print(f"step {step}: loss {loss.item():.4f}")
