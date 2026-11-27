# Train a 2-layer MLP twice from the same initialization: (1) with a custom CUDA Linear+ReLU layer (forward and backward
# kernels) and a CUDA Adam optimizer kernel, (2) with plain tinygrad ops and tinygrad's Adam. The losses must agree.
import json, time
import numpy as np
from tinygrad import Tensor, nn
from tinygrad.helpers import Context
from tinygrad.cuda import CUDAKernel, CUDAFunction, library
src = r"""
// y = relu(x @ W + b): x[M,K], W[K,N]
__global__ void linear_relu_fwd(const float* x, const float* W, const float* b, float* y, int M, int N, int K) {
  int n = blockIdx.x * blockDim.x + threadIdx.x, m = blockIdx.y;
  if (n >= N) return;
  float acc = b[n];
  for (int k = 0; k < K; ++k) acc += x[m * K + k] * W[k * N + n];
  y[m * N + n] = fmaxf(acc, 0.0f);
}
// gz = gy * (y > 0); gx = gz @ W^T; gW = x^T @ gz; gb = sum_m gz
__global__ void linear_relu_bwd_x(const float* gy, const float* y, const float* W, float* gx, int M, int N, int K) {
  int k = blockIdx.x * blockDim.x + threadIdx.x, m = blockIdx.y;
  if (k >= K) return;
  float acc = 0.0f;
  for (int n = 0; n < N; ++n) acc += (y[m * N + n] > 0.0f ? gy[m * N + n] : 0.0f) * W[k * N + n];
  gx[m * K + k] = acc;
}
__global__ void linear_relu_bwd_w(const float* gy, const float* y, const float* x, float* gW, float* gb, int M, int N, int K) {
  int n = blockIdx.x * blockDim.x + threadIdx.x, k = blockIdx.y;
  if (n >= N) return;
  float acc = 0.0f, accb = 0.0f;
  for (int m = 0; m < M; ++m) { float g = y[m * N + n] > 0.0f ? gy[m * N + n] : 0.0f; acc += x[m * K + k] * g; accb += g; }
  gW[k * N + n] = acc;
  if (k == 0) gb[n] = accb;
}
"""
L = CUDAKernel(src, "linear_relu_fwd")
L_bx, L_bw = L["linear_relu_bwd_x"], L["linear_relu_bwd_w"]
adam = CUDAKernel(library("optimizers"), "adam_step")

class LinearReLU(CUDAFunction):
  @staticmethod
  def forward(ctx, x, W, b):
    M, K = x.shape; N = W.shape[1]
    y = Tensor.empty(M, N, device=x.device)
    L.launch(((N + 63) // 64, M), 64, [x.contiguous(), W, b, y, M, N, K])
    ctx.save_for_backward(x, W, y)
    return y
  @staticmethod
  def backward(ctx, gy):
    x, W, y = ctx.saved_tensors
    M, K = x.shape; N = W.shape[1]
    gy = gy.contiguous()
    gx, gW, gb = Tensor.empty(M, K, device=x.device), Tensor.empty(K, N, device=x.device), Tensor.empty(N, device=x.device)
    L_bx.launch(((K + 63) // 64, M), 64, [gy, y, W, gx, M, N, K])
    L_bw.launch(((N + 63) // 64, K), 64, [gy, y, x.contiguous(), gW, gb, M, N, K])
    return gx, gW, gb

class CudaAdam:
  def __init__(self, params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
    for p in params: p.requires_grad = True
    self.params, self.lr, self.b1, self.b2, self.eps, self.t = params, lr, b1, b2, eps, 0
    self.m = [Tensor.zeros(*p.shape, device=p.device).contiguous().realize() for p in params]
    self.v = [Tensor.zeros(*p.shape, device=p.device).contiguous().realize() for p in params]
  def zero_grad(self):
    for p in self.params: p.grad = None
  def step(self):
    self.t += 1
    for p, m, v in zip(self.params, self.m, self.v):
      n = p.numel()
      adam.launch((n + 255) // 256, 256, [p, p.grad.contiguous(), m, v, n, self.lr, self.b1, self.b2, self.eps, 0.0, self.t, 0])

rng = np.random.default_rng(1)
X = rng.standard_normal((64, 2)).astype(np.float32)
Y = (X[:, :1] * X[:, 1:] > 0).astype(np.float32)
W1, b1 = (rng.standard_normal((2, 32)) * 0.5).astype(np.float32), np.zeros(32, np.float32)
W2, b2 = (rng.standard_normal((32, 1)) * 0.5).astype(np.float32), np.zeros(1, np.float32)
STEPS = 25

def run(use_cuda):
  P = [Tensor(a.copy()).contiguous().realize() for a in (W1, b1, W2, b2)]
  opt = CudaAdam(P) if use_cuda else nn.optim.Adam(P, lr=1e-2)
  xs, ys = Tensor(X), Tensor(Y)
  losses = []
  t0 = time.perf_counter()
  with Context(TRAINING=1):
    for i in range(STEPS):
      opt.zero_grad()
      h = LinearReLU.apply(xs, P[0], P[1]) if use_cuda else (xs @ P[0] + P[1]).relu()
      out = (h @ P[2] + P[3]).sigmoid()
      loss = ((out - ys) ** 2).mean()
      loss.backward()
      losses.append(loss.item())  # before step(): the lazy loss would otherwise read the already-updated parameters
      opt.step()
  return losses, (time.perf_counter() - t0) / STEPS * 1000

lc, tc = run(True)
lt, tt = run(False)
print("RESULT_JSON:" + json.dumps({"losses_cuda": lc, "losses_tinygrad": lt, "max_loss_diff": float(np.abs(np.array(lc) - np.array(lt)).max()),
                                    "ms_per_step_cuda": tc, "ms_per_step_tinygrad": tt, "device": Tensor([1.0]).device}))
