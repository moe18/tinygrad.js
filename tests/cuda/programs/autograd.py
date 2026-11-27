import json
import numpy as np
from tinygrad import Tensor
from tinygrad.cuda import CUDAKernel, CUDAFunction, library
act = CUDAKernel(library("activations"), "gelu_fwd")
gelu_bwd = act["gelu_bwd"]

class GELU(CUDAFunction):
  @staticmethod
  def forward(ctx, x):
    n = x.numel()
    y = Tensor.empty(*x.shape, device=x.device)
    act.launch((n + 255) // 256, 256, [x.contiguous(), y, n, 1])
    ctx.save_for_backward(x)
    return y
  @staticmethod
  def backward(ctx, gy):
    (x,) = ctx.saved_tensors
    n = x.numel()
    gx = Tensor.empty(*x.shape, device=x.device)
    gelu_bwd.launch((n + 255) // 256, 256, [x.contiguous(), gy.contiguous(), gx, n, 1])
    return gx

sq = CUDAKernel("""
__global__ void two_outputs(const float* x, float* a, float* b, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { a[i] = 2.0f * x[i]; b[i] = x[i] * x[i]; } }
__global__ void two_outputs_bwd(const float* x, const float* ga, const float* gb, float* gx, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) gx[i] = 2.0f * ga[i] + 2.0f * x[i] * gb[i]; }
""", "two_outputs")
sq_bwd = sq["two_outputs_bwd"]
class TwoOut(CUDAFunction):
  @staticmethod
  def forward(ctx, x):
    a, b = Tensor.empty(*x.shape, device=x.device), Tensor.empty(*x.shape, device=x.device)
    sq.launch(1, 64, [x, a, b, x.numel()])
    ctx.save_for_backward(x)
    return a, b
  @staticmethod
  def backward(ctx, ga, gb):
    (x,) = ctx.saved_tensors
    gx = Tensor.empty(*x.shape, device=x.device)
    sq_bwd.launch(1, 64, [x, ga.contiguous(), gb.contiguous(), gx, x.numel()])
    return gx

rng = np.random.default_rng(0)
xs = rng.standard_normal((4, 37)).astype(np.float32)
w = Tensor(rng.standard_normal((37, 5)).astype(np.float32))
# CUDA GELU inside a tinygrad graph (matmul before, sum after), gradients through both
x1 = Tensor(xs)
loss1 = (GELU.apply(x1 * 1.5) @ w).square().sum()
loss1.backward()
x2 = Tensor(xs)
loss2 = ((x2 * 1.5).gelu() @ w).square().sum()
loss2.backward()
g_err = float(np.abs(x1.grad.numpy() - x2.grad.numpy()).max() / (np.abs(x2.grad.numpy()).max() + 1e-9))
o_err = abs(loss1.item() - loss2.item()) / abs(loss2.item())
# multiple outputs: both gradients contribute (backward is linear in the output gradients)
x3 = Tensor([1.0, -2.0, 0.5])
a, b = TwoOut.apply(x3)
(a.sum() * 3 + (b * Tensor([1.0, 2.0, 3.0])).sum()).backward()
want = 3 * 2 + 2 * np.array([1.0, -2.0, 0.5]) * np.array([1.0, 2.0, 3.0])
g_err = max(g_err, float(np.abs(x3.grad.numpy() - want).max()))
# gradient accumulation across two backward passes
x4 = Tensor([1.0, 2.0])
GELU.apply(x4).sum().backward()
g1 = x4.grad.numpy().copy()
print("RESULT_JSON:" + json.dumps({"max_grad_err": g_err, "max_out_err": o_err, "device": x1.device}))
