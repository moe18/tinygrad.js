import mlpTraining from "../../python/mlp_training.py?raw";

export const EXAMPLES: Record<string, string> = {
  "Tensor basics": `from tinygrad import Tensor, Device
print("default device:", Device.DEFAULT)

a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])
print("a + b =", (a + b).numpy())
print("a * b =", (a * b).numpy())

m = Tensor.arange(12).reshape(3, 4).float()
print("broadcast:\\n", (m - m.mean(axis=1, keepdim=True)).numpy())
print("softmax row 0:", m.softmax(axis=1).numpy()[0])
`,
  "Matrix multiplication": `import time
from tinygrad import Tensor

for n in (64, 256, 512):
  a, b = Tensor.randn(n, n).realize(), Tensor.randn(n, n).realize()
  (a @ b).realize()                      # first call compiles the kernel
  st = time.perf_counter()
  c = (a @ b).numpy()                    # realize on the GPU + copy back
  dt = time.perf_counter() - st
  print(f"{n}x{n}: {dt*1000:7.1f} ms  ({2*n**3/dt/1e9:6.2f} GFLOP/s incl. readback)  c[0,0]={c[0,0]:.4f}")
`,
  "Automatic differentiation": `from tinygrad import Tensor
x = Tensor([2.0])               # tinygrad >= 0.13: every tensor can receive a gradient
y = (x * x).sum()
y.backward()
print("dy/dx at x=2:", x.grad.numpy())

w = Tensor([[1.0, -2.0], [0.5, 3.0]])
loss = (w @ Tensor([[1.0], [2.0]])).relu().sum()
loss.backward()
print("dloss/dw:\\n", w.grad.numpy())
`,
  "Train an MLP (TinyJit + Adam)": mlpTraining,
  "Show generated WGSL": `# DEBUG=4 makes tinygrad print the kernels it generates: these WGSL compute shaders run on your GPU
from tinygrad import Tensor
from tinygrad.helpers import Context
with Context(DEBUG=4):
  (Tensor([1.0, 2.0, 3.0]) * 2 + 1).realize()
`,
  "WebGPU vs CPU": `# CPU here is tinygrad's pure-Python device: correct but interpreted, so only use it for tiny tensors
import time
from tinygrad import Tensor
for n in (8, 32):
  for dev in ("WEBGPU", "CPU"):
    a, b = Tensor.ones(n, n, device=dev).contiguous().realize(), Tensor.ones(n, n, device=dev).contiguous().realize()
    (a @ b).realize()
    st = time.perf_counter(); (a @ b).numpy(); dt = time.perf_counter() - st
    print(f"{n:3d}x{n:<3d} {dev:7s} {dt*1000:8.2f} ms")
`,
  "GPU stats": `import tinygrad_web
from tinygrad import Tensor
(Tensor.rand(256, 256) @ Tensor.rand(256, 256)).sum().item()
for k, v in tinygrad_web.stats().items(): print(f"{k:22s} {v}")
`,
};
