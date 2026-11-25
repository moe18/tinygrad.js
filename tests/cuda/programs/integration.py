import json
import numpy as np
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.cuda import CUDAKernel, CUDAError, synchronize, device_info
R = {}
src = r"""
__global__ void scale(float* x, float s, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] *= s; }
__global__ void axpy(const float* x, float* y, float a, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] += a * x[i]; }
__global__ void add_alias(const float* a, float* b, int n) { int i = threadIdx.x; if (i < n) b[i] = a[i] + 1.0f; }
__global__ void iota(int* o, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = i * i - 3; }
template <int B> __global__ void rowsum(const float* x, float* out, int cols) {
  __shared__ float s[B];
  float a = 0.0f;
  for (int i = threadIdx.x; i < cols; i += B) a += x[blockIdx.x * cols + i];
  s[threadIdx.x] = a; __syncthreads();
  for (int st = B / 2; st > 0; st >>= 1) { if (threadIdx.x < st) s[threadIdx.x] += s[threadIdx.x + st]; __syncthreads(); }
  if (threadIdx.x == 0) out[blockIdx.x] = s[0];
}
struct Affine { float a; float b; };
__global__ void affine(float* x, Affine p, int n) { int i = threadIdx.x; if (i < n) x[i] = p.a * x[i] + p.b; if (i == 0) printf("affine a=%.1f b=%.1f\n", p.a, p.b); }
"""
K = CUDAKernel(src, "scale")
scale, axpy, alias, iota, rowsum, affine = K, K["axpy"], K["add_alias"], K["iota"], K["rowsum<64>"], K["affine"]

# in place on a whole buffer, and through a strided view (written back with assign)
z = Tensor.ones(8).contiguous().realize()
scale.launch(1, 32, [z, 3.0, 8])
R["inplace"] = z.numpy().tolist() == [3.0] * 8
m = Tensor(np.arange(12, dtype=np.float32).reshape(3, 4)).realize()
scale.launch(1, 32, [m[:, 1], 10.0, 3])
R["view_writeback"] = m.numpy()[:, 1].tolist() == [10.0, 50.0, 90.0] and m.numpy()[:, 0].tolist() == [0.0, 4.0, 8.0]
# writes through views of tensors that are computed rather than stored can't reach the base in tinygrad: explicit error
a = Tensor.arange(8, dtype="float32").reshape(2, 4).contiguous()
try: scale.launch(1, 32, [a[1], 10.0, 4]); R["lazy_view_write_error"] = False
except CUDAError as e: R["lazy_view_write_error"] = "argument 'x' is a view" in str(e) and "clone().realize()" in str(e)
a = a.clone().realize()                       # the documented fix
scale.launch(1, 32, [a[1], 10.0, 4])
R["materialized_view_write"] = a.numpy().tolist() == [[0, 1, 2, 3], [40, 50, 60, 70]]
tt = Tensor(np.arange(12, dtype=np.float32).reshape(3, 4))
scale.launch(1, 32, [tt.T[1], 2.0, 3])         # a row of the transpose = a strided column
R["transposed_view_write"] = tt.numpy()[:, 1].tolist() == [2.0, 10.0, 18.0] and tt.numpy()[:, 0].tolist() == [0.0, 4.0, 8.0]
zz = Tensor.ones(2, 4).contiguous().realize()
zz = zz + 0                                    # a lazy value: writing through its view is refused, not lost
try: scale.launch(1, 32, [zz[1], 5.0, 4]); R["lazy_view_write_error2"] = False
except CUDAError: R["lazy_view_write_error2"] = True
zz = Tensor.ones(2, 4).clone().realize()
scale.launch(1, 32, [zz[1], 5.0, 4])
R["row_view_write"] = zz.numpy().tolist() == [[1.0] * 4, [5.0] * 4]
# non-contiguous input (transpose) is read correctly
t = Tensor(np.arange(6, dtype=np.float32).reshape(2, 3))
y = Tensor.zeros(6).contiguous()
axpy.launch(1, 32, [t.T.flatten(), y, 2.0, 6])
R["noncontig_input"] = y.numpy().tolist() == (2 * np.arange(6, dtype=np.float32).reshape(2, 3).T.flatten()).tolist()
# the same tensor passed twice (aliased pointers)
w = Tensor([1.0, 2.0, 3.0]).contiguous().realize()
alias.launch(1, 32, [w, w, 3])
R["aliased_args"] = w.numpy().tolist() == [2.0, 3.0, 4.0]
# integer tensors
o = Tensor.zeros(10, dtype=dtypes.int32).contiguous()
iota.launch(1, 32, [o, 10])
R["int_tensor"] = o.numpy().tolist() == [i * i - 3 for i in range(10)]
# templates + shared memory
a = Tensor.rand(5, 300)
out = Tensor.zeros(5)
rowsum.launch(5, 64, [a, out, 300])
R["template_shared"] = float(np.abs(out.numpy() - a.numpy().sum(1)).max()) < 1e-4
# struct argument + printf (output delivered by synchronize())
v = Tensor.ones(4).contiguous()
affine.launch(1, 4, [v, {"a": 2.0, "b": 0.5}, 4])
R["struct_arg"] = v.numpy().tolist() == [2.5] * 4
R["printf"] = "affine a=2.0 b=0.5" in synchronize()
# lazy launches join tinygrad's graph: nothing runs until the result is needed
q = Tensor.ones(4).contiguous().realize()
d = scale.launch(1, 32, [q, 5.0, 4], lazy=True)
r2 = (q + 1).numpy().tolist()
R["lazy_dependency"] = r2 == [6.0] * 4 and len(d.outputs) == 1
# mixing tinygrad ops before and after the kernel
p = (Tensor.arange(8).float() * 0.5).contiguous()
scale.launch(1, 32, [p, 2.0, 8])
R["mixed_ops"] = float((p.sum() * 2).item()) == float(np.arange(8).sum() * 2.0)
# TinyJit captures CUDA kernels and replays them
@TinyJit
def step(x):
  ob = Tensor.zeros(4).contiguous()
  axpy.launch(1, 32, [x, ob, 3.0, 4], lazy=True)   # ob = 3x
  return (ob + 1).realize()
for i in range(4): rj = step(Tensor([1.0, 2.0, 3.0, float(i)]))
R["tinyjit"] = rj.numpy().tolist() == [4.0, 7.0, 10.0, 10.0]
# errors are reported as CUDAError with diagnostics
try: CUDAKernel("__global__ void k(float* x) { x[0] = undefined_thing; }"); R["compile_error"] = False
except CUDAError as e: R["compile_error"] = "undeclared identifier 'undefined_thing'" in str(e)
try: scale.launch(1, 32, [Tensor.ones(4), 2.0]); R["arg_error"] = False
except CUDAError as e: R["arg_error"] = "takes 3 arguments" in str(e)
try: scale.launch(1, 32, [1.0, 2.0, 4]); R["type_error"] = False
except CUDAError as e: R["type_error"] = "must be a Tensor" in str(e)
R["device_info"] = device_info()["limits"]["warpSize"] == 32
R["device"] = z.device
print("RESULT_JSON:" + json.dumps(R))
