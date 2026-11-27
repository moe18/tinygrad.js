# CUDA kernels on tinygrad tensors (run inside tinygrad.js: rt.run(source), the playground's Python box, or a notebook).
# Works on the WEBGPU device (zero-copy on tinygrad's GPU buffers) and on the CPU device (WebAssembly backend).
from tinygrad import Tensor, Device
from tinygrad.cuda import CUDAKernel, library, device_info

kernel = CUDAKernel("""
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
""")
a = Tensor.rand(1024); b = Tensor.rand(1024); c = Tensor.zeros(1024)
await kernel.launch(grid=(4, 1, 1), block=(256, 1, 1), args=[a, b, c, 1024])
print(Device.DEFAULT, device_info()["name"])
print("max |c - (a + b)| =", (c - (a + b)).abs().max().item())

# library kernels mix with tinygrad ops
mm = CUDAKernel(library("matmul"), "matmul_tiled<16>")
A, B = Tensor.rand(64, 64), Tensor.rand(64, 64)
C = Tensor.empty(64, 64)
mm.launch((4, 4), (16, 16), [A, B, C, 64, 64, 64])
print("matmul_tiled vs A @ B:", (C - A @ B).abs().max().item())

scale = CUDAKernel("__global__ void scale(float* x, float s, int n) { int i = threadIdx.x; if (i < n) x[i] *= s; }")
# writes through views (slices, transposes) update the tensor they view, which must be stored in a buffer:
# a lazy result (here: arange) is materialized with .clone().realize() first, otherwise launch raises CUDAError
x = Tensor.arange(8, dtype="float32").reshape(2, 4).clone().realize()
scale.launch(1, 32, [x[1], 10.0, 4])   # row 1 only
print(x.numpy())
