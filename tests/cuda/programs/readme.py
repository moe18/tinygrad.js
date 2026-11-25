import json
import numpy as np
from tinygrad import Tensor
from tinygrad.cuda import CUDAKernel
kernel = CUDAKernel("""
    __global__ void vector_add(
        float* a,
        float* b,
        float* c,
        int n
    ) {
        int i = blockIdx.x * blockDim.x
              + threadIdx.x;
        if (i < n) {
            c[i] = a[i] + b[i];
        }
    }
""")
a = Tensor.rand(1024)
b = Tensor.rand(1024)
c = Tensor.zeros(1024)
await kernel.launch(
    grid=(4, 1, 1),
    block=(256, 1, 1),
    args=[a, b, c, 1024]
)
err = float(np.abs(c.numpy() - (a.numpy() + b.numpy())).max())
print("RESULT_JSON:" + json.dumps({"max_err": err, "device": c.device}))
