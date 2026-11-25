# CUDA kernels on tinygrad tensors (tinygrad.cuda) vs the equivalent standard tinygrad operations, in the same runtime.
# Each pair is checked for equal results first. Times are per call: realize + device synchronize, median of runs,
# kernels already compiled (warmup). Prints one line: BENCH_JSON:{...}
import json, statistics, time
import numpy as np
from tinygrad import Tensor, Device
from tinygrad.cuda import CUDAKernel, library

QUICK = bool(globals().get("BENCH_QUICK", 0))
DEVICES = globals().get("BENCH_DEVICES", Device.DEFAULT).split(",")
out = {"runs": []}

def timeit(fn, dev, runs, warmup=2):
  for _ in range(warmup): fn(); Device[dev].synchronize()
  ts = []
  for _ in range(runs):
    t = time.perf_counter(); fn(); Device[dev].synchronize(); ts.append((time.perf_counter() - t) * 1e3)
  return statistics.median(ts)

vadd = CUDAKernel(library("vectorAdd"), "vector_add")
mm = CUDAKernel(library("matmul"), "matmul_tiled<16>")
mm_reg = mm["matmul_regtile<64, 64, 8, 4, 4>"]
smax = CUDAKernel(library("softmax"), "softmax_rows<128>")

for dev in DEVICES:
  cpu = dev in ("CPU", "PYTHON")
  runs = 3 if cpu or QUICK else 10
  def pair(name, cuda_fn, tg_fn, check, flops=None, nbytes=None):
    c, t = cuda_fn(), tg_fn()
    err = float(np.abs(c.numpy() - t.numpy()).max() / max(1e-30, np.abs(t.numpy()).max()))
    ok = err <= check
    rec = {"device": dev, "workload": name, "maxRelErr": err, "ok": ok}
    if ok:
      rec["cuda_ms"] = timeit(lambda: cuda_fn().realize(), dev, runs)
      rec["tinygrad_ms"] = timeit(lambda: tg_fn().realize(), dev, runs)
      if flops: rec["cuda_gflops"], rec["tinygrad_gflops"] = flops / rec["cuda_ms"] / 1e6, flops / rec["tinygrad_ms"] / 1e6
      if nbytes: rec["cuda_gbps"], rec["tinygrad_gbps"] = nbytes / rec["cuda_ms"] / 1e6, nbytes / rec["tinygrad_ms"] / 1e6
    out["runs"].append(rec)
    print(name, dev, rec, flush=True)

  n = 1 << 14 if cpu else 1 << 20
  a, b = Tensor.rand(n, device=dev).realize(), Tensor.rand(n, device=dev).realize()
  def cuda_add():
    c = Tensor.empty(n, device=dev)
    vadd.launch((n + 255) // 256, 256, [a, b, c, n])
    return c
  pair(f"vector add n={n}", cuda_add, lambda: a + b, 0, nbytes=12 * n)

  for N in ([64] if cpu else [256] if QUICK else [256, 512]):
    A, B = Tensor.rand(N, N, device=dev).realize(), Tensor.rand(N, N, device=dev).realize()
    def cuda_mm(k=mm, tile=16):
      C = Tensor.empty(N, N, device=dev)
      k.launch(((N + tile - 1) // tile, (N + tile - 1) // tile), (16, 16), [A, B, C, N, N, N])
      return C
    pair(f"matmul {N} (matmul_tiled<16>)", cuda_mm, lambda: A @ B, 1e-5, flops=2 * N ** 3)
    if not cpu: pair(f"matmul {N} (matmul_regtile)", lambda: cuda_mm(mm_reg, 64), lambda: A @ B, 1e-5, flops=2 * N ** 3)

  rows, cols = (16, 256) if cpu else (1024, 1024)
  X = (Tensor.rand(rows, cols, device=dev) * 8).realize()
  def cuda_softmax():
    Y = Tensor.empty(rows, cols, device=dev)
    smax.launch(rows, 128, [X, Y, cols])
    return Y
  pair(f"softmax {rows}x{cols}", cuda_softmax, lambda: X.softmax(-1), 1e-5, nbytes=8 * rows * cols)

print("BENCH_JSON:" + json.dumps(out))
