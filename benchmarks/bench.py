# tinygrad.js benchmarks. Runs unchanged in the browser (tinygrad.js) and natively (`DEV=CPU python bench.py`).
# Prints one line: BENCH_JSON:{...}. Configure with BENCH_DEVICES / BENCH_QUICK in the environment or globals.
import json, os, statistics, time
from tinygrad import Tensor, Device, TinyJit, nn
from tinygrad.helpers import Context
from tinygrad.nn.optim import Adam

QUICK = bool(int(globals().get("BENCH_QUICK", os.environ.get("BENCH_QUICK", "0"))))
DEVICES = globals().get("BENCH_DEVICES", os.environ.get("BENCH_DEVICES", Device.DEFAULT)).split(",")
training = Tensor.train if hasattr(Tensor, "train") else (lambda: Context(TRAINING=1))

def sync(dev): Device[dev].synchronize()

def timeit(fn, dev, repeat, warmup=1):
  for _ in range(warmup): fn(); sync(dev)
  times = []
  for _ in range(repeat):
    st = time.perf_counter(); fn(); sync(dev); times.append(time.perf_counter() - st)
  return statistics.median(times)

results = {"device_default": Device.DEFAULT, "quick": QUICK, "runs": []}
def record(**kw):
  results["runs"].append(kw)

for dev in DEVICES:
  is_slow_cpu = dev in ("CPU", "PYTHON") and "pyodide" in __import__("sys").modules
  # 1. first kernel (compile + run) vs warm run of the same kernel, unique constant -> guaranteed new kernel
  salt = time.time_ns() % 100003 / 1e6
  a = Tensor.ones(1024, device=dev).contiguous().realize()
  st = time.perf_counter(); (a * (2.0 + salt) + 1).realize(); sync(dev); first = time.perf_counter() - st
  warm = timeit(lambda: (a * (2.0 + salt) + 1).realize(), dev, 20)
  record(bench="first_kernel", device=dev, first_ms=first * 1e3, warm_ms=warm * 1e3)

  # 2. elementwise throughput
  for n in ([1 << 10, 1 << 14] if is_slow_cpu else [1 << 10, 1 << 16, 1 << 20] + ([] if QUICK else [1 << 22])):
    x, y = Tensor.rand(n, device=dev).realize(), Tensor.rand(n, device=dev).realize()
    t = timeit(lambda: (x * y + x).realize(), dev, 3 if is_slow_cpu else 10)
    record(bench="elementwise", device=dev, n=n, ms=t * 1e3, gbps=3 * 4 * n / t / 1e9)

  # 3. matmul throughput (realize + synchronize, no readback); kernel compiled in warmup
  for n in ([16, 32] if is_slow_cpu else [64, 128, 256, 512] + ([] if QUICK else [1024])):
    x, y = Tensor.rand(n, n, device=dev).realize(), Tensor.rand(n, n, device=dev).realize()
    t = timeit(lambda: (x @ y).realize(), dev, 2 if is_slow_cpu else 5)
    record(bench="matmul", device=dev, n=n, ms=t * 1e3, gflops=2 * n ** 3 / t / 1e9)

  # 4. readback (GPU -> CPU) bandwidth
  if not is_slow_cpu:
    for n in [1 << 10, 1 << 20, 1 << 22]:
      x = Tensor.rand(n, device=dev).realize()
      x.numpy(); sync(dev)                          # warm up (staging buffers, first-call paths)
      st = time.perf_counter(); reps = 5
      for _ in range(reps): x.numpy()
      t = (time.perf_counter() - st) / reps
      record(bench="readback", device=dev, bytes=4 * n, ms=t * 1e3, mbps=4 * n / t / 1e6)

  # 5. small MLP training: steps/s with and without TinyJit
  Tensor.manual_seed(0)
  class MLP:
    def __init__(self): self.l1, self.l2 = nn.Linear(2, 64), nn.Linear(64, 1)
    def __call__(self, v): return self.l2(self.l1(v).relu())
  X = (Tensor.rand(256, 2, device=dev) * 2 - 1).realize()
  Y = ((X * X).sum(axis=1, keepdim=True) < 0.4).float().realize()
  for jit in ([False] if is_slow_cpu else [False, True]):
    model = MLP()
    for p in nn.state.get_parameters(model): p.replace(p.to(dev)).realize()
    opt = Adam(nn.state.get_parameters(model), lr=0.01)
    def step():
      with training():
        opt.zero_grad()
        loss = (model(X).sigmoid() - Y).square().mean()
        loss.backward(); opt.step()
        return loss.realize()
    stepf = TinyJit(step) if jit else step
    steps = 3 if is_slow_cpu else (20 if QUICK else 50)
    first_loss = stepf().item()                     # includes compile
    if jit: stepf(); stepf()                        # TinyJit captures on the 2nd call
    sync(dev); st = time.perf_counter()
    for _ in range(steps): loss = stepf()
    last_loss = loss.item(); t = time.perf_counter() - st
    record(bench="mlp_train", device=dev, jit=jit, steps=steps, ms_per_step=t / steps * 1e3, first_loss=first_loss, last_loss=last_loss)

print("BENCH_JSON:" + json.dumps(results))
