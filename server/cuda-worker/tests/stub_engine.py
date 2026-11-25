"""TEST-ONLY engine for the gateway tests: host memory, no kernels. Kernel names starting with "sleep" sleep for the number
of milliseconds in their first scalar argument (little-endian int32), so tests can hold a request in flight."""
from __future__ import annotations
import os, struct, time
from tgjs_cuda_worker.errors import WorkerError

class StubEngine:
  def __init__(self):
    self.mem: dict[str, bytearray] = {}
  def device_info(self) -> dict:
    return {"name": f"stub {os.getpid()}", "totalMem": 1 << 30, "major": 7, "minor": 0, "multiProcessorCount": 1,
            "sharedMemPerBlock": 48 * 1024, "maxThreadsPerBlock": 1024}
  def _get(self, h, off, n):
    if h not in self.mem: raise WorkerError("cudaErrorInvalidDevicePointer", f"unknown allocation {h!r}")
    if off + n > len(self.mem[h]): raise WorkerError("cudaErrorInvalidValue", "out of range")
    return self.mem[h]
  def alloc(self, h, n):
    time.sleep(float(os.environ.get("STUB_ALLOC_SLEEP_S", "0")))  # widens the window between the quota check and the reply
    self.mem[h] = bytearray(n)
  def free(self, h): self.mem.pop(h, None)
  def write(self, h, off, data): self._get(h, off, len(data))[off:off + len(data)] = data
  def read(self, h, off, n): return bytes(self._get(h, off, n)[off:off + n])
  def copy(self, d, do, s, so, n): self._get(d, do, n)[do:do + n] = self._get(s, so, n)[so:so + n]
  def memset(self, h, off, v, n): self._get(h, off, n)[off:off + n] = bytes([v]) * n
  def launch(self, source, kernel, options, grid, block, shared_mem, args, timing):
    if kernel.startswith("sleep"):
      ms = struct.unpack("<i", args[0]["scalar"]["bytes"][:4])[0]
      time.sleep(ms / 1000)
    return {"ms": None}
  def close(self): pass
