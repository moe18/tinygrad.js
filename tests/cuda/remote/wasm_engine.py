"""TEST-ONLY engine for the remote CUDA worker: executes kernels with tinygrad.js's own WebAssembly backend (via Node) instead
of a GPU, so the full remote path (TypeScript client -> HTTP gateway -> isolated session process -> engine, including the
argument wire encoding) can be tested on machines without NVIDIA hardware. Production workers use tgjs_cuda_worker.cuda.
"""
from __future__ import annotations
import base64, json, os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "server", "cuda-worker"))
from tgjs_cuda_worker.errors import WorkerError

RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run-wasm.mjs")

class WasmEngine:
  def __init__(self):
    self.mem: dict[str, bytearray] = {}
  def device_info(self) -> dict:
    return {"name": "tinygrad.js WebAssembly test engine", "totalMem": 1 << 30, "major": 7, "minor": 0, "multiProcessorCount": 1,
            "sharedMemPerBlock": 48 * 1024, "maxThreadsPerBlock": 1024}
  def _get(self, h, off, n):
    if h not in self.mem: raise WorkerError("cudaErrorInvalidDevicePointer", f"unknown allocation {h!r}")
    if off + n > len(self.mem[h]): raise WorkerError("cudaErrorInvalidValue", "out of range")
    return self.mem[h]
  def alloc(self, h, n): self.mem[h] = bytearray(n)
  def free(self, h): self.mem.pop(h, None)
  def write(self, h, off, data): self._get(h, off, len(data))[off:off + len(data)] = data
  def read(self, h, off, n): return bytes(self._get(h, off, n)[off:off + n])
  def copy(self, d, do, s, so, n): self._get(d, do, n)[do:do + n] = self._get(s, so, n)[so:so + n]
  def memset(self, h, off, v, n): self._get(h, off, n)[off:off + n] = bytes([v]) * n
  def launch(self, source, kernel, options, grid, block, shared_mem, args, timing):
    used = sorted({a["ptr"]["handle"] for a in args if "ptr" in a})
    job = {"source": source, "kernel": kernel, "defines": options.get("defines"), "grid": grid, "block": block, "sharedMem": shared_mem,
           "mem": {h: base64.b64encode(bytes(self.mem[h])).decode() for h in used},
           "args": [a if "ptr" in a else {"scalar": {"ctype": a["scalar"]["ctype"], "bytes": base64.b64encode(a["scalar"]["bytes"]).decode()}} for a in args]}
    r = subprocess.run(["node", RUNNER], input=json.dumps(job).encode(), capture_output=True, timeout=60)
    if r.returncode != 0: raise WorkerError("cudaErrorUnknown", r.stderr.decode()[-2000:], 500)
    out = json.loads(r.stdout)
    if out.get("error"):
      e = WorkerError(out["error"]["name"], out["error"]["message"], 500)
      e.sticky = out["error"]["name"] == "cudaErrorIllegalAddress"  # type: ignore[attr-defined]
      raise e
    for h, b in out["mem"].items(): self.mem[h][:] = base64.b64decode(b)
    if out.get("printf"): sys.stdout.write(out["printf"]); sys.stdout.flush()
    return {"ms": out["ms"] if timing else None}
  def close(self): self.mem.clear()
