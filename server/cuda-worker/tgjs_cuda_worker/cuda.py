"""Native CUDA engine: NVRTC (runtime compilation) + the CUDA driver API through ctypes. Runs inside a session's child process.

The kernel source is the user's original CUDA C++; it is compiled for the GPU's own architecture. Kernel names are looked
up with NVRTC name expressions, so template instantiations ("reduce<256>") and C++ mangling work as with nvcc.
"""
from __future__ import annotations
import ctypes, ctypes.util, hashlib, os
from .errors import WorkerError, CU_TO_CUDA, STICKY
from .source import check_includes

def _load(names: list[str]) -> ctypes.CDLL:
  errs = []
  for n in names:
    try: return ctypes.CDLL(n)
    except OSError as e: errs.append(str(e))
  raise WorkerError("cudaErrorNoDevice", f"CUDA library not found ({names[0]}): {'; '.join(errs[-1:])}", 503)

def _find_nvrtc() -> list[str]:
  names = [os.environ.get("TGJS_CUDA_NVRTC", "")] if os.environ.get("TGJS_CUDA_NVRTC") else []
  found = ctypes.util.find_library("nvrtc")
  if found: names.append(found)
  home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
  for d in (f"{home}/lib64", f"{home}/lib", "/usr/lib/x86_64-linux-gnu"):
    if os.path.isdir(d):
      names += sorted((os.path.join(d, f) for f in os.listdir(d) if f.startswith("libnvrtc.so")), reverse=True)
  return names + ["libnvrtc.so", "libnvrtc.so.12", "libnvrtc.so.11.2"]

# CUdevice_attribute values
ATTR = {"maxThreadsPerBlock": 1, "sharedMemPerBlock": 8, "multiProcessorCount": 16, "major": 75, "minor": 76, "sharedMemPerBlockOptin": 97}
CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8

class CudaEngine:
  def __init__(self, device_index: int = 0):
    self.cu = _load(["libcuda.so.1", "libcuda.so"])
    self.nvrtc = _load(_find_nvrtc())
    self.cu.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    self._chk(self.cu.cuInit(0), "cuInit")
    self.dev = ctypes.c_int()
    self._chk(self.cu.cuDeviceGet(ctypes.byref(self.dev), device_index), "cuDeviceGet")
    self.ctx = ctypes.c_void_p()
    self._chk(self.cu.cuDevicePrimaryCtxRetain(ctypes.byref(self.ctx), self.dev), "cuDevicePrimaryCtxRetain")
    self._chk(self.cu.cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent")
    self.mem: dict[str, tuple[int, int]] = {}  # handle -> (device pointer, bytes)
    self.modules: dict[str, ctypes.c_void_p] = {}
    self.functions: dict[tuple[str, str], ctypes.c_void_p] = {}
    for f in ("cuMemAlloc_v2", "cuMemFree_v2", "cuMemcpyHtoD_v2", "cuMemcpyDtoH_v2", "cuMemcpyDtoD_v2", "cuMemsetD8_v2"):
      getattr(self.cu, f).restype = ctypes.c_int
    self.cu.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
    self.cu.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
    self.cu.cuMemcpyDtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_size_t]
    self.cu.cuMemsetD8_v2.argtypes = [ctypes.c_uint64, ctypes.c_ubyte, ctypes.c_size_t]
    self.cu.cuMemFree_v2.argtypes = [ctypes.c_uint64]
    self.cu.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]

  # ------------------------------------------------------------------ helpers
  def _chk(self, r: int, what: str) -> None:
    if r == 0: return
    s = ctypes.c_char_p()
    try: self.cu.cuGetErrorName(r, ctypes.byref(s)); name = (s.value or b"?").decode()
    except Exception: name = str(r)
    err = WorkerError(CU_TO_CUDA.get(r, "cudaErrorUnknown"), f"{what} failed: {name} ({r})", 500 if r not in (1, 2) else 400)
    err.sticky = r in STICKY  # type: ignore[attr-defined]
    raise err

  def _attr(self, a: str) -> int:
    v = ctypes.c_int()
    self._chk(self.cu.cuDeviceGetAttribute(ctypes.byref(v), ATTR[a], self.dev), "cuDeviceGetAttribute")
    return v.value

  def device_info(self) -> dict:
    name = ctypes.create_string_buffer(256)
    self._chk(self.cu.cuDeviceGetName(name, 256, self.dev), "cuDeviceGetName")
    total = ctypes.c_size_t()
    self._chk(self.cu.cuDeviceTotalMem_v2(ctypes.byref(total), self.dev), "cuDeviceTotalMem")
    return {"name": name.value.decode(), "totalMem": total.value, "major": self._attr("major"), "minor": self._attr("minor"),
            "multiProcessorCount": self._attr("multiProcessorCount"), "sharedMemPerBlock": self._attr("sharedMemPerBlock"),
            "maxThreadsPerBlock": self._attr("maxThreadsPerBlock")}

  def _ptr(self, handle: str, offset: int, n: int) -> int:
    if handle not in self.mem: raise WorkerError("cudaErrorInvalidDevicePointer", f"unknown allocation {handle!r}")
    base, size = self.mem[handle]
    if offset < 0 or n < 0 or offset + n > size: raise WorkerError("cudaErrorInvalidValue", f"{n} bytes at offset {offset} exceed the allocation of {size} bytes")
    return base + offset

  # ------------------------------------------------------------------ memory
  def alloc(self, handle: str, nbytes: int) -> None:
    p = ctypes.c_uint64()
    self._chk(self.cu.cuMemAlloc_v2(ctypes.byref(p), max(1, nbytes)), "cuMemAlloc")
    # device memory is not cleared by the driver: zero it so no data from earlier sessions on this GPU can be read
    self._chk(self.cu.cuMemsetD8_v2(p.value, 0, max(1, nbytes)), "cuMemsetD8")
    self.mem[handle] = (p.value, nbytes)
  def free(self, handle: str) -> None:
    base, _ = self.mem.pop(handle, (0, 0))
    if base: self._chk(self.cu.cuMemFree_v2(base), "cuMemFree")
  def write(self, handle: str, offset: int, data: bytes) -> None:
    if data: self._chk(self.cu.cuMemcpyHtoD_v2(self._ptr(handle, offset, len(data)), data, len(data)), "cuMemcpyHtoD")
  def read(self, handle: str, offset: int, n: int) -> bytes:
    buf = ctypes.create_string_buffer(n)
    if n: self._chk(self.cu.cuMemcpyDtoH_v2(buf, self._ptr(handle, offset, n), n), "cuMemcpyDtoH")
    return buf.raw
  def copy(self, dst: str, dst_off: int, src: str, src_off: int, n: int) -> None:
    if n: self._chk(self.cu.cuMemcpyDtoD_v2(self._ptr(dst, dst_off, n), self._ptr(src, src_off, n), n), "cuMemcpyDtoD")
  def memset(self, handle: str, offset: int, value: int, n: int) -> None:
    if n: self._chk(self.cu.cuMemsetD8_v2(self._ptr(handle, offset, n), value & 0xFF, n), "cuMemsetD8")

  # ------------------------------------------------------------------ compilation
  def _compile(self, source: str, kernel: str, options: dict) -> ctypes.c_void_p:
    check_includes(source)  # also checked by the gateway; here too, for engines driven some other way
    major, minor = self._attr("major"), self._attr("minor")
    opts = [f"--gpu-architecture=sm_{major}{minor}", "-std=c++17", "-default-device", "-DTINYGRAD_JS_REMOTE=1"]
    for k, v in (options.get("defines") or {}).items():
      if not str(k).replace("_", "").isalnum(): raise WorkerError("cudaErrorInvalidValue", f"invalid macro name {k!r}")
      opts.append(f"-D{k}" if v is None or v == "" else f"-D{k}={v}")
    inc = os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "include")
    if os.path.isdir(inc): opts.append(f"--include-path={inc}")
    key = hashlib.sha256(("\0".join(opts) + "\0" + source).encode()).hexdigest()
    fkey = (key, kernel)
    if fkey in self.functions: return self.functions[fkey]
    nv = self.nvrtc
    prog = ctypes.c_void_p()
    r = nv.nvrtcCreateProgram(ctypes.byref(prog), source.encode(), b"kernel.cu", 0, None, None)
    if r: raise WorkerError("cudaErrorInvalidSource", f"nvrtcCreateProgram failed ({r})")
    try:
      if nv.nvrtcAddNameExpression(prog, kernel.encode()): raise WorkerError("cudaErrorInvalidValue", f"invalid kernel name {kernel!r}")
      arr = (ctypes.c_char_p * len(opts))(*[o.encode() for o in opts])
      r = nv.nvrtcCompileProgram(prog, len(opts), arr)
      n = ctypes.c_size_t(); nv.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
      log = ctypes.create_string_buffer(n.value); nv.nvrtcGetProgramLog(prog, log)
      if r: raise WorkerError("cudaErrorInvalidSource", f"NVRTC compilation failed:\n{log.value.decode(errors='replace').strip()}")
      if key not in self.modules:
        n = ctypes.c_size_t(); nv.nvrtcGetCUBINSize(prog, ctypes.byref(n))
        cubin = ctypes.create_string_buffer(n.value); nv.nvrtcGetCUBIN(prog, cubin)
        mod = ctypes.c_void_p()
        self._chk(self.cu.cuModuleLoadData(ctypes.byref(mod), cubin), "cuModuleLoadData")
        self.modules[key] = mod
      lowered = ctypes.c_char_p()
      if nv.nvrtcGetLoweredName(prog, kernel.encode(), ctypes.byref(lowered)): raise WorkerError("cudaErrorInvalidDeviceFunction", f"kernel {kernel!r} not found")
      fn = ctypes.c_void_p()
      self._chk(self.cu.cuModuleGetFunction(ctypes.byref(fn), self.modules[key], lowered.value), "cuModuleGetFunction")
    finally:
      nv.nvrtcDestroyProgram(ctypes.byref(prog))
    self.functions[fkey] = fn
    return fn

  # ------------------------------------------------------------------ launch
  def launch(self, source: str, kernel: str, options: dict, grid: list[int], block: list[int], shared_mem: int, args: list[dict], timing: bool) -> dict:
    fn = self._compile(source, kernel, options or {})
    if shared_mem > 48 * 1024:
      self._chk(self.cu.cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_mem), "cuFuncSetAttribute")
    holders = []
    for a in args:
      if "ptr" in a:
        holders.append(ctypes.c_uint64(self._ptr(a["ptr"]["handle"], int(a["ptr"]["offset"]), 0)))
      else:
        holders.append(ctypes.create_string_buffer(a["scalar"]["bytes"], len(a["scalar"]["bytes"])))
    params = (ctypes.c_void_p * max(1, len(holders)))(*[ctypes.cast(ctypes.byref(h), ctypes.c_void_p) for h in holders])
    ev = [ctypes.c_void_p(), ctypes.c_void_p()]
    if timing:
      for e in ev: self._chk(self.cu.cuEventCreate(ctypes.byref(e), 0), "cuEventCreate")
      self._chk(self.cu.cuEventRecord(ev[0], None), "cuEventRecord")
    g, b = (list(grid) + [1, 1, 1])[:3], (list(block) + [1, 1, 1])[:3]
    self._chk(self.cu.cuLaunchKernel(fn, *g, *b, shared_mem, None, params, None), "cuLaunchKernel")
    ms = None
    if timing: self._chk(self.cu.cuEventRecord(ev[1], None), "cuEventRecord")
    self._chk(self.cu.cuCtxSynchronize(), "kernel execution")
    if timing:
      t = ctypes.c_float()
      self._chk(self.cu.cuEventElapsedTime(ctypes.byref(t), ev[0], ev[1]), "cuEventElapsedTime")
      ms = t.value
      for e in ev: self.cu.cuEventDestroy_v2(e)
    return {"ms": ms}

  def close(self) -> None:
    for h in list(self.mem): self.free(h)
    for m in self.modules.values(): self.cu.cuModuleUnload(m)
    self.cu.cuDevicePrimaryCtxRelease(self.dev)
