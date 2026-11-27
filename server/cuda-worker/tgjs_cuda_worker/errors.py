"""Errors carry a CUDA runtime error name (cudaError*) that the tinygrad.js client turns into a CudaError."""
from __future__ import annotations

class WorkerError(Exception):
  def __init__(self, name: str, message: str, status: int = 400):
    super().__init__(message)
    self.name, self.message, self.status = name, message, status
  def to_json(self) -> dict: return {"error": {"name": self.name, "message": self.message}}

# CUresult (driver API) -> cudaError_t name (runtime API), for the errors kernels and memory operations can produce
CU_TO_CUDA = {
  1: "cudaErrorInvalidValue", 2: "cudaErrorMemoryAllocation", 3: "cudaErrorInitializationError", 4: "cudaErrorCudartUnloading",
  100: "cudaErrorNoDevice", 101: "cudaErrorInvalidDevice", 200: "cudaErrorInvalidKernelImage", 201: "cudaErrorDeviceUninitialized",
  209: "cudaErrorNoKernelImageForDevice", 218: "cudaErrorInvalidPtx", 300: "cudaErrorInvalidSource", 400: "cudaErrorInvalidResourceHandle",
  500: "cudaErrorSymbolNotFound", 700: "cudaErrorIllegalAddress", 701: "cudaErrorLaunchOutOfResources", 702: "cudaErrorLaunchTimeout",
  710: "cudaErrorAssert", 714: "cudaErrorHardwareStackError", 715: "cudaErrorIllegalInstruction", 716: "cudaErrorMisalignedAddress",
  717: "cudaErrorInvalidAddressSpace", 718: "cudaErrorInvalidPc", 719: "cudaErrorLaunchFailure", 999: "cudaErrorUnknown",
}
# errors after which the CUDA context is unusable: the session is ended (as a native CUDA process would have to exit)
STICKY = {700, 702, 710, 714, 715, 716, 717, 718, 719}
