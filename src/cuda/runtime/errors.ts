/** CUDA-style error codes. The numeric values follow cudaError_t so code ported from CUDA can compare them. */
export const cudaError = {
  cudaSuccess: 0,
  cudaErrorInvalidValue: 1,
  cudaErrorMemoryAllocation: 2,
  cudaErrorInitializationError: 3,
  cudaErrorCudartUnloading: 4,
  cudaErrorInvalidConfiguration: 9,
  cudaErrorInvalidDevicePointer: 17,
  cudaErrorInvalidMemcpyDirection: 21,
  cudaErrorInvalidDeviceFunction: 98,
  cudaErrorInvalidSymbol: 13,
  cudaErrorNoDevice: 100,
  cudaErrorInvalidDevice: 101,
  cudaErrorInvalidKernelImage: 200,
  cudaErrorDeviceUninitialized: 201,
  cudaErrorNoKernelImageForDevice: 209,
  cudaErrorInvalidPtx: 218,
  cudaErrorInvalidSource: 300,
  cudaErrorInvalidResourceHandle: 400,
  cudaErrorSymbolNotFound: 500,
  cudaErrorNotReady: 600,
  cudaErrorIllegalAddress: 700,
  cudaErrorLaunchOutOfResources: 701,
  cudaErrorLaunchTimeout: 702,
  cudaErrorAssert: 710,
  cudaErrorHardwareStackError: 714,
  cudaErrorIllegalInstruction: 715,
  cudaErrorMisalignedAddress: 716,
  cudaErrorInvalidAddressSpace: 717,
  cudaErrorInvalidPc: 718,
  cudaErrorLaunchFailure: 719,
  cudaErrorNotSupported: 801,
  cudaErrorDevicesUnavailable: 46,
  cudaErrorUnknown: 999,
} as const;
export type CudaErrorName = keyof typeof cudaError;

export class CudaError extends Error {
  readonly code: number;
  constructor(readonly errorName: CudaErrorName, message: string, readonly details?: unknown) {
    super(`${errorName}: ${message}`);
    this.name = "CudaError";
    this.code = cudaError[errorName];
  }
}

export function cudaGetErrorName(code: number): string {
  return (Object.entries(cudaError).find(([, v]) => v === code)?.[0]) ?? "cudaErrorUnknown";
}
export function cudaGetErrorString(code: number): string {
  const m: Record<string, string> = {
    cudaSuccess: "no error", cudaErrorInvalidValue: "invalid argument", cudaErrorMemoryAllocation: "out of memory",
    cudaErrorIllegalAddress: "an illegal memory access was encountered", cudaErrorAssert: "device-side assert triggered",
    cudaErrorLaunchFailure: "unspecified launch failure", cudaErrorInvalidConfiguration: "invalid configuration argument",
    cudaErrorNotSupported: "operation not supported", cudaErrorNoDevice: "no CUDA-capable device is detected",
    cudaErrorInvalidDevicePointer: "invalid device pointer", cudaErrorInvalidSource: "device kernel image is invalid",
    cudaErrorLaunchOutOfResources: "too many resources requested for launch", cudaErrorDevicesUnavailable: "all CUDA-capable devices are busy or unavailable",
    cudaErrorLaunchTimeout: "the launch timed out and was terminated", cudaErrorInitializationError: "initialization error",
    cudaErrorInvalidDeviceFunction: "invalid device function", cudaErrorInvalidResourceHandle: "invalid resource handle",
    cudaErrorMisalignedAddress: "misaligned address", cudaErrorIllegalInstruction: "an illegal instruction was encountered",
    cudaErrorNoKernelImageForDevice: "no kernel image is available for execution on the device", cudaErrorNotReady: "device not ready",
  };
  return m[cudaGetErrorName(code)] ?? "unknown error";
}
