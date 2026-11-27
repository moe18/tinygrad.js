/**
 * tinygrad.js CUDA: compile and run CUDA C/C++ kernels in the browser on WebGPU, on the CPU (WebAssembly), or on a remote
 * NVIDIA GPU.
 *
 * ```ts
 * import { CUDA } from "tinygrad.js/cuda";
 * const kernel = await CUDA.compile(`__global__ void vector_add(float* a, float* b, float* c, int n) {
 *   int i = blockIdx.x * blockDim.x + threadIdx.x;
 *   if (i < n) c[i] = a[i] + b[i];
 * }`);
 * const ctx = kernel.ctx;
 * const a = ctx.rand([1024]), b = ctx.rand([1024]), c = ctx.zeros([1024]);
 * await kernel.launch({ grid: [4, 1, 1], block: [256, 1, 1], args: [a, b, c, 1024] });
 * console.log(await c.toArray());
 * ```
 */
import { cudaContext, cudaConfigure } from "./api.js";
import { createCudaContext, type ContextOptions } from "./device.js";
import type { CompileOptions, CudaModule } from "./runtime/context.js";

export * from "./api.js";
export { createCudaContext, checkRequirements, Devices, type ContextOptions, type DeviceKind } from "./device.js";
export { CudaContext, CudaModule, CudaFunction, DeviceTensor, Stream, CudaEvent, deviceProp, toHost } from "./runtime/context.js";
export type { CompileOptions, LaunchParams, LaunchInfo, KernelArg, HostArray, Dim3, CudaDeviceProp, DeviceSelection } from "./runtime/context.js";
export { DevicePtr, Allocation, DTYPE_SIZE, type DType } from "./runtime/memory.js";
export { CudaError, cudaError } from "./runtime/errors.js";
export type { Backend, DeviceInfo, FeatureSupport, SyncReport, LaunchConfig } from "./runtime/backend.js";
export { WebGPUBackend, OwnQueue, type GpuQueue } from "./runtime/webgpu.js";
export { WasmBackend } from "./runtime/wasm.js";
export { RemoteCudaBackend, type RemoteCudaConfig } from "./runtime/remote.js";
export { CompileError, type Diagnostic } from "./compiler/diagnostics.js";
export { parseModule, COMPILER_VERSION, type ParsedModule, type KernelSignature } from "./compiler/module.js";
export { printKernel, type Kernel } from "./compiler/ir.js";
export { generateWGSL, type WGSLCaps, DEFAULT_WGSL_CAPS } from "./compiler/wgsl.js";
export { generateWasm } from "./compiler/wasm/codegen.js";
export { serializeTensors, deserializeTensors, serializeKernel, deserializeKernel, deviceReport, type KernelBundle } from "./serialize.js";
export { autotune, type TuneCandidate, type TuneResult } from "./autotune.js";
export { formatPrintf } from "./runtime/encode.js";
export * as kernels from "./kernels/index.js";
export * as cublas from "./libs/cublas.js";
export * as cudnn from "./libs/cudnn.js";
export * as curand from "./libs/curand.js";

/** The convenience namespace from the README examples. */
export const CUDA = {
  /** configure the default device context (device, memory limit, threads, remote) before first use */
  configure(opts: ContextOptions): void { cudaConfigure(opts); },
  /** create an independent device context */
  context(opts: ContextOptions = {}) { return createCudaContext(opts); },
  /** the default context (created on first use with AUTO device selection) */
  defaultContext() { return cudaContext(); },
  /** compile CUDA source in the default context; `module.launch(...)` launches its kernel */
  async compile(source: string, options: CompileOptions = {}): Promise<CudaModule> {
    return (await cudaContext()).compile(source, options);
  },
};
export default CUDA;
