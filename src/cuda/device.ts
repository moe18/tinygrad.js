/**
 * Device selection: AUTO, WEBGPU, CPU / WASM, CUDA_REMOTE.
 *
 * AUTO uses WebGPU when a hardware adapter is available, and the WebAssembly CPU backend otherwise (no GPU, no WebGPU, or
 * only a software WebGPU adapter). It never selects a remote GPU: CUDA_REMOTE must be requested explicitly with a
 * configured endpoint. The selection and its reasons are reported in `ctx.selection`.
 */
import { CudaContext, type DeviceSelection } from "./runtime/context.js";
import { WebGPUBackend, type GpuQueue } from "./runtime/webgpu.js";
import { WasmBackend } from "./runtime/wasm.js";
import { RemoteCudaBackend, type RemoteCudaConfig } from "./runtime/remote.js";
import { CudaError } from "./runtime/errors.js";
import type { Backend } from "./runtime/backend.js";
import { parseModule } from "./compiler/module.js";
import { generateWGSL, type WGSLCaps } from "./compiler/wgsl.js";
import { CompileError, DiagnosticSink } from "./compiler/diagnostics.js";

export type DeviceKind = "auto" | "webgpu" | "cpu" | "wasm" | "cuda_remote";
export const Devices = { AUTO: "auto", WEBGPU: "webgpu", CPU: "cpu", WASM: "wasm", CUDA_REMOTE: "cuda_remote" } as const;

export interface ContextOptions {
  device?: DeviceKind;
  /** device memory limit in bytes */
  memoryLimit?: number;
  /** CPU backend worker threads (0 = calling thread) */
  threads?: number;
  /** URL of dist/cuda-worker.js when bundlers can't resolve it */
  workerURL?: string | URL;
  /** AUTO: also use a software WebGPU adapter (SwiftShader) instead of the CPU backend */
  allowSoftwareWebGPU?: boolean;
  /** share an existing GPUDevice (and its command stream) instead of creating one */
  gpu?: { queue: GpuQueue; adapterInfo?: ConstructorParameters<typeof WebGPUBackend>[1] };
  /** required for device: "cuda_remote" */
  remote?: RemoteCudaConfig;
  /**
   * AUTO: kernels the program will launch, checked against the GPU's capabilities before choosing it
   * (e.g. double precision or blocks larger than the GPU's workgroup limit select the CPU backend).
   */
  requirements?: { source: string; kernels: { name: string; block: [number, number?, number?] }[] }[];
}

export async function createCudaContext(opts: ContextOptions = {}): Promise<CudaContext> {
  const requested = opts.device ?? "auto";
  const reasons: string[] = [];
  let backend: Backend | null = null;
  const sel = (b: Backend, name: string): CudaContext => new CudaContext(b, { requested, selected: name, reasons } satisfies DeviceSelection);
  if (requested === "cuda_remote") {
    if (!opts.remote) throw new CudaError("cudaErrorInvalidValue", "device 'cuda_remote' needs a remote configuration ({ remote: { endpoint, token } })");
    backend = await RemoteCudaBackend.connect(opts.remote);
    reasons.push(`remote NVIDIA GPU requested explicitly (${new URL(opts.remote.endpoint).host})`);
    return sel(backend, "cuda_remote");
  }
  if (requested === "cpu" || requested === "wasm") {
    backend = new WasmBackend({ memoryLimit: opts.memoryLimit, threads: opts.threads, workerURL: opts.workerURL });
    reasons.push("CPU backend requested explicitly");
    return sel(backend, "wasm");
  }
  // webgpu or auto
  let gpuError: string | null = null;
  let gpu: WebGPUBackend | null = null;
  try {
    gpu = opts.gpu ? new WebGPUBackend(opts.gpu.queue, opts.gpu.adapterInfo ?? {}, { memoryLimit: opts.memoryLimit }) : await WebGPUBackend.create({ memoryLimit: opts.memoryLimit });
  } catch (e) { gpuError = (e as Error).message; }
  if (requested === "webgpu") {
    if (!gpu) throw new CudaError("cudaErrorNoDevice", `WebGPU requested but unavailable: ${gpuError}`);
    reasons.push("WebGPU requested explicitly");
    if (gpu.info().isFallbackAdapter) reasons.push("note: the WebGPU adapter is a software implementation (runs on the CPU)");
    return sel(gpu, "webgpu");
  }
  // auto
  if (!gpu) reasons.push(`WebGPU unavailable (${gpuError}): using the CPU backend`);
  else if (gpu.info().isFallbackAdapter && !opts.allowSoftwareWebGPU) {
    reasons.push(`the only WebGPU adapter is a software implementation (${gpu.info().name}): the WebAssembly CPU backend runs kernels directly on the CPU cores instead`);
    gpu.destroy(); gpu = null;
  } else {
    const blockers = gpu ? checkRequirements(gpu.caps, opts.requirements ?? []) : [];
    if (blockers.length) {
      reasons.push(...blockers.map((b) => `kernel needs the CPU backend: ${b}`));
      gpu!.destroy(); gpu = null;
    }
  }
  if (gpu) { reasons.push(`WebGPU adapter: ${gpu.info().name}`); return sel(gpu, "webgpu"); }
  backend = new WasmBackend({ memoryLimit: opts.memoryLimit, threads: opts.threads, workerURL: opts.workerURL });
  return sel(backend, "wasm");
}

/** Check kernels against WebGPU capabilities without a device (reasons why WebGPU can't run them). */
export function checkRequirements(caps: WGSLCaps, reqs: NonNullable<ContextOptions["requirements"]>): string[] {
  const out: string[] = [];
  for (const r of reqs) {
    let m;
    try { m = parseModule(r.source, { backend: "wgsl" }); } catch (e) { out.push(`compile error: ${(e as Error).message}`); continue; }
    for (const k of r.kernels) {
      const block: [number, number, number] = [k.block[0], k.block[1] ?? 1, k.block[2] ?? 1];
      try {
        const ir = m.instantiate(k.name, { block, dynamicShared: 0 });
        generateWGSL(ir, caps, {}, new DiagnosticSink());
      } catch (e) {
        if (e instanceof CompileError && e.code === "backend") out.push(`${k.name}: ${e.diagnostics[0].message}`);
        else if (!(e instanceof CompileError)) out.push(`${k.name}: ${(e as Error).message}`);
      }
    }
  }
  return out;
}
