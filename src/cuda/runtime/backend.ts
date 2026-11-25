/** The interface every execution backend (WebGPU, WebAssembly CPU, remote native CUDA) implements. */
import type { Kernel, VT } from "../compiler/ir.js";
import type { DevicePtr, MemoryStats } from "./memory.js";
import type { DiagnosticSink } from "../compiler/diagnostics.js";

export type BackendKind = "webgpu" | "wasm" | "cuda_remote";

export interface LaunchConfig {
  grid: [number, number, number];
  block: [number, number, number];
  /** dynamic shared memory in bytes (extern __shared__) */
  sharedMem: number;
}

export type ScalarValue = number | bigint | boolean | number[] | bigint[];
export type ResolvedArg = { kind: "ptr"; ptr: DevicePtr } | { kind: "scalar"; value: ScalarValue; t: VT };

export interface FeatureSupport {
  fp16: "native" | "emulated";
  bf16: "emulated";
  fp64: "native" | "unsupported";
  int64: "native" | "emulated";
  atomics64: boolean;
  /** logical warps of 32 lanes: how warp collectives execute */
  warps: "subgroups" | "emulated" | "sequential";
  subgroupSize: [number, number] | null;
  timestamps: boolean;
  printf: boolean;
  boundsCheck: "report" | "enforced";
}

export interface DeviceInfo {
  backend: BackendKind;
  name: string;
  vendor: string;
  architecture: string;
  /** true when "GPU" code actually runs on the CPU (WASM backend, or a software WebGPU adapter like SwiftShader) */
  emulated: boolean;
  isFallbackAdapter: boolean;
  limits: {
    maxThreadsPerBlock: number;
    maxBlockDim: [number, number, number];
    maxGridDim: [number, number, number];
    sharedMemPerBlock: number;
    memoryLimit: number;
    maxBufferSize: number;
    maxPointerArgs: number;
    warpSize: 32;
  };
  features: FeatureSupport;
  /** CPU worker threads (WASM backend) */
  threads?: number;
  notes: string[];
}

export interface Executable {
  readonly kernel: Kernel;
  /** human-readable generated code: WGSL source, WebAssembly text summary, or the original CUDA for remote */
  readonly code: string;
  readonly notes: string[];
}

export interface SyncReport {
  printf: string[];
  /** GPU kernel time of timed launches since the previous synchronize, in ms (null if timestamps are unavailable) */
  kernelTimes: { name: string; ms: number | null; wallMs?: number }[];
}

export interface LaunchOptions {
  /** measure this launch's execution time (GPU timestamp queries on WebGPU, worker wall time on WASM) */
  timing?: boolean;
  label?: string;
  /** device memory of the module's __constant__ / __device__ variables, by name */
  symbols?: Map<string, DevicePtr>;
}

export interface PrepareRequest {
  kernel: Kernel;
  /** original CUDA source and kernel reference (used by the remote backend, which compiles with NVRTC) */
  source?: string;
  ref?: string;
  defines?: Record<string, string>;
  spec: SpecKey;
  options: BackendCompileOptions;
  sink: DiagnosticSink;
}

export interface BackendCompileOptions {
  boundsCheck?: boolean;
  fp64?: "error" | "demote";
  warps?: "auto" | "emulate" | "subgroups";
  simd?: boolean;
}

/** Backend-specific parts of the specialization key, computed from the actual launch arguments. */
export interface SpecKey { aliasGroups?: number[]; zeroBase?: boolean[]; key: string }

export interface Backend {
  readonly kind: BackendKind;
  readonly id: number;
  info(): DeviceInfo;
  malloc(bytes: number, label?: string): DevicePtr;
  free(p: DevicePtr): void;
  /** enqueue a host-to-device copy (the data is captured synchronously) */
  upload(dst: DevicePtr, data: Uint8Array): void;
  /** device-to-host copy ordered after all previously enqueued work */
  download(src: DevicePtr, bytes: number): Promise<Uint8Array>;
  copy(dst: DevicePtr, src: DevicePtr, bytes: number): void;
  fill(dst: DevicePtr, byte: number, bytes: number): void;
  /** backend-specific specialization for these arguments (e.g. aliasing on WebGPU) */
  specKey(args: ResolvedArg[]): SpecKey;
  prepare(req: PrepareRequest): Promise<Executable>;
  launch(exe: Executable, cfg: LaunchConfig, args: ResolvedArg[], opts?: LaunchOptions): void;
  synchronize(): Promise<SyncReport>;
  /** module-scope __constant__ / __device__ variables */
  symbol(moduleId: number, name: string, bytes: number, init?: Uint8Array): DevicePtr;
  memoryStats(): MemoryStats;
  destroy(): void;
}
