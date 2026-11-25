/**
 * The JavaScript side of `tinygrad.cuda` (python/tinygrad_js/cuda.py), running in the tinygrad.js worker next to Pyodide.
 *
 * WEBGPU tensors: kernels run on tinygrad's own GPUDevice, recorded into the WebGPUBridge's command stream, reading and
 * writing the tensors' GPU buffers directly (zero-copy). CPU (PYTHON device) tensors: buffers are copied into the
 * WebAssembly backend's memory, the kernel runs, written buffers are copied back (an explicit transfer).
 */
import type { WebGPUBridge } from "../bridge.js";
import { CudaContext, type CudaModule, type KernelArg } from "./runtime/context.js";
import { WebGPUBackend } from "./runtime/webgpu.js";
import { WasmBackend } from "./runtime/wasm.js";
import { DevicePtr } from "./runtime/memory.js";
import { CompileError } from "./compiler/diagnostics.js";
import { CudaError } from "./runtime/errors.js";
import { printKernel, walkStmts, type Kernel } from "./compiler/ir.js";
import { sources } from "./kernels/index.js";

interface PyBufferProxy { getBuffer(t?: string): { data: Uint8Array; release(): void } }
type Dev = "webgpu" | "cpu";

export class TinygradCudaHost {
  private gpu: CudaContext | null = null;
  private cpu: CudaContext | null = null;
  private modules = new Map<number, { source: string; options: Record<string, unknown>; byDev: Map<Dev, CudaModule> }>();
  private nextModule = 1;
  private wrapped = new Map<number, DevicePtr>();
  private wrappedBuffers = new WeakMap<GPUBuffer, number>();
  readonly output: string[] = [];

  constructor(private bridge: WebGPUBridge | null) {}

  private ctx(dev: Dev): CudaContext {
    if (dev === "webgpu") {
      if (!this.bridge) throw new CudaError("cudaErrorNoDevice", "tinygrad.cuda: this runtime has no WebGPU device (tensors are on the CPU device)");
      if (!this.gpu) {
        const be = new WebGPUBackend(this.bridge.cudaQueue(), this.bridge.adapterDetails);
        this.gpu = new CudaContext(be, { requested: "webgpu", selected: "webgpu", reasons: ["shares tinygrad.js's WebGPU device"] });
        this.gpu.onPrintf = (t) => this.output.push(t);
      }
      return this.gpu;
    }
    if (!this.cpu) {
      // kernels run on this (worker) thread: the Pyodide worker is already off the main thread
      this.cpu = new CudaContext(new WasmBackend({ threads: 0 }), { requested: "cpu", selected: "wasm", reasons: ["tensors on tinygrad's CPU device"] });
      this.cpu.onPrintf = (t) => this.output.push(t);
    }
    return this.cpu;
  }

  private module(mid: number, dev: Dev): CudaModule {
    const m = this.modules.get(mid);
    if (!m) throw new CudaError("cudaErrorInvalidResourceHandle", `unknown CUDA module ${mid}`);
    let cm = m.byDev.get(dev);
    if (!cm) { cm = this.ctx(dev).compile(m.source, m.options); m.byDev.set(dev, cm); }
    return cm;
  }

  /** Parse and check a module. Returns its id; throws an Error with compiler diagnostics. */
  compile(source: string, optionsJSON: string): number {
    const options = JSON.parse(optionsJSON || "{}") as Record<string, unknown>;
    // parse once eagerly (device independent) so errors surface at CUDAKernel(...)
    const probe = new CudaContext(new NullBackend(), { requested: "", selected: "", reasons: [] }).compile(source, options as never);
    void probe;
    const id = this.nextModule++;
    this.modules.set(id, { source, options, byDev: new Map() });
    return id;
  }

  signatures(mid: number): string {
    const m = this.modules.get(mid)!;
    const probe = new CudaContext(new NullBackend(), { requested: "", selected: "", reasons: [] }).compile(m.source, m.options as never);
    return JSON.stringify(probe.kernels);
  }

  /**
   * Lower a kernel for a launch configuration and report which pointer parameters it reads and writes
   * (tinygrad needs this to order the kernel against other work).
   */
  describe(mid: number, ref: string, blockJSON: string, smem: number, dev: Dev): string {
    const block = JSON.parse(blockJSON) as [number, number, number];
    const f = this.module(mid, dev).getFunction(ref);
    const k = f.instantiate(block, smem);
    const written = new Set<number>(), read = new Set<number>();
    walkStmts(k.body, (s) => {
      if ((s.s === "store" || s.s === "atomic") && s.root.param !== undefined) written.add(s.root.param);
      if (s.s === "atomic" && s.root.param !== undefined) read.add(s.root.param);
    });
    walkStmts(k.body, (s) => {
      const visit = (e: unknown): void => {
        const x = e as { op: string; root?: { param?: number }; [k: string]: unknown };
        if (x.op === "load" && x.root?.param !== undefined) read.add(x.root.param);
        for (const v of Object.values(x)) if (v && typeof v === "object" && "op" in (v as object)) visit(v);
        for (const v of Object.values(x)) if (Array.isArray(v)) for (const y of v) if (y && typeof y === "object" && "op" in y) visit(y);
      };
      for (const e of exprsOf(s)) visit(e);
    });
    return JSON.stringify({
      kernel: k.displayName,
      params: k.params.map((p, i) => ({ name: p.name, kind: p.kind, ctype: p.ctype, origin: p.origin ?? null, written: written.has(i), read: read.has(i) })),
      cudaParams: f.signature.params,
    });
  }

  private ptrFor(handle: number): DevicePtr {
    const buf = this.bridge!.gpuBuffer(handle);
    const have = this.wrapped.get(handle);
    if (have && this.wrappedBuffers.get(buf) === handle && !have.alloc.freed) return have;
    const be = this.ctx("webgpu").backend as WebGPUBackend;
    const p = be.wrapBuffer(buf, buf.size, `tinygrad buffer ${handle}`);
    this.wrapped.set(handle, p);
    this.wrappedBuffers.set(buf, handle);
    if (this.wrapped.size > 4096) this.wrapped.clear();
    return p;
  }

  /** Launch on WEBGPU tensors: `handles` are WebGPUBridge buffer handles in pointer-parameter order. */
  async launchWebGPU(mid: number, ref: string, gridJSON: string, blockJSON: string, smem: number, handles: number[] | { toJs(): number[] }, argsJSON: string): Promise<void> {
    const hs = Array.isArray(handles) ? handles : handles.toJs();
    const m = this.module(mid, "webgpu");
    const args = this.buildArgs(m, ref, JSON.parse(argsJSON), hs.map((h) => this.ptrFor(Number(h))));
    await m.getFunction(ref).launch({ grid: JSON.parse(gridJSON), block: JSON.parse(blockJSON), sharedMem: smem, args });
  }

  /** Launch on CPU-device tensors: copies each buffer (memoryview proxies) in, runs the kernel, copies written ones back. */
  /**
   * Runs a kernel on the CPU backend over tinygrad buffers. Inputs are read through the buffer protocol; written
   * buffers are returned (in order) and copied back by the Python side, because Pyodide's getBuffer() may hand out a
   * temporary copy (e.g. for memoryviews of numpy-backed storage), in which case writes through it would be lost.
   */
  async launchCPU(mid: number, ref: string, gridJSON: string, blockJSON: string, smem: number, bufs: PyBufferProxy[] | { toJs(o?: unknown): PyBufferProxy[] }, writtenJSON: string, argsJSON: string): Promise<Uint8Array[]> {
    const proxies = Array.isArray(bufs) ? bufs : bufs.toJs({ depth: 1 });
    const ctx = this.ctx("cpu");
    const written = JSON.parse(writtenJSON) as boolean[];
    const ptrs: DevicePtr[] = [];
    const sizes: number[] = [];
    try {
      for (const p of proxies) {
        const b = p.getBuffer("u8");
        try {
          const ptr = ctx.malloc(Math.max(4, b.data.byteLength), "tinygrad cpu buffer");
          ptrs.push(ptr);
          sizes.push(b.data.byteLength);
          ctx.backend.upload(ptr, b.data);
        } finally { b.release(); }
      }
      const m = this.module(mid, "cpu");
      const args = this.buildArgs(m, ref, JSON.parse(argsJSON), ptrs);
      await m.getFunction(ref).launch({ grid: JSON.parse(gridJSON), block: JSON.parse(blockJSON), sharedMem: smem, args });
      const out: Uint8Array[] = [];
      for (let i = 0; i < ptrs.length; i++) if (written[i]) out.push(await ctx.backend.download(ptrs[i], sizes[i]));
      await ctx.synchronize();
      return out;
    } finally {
      for (const p of ptrs) ctx.free(p);
    }
  }

  /** CUDA-level argument list: pointer params take the next device pointer, others the JSON scalar / struct value. */
  private buildArgs(m: CudaModule, ref: string, scalars: unknown[], ptrs: DevicePtr[]): KernelArg[] {
    const sig = m.getFunction(ref).signature;
    let pi = 0, si = 0;
    return sig.params.map((p) => (p.pointer ? ptrs[pi++] : scalars[si++]) as KernelArg);
  }

  async synchronize(dev: Dev): Promise<string> {
    const c = dev === "webgpu" ? this.gpu : this.cpu;
    if (c) await c.synchronize();
    const out = this.output.join(""); this.output.length = 0;
    return out;
  }

  async inspect(mid: number, ref: string, blockJSON: string, smem: number, dev: Dev, what: string): Promise<string> {
    const r = await this.module(mid, dev).inspect(ref, JSON.parse(blockJSON), what === "ir" ? "ir" : "all", smem);
    return what === "ir" ? r.ir : r.code;
  }

  info(dev: Dev): string { return JSON.stringify(this.ctx(dev).info()); }

  /** CUDA source of a kernel library group (src/cuda/kernels), e.g. "matmul", "attention" */
  library(name: string): string {
    const src = (sources as Record<string, string>)[name];
    if (src === undefined) throw new CudaError("cudaErrorInvalidValue", `no kernel library '${name}' (have: ${Object.keys(sources).join(", ")})`);
    return src;
  }

  /** Format a compile/runtime error for Python. */
  static describeError(e: unknown): string {
    if (e instanceof CompileError) return `CompileError: ${e.message}`;
    if (e instanceof CudaError) return e.message;
    return e instanceof Error ? e.message : String(e);
  }
}

function exprsOf(s: unknown): unknown[] {
  const x = s as Record<string, unknown>;
  return ["e", "off", "c", "breakIf"].map((k) => x[k]).concat((x.args as unknown[]) ?? []).filter((v) => v && typeof v === "object");
}

/** A backend that only exists so modules can be parsed and checked without a device. */
class NullBackend {
  readonly kind = "wasm" as const;
  readonly id = -1;
  info() { return { limits: { maxThreadsPerBlock: 1024, maxBlockDim: [1024, 1024, 64], maxGridDim: [2 ** 31 - 1, 65535, 65535] } } as never; }
  malloc(): never { throw new Error("no device"); }
  free(): void {}
  upload(): void {}
  download(): never { throw new Error("no device"); }
  copy(): void {}
  fill(): void {}
  specKey() { return { key: "" }; }
  prepare(): never { throw new Error("no device"); }
  launch(): void {}
  synchronize() { return Promise.resolve({ printf: [], kernelTimes: [] }); }
  symbol(): never { throw new Error("no device"); }
  memoryStats() { return { allocations: 0, bytesAllocated: 0, peakBytes: 0, bytesReserved: 0, limit: 0 }; }
  destroy(): void {}
}

export { printKernel, type Kernel };
