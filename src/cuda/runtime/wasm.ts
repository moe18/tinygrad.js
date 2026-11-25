/**
 * WebAssembly CPU backend: runs CUDA kernels on the CPU, inside the browser (or Node.js).
 *
 * Device memory is one WebAssembly.Memory. When SharedArrayBuffer is available (cross-origin isolated pages, Node.js) it
 * is shared with a pool of workers (one per core) that execute thread blocks in parallel; otherwise kernels run on the
 * calling thread (call the API from a Web Worker to keep a page responsive). This is CPU emulation of the CUDA execution
 * model, not GPU acceleration.
 */
import { Allocation, AllocationTracker, DevicePtr, type MemoryStats } from "./memory.js";
import { CudaError } from "./errors.js";
import type { Backend, DeviceInfo, Executable, LaunchConfig, LaunchOptions, PrepareRequest, ResolvedArg, SpecKey, SyncReport } from "./backend.js";
import { generateWasm, type WasmProgram } from "../compiler/wasm/codegen.js";
import { encodeScalarBytes } from "./encode.js";
import { Executor, KernelFaultError, type KernelFault } from "./wasm-exec.js";
import { hashString } from "./hash.js";

const PAGE = 65536;
const HEAP_START = 64 * 1024;
const ALIGN = 256;

export interface WasmBackendOptions {
  /** maximum device memory in bytes (default 1 GiB; the WebAssembly memory reserves address space up to this) */
  memoryLimit?: number;
  /** worker threads: 0 = run on the calling thread; default: hardwareConcurrency (when SharedArrayBuffer is available) */
  threads?: number;
  /** URL / path of the bundled worker script (dist/cuda-worker.js) */
  workerURL?: string | URL;
}

interface WasmExe extends Executable { prog: WasmProgram; module: WebAssembly.Module; id: number }

interface PoolWorker { post(m: unknown, transfer?: unknown[]): void; terminate(): void; sent: Set<number>; onDone: ((m: DoneMsg) => void) | null }
interface DoneMsg { type: "done"; job: number; printf?: string[]; fault?: KernelFault; error?: { message: string }; ms?: number }

let nextId = 1;

export class WasmBackend implements Backend {
  readonly kind = "wasm" as const;
  readonly id = nextId++;
  readonly memory: WebAssembly.Memory;
  readonly shared: boolean;
  readonly tracker: AllocationTracker;
  private free_: { addr: number; size: number }[] = [];
  private top = HEAP_START;
  private maxPages: number;
  private tail: Promise<void> = Promise.resolve();
  private executor: Executor;
  private workers: PoolWorker[] = [];
  private workersReady: Promise<void> | null = null;
  private threads: number;
  private modules = new Map<string, WebAssembly.Module>();
  private moduleIds = new Map<WebAssembly.Module, number>();
  private nextModuleId = 1;
  private scratch: { addr: number; size: number } | null = null;
  private printfOut: string[] = [];
  private times: SyncReport["kernelTimes"] = [];
  private stickyError: CudaError | null = null;
  private symbols = new Map<string, DevicePtr>();
  private infoCache: DeviceInfo;
  private job = 0;

  constructor(private opts: WasmBackendOptions = {}) {
    const limit = opts.memoryLimit ?? 1024 ** 3;
    this.maxPages = Math.min(65536, Math.ceil(limit / PAGE) + 2);
    const canShare = typeof SharedArrayBuffer !== "undefined" && (typeof crossOriginIsolated === "undefined" || crossOriginIsolated || isNode());
    let mem: WebAssembly.Memory | null = null;
    if (canShare) { try { mem = new WebAssembly.Memory({ initial: 32, maximum: this.maxPages, shared: true }); } catch { mem = null; } }
    this.shared = mem !== null;
    this.memory = mem ?? new WebAssembly.Memory({ initial: 32, maximum: this.maxPages });
    this.tracker = new AllocationTracker({ id: this.id, kind: "wasm" }, limit);
    const hw = (globalThis.navigator as Navigator | undefined)?.hardwareConcurrency ?? 4;
    this.threads = this.shared && hasWorkers() ? Math.max(0, opts.threads ?? Math.min(16, hw)) : 0;
    this.executor = new Executor(this.memory);
    this.infoCache = {
      backend: "wasm", name: `CPU (WebAssembly, ${this.threads || 1} thread${this.threads > 1 ? "s" : ""})`, vendor: "", architecture: "wasm32",
      emulated: true, isFallbackAdapter: false, threads: Math.max(1, this.threads),
      limits: {
        maxThreadsPerBlock: 1024, maxBlockDim: [1024, 1024, 64], maxGridDim: [2 ** 31 - 1, 65535, 65535], sharedMemPerBlock: 48 * 1024,
        memoryLimit: limit, maxBufferSize: limit, maxPointerArgs: 64, warpSize: 32,
      },
      features: {
        fp16: "emulated", bf16: "emulated", fp64: "native", int64: "native", atomics64: true, warps: "sequential", subgroupSize: null,
        timestamps: false, printf: true, boundsCheck: "enforced",
      },
      notes: [
        "CPU emulation of the CUDA execution model: thread blocks run on CPU cores, CUDA threads are simulated in loops",
        this.threads > 0 ? `${this.threads} worker threads share the device memory (SharedArrayBuffer)` : "single-threaded: no SharedArrayBuffer/Workers (serve the page cross-origin isolated for multi-core execution); kernels run on the calling thread",
        "every memory access is bounds checked",
      ],
    };
  }

  info(): DeviceInfo { return this.infoCache; }
  private check(): void { if (this.stickyError) throw this.stickyError; }
  /** order an operation after all previously enqueued ones */
  private enqueue<T>(f: () => Promise<T> | T): Promise<T> {
    const p = this.tail.then(f);
    this.tail = p.then(() => {}, () => {});
    return p;
  }

  // ------------------------------------------------------------------ memory
  private allocRaw(bytes: number): number {
    const size = Math.ceil(Math.max(bytes, 1) / ALIGN) * ALIGN;
    for (let i = 0; i < this.free_.length; i++) {
      const b = this.free_[i];
      if (b.size >= size) {
        const addr = b.addr;
        if (b.size === size) this.free_.splice(i, 1); else { b.addr += size; b.size -= size; }
        return addr;
      }
    }
    const addr = this.top;
    const end = addr + size;
    const need = Math.ceil(end / PAGE);
    const have = this.memory.buffer.byteLength / PAGE;
    if (need > have) {
      if (need > this.maxPages) throw new CudaError("cudaErrorMemoryAllocation", `out of CPU device memory (limit ${this.tracker.limit} bytes)`);
      this.memory.grow(Math.max(need - have, Math.min(this.maxPages - have, Math.ceil(have / 4))));
    }
    this.top = end;
    return addr;
  }
  private freeRaw(addr: number, bytes: number): void {
    const size = Math.ceil(Math.max(bytes, 1) / ALIGN) * ALIGN;
    this.free_.push({ addr, size });
    this.free_.sort((a, b) => a.addr - b.addr);
    const merged: { addr: number; size: number }[] = [];
    for (const b of this.free_) {
      const last = merged[merged.length - 1];
      if (last && last.addr + last.size === b.addr) last.size += b.size; else merged.push({ ...b });
    }
    if (merged.length && merged[merged.length - 1].addr + merged[merged.length - 1].size === this.top) { this.top = merged[merged.length - 1].addr; merged.pop(); }
    this.free_ = merged;
  }

  malloc(bytes: number, label = ""): DevicePtr {
    this.check();
    this.tracker.reserve(bytes);
    const addr = this.allocRaw(bytes);
    const a = new Allocation(this.tracker.owner, bytes, addr, label);
    this.tracker.register(a);
    return new DevicePtr(a, 0);
  }
  free(p: DevicePtr): void {
    if (p.offset !== 0) throw new CudaError("cudaErrorInvalidDevicePointer", "cudaFree needs the pointer returned by cudaMalloc (offset 0)");
    if (p.alloc.freed) throw new CudaError("cudaErrorInvalidDevicePointer", `double free of allocation #${p.alloc.id}`);
    const a = p.alloc;
    this.tracker.release(a);
    // the memory is reused only after work that may still use it completes
    void this.enqueue(() => this.freeRaw(a.native as number, a.size));
  }
  private addr(p: DevicePtr): number { return (p.alloc.native as number) + p.offset; }

  upload(dst: DevicePtr, data: Uint8Array): void {
    this.check();
    this.tracker.validate(dst, data.byteLength, "cudaMemcpy (host to device)");
    const copy = data.slice();
    const at = this.addr(dst);
    void this.enqueue(() => { new Uint8Array(this.memory.buffer, at, copy.byteLength).set(copy); });
  }
  download(src: DevicePtr, bytes: number): Promise<Uint8Array> {
    this.check();
    this.tracker.validate(src, bytes, "cudaMemcpy (device to host)");
    const at = this.addr(src);
    return this.enqueue(() => { this.check(); return new Uint8Array(this.memory.buffer, at, bytes).slice(); });
  }
  copy(dst: DevicePtr, src: DevicePtr, bytes: number): void {
    this.check();
    this.tracker.validate(dst, bytes, "cudaMemcpy (device to device) destination");
    this.tracker.validate(src, bytes, "cudaMemcpy (device to device) source");
    const d = this.addr(dst), s = this.addr(src);
    void this.enqueue(() => { new Uint8Array(this.memory.buffer).copyWithin(d, s, s + bytes); });
  }
  fill(dst: DevicePtr, byte: number, bytes: number): void {
    this.check();
    this.tracker.validate(dst, bytes, "cudaMemset");
    const d = this.addr(dst);
    void this.enqueue(() => { new Uint8Array(this.memory.buffer, d, bytes).fill(byte & 0xff); });
  }
  /** direct view of device memory (synchronous; only valid while no kernel is running) */
  view(p: DevicePtr, bytes: number): Uint8Array { return new Uint8Array(this.memory.buffer, this.addr(p), bytes); }

  symbol(moduleId: number, name: string, bytes: number, init?: Uint8Array): DevicePtr {
    const key = `${moduleId}:${name}`;
    let p = this.symbols.get(key);
    if (!p) {
      p = this.malloc(Math.max(1, bytes), `symbol ${name}`);
      if (init) this.upload(p, init); else this.fill(p, 0, p.alloc.size);
      this.symbols.set(key, p);
    }
    return p;
  }

  memoryStats(): MemoryStats {
    return { allocations: this.tracker.count, bytesAllocated: this.tracker.bytes, peakBytes: this.tracker.peak, bytesReserved: this.memory.buffer.byteLength, limit: this.tracker.limit };
  }

  // ------------------------------------------------------------------ compilation
  specKey(_args: ResolvedArg[]): SpecKey { return { key: "" }; }

  async prepare(req: PrepareRequest): Promise<Executable> {
    const prog = generateWasm(req.kernel, { shared: this.shared, maxMemoryPages: this.maxPages }, req.sink);
    const key = hashString(String.fromCharCode(...prog.bytes.subarray(0, 0)) + Array.prototype.join.call(prog.bytes, ","));
    let module = this.modules.get(key);
    if (!module) {
      try { module = await WebAssembly.compile(prog.bytes as BufferSource); }
      catch (e) { throw new CudaError("cudaErrorInvalidSource", `${req.kernel.displayName}: the generated WebAssembly was rejected (compiler bug): ${(e as Error).message}`); }
      this.modules.set(key, module);
      this.moduleIds.set(module, this.nextModuleId++);
    }
    return { kernel: req.kernel, code: prog.text, notes: prog.notes, prog, module, id: this.moduleIds.get(module)! } as WasmExe;
  }

  // ------------------------------------------------------------------ workers
  private startWorkers(): Promise<void> {
    if (this.workersReady) return this.workersReady;
    this.workersReady = (async () => {
      const url = this.opts.workerURL ?? new URL("./cuda-worker.js", import.meta.url);
      const ready: Promise<void>[] = [];
      for (let i = 0; i < this.threads; i++) {
        const w = await spawnWorker(url);
        this.workers.push(w);
        ready.push(new Promise<void>((res) => { w.onDone = () => res(); }));
        w.post({ type: "init", memory: this.memory, index: i });
      }
      await Promise.all(ready);
      for (const w of this.workers) w.onDone = null;
    })().catch((e) => {
      // fall back to running on this thread
      this.threads = 0;
      this.infoCache.notes.push(`worker pool unavailable (${(e as Error).message}); running kernels on the calling thread`);
      this.infoCache.threads = 1;
    });
    return this.workersReady;
  }

  // ------------------------------------------------------------------ launch
  launch(exe0: Executable, cfg: LaunchConfig, args: ResolvedArg[], opts: LaunchOptions = {}): void {
    this.check();
    const exe = exe0 as WasmExe;
    const prog = exe.prog;
    // validate pointers now (synchronously), run later in order
    for (const p of prog.args.pointers) {
      const a = args[p.param];
      if (a.kind !== "ptr") throw new CudaError("cudaErrorInvalidValue", `argument ${p.param + 1} must be a device pointer`);
      a.ptr.alloc.check();
    }
    const block = new Uint8Array(prog.args.size);
    const dv = new DataView(block.buffer);
    cfg.grid.forEach((g, i) => dv.setUint32(prog.args.grid + 4 * i, g, true));
    const blocks = cfg.grid[0] * cfg.grid[1] * cfg.grid[2];
    const workers = Math.max(1, this.threads);
    dv.setUint32(prog.args.chunk, Math.max(1, Math.min(64, Math.floor(blocks / (workers * 8)))), true);
    for (const p of prog.args.pointers) {
      const a = args[p.param] as ResolvedArg & { kind: "ptr" };
      dv.setUint32(p.base, this.addr(a.ptr), true);
      dv.setUint32(p.limit, a.ptr.remaining, true);
    }
    for (const s of prog.args.symbols) {
      const sp = opts.symbols?.get(s.symbol);
      if (!sp) throw new CudaError("cudaErrorInvalidSymbol", `symbol ${s.symbol} has no device memory`);
      dv.setUint32(s.base, this.addr(sp), true);
      dv.setUint32(s.limit, sp.remaining, true);
    }
    for (const s of prog.args.scalars) {
      const a = args[s.param];
      if (a.kind !== "scalar") throw new CudaError("cudaErrorInvalidValue", `argument ${s.param + 1} must be a scalar`);
      block.set(encodeScalarBytes(s.t, a.value), s.off);
    }
    const timing = opts.timing;
    void this.enqueue(() => this.execute(exe, block, timing));
  }

  private async execute(exe: WasmExe, block: Uint8Array, timing?: boolean): Promise<void> {
    if (this.stickyError) return;
    const prog = exe.prog;
    if (this.threads > 0) await this.startWorkers();
    const nw = Math.max(1, this.threads);
    const argsAddr = this.allocRaw(block.byteLength);
    new Uint8Array(this.memory.buffer, argsAddr, block.byteLength).set(block);
    const per = Math.ceil(Math.max(prog.scratchBytes, 16) / ALIGN) * ALIGN;
    if (!this.scratch || this.scratch.size < per * nw) {
      if (this.scratch) this.freeRaw(this.scratch.addr, this.scratch.size);
      this.scratch = { addr: this.allocRaw(per * nw), size: per * nw };
    }
    const t0 = performance.now();
    let fault: KernelFault | null = null, error: string | null = null;
    try {
      if (this.threads === 0) {
        try { this.printfOut.push(...this.executor.run(exe.id, { module: exe.module, printf: prog.printf }, argsAddr, this.scratch.addr, 0)); }
        catch (e) { if (e instanceof KernelFaultError) fault = e.fault; else error = (e as Error).message; }
      } else {
        const job = ++this.job;
        const results = await Promise.all(this.workers.map((w, i) => new Promise<DoneMsg>((res) => {
          w.onDone = (m) => { if (m.job === job) { w.onDone = null; res(m); } };
          const first = !w.sent.has(exe.id);
          w.sent.add(exe.id);
          w.post({ type: "run", job, id: exe.id, module: first ? exe.module : undefined, printf: first ? prog.printf : undefined, args: argsAddr, scratch: this.scratch!.addr + i * per });
        })));
        for (const r of results) {
          if (r.printf) this.printfOut.push(...r.printf);
          if (r.fault && !fault) fault = r.fault;
          if (r.error && !error) error = r.error.message;
        }
      }
    } finally {
      this.freeRaw(argsAddr, block.byteLength);
    }
    if (timing) this.times.push({ name: exe.kernel.displayName, ms: null, wallMs: performance.now() - t0 });
    if (fault) this.stickyError = this.faultError(exe, fault, block);
    else if (error) this.stickyError = new CudaError("cudaErrorLaunchFailure", `${exe.kernel.displayName}: ${error}`);
  }

  private faultError(exe: WasmExe, f: KernelFault, block: Uint8Array): CudaError {
    const dv = new DataView(block.buffer);
    const g = [0, 1, 2].map((i) => dv.getUint32(exe.prog.args.grid + 4 * i, true));
    const [bx, by] = exe.kernel.block;
    const tid = f.tid >= 0 ? `(${f.tid % bx},${Math.floor(f.tid / bx) % by},${Math.floor(f.tid / (bx * by))})` : "?";
    const bid = f.bid >= 0 ? `(${f.bid % g[0]},${Math.floor(f.bid / g[0]) % g[1]},${Math.floor(f.bid / (g[0] * g[1]))})` : "?";
    const where = `thread ${tid} in block ${bid}`;
    if (f.site === -2) return new CudaError("cudaErrorLaunchFailure", `${exe.kernel.displayName}: integer division by zero (or INT_MIN / -1)`);
    const site = exe.prog.sites[f.site];
    if (!site) return new CudaError("cudaErrorLaunchFailure", `${exe.kernel.displayName}: kernel fault`);
    if (site.kind === "assert") return new CudaError("cudaErrorAssert", `${exe.kernel.displayName}: ${site.msg} [${where}]`);
    if (site.kind === "trap") return new CudaError("cudaErrorLaunchFailure", `${exe.kernel.displayName}: ${site.msg} [${where}]`);
    const verb = site.kind === "oob-store" ? "store to" : site.kind === "oob-atomic" ? "atomic on" : "load from";
    return new CudaError("cudaErrorIllegalAddress", `${exe.kernel.displayName}: out-of-bounds ${verb} '${site.root}' at byte offset ${f.value} by ${where}`);
  }

  async synchronize(): Promise<SyncReport> {
    await this.tail;
    const printf = this.printfOut; this.printfOut = [];
    const kernelTimes = this.times; this.times = [];
    if (this.stickyError) throw this.stickyError;
    return { printf, kernelTimes };
  }

  resetError(): void { this.stickyError = null; }

  destroy(): void {
    for (const w of this.workers) w.terminate();
    this.workers = [];
  }
}

function isNode(): boolean { return Boolean((globalThis as { process?: { versions?: { node?: string } } }).process?.versions?.node); }
function hasWorkers(): boolean { return typeof Worker !== "undefined" || isNode(); }

async function spawnWorker(url: string | URL): Promise<PoolWorker> {
  const sent = new Set<number>();
  if (typeof Worker !== "undefined" && !isNode()) {
    const w = new Worker(url, { type: "module", name: "tinygrad.js-cuda-cpu" });
    const pw: PoolWorker = { post: (m) => w.postMessage(m), terminate: () => w.terminate(), sent, onDone: null };
    w.onmessage = (ev) => pw.onDone?.(ev.data);
    w.onerror = (ev) => { ev.preventDefault?.(); pw.onDone?.({ type: "done", job: -1, error: { message: ev.message || "worker failed to load" } }); };
    return pw;
  }
  interface NodeWorkerLike { postMessage(m: unknown): void; terminate(): unknown; on(ev: string, f: (x: never) => void): void }
  const NODE_WT = "node:worker_threads";
  const { Worker: NodeWorker } = await import(/* webpackIgnore: true */ /* @vite-ignore */ NODE_WT) as { Worker: new (u: string | URL) => NodeWorkerLike };
  const path = typeof url === "string" ? url : url.protocol === "file:" ? new URL(url) : url.href;
  const w = new NodeWorker(path as string | URL);
  const pw: PoolWorker = { post: (m) => w.postMessage(m), terminate: () => void w.terminate(), sent, onDone: null };
  w.on("message", (m: DoneMsg) => pw.onDone?.(m));
  w.on("error", (e: Error) => pw.onDone?.({ type: "done", job: -1, error: { message: e.message } }));
  return pw;
}
