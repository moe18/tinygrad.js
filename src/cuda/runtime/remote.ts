/**
 * Remote native CUDA backend: runs the original CUDA source with NVIDIA's toolchain on a remote GPU worker.
 *
 * Provider independent: the browser talks to a CUDA worker gateway with the small session protocol below. The reference
 * worker (server/cuda-worker/) runs self-hosted, on Modal or on RunPod; its provider credentials stay on the server. The
 * browser only holds a short-lived session token issued by your own backend. Nothing here is used unless a remote device
 * is configured explicitly; AUTO device selection never picks it.
 *
 *   POST   /v1/sessions                                   -> { session, device, limits }
 *   POST   /v1/sessions/:s/alloc        { handle, bytes }
 *   PUT    /v1/sessions/:s/mem/:handle?offset=N            body: raw bytes
 *   GET    /v1/sessions/:s/mem/:handle?offset=N&bytes=M    -> raw bytes
 *   POST   /v1/sessions/:s/free         { handle }
 *   POST   /v1/sessions/:s/copy         { dst, dstOffset, src, srcOffset, bytes }
 *   POST   /v1/sessions/:s/memset       { handle, offset, value, bytes }
 *   POST   /v1/sessions/:s/launch       { source, kernel, options, grid, block, sharedMem, args } -> { ms, printf, error? }
 *   DELETE /v1/sessions/:s
 */
import { Allocation, AllocationTracker, DevicePtr, type MemoryStats } from "./memory.js";
import { CudaError } from "./errors.js";
import type { Backend, DeviceInfo, Executable, LaunchConfig, LaunchOptions, PrepareRequest, ResolvedArg, SpecKey, SyncReport } from "./backend.js";
import { encodeScalarBytes } from "./encode.js";

export interface RemoteCudaConfig {
  /** base URL of the CUDA worker gateway (your server, or a self-hosted worker) */
  endpoint: string;
  /** session token from your own backend (never a cloud provider API key) */
  token?: string | (() => Promise<string>);
  /** maximum device memory for this session (the worker enforces its own limit too) */
  memoryLimit?: number;
  /** per-launch execution time limit requested from the worker (ms) */
  timeLimitMs?: number;
  /** GPU selection hint passed to the worker (e.g. "A10G", "any") */
  gpu?: string;
  /** fetch implementation (tests, Node) */
  fetch?: typeof fetch;
}

interface RemoteExe extends Executable { source: string; ref: string; options: unknown }

let nextId = 1;

export class RemoteCudaBackend implements Backend {
  readonly kind = "cuda_remote" as const;
  readonly id = nextId++;
  readonly tracker: AllocationTracker;
  private tail: Promise<void> = Promise.resolve();
  private sticky: CudaError | null = null;
  private printf: string[] = [];
  private times: SyncReport["kernelTimes"] = [];
  private nextHandle = 1;
  private symbols = new Map<string, DevicePtr>();
  private constructor(private cfg: RemoteCudaConfig, private session: string, private infoCache: DeviceInfo) {
    this.tracker = new AllocationTracker({ id: this.id, kind: "cuda_remote" }, cfg.memoryLimit ?? infoCache.limits.memoryLimit);
  }

  static async connect(cfg: RemoteCudaConfig): Promise<RemoteCudaBackend> {
    if (!cfg.endpoint) throw new CudaError("cudaErrorInitializationError", "remote CUDA needs an endpoint");
    const res = await request(cfg, "POST", "/v1/sessions", { gpu: cfg.gpu ?? "any", memoryLimit: cfg.memoryLimit, timeLimitMs: cfg.timeLimitMs });
    const j = (await res.json()) as { session: string; device: { name: string; totalMem: number; major: number; minor: number; multiProcessorCount?: number; sharedMemPerBlock?: number; maxThreadsPerBlock?: number }; limits: { memoryBytes: number; timeLimitMs: number } };
    const d = j.device;
    const info: DeviceInfo = {
      backend: "cuda_remote", name: `${d.name} (remote, sm_${d.major}${d.minor})`, vendor: "NVIDIA", architecture: `sm_${d.major}${d.minor}`, emulated: false, isFallbackAdapter: false,
      limits: {
        maxThreadsPerBlock: d.maxThreadsPerBlock ?? 1024, maxBlockDim: [1024, 1024, 64], maxGridDim: [2 ** 31 - 1, 65535, 65535],
        sharedMemPerBlock: d.sharedMemPerBlock ?? 48 * 1024, memoryLimit: Math.min(j.limits.memoryBytes, d.totalMem), maxBufferSize: j.limits.memoryBytes, maxPointerArgs: 256, warpSize: 32,
      },
      features: { fp16: "native", bf16: "emulated", fp64: "native", int64: "native", atomics64: true, warps: "subgroups", subgroupSize: [32, 32], timestamps: true, printf: true, boundsCheck: "report" },
      threads: d.multiProcessorCount,
      notes: [`remote NVIDIA GPU via ${new URL(cfg.endpoint).host}: every transfer crosses the network`, `worker time limit ${j.limits.timeLimitMs} ms per launch`],
    };
    return new RemoteCudaBackend(cfg, j.session, info);
  }

  info(): DeviceInfo { return this.infoCache; }
  private enqueue<T>(f: () => Promise<T>): Promise<T> {
    const p = this.tail.then(() => { if (this.sticky) throw this.sticky; return f(); });
    this.tail = p.then(() => {}, (e) => { if (e instanceof CudaError && !this.sticky) this.sticky = e; });
    return p;
  }
  private path(p: string): string { return `/v1/sessions/${encodeURIComponent(this.session)}${p}`; }

  malloc(bytes: number, label = ""): DevicePtr {
    this.tracker.reserve(bytes);
    const handle = `h${this.nextHandle++}`;
    const a = new Allocation(this.tracker.owner, bytes, handle, label);
    this.tracker.register(a);
    void this.enqueue(() => request(this.cfg, "POST", this.path("/alloc"), { handle, bytes }).then(() => {}));
    return new DevicePtr(a, 0);
  }
  free(p: DevicePtr): void {
    this.tracker.release(p.alloc);
    void this.enqueue(() => request(this.cfg, "POST", this.path("/free"), { handle: p.alloc.native }).then(() => {}));
  }
  upload(dst: DevicePtr, data: Uint8Array): void {
    this.tracker.validate(dst, data.byteLength, "cudaMemcpy (host to device)");
    const copy = data.slice();
    void this.enqueue(() => request(this.cfg, "PUT", this.path(`/mem/${dst.alloc.native}?offset=${dst.offset}`), copy).then(() => {}));
  }
  download(src: DevicePtr, bytes: number): Promise<Uint8Array> {
    this.tracker.validate(src, bytes, "cudaMemcpy (device to host)");
    return this.enqueue(async () => new Uint8Array(await (await request(this.cfg, "GET", this.path(`/mem/${src.alloc.native}?offset=${src.offset}&bytes=${bytes}`))).arrayBuffer()));
  }
  copy(dst: DevicePtr, src: DevicePtr, bytes: number): void {
    void this.enqueue(() => request(this.cfg, "POST", this.path("/copy"), { dst: dst.alloc.native, dstOffset: dst.offset, src: src.alloc.native, srcOffset: src.offset, bytes }).then(() => {}));
  }
  fill(dst: DevicePtr, byte: number, bytes: number): void {
    void this.enqueue(() => request(this.cfg, "POST", this.path("/memset"), { handle: dst.alloc.native, offset: dst.offset, value: byte & 0xff, bytes }).then(() => {}));
  }
  symbol(moduleId: number, name: string, bytes: number, init?: Uint8Array): DevicePtr {
    const key = `${moduleId}:${name}`;
    let p = this.symbols.get(key);
    if (!p) { p = this.malloc(Math.max(4, bytes), `symbol ${name}`); if (init) this.upload(p, init); this.symbols.set(key, p); }
    return p;
  }
  memoryStats(): MemoryStats { return { allocations: this.tracker.count, bytesAllocated: this.tracker.bytes, peakBytes: this.tracker.peak, bytesReserved: 0, limit: this.tracker.limit }; }
  specKey(): SpecKey { return { key: "" }; }

  async prepare(req: PrepareRequest): Promise<Executable> {
    if (!req.source || !req.ref) throw new CudaError("cudaErrorInvalidSource", "remote execution needs the module source");
    if (req.kernel.roots.some((r) => r.symbol)) throw new CudaError("cudaErrorNotSupported", "__constant__/__device__ variables are not supported by the remote backend yet");
    if (req.kernel.params.some((p) => p.origin)) throw new CudaError("cudaErrorNotSupported", "struct-by-value kernel parameters are not supported by the remote backend yet");
    return { kernel: req.kernel, code: req.source, notes: ["compiled with NVRTC on the remote worker"], source: req.source, ref: req.ref, options: { defines: req.defines } } as RemoteExe;
  }

  launch(exe0: Executable, cfg: LaunchConfig, args: ResolvedArg[], opts: LaunchOptions = {}): void {
    const exe = exe0 as RemoteExe;
    const wire = args.map((a, i) => {
      if (a.kind === "ptr") { a.ptr.alloc.check(); return { ptr: { handle: a.ptr.alloc.native, offset: a.ptr.offset } }; }
      const p = exe.kernel.params[i];
      return { scalar: { ctype: p.ctype, bytes: b64(encodeScalarBytes(a.t, a.value)) } };
    });
    void this.enqueue(async () => {
      const res = await request(this.cfg, "POST", this.path("/launch"), {
        source: exe.source, kernel: exe.ref, options: exe.options, grid: cfg.grid, block: cfg.block, sharedMem: cfg.sharedMem, args: wire, timing: Boolean(opts.timing),
      });
      const j = (await res.json()) as { ms?: number; printf?: string; error?: { name: string; message: string } };
      if (j.printf) this.printf.push(j.printf);
      if (opts.timing) this.times.push({ name: exe.kernel.displayName, ms: j.ms ?? null });
      if (j.error) throw new CudaError((j.error.name as never) ?? "cudaErrorLaunchFailure", `remote: ${j.error.message}`);
    });
  }

  async synchronize(): Promise<SyncReport> {
    await this.tail;
    if (this.sticky) throw this.sticky;
    const r = { printf: this.printf, kernelTimes: this.times };
    this.printf = []; this.times = [];
    return r;
  }

  destroy(): void { void request(this.cfg, "DELETE", this.path("")).catch(() => {}); }
}

async function request(cfg: RemoteCudaConfig, method: string, path: string, body?: unknown): Promise<Response> {
  const f = cfg.fetch ?? fetch;
  const token = typeof cfg.token === "function" ? await cfg.token() : cfg.token;
  const headers: Record<string, string> = {};
  if (token) headers.Authorization = `Bearer ${token}`;
  let data: BodyInit | undefined;
  if (body instanceof Uint8Array) { headers["Content-Type"] = "application/octet-stream"; data = body as BodyInit; }
  else if (body !== undefined) { headers["Content-Type"] = "application/json"; data = JSON.stringify(body); }
  let res: Response;
  try { res = await f(cfg.endpoint.replace(/\/$/, "") + path, { method, headers, body: data }); }
  catch (e) { throw new CudaError("cudaErrorDevicesUnavailable", `remote CUDA worker unreachable: ${(e as Error).message}`); }
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try { const j = await res.json() as { error?: { message?: string; name?: string } }; if (j.error?.message) msg = j.error.message; } catch { /* not json */ }
    const name = res.status === 401 || res.status === 403 ? "cudaErrorInitializationError" : res.status === 413 ? "cudaErrorMemoryAllocation" : res.status === 503 ? "cudaErrorDevicesUnavailable" : "cudaErrorLaunchFailure";
    throw new CudaError(name, `remote CUDA worker: ${msg}`);
  }
  return res;
}

function b64(b: Uint8Array): string {
  let s = "";
  for (const x of b) s += String.fromCharCode(x);
  return btoa(s);
}
