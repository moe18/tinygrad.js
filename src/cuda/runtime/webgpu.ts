/// <reference types="@webgpu/types" />
/**
 * WebGPU backend: runs CUDA kernels compiled to WGSL on the browser's GPU.
 *
 * It can own its GPUDevice or share tinygrad.js's (WebGPUBridge.cudaQueue()): then CUDA kernels and tinygrad kernels are
 * recorded into the same ordered command stream and tensors' buffers are used directly (zero-copy).
 */
import { DevicePtr, Allocation, AllocationTracker, type MemoryStats } from "./memory.js";
import { CudaError } from "./errors.js";
import type { Backend, DeviceInfo, Executable, LaunchConfig, LaunchOptions, PrepareRequest, ResolvedArg, SpecKey, SyncReport } from "./backend.js";
import { generateWGSL, DIAG_HEADER_WORDS, type WGSLCaps, type WGSLProgram } from "../compiler/wgsl.js";
import { encodeScalarWords, formatPrintf } from "./encode.js";
import type { Kernel } from "../compiler/ir.js";

const STORAGE = 0x80, COPY_SRC = 0x04, COPY_DST = 0x08, MAP_READ = 0x01, UNIFORM = 0x40, QUERY_RESOLVE = 0x200;

/** Ordered GPU command submission (the WebGPUBridge provides one that shares tinygrad's command stream). */
export interface GpuQueue {
  readonly device: GPUDevice;
  dispatch(pipeline: GPUComputePipeline, bindGroup: GPUBindGroup, x: number, y: number, z: number, timestampWrites?: GPUComputePassTimestampWrites): void;
  /** commands outside a compute pass (copies, query resolves), in order */
  encode(f: (enc: GPUCommandEncoder) => void): void;
  /** queue.writeBuffer, ordered after everything recorded so far */
  writeBuffer(buf: GPUBuffer, offset: number, data: Uint8Array): void;
  flush(): void;
  /** buffers that recorded-but-unsubmitted work may still use are destroyed after the next submit */
  deferDestroy(buf: GPUBuffer): void;
}

export class OwnQueue implements GpuQueue {
  private encoder: GPUCommandEncoder | null = null;
  private pass: GPUComputePassEncoder | null = null;
  private pending = 0;
  private deferred: GPUBuffer[] = [];
  submits = 0;
  constructor(readonly device: GPUDevice, private maxBatch = 256) {}
  private enc(): GPUCommandEncoder { return (this.encoder ??= this.device.createCommandEncoder()); }
  dispatch(p: GPUComputePipeline, bg: GPUBindGroup, x: number, y: number, z: number, ts?: GPUComputePassTimestampWrites): void {
    if (ts) {
      this.pass?.end(); this.pass = null;
      const pass = this.enc().beginComputePass({ timestampWrites: ts });
      pass.setPipeline(p); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(x, y, z); pass.end();
    } else {
      if (!this.pass) this.pass = this.enc().beginComputePass();
      this.pass.setPipeline(p); this.pass.setBindGroup(0, bg); this.pass.dispatchWorkgroups(x, y, z);
    }
    if (++this.pending >= this.maxBatch) this.flush();
  }
  encode(f: (enc: GPUCommandEncoder) => void): void { this.pass?.end(); this.pass = null; f(this.enc()); this.pending++; }
  writeBuffer(buf: GPUBuffer, off: number, data: Uint8Array): void {
    this.flush();
    this.device.queue.writeBuffer(buf, off, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
  }
  flush(): void {
    if (!this.encoder) return;
    this.pass?.end(); this.pass = null;
    this.device.queue.submit([this.encoder.finish()]);
    this.encoder = null; this.pending = 0; this.submits++;
    for (const b of this.deferred) b.destroy();
    this.deferred = [];
  }
  deferDestroy(buf: GPUBuffer): void { if (this.encoder) this.deferred.push(buf); else buf.destroy(); }
}

export interface WebGPUBackendOptions {
  /** cap on device memory (bytes) allocated through this backend */
  memoryLimit?: number;
  /** adapter preference when the backend creates its own device */
  powerPreference?: GPUPowerPreference;
  /** resolve a buffer handle of another subsystem (tinygrad.js tensors) */
  label?: string;
}

interface WebGPUExe extends Executable {
  prog: WGSLProgram;
  pipeline: GPUComputePipeline;
  layout: GPUBindGroupLayout;
}

interface PendingDiag { buf: GPUBuffer; exe: WebGPUExe; launch: number; cfg: LaunchConfig }

let nextBackendId = 1;

export class WebGPUBackend implements Backend {
  readonly kind = "webgpu" as const;
  readonly id = nextBackendId++;
  readonly tracker: AllocationTracker;
  private pool = new Map<number, GPUBuffer[]>();
  private pooledBytes = 0;
  private pipelines = new Map<string, Promise<{ pipeline: GPUComputePipeline; layout: GPUBindGroupLayout }>>();
  private uniforms = new Map<string, GPUBuffer>();
  private bindGroups = new Map<string, GPUBindGroup>();
  private bufferIds = new WeakMap<GPUBuffer, number>();
  private nextBufferId = 1;
  private diagPool: GPUBuffer[] = [];
  private pendingDiag: PendingDiag[] = [];
  /** one entry per timed launch; a launch split into several dispatches (grid beyond the per-dimension limit) has one query pair per piece */
  private timed: { name: string; queries: number[] }[] = [];
  private querySet: GPUQuerySet | null = null;
  private queryResolve: GPUBuffer | null = null;
  private queryCount = 0;
  private launches = 0;
  private uncaptured: string[] = [];
  private lostReason: string | null = null;
  private stickyError: CudaError | null = null;
  readonly caps: WGSLCaps;
  private infoCache: DeviceInfo;

  constructor(readonly queue: GpuQueue, adapterInfo: { vendor?: string; architecture?: string; description?: string; isFallbackAdapter?: boolean; subgroupMinSize?: number; subgroupMaxSize?: number }, opts: WebGPUBackendOptions = {}) {
    const device = queue.device;
    const L = device.limits;
    this.tracker = new AllocationTracker({ id: this.id, kind: "webgpu" }, opts.memoryLimit ?? Math.min(L.maxBufferSize * 16, 4 * 1024 ** 3));
    const sgMin = adapterInfo.subgroupMinSize ?? 0, sgMax = adapterInfo.subgroupMaxSize ?? 0;
    this.caps = {
      shaderF16: device.features.has("shader-f16"), subgroups: device.features.has("subgroups" as GPUFeatureName),
      subgroupMinSize: sgMin, subgroupMaxSize: sgMax, maxComputeInvocationsPerWorkgroup: L.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: L.maxComputeWorkgroupSizeX, maxComputeWorkgroupSizeY: L.maxComputeWorkgroupSizeY, maxComputeWorkgroupSizeZ: L.maxComputeWorkgroupSizeZ,
      maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize, maxStorageBuffersPerShaderStage: L.maxStorageBuffersPerShaderStage,
      maxComputeWorkgroupsPerDimension: L.maxComputeWorkgroupsPerDimension,
    };
    const fallback = Boolean(adapterInfo.isFallbackAdapter) || /swiftshader|llvmpipe|warp|software/i.test(`${adapterInfo.architecture} ${adapterInfo.description} ${adapterInfo.vendor}`);
    const native32 = this.caps.subgroups && sgMin === 32 && sgMax === 32;
    this.infoCache = {
      backend: "webgpu", name: adapterInfo.description || `${adapterInfo.vendor ?? ""} ${adapterInfo.architecture ?? ""}`.trim() || "WebGPU adapter",
      vendor: adapterInfo.vendor ?? "", architecture: adapterInfo.architecture ?? "", emulated: fallback, isFallbackAdapter: fallback,
      limits: {
        maxThreadsPerBlock: L.maxComputeInvocationsPerWorkgroup, maxBlockDim: [L.maxComputeWorkgroupSizeX, L.maxComputeWorkgroupSizeY, L.maxComputeWorkgroupSizeZ],
        // larger grids are split into several dispatches
        maxGridDim: [2 ** 31 - 1, 65535 * Math.max(1, Math.floor(L.maxComputeWorkgroupsPerDimension / 65535)), 65535],
        sharedMemPerBlock: L.maxComputeWorkgroupStorageSize, memoryLimit: this.tracker.limit, maxBufferSize: L.maxBufferSize,
        maxPointerArgs: L.maxStorageBuffersPerShaderStage - 1, warpSize: 32,
      },
      features: {
        fp16: this.caps.shaderF16 ? "native" : "emulated", bf16: "emulated", fp64: "unsupported", int64: "emulated", atomics64: false,
        warps: native32 ? "subgroups" : "emulated", subgroupSize: this.caps.subgroups ? [sgMin, sgMax] : null,
        timestamps: device.features.has("timestamp-query"), printf: true, boundsCheck: "report",
      },
      notes: [
        ...(fallback ? ["software WebGPU adapter: kernels execute on the CPU, timings do not reflect GPU hardware"] : []),
        ...(native32 ? [] : [`warp collectives are emulated with workgroup memory (subgroups: ${this.caps.subgroups ? `${sgMin}-${sgMax} lanes` : "unavailable"})`]),
        "out-of-bounds accesses are clamped by WebGPU; enable boundsCheck to report them",
      ],
    };
    device.addEventListener?.("uncapturederror", (ev) => {
      const e = (ev as GPUUncapturedErrorEvent).error;
      if (this.uncaptured.length < 16) this.uncaptured.push(e.message);
    });
    device.lost?.then((l) => { this.lostReason = `${l.reason ?? "unknown"}: ${l.message}`; });
  }

  /** Create a backend with its own adapter and device. */
  static async create(opts: WebGPUBackendOptions = {}): Promise<WebGPUBackend> {
    const gpu = (globalThis.navigator as Navigator | undefined)?.gpu;
    if (!gpu) throw new CudaError("cudaErrorNoDevice", "WebGPU is not available (navigator.gpu is undefined)");
    const adapter = await gpu.requestAdapter({ powerPreference: opts.powerPreference ?? "high-performance" });
    if (!adapter) throw new CudaError("cudaErrorNoDevice", "WebGPU is available but no adapter was returned");
    const wanted = ["shader-f16", "timestamp-query", "subgroups"] as GPUFeatureName[];
    const requiredFeatures = wanted.filter((f) => adapter.features.has(f));
    const requiredLimits: Record<string, number> = {};
    for (const key of ["maxBufferSize", "maxStorageBufferBindingSize", "maxComputeWorkgroupStorageSize", "maxComputeInvocationsPerWorkgroup",
      "maxComputeWorkgroupSizeX", "maxComputeWorkgroupSizeY", "maxComputeWorkgroupSizeZ", "maxStorageBuffersPerShaderStage", "maxComputeWorkgroupsPerDimension"]) {
      const v = (adapter.limits as unknown as Record<string, number>)[key];
      if (typeof v === "number") requiredLimits[key] = v;
    }
    const device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
    const ai = (adapter as GPUAdapter & { info?: GPUAdapterInfo }).info as (GPUAdapterInfo & { subgroupMinSize?: number; subgroupMaxSize?: number; isFallbackAdapter?: boolean }) | undefined;
    return new WebGPUBackend(new OwnQueue(device), {
      vendor: ai?.vendor, architecture: ai?.architecture, description: ai?.description, isFallbackAdapter: ai?.isFallbackAdapter ?? (adapter as unknown as { isFallbackAdapter?: boolean }).isFallbackAdapter,
      subgroupMinSize: ai?.subgroupMinSize, subgroupMaxSize: ai?.subgroupMaxSize,
    }, opts);
  }

  get device(): GPUDevice { return this.queue.device; }
  info(): DeviceInfo { return this.infoCache; }

  private check(): void {
    if (this.lostReason) throw new CudaError("cudaErrorDevicesUnavailable", `WebGPU device lost (${this.lostReason}); create a new context`);
    if (this.stickyError) throw this.stickyError;
  }

  private bufId(b: GPUBuffer): number {
    let id = this.bufferIds.get(b);
    if (id === undefined) { id = this.nextBufferId++; this.bufferIds.set(b, id); }
    return id;
  }

  // ------------------------------------------------------------------ memory
  malloc(bytes: number, label = ""): DevicePtr {
    this.check();
    this.tracker.reserve(bytes);
    if (bytes > this.device.limits.maxBufferSize) throw new CudaError("cudaErrorMemoryAllocation", `allocation of ${bytes} bytes exceeds this GPU's maxBufferSize (${this.device.limits.maxBufferSize})`);
    const cap = bucketSize(bytes);
    const list = this.pool.get(cap);
    let buf = list?.pop();
    if (buf) this.pooledBytes -= cap;
    else buf = this.device.createBuffer({ size: cap, usage: STORAGE | COPY_SRC | COPY_DST, label });
    const a = new Allocation(this.tracker.owner, bytes, buf, label);
    this.tracker.register(a);
    return new DevicePtr(a, 0);
  }

  /** Wrap a GPUBuffer owned by someone else (tinygrad.js tensor storage) as a device pointer, without copying. */
  wrapBuffer(buf: GPUBuffer, bytes: number, label = "external"): DevicePtr {
    const a = new Allocation(this.tracker.owner, bytes, buf, label);
    (a as Allocation & { external?: boolean }).external = true;
    return new DevicePtr(a, 0);
  }

  free(p: DevicePtr): void {
    if (p.offset !== 0) throw new CudaError("cudaErrorInvalidDevicePointer", "cudaFree needs the pointer returned by cudaMalloc (offset 0)");
    const a = p.alloc;
    if (a.freed) throw new CudaError("cudaErrorInvalidDevicePointer", `double free of allocation #${a.id}`);
    if ((a as Allocation & { external?: boolean }).external) { a.freed = true; return; }
    this.tracker.release(a);
    const buf = a.native as GPUBuffer;
    for (const [k] of this.bindGroups) if (k.includes(`|${this.bufId(buf)}:`)) this.bindGroups.delete(k);
    const cap = buf.size;
    if (this.pooledBytes + cap <= 256 * 1024 * 1024) {
      const list = this.pool.get(cap) ?? [];
      list.push(buf);
      this.pool.set(cap, list);
      this.pooledBytes += cap;
    } else this.queue.deferDestroy(buf);
  }

  upload(dst: DevicePtr, data: Uint8Array): void {
    this.check();
    this.tracker.validate(dst, data.byteLength, "cudaMemcpy (host to device)");
    const buf = dst.alloc.native as GPUBuffer;
    // WebGPU copies must be 4-byte aligned in offset and size. The aligned middle is written directly; the partial words at
    // the edges are merged on the GPU by a one-thread-per-word dispatch, recorded in queue order like any launch, so a
    // later launch, copy or write always sees this upload (a CPU read-modify-write would land after them)
    const start = dst.offset, end = dst.offset + data.byteLength;
    if (data.byteLength === 0) return;
    if (start % 4 === 0 && data.byteLength % 4 === 0) { this.queue.writeBuffer(buf, start, data); return; }
    const i0 = (start + 3) & ~3, i1 = end & ~3;  // aligned interior [i0, i1)
    if (i1 > i0) this.queue.writeBuffer(buf, i0, data.subarray(i0 - start, i1 - start));
    const words = new Set<number>([start >> 2, (end - 1) >> 2]);
    for (const w of [...words]) if (w * 4 >= i0 && w * 4 + 4 <= i1) words.delete(w);
    const edits: number[] = [];
    for (const w of words) {
      let mask = 0, value = 0;
      for (let b = 0; b < 4; b++) {
        const at = w * 4 + b;
        if (at >= start && at < end) { mask |= 0xff << (8 * b); value |= data[at - start] << (8 * b); }
      }
      edits.push(w, mask >>> 0, value >>> 0, 0);
    }
    this.mergeWords(buf, edits);
  }

  private mergePipeline: GPUComputePipeline | null = null;
  /** dst word = (dst & ~mask) | (value & mask) for each (word, mask, value, _) quadruple, in queue order */
  private mergeWords(buf: GPUBuffer, edits: number[]): void {
    const n = edits.length / 4;
    const lo = Math.min(...edits.filter((_, i) => i % 4 === 0)), hi = Math.max(...edits.filter((_, i) => i % 4 === 0));
    // bind a small window of the buffer: binding offsets must be aligned, and a whole buffer may exceed the binding limit
    const align = this.device.limits.minStorageBufferOffsetAlignment;
    const base = Math.floor((lo * 4) / align) * align;
    const size = Math.min(buf.size - base, hi * 4 + 4 - base);
    this.mergePipeline ??= this.device.createComputePipeline({
      layout: "auto", label: "unaligned upload",
      compute: { entryPoint: "main", module: this.device.createShaderModule({ label: "unaligned upload", code: `
@group(0) @binding(0) var<storage, read_write> dst: array<u32>;
@group(0) @binding(1) var<storage, read> e: array<vec4<u32>>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let x = e[id.x];
  dst[x.x] = (dst[x.x] & ~x.y) | (x.z & x.y);
}` }) },
    });
    const list = this.device.createBuffer({ size: edits.length * 4, usage: STORAGE, mappedAtCreation: true, label: "unaligned upload edits" });
    const rel = edits.map((v, i) => (i % 4 === 0 ? v - base / 4 : v));
    new Uint32Array(list.getMappedRange()).set(rel);
    list.unmap();
    const bg = this.device.createBindGroup({ layout: this.mergePipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: buf, offset: base, size } }, { binding: 1, resource: { buffer: list } }] });
    this.queue.dispatch(this.mergePipeline, bg, n, 1, 1);
    this.queue.deferDestroy(list);
  }

  async download(src: DevicePtr, bytes: number): Promise<Uint8Array> {
    this.check();
    this.tracker.validate(src, bytes, "cudaMemcpy (device to host)");
    const buf = src.alloc.native as GPUBuffer;
    const a0 = src.offset & ~3, a1 = (src.offset + bytes + 3) & ~3;
    const size = Math.max(4, a1 - a0);
    const staging = this.device.createBuffer({ size, usage: MAP_READ | COPY_DST });
    this.queue.encode((enc) => enc.copyBufferToBuffer(buf, a0, staging, 0, Math.min(size, buf.size - a0)));
    this.queue.flush();
    try { await staging.mapAsync(MAP_READ); }
    catch (e) { staging.destroy(); throw new CudaError("cudaErrorLaunchFailure", `reading device memory failed: ${this.lostReason ?? (e as Error).message}`); }
    const out = new Uint8Array(staging.getMappedRange().slice(src.offset - a0, src.offset - a0 + bytes));
    staging.unmap(); staging.destroy();
    await this.checkDiagnostics();
    return out;
  }

  copy(dst: DevicePtr, src: DevicePtr, bytes: number): void {
    this.check();
    this.tracker.validate(dst, bytes, "cudaMemcpy (device to device) destination");
    this.tracker.validate(src, bytes, "cudaMemcpy (device to device) source");
    if (dst.offset % 4 || src.offset % 4 || bytes % 4) throw new CudaError("cudaErrorInvalidValue", "device-to-device copies on WebGPU need 4-byte aligned offsets and sizes");
    if (bytes === 0) return;
    this.queue.encode((enc) => enc.copyBufferToBuffer(src.alloc.native as GPUBuffer, src.offset, dst.alloc.native as GPUBuffer, dst.offset, bytes));
  }

  fill(dst: DevicePtr, byte: number, bytes: number): void {
    this.check();
    this.tracker.validate(dst, bytes, "cudaMemset");
    if (bytes === 0) return;
    const buf = dst.alloc.native as GPUBuffer;
    if ((byte & 0xff) === 0 && dst.offset % 4 === 0 && bytes % 4 === 0) { this.queue.encode((enc) => enc.clearBuffer(buf, dst.offset, bytes)); return; }
    this.upload(dst, new Uint8Array(bytes).fill(byte & 0xff));
  }

  symbol(moduleId: number, name: string, bytes: number, init?: Uint8Array): DevicePtr {
    const key = `${moduleId}:${name}`;
    let p = this.symbols.get(key);
    if (!p) {
      p = this.malloc(Math.ceil(Math.max(4, bytes) / 4) * 4, `symbol ${name}`);
      if (init) this.upload(p, padTo4(init)); else this.fill(p, 0, p.alloc.size);
      this.symbols.set(key, p);
    }
    return p;
  }
  private symbols = new Map<string, DevicePtr>();

  memoryStats(): MemoryStats {
    return { allocations: this.tracker.count, bytesAllocated: this.tracker.bytes, peakBytes: this.tracker.peak, bytesReserved: this.pooledBytes, limit: this.tracker.limit };
  }

  // ------------------------------------------------------------------ compilation
  specKey(args: ResolvedArg[]): SpecKey {
    // pointer params bound to the same GPUBuffer share one binding; params at offset 0 skip the base add
    const groups: number[] = [], zero: boolean[] = [];
    const seen = new Map<GPUBuffer, number>();
    args.forEach((a, i) => {
      if (a.kind !== "ptr") { groups.push(i); zero.push(true); return; }
      const buf = a.ptr.alloc.native as GPUBuffer;
      const g = seen.get(buf);
      if (g === undefined) seen.set(buf, i);
      groups.push(g ?? i);
      zero.push(a.ptr.offset === 0);
    });
    const key = `a${groups.join(".")}z${zero.map((z) => (z ? 1 : 0)).join("")}`;
    return { aliasGroups: groups, zeroBase: zero, key };
  }

  async prepare(req: PrepareRequest): Promise<Executable> {
    this.check();
    const prog = generateWGSL(req.kernel, this.caps, {
      boundsCheck: req.options.boundsCheck, fp64: req.options.fp64, warps: req.options.warps,
      aliasGroups: req.spec.aliasGroups, zeroBase: req.spec.zeroBase,
    }, req.sink);
    const { pipeline, layout } = await this.pipeline(prog, req.kernel);
    return { kernel: req.kernel, code: prog.code, notes: prog.notes, prog, pipeline, layout } as WebGPUExe;
  }

  private pipeline(prog: WGSLProgram, kernel: Kernel): Promise<{ pipeline: GPUComputePipeline; layout: GPUBindGroupLayout }> {
    const key = prog.entry + "\n" + prog.code;
    let p = this.pipelines.get(key);
    if (!p) {
      p = this.createPipeline(prog, kernel);
      this.pipelines.set(key, p);
      p.catch(() => this.pipelines.delete(key));
    }
    return p;
  }

  private async createPipeline(prog: WGSLProgram, kernel: Kernel): Promise<{ pipeline: GPUComputePipeline; layout: GPUBindGroupLayout }> {
    const device = this.device;
    const entries: GPUBindGroupLayoutEntry[] = prog.bindings.map((b) => ({
      binding: b.binding, visibility: 0x4,
      buffer: { type: b.kind === "params" ? "uniform" : b.kind === "diag" ? "storage" : (b as { access: string }).access === "read" && !(b as { layout: string }).layout?.includes("atomic") ? "read-only-storage" : "storage" },
    }));
    device.pushErrorScope("validation");
    const module = device.createShaderModule({ code: prog.code, label: kernel.displayName });
    const layout = device.createBindGroupLayout({ entries });
    const scope = device.popErrorScope();
    const ci = await module.getCompilationInfo();
    const errs = ci.messages.filter((m) => m.type === "error");
    await scope;
    if (errs.length) {
      const lines = prog.code.split("\n");
      const detail = errs.map((m) => `  WGSL line ${m.lineNum}: ${m.message}\n    ${lines[m.lineNum - 1]?.trim() ?? ""}`).join("\n");
      const uniform = errs.some((m) => /uniform/i.test(m.message));
      throw new CudaError("cudaErrorInvalidSource", `${kernel.displayName}: the generated WGSL was rejected by the browser${uniform ? " (WGSL requires __syncthreads/warp operations to be reached by all threads of the block: barriers inside thread-dependent conditions are not allowed)" : " (this is a tinygrad.js compiler bug unless noted otherwise)"}:\n${detail}`);
    }
    device.pushErrorScope("validation");
    let pipeline: GPUComputePipeline;
    try {
      pipeline = await device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }), compute: { module, entryPoint: prog.entry }, label: kernel.displayName });
    } catch (e) {
      await device.popErrorScope();
      throw new CudaError("cudaErrorInvalidSource", `${kernel.displayName}: creating the compute pipeline failed: ${(e as Error).message}`);
    }
    const perr = await device.popErrorScope();
    if (perr) throw new CudaError("cudaErrorInvalidSource", `${kernel.displayName}: ${perr.message}`);
    return { pipeline, layout };
  }

  // ------------------------------------------------------------------ launch
  launch(exe0: Executable, cfg: LaunchConfig, args: ResolvedArg[], opts: LaunchOptions = {}): void {
    this.check();
    const exe = exe0 as WebGPUExe;
    const prog = exe.prog;
    const words = new Uint32Array(prog.paramWords);
    words.set(cfg.grid, 0);
    words[3] = 0x7f800000; // +inf bits: WGSL builds inf/NaN constants from this word
    for (const slot of prog.params) {
      const a = args[slot.param];
      if (a.kind !== "scalar") throw new CudaError("cudaErrorInvalidValue", `argument ${slot.param + 1} must be a scalar`);
      words.set(encodeScalarWords(slot.t, a.value), slot.word);
    }
    for (const pw of prog.pointerWords) {
      const a = args[pw.param];
      if (a.kind !== "ptr") throw new CudaError("cudaErrorInvalidValue", `argument ${pw.param + 1} must be a device pointer`);
      if (a.ptr.offset % pw.elemSize) throw new CudaError("cudaErrorInvalidValue", `argument ${pw.param + 1} (${exe.kernel.params[pw.param].name}): pointer offset ${a.ptr.offset} is not aligned to ${pw.elemSize} bytes`);
      words[pw.baseWord] = a.ptr.offset / pw.elemSize;
      words[pw.limitWord] = Math.floor(a.ptr.remaining / pw.elemSize) + a.ptr.offset / pw.elemSize;
    }
    // bindings
    const resources: GPUBuffer[] = [];
    let diagBuf: GPUBuffer | null = null;
    for (const b of prog.bindings) {
      if (b.kind === "params") { resources.push(null as unknown as GPUBuffer); continue; }
      if (b.kind === "diag") { diagBuf = this.diagBuffer(prog); resources.push(diagBuf); continue; }
      if (b.kind === "symbol") {
        const p = opts.symbols?.get(b.symbol);
        if (!p) throw new CudaError("cudaErrorInvalidSymbol", `symbol ${b.symbol} has no device memory`);
        resources.push(p.alloc.native as GPUBuffer);
        continue;
      }
      const a = args[b.params[0]] as ResolvedArg & { kind: "ptr" };
      a.ptr.alloc.check();
      resources.push(a.ptr.alloc.native as GPUBuffer);
    }
    if (prog.diag) words[7] = prog.diag.words;
    // split grids larger than maxComputeWorkgroupsPerDimension
    const maxD = this.caps.maxComputeWorkgroupsPerDimension;
    const [gx, gy, gz] = cfg.grid;
    const launchNo = ++this.launches;
    let timestamp: GPUComputePassTimestampWrites | undefined;
    const timedEntry = opts.timing ? { name: exe.kernel.displayName, queries: [] as number[] } : null;
    if (timedEntry) this.timed.push(timedEntry);
    for (let z0 = 0; z0 < gz; z0 += maxD) for (let y0 = 0; y0 < gy; y0 += maxD) for (let x0 = 0; x0 < gx; x0 += maxD) {
      words[4] = x0; words[5] = y0; words[6] = z0;
      const ub = this.uniformBuffer(words);
      resources[0] = ub;
      const bg = this.bindGroup(exe, resources);
      if (opts.timing && this.infoCache.features.timestamps) {
        const q = this.allocQuery();
        timestamp = { querySet: this.querySet!, beginningOfPassWriteIndex: q, endOfPassWriteIndex: q + 1 };
        timedEntry!.queries.push(q);
      }
      this.queue.dispatch(exe.pipeline, bg, Math.min(maxD, gx - x0), Math.min(maxD, gy - y0), Math.min(maxD, gz - z0), timestamp);
    }
    if (diagBuf) this.pendingDiag.push({ buf: diagBuf, exe, launch: launchNo, cfg });
  }

  private allocQuery(): number {
    if (!this.querySet || this.queryCount + 2 > 512) {
      if (this.querySet && this.queryCount) throw new CudaError("cudaErrorLaunchOutOfResources", "too many timed launches between synchronizations (max 256)");
      this.querySet ??= this.device.createQuerySet({ type: "timestamp", count: 512 });
      this.queryResolve ??= this.device.createBuffer({ size: 512 * 8, usage: QUERY_RESOLVE | COPY_SRC });
    }
    const q = this.queryCount;
    this.queryCount += 2;
    return q;
  }

  private uniformBuffer(words: Uint32Array): GPUBuffer {
    const key = Array.prototype.join.call(words, ",");
    let b = this.uniforms.get(key);
    if (!b) {
      if (this.uniforms.size > 4096) { this.uniforms.clear(); this.bindGroups.clear(); }
      b = this.device.createBuffer({ size: Math.max(16, words.byteLength), usage: UNIFORM | COPY_DST, mappedAtCreation: true });
      new Uint32Array(b.getMappedRange()).set(words);
      b.unmap();
      this.uniforms.set(key, b);
    }
    return b;
  }

  private bindGroup(exe: WebGPUExe, resources: GPUBuffer[]): GPUBindGroup {
    const key = `${exe.prog.entry}#${exe.prog.code.length}|` + resources.map((r) => this.bufId(r) + ":").join("");
    let bg = this.bindGroups.get(key);
    if (!bg) {
      bg = this.device.createBindGroup({ layout: exe.layout, entries: resources.map((buffer, binding) => ({ binding, resource: { buffer } })) });
      if (this.bindGroups.size > 8192) this.bindGroups.clear();
      this.bindGroups.set(key, bg);
    }
    return bg;
  }

  private diagBuffer(prog: WGSLProgram): GPUBuffer {
    const bytes = prog.diag!.words * 4;
    const b = this.diagPool.find((x) => x.size === bytes);
    if (b) { this.diagPool.splice(this.diagPool.indexOf(b), 1); return b; }
    const nb = this.device.createBuffer({ size: bytes, usage: STORAGE | COPY_SRC | COPY_DST, mappedAtCreation: true });
    nb.unmap(); // zero-initialized
    return nb;
  }

  private printfOut: string[] = [];

  /** Read back diagnostics buffers of finished launches: printf output, device asserts, out-of-bounds reports. */
  private async checkDiagnostics(): Promise<void> {
    if (!this.pendingDiag.length) return;
    const pend = this.pendingDiag;
    this.pendingDiag = [];
    const stagings = pend.map((p) => {
      const st = this.device.createBuffer({ size: p.buf.size, usage: MAP_READ | COPY_DST });
      this.queue.encode((enc) => enc.copyBufferToBuffer(p.buf, 0, st, 0, p.buf.size));
      return st;
    });
    this.queue.flush();
    let firstError: CudaError | null = null;
    for (let i = 0; i < pend.length; i++) {
      const p = pend[i], st = stagings[i];
      await st.mapAsync(MAP_READ);
      const w = new Uint32Array(st.getMappedRange().slice(0));
      st.unmap(); st.destroy();
      const info = p.exe.prog.diag!;
      // printf records
      const used = w[0];
      let pos = DIAG_HEADER_WORDS;
      const end = Math.min(DIAG_HEADER_WORDS + used, w.length);
      const records: { id: number; words: Uint32Array }[] = [];
      while (pos < end) {
        const n = w[pos];
        if (n < 2 || pos + n > w.length) break;
        records.push({ id: w[pos + 1], words: w.subarray(pos + 2, pos + n) });
        pos += n;
      }
      for (const r of records) {
        const f = info.printf[r.id];
        if (f) this.printfOut.push(formatPrintf(f.fmt, f.types, r.words));
      }
      if (DIAG_HEADER_WORDS + used > w.length) this.printfOut.push(`[tinygrad.js: printf buffer full; ${DIAG_HEADER_WORDS + used - w.length} words of output from ${p.exe.kernel.displayName} were dropped]\n`);
      if (w[1] > 0 && !firstError) {
        const site = info.sites[w[2]];
        const where = `thread (${w[4]},${w[5]},${w[6]}) in block (${w[7]},${w[8]},${w[9]})`;
        const count = w[1] > 1 ? ` (and ${w[1] - 1} more)` : "";
        if (site?.kind === "assert") firstError = new CudaError("cudaErrorAssert", `${p.exe.kernel.displayName}: ${site.msg} [${where}]${count}`);
        else if (site?.kind === "trap") firstError = new CudaError("cudaErrorLaunchFailure", `${p.exe.kernel.displayName}: ${site.msg} [${where}]${count}`);
        else firstError = new CudaError("cudaErrorIllegalAddress", `${p.exe.kernel.displayName}: out-of-bounds ${site?.kind === "oob-store" ? "store to" : site?.kind === "oob-atomic" ? "atomic on" : "load from"} '${site?.root}' at element/byte ${w[3]} by ${where}${count}`);
      }
      // reset for reuse
      this.device.queue.writeBuffer(p.buf, 0, new Uint32Array(Math.min(p.buf.size / 4, DIAG_HEADER_WORDS + used + 16)));
      if (this.diagPool.length < 16) this.diagPool.push(p.buf); else p.buf.destroy();
    }
    if (firstError) { this.stickyError = firstError; throw firstError; }
  }

  async synchronize(): Promise<SyncReport> {
    this.check();
    let times: SyncReport["kernelTimes"] = [];
    if (this.timed.length) {
      const timed = this.timed; this.timed = [];
      if (this.querySet && this.queryCount) {
        const n = this.queryCount;
        const st = this.device.createBuffer({ size: n * 8, usage: MAP_READ | COPY_DST });
        this.queue.encode((enc) => { enc.resolveQuerySet(this.querySet!, 0, n, this.queryResolve!, 0); enc.copyBufferToBuffer(this.queryResolve!, 0, st, 0, n * 8); });
        this.queue.flush();
        await st.mapAsync(MAP_READ);
        const ts = new BigUint64Array(st.getMappedRange().slice(0));
        st.unmap(); st.destroy();
        this.queryCount = 0;
        // a split launch is the sum of its pieces
        times = timed.map((t) => ({ name: t.name, ms: t.queries.length ? t.queries.reduce((a, q) => a + Number(ts[q + 1] - ts[q]), 0) / 1e6 : null }));
      } else times = timed.map((t) => ({ name: t.name, ms: null }));
    }
    this.queue.flush();
    await this.device.queue.onSubmittedWorkDone();
    if (this.uncaptured.length) { const m = this.uncaptured.join("\n"); this.uncaptured = []; throw new CudaError("cudaErrorLaunchFailure", `WebGPU error: ${m}`); }
    await this.checkDiagnostics();
    this.check();
    const printf = this.printfOut; this.printfOut = [];
    return { printf, kernelTimes: times };
  }

  /** Clear a sticky device error (CUDA contexts can't recover from them; here the next launch may proceed). */
  resetError(): void { this.stickyError = null; }

  destroy(): void {
    this.queue.flush();
    for (const a of this.tracker.all()) if (!(a as Allocation & { external?: boolean }).external) (a.native as GPUBuffer).destroy();
    for (const l of this.pool.values()) for (const b of l) b.destroy();
    for (const b of this.uniforms.values()) b.destroy();
    for (const b of this.diagPool) b.destroy();
    this.querySet?.destroy(); this.queryResolve?.destroy();
    this.pool.clear(); this.uniforms.clear(); this.bindGroups.clear(); this.pipelines.clear();
  }
}

function bucketSize(bytes: number): number {
  const b = Math.max(16, Math.ceil(bytes / 4) * 4);
  if (b <= 4096) return Math.ceil(b / 256) * 256;
  // round up to 1/8 of the next power of two: bounded waste (<12.5%), good reuse
  const p = 2 ** Math.ceil(Math.log2(b));
  const step = p / 8;
  return Math.ceil(b / step) * step;
}

function padTo4(d: Uint8Array): Uint8Array {
  if (d.byteLength % 4 === 0) return d;
  const r = new Uint8Array(Math.ceil(d.byteLength / 4) * 4); r.set(d); return r;
}
