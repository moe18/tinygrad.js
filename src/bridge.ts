/// <reference types="@webgpu/types" />
/**
 * WebGPU bridge: the JavaScript half of tinygrad.js's WEBGPU device.
 *
 * The Python device (python/tinygrad_js/webgpu.py) calls these methods through Pyodide's FFI. GPU objects stay
 * on the JS side and Python refers to them by integer handles, so no JS objects need to be kept alive from Python.
 *
 * Synchronous methods are used on the hot path (createBuffer, writeBuffer, dispatch). Methods returning a Promise
 * (readBuffer, sync, compile) are awaited from Python with `pyodide.ffi.run_sync` (JSPI).
 *
 * Dispatches are recorded into one compute pass and submitted in batches. The pending batch is flushed before any
 * operation whose queue ordering matters (queue.writeBuffer, readback, sync) and when it grows past `maxBatch`.
 */
import type { DeviceInfo, GpuStats } from "./types.js";

const STORAGE = 0x80, COPY_SRC = 0x04, COPY_DST = 0x08, MAP_READ = 0x01, UNIFORM = 0x40, COMPUTE = 0x4;

interface ProgramEntry { name: string; pipeline: GPUComputePipeline; layout: GPUBindGroupLayout; kinds: ("storage" | "uniform")[] }

export interface BridgeOptions {
  powerPreference?: GPUPowerPreference;
  /** Upper bound on bytes allocated in tinygrad storage buffers. Allocation beyond it raises MemoryError in Python. */
  maxBufferMemory?: number;
  /** Dispatches recorded before an automatic submit. */
  maxBatch?: number;
}

/** Parse the binding declarations tinygrad's WGSLRenderer emits: `@group(0) @binding(N)var<storage,read_write>...`. */
export function parseBindings(code: string): ("storage" | "uniform")[] {
  const kinds: ("storage" | "uniform")[] = [];
  const re = /@binding\((\d+)\)\s*var<\s*(storage|uniform)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(code)) !== null) kinds[Number(m[1])] = m[2] as "storage" | "uniform";
  for (let i = 0; i < kinds.length; i++) if (kinds[i] === undefined) throw new Error(`kernel has a gap at binding ${i}`);
  return kinds;
}

export class WebGPUBridge {
  readonly device: GPUDevice;
  private readonly infoData: DeviceInfo;
  private buffers = new Map<number, GPUBuffer>();
  private programs = new Map<number, ProgramEntry>();
  private nextId = 1;
  private encoder: GPUCommandEncoder | null = null;
  private pass: GPUComputePassEncoder | null = null;
  private pending = 0;
  private deferredDestroy: GPUBuffer[] = [];
  private uniforms = new Map<string, GPUBuffer>();
  private bindGroups = new Map<string, GPUBindGroup>();
  private bufferKeys = new Map<number, string[]>();
  private staging = new Map<number, GPUBuffer[]>();
  private errors: string[] = [];
  private lostReason: string | null = null;
  private maxBatch: number;
  private maxBufferMemory: number;
  readonly counters: GpuStats = {
    kernelsCompiled: 0, kernelsDispatched: 0, submits: 0, buffersAllocated: 0, buffersLive: 0,
    bytesAllocated: 0, peakBytesAllocated: 0, bytesUploaded: 0, bytesDownloaded: 0, readbacks: 0, syncs: 0,
    bindGroupCacheHits: 0,
  };

  private constructor(device: GPUDevice, info: DeviceInfo, opts: BridgeOptions) {
    this.device = device;
    this.infoData = info;
    this.maxBatch = opts.maxBatch ?? 128;
    this.maxBufferMemory = opts.maxBufferMemory ?? Infinity;
    device.addEventListener("uncapturederror", (ev) => {
      const e = (ev as GPUUncapturedErrorEvent).error;
      if (this.errors.length < 16) this.errors.push(`${e.constructor?.name ?? "GPUError"}: ${e.message}`);
    });
    device.lost.then((l) => { this.lostReason = `${l.reason ?? "unknown"}: ${l.message}`; });
  }

  /** Request an adapter and device. Throws a descriptive Error when WebGPU is unavailable. */
  static async create(opts: BridgeOptions = {}): Promise<WebGPUBridge> {
    const gpu: GPU | undefined = (globalThis.navigator as Navigator | undefined)?.gpu;
    if (!gpu) throw new Error("WebGPU is not available in this context (navigator.gpu is undefined)");
    const adapter = await gpu.requestAdapter({ powerPreference: opts.powerPreference ?? "high-performance" });
    if (!adapter) throw new Error("WebGPU is available but no GPU adapter was returned (requestAdapter() gave null)");
    const wanted: GPUFeatureName[] = ["shader-f16", "timestamp-query"];
    const features = wanted.filter((f) => adapter.features.has(f));
    // ask for the adapter's full limits, as upstream tinygrad's native WEBGPU backend does
    const requiredLimits: Record<string, number> = {};
    for (const key in adapter.limits) {
      const v = (adapter.limits as unknown as Record<string, unknown>)[key];
      if (typeof v === "number") requiredLimits[key] = v;
    }
    const device = await adapter.requestDevice({ requiredFeatures: features, requiredLimits });
    const ai = (adapter as GPUAdapter & { info?: GPUAdapterInfo }).info;
    const info: DeviceInfo = {
      vendor: ai?.vendor ?? "", architecture: ai?.architecture ?? "", device: ai?.device ?? "", description: ai?.description ?? "",
      isFallbackAdapter: Boolean((ai as unknown as { isFallbackAdapter?: boolean })?.isFallbackAdapter ??
        (adapter as unknown as { isFallbackAdapter?: boolean }).isFallbackAdapter),
      features: [...device.features] as string[],
      limits: {
        maxBufferSize: device.limits.maxBufferSize,
        maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
        maxStorageBuffersPerShaderStage: device.limits.maxStorageBuffersPerShaderStage,
        maxComputeWorkgroupStorageSize: device.limits.maxComputeWorkgroupStorageSize,
        maxComputeInvocationsPerWorkgroup: device.limits.maxComputeInvocationsPerWorkgroup,
        maxComputeWorkgroupsPerDimension: device.limits.maxComputeWorkgroupsPerDimension,
      },
    };
    return new WebGPUBridge(device, info, opts);
  }

  // ------------------------------------------------------------------ info / errors

  info(): DeviceInfo { return this.infoData; }
  stats(): GpuStats & { pendingDispatches: number; deviceLost: string } {
    return { ...this.counters, pendingDispatches: this.pending, deviceLost: this.lostReason ?? "" };
  }
  get lost(): string | null { return this.lostReason; }

  private check(): void {
    if (this.lostReason) throw new Error(`WebGPU device lost (${this.lostReason}). Restart the runtime to get a new device.`);
  }
  private takeErrors(): string | null {
    if (!this.errors.length) return null;
    const msg = this.errors.join("\n");
    this.errors = [];
    return msg;
  }

  // ------------------------------------------------------------------ buffers

  createBuffer(size: number): number {
    this.check();
    if (this.counters.bytesAllocated + size > this.maxBufferMemory)
      throw new Error(`tinygrad.js GPU memory limit exceeded: ${this.counters.bytesAllocated + size} > ${this.maxBufferMemory} bytes`);
    if (size > this.device.limits.maxBufferSize)
      throw new Error(`buffer of ${size} bytes exceeds this device's maxBufferSize (${this.device.limits.maxBufferSize})`);
    const buf = this.device.createBuffer({ size, usage: STORAGE | COPY_SRC | COPY_DST });
    const id = this.nextId++;
    this.buffers.set(id, buf);
    const c = this.counters;
    c.buffersAllocated++; c.buffersLive++; c.bytesAllocated += size;
    c.peakBytesAllocated = Math.max(c.peakBytesAllocated, c.bytesAllocated);
    return id;
  }

  destroyBuffer(id: number): void {
    const buf = this.buffers.get(id);
    if (!buf) return;
    this.buffers.delete(id);
    for (const key of this.bufferKeys.get(id) ?? []) this.bindGroups.delete(key);
    this.bufferKeys.delete(id);
    this.counters.buffersLive--; this.counters.bytesAllocated -= buf.size;
    // a recorded-but-unsubmitted dispatch may still use it: destroy after the next submit
    if (this.pending) this.deferredDestroy.push(buf); else buf.destroy();
  }

  /** Upload from a Python buffer (a PyProxy of a memoryview) or any ArrayBufferView. */
  writeBuffer(id: number, src: unknown): void {
    this.check();
    const buf = this.getBuffer(id);
    this.flush(); // queue.writeBuffer is ordered against submits, so earlier dispatches must be submitted first
    const proxy = src as { getBuffer?: (t?: string) => { data: ArrayBufferView; c_contiguous: boolean; release(): void } };
    if (typeof proxy.getBuffer === "function") {
      const pb = proxy.getBuffer("u8");
      try {
        if (!pb.c_contiguous) throw new Error("writeBuffer needs a C-contiguous buffer");
        this.upload(buf, new Uint8Array(pb.data.buffer, pb.data.byteOffset, pb.data.byteLength));
      } finally { pb.release(); }
    } else {
      const v = src as ArrayBufferView;
      this.upload(buf, new Uint8Array(v.buffer, v.byteOffset, v.byteLength));
    }
  }

  private upload(buf: GPUBuffer, bytes: Uint8Array): void {
    let data = bytes;
    if (data.byteLength % 4) { // WebGPU copies must be multiples of 4 bytes
      data = new Uint8Array(Math.ceil(bytes.byteLength / 4) * 4);
      data.set(bytes);
    }
    if (data.byteLength > buf.size) throw new Error(`writeBuffer: ${data.byteLength} bytes into a ${buf.size}-byte buffer`);
    this.device.queue.writeBuffer(buf, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
    this.counters.bytesUploaded += bytes.byteLength;
  }

  /** Copy `nbytes` from a storage buffer to the CPU. Resolves to a Uint8Array of exactly `nbytes`. */
  async readBuffer(id: number, nbytes: number): Promise<Uint8Array> {
    this.check();
    const src = this.getBuffer(id);
    const size = Math.ceil(nbytes / 4) * 4;
    const pool = this.staging.get(size) ?? [];
    const staging = pool.pop() ?? this.device.createBuffer({ size, usage: MAP_READ | COPY_DST });
    this.flush((enc) => enc.copyBufferToBuffer(src, 0, staging, 0, size));
    try {
      await staging.mapAsync(MAP_READ, 0, size);
    } catch (e) {
      staging.destroy();
      throw new Error(this.takeErrors() ?? this.lostReason ?? String(e));
    }
    const out = new Uint8Array(nbytes);
    out.set(new Uint8Array(staging.getMappedRange(0, size), 0, nbytes));
    staging.unmap();
    if (pool.length < 4) { pool.push(staging); this.staging.set(size, pool); } else staging.destroy();
    this.counters.bytesDownloaded += nbytes; this.counters.readbacks++;
    const err = this.takeErrors();
    if (err) throw new Error(err);
    return out;
  }

  private getBuffer(id: number): GPUBuffer {
    const b = this.buffers.get(id);
    if (!b) throw new Error(`invalid or freed buffer handle ${id}`);
    return b;
  }

  // ------------------------------------------------------------------ programs

  /** Compile a WGSL kernel into a compute pipeline. Rejects with the WGSL compiler messages on failure. */
  async compile(name: string, code: string): Promise<number> {
    this.check();
    const kinds = parseBindings(code);
    this.device.pushErrorScope("validation");
    const module = this.device.createShaderModule({ code, label: name });
    const layout = this.device.createBindGroupLayout({
      entries: kinds.map((type, binding) => ({ binding, visibility: COMPUTE, buffer: { type } })),
    });
    const scopeErr = this.device.popErrorScope();
    const info = await module.getCompilationInfo();
    const msgs = info.messages.filter((m) => m.type === "error");
    if (msgs.length) {
      await scopeErr;
      throw new Error(`WGSL compilation of ${name} failed:\n` + msgs.map((m) => `  line ${m.lineNum}:${m.linePos} ${m.message}`).join("\n"));
    }
    const se = await scopeErr;
    if (se) throw new Error(`creating ${name}: ${se.message}`);
    let pipeline: GPUComputePipeline;
    try {
      pipeline = await this.device.createComputePipelineAsync({
        label: name, layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout] }), compute: { module, entryPoint: name },
      });
    } catch (e) {
      throw new Error(`creating compute pipeline ${name}: ${(e as Error).message ?? e}`);
    }
    const id = this.nextId++;
    this.programs.set(id, { name, pipeline, layout, kinds });
    this.counters.kernelsCompiled++;
    return id;
  }

  releaseProgram(id: number): void { this.programs.delete(id); }

  private uniform(kind: "f" | "i", value: number): GPUBuffer {
    const key = kind + value;
    let buf = this.uniforms.get(key);
    if (!buf) {
      if (this.uniforms.size > 1024) this.uniforms.clear(); // dropped buffers are garbage-collected by the browser
      buf = this.device.createBuffer({ size: 4, usage: UNIFORM | COPY_DST, mappedAtCreation: true });
      const range = buf.getMappedRange();
      if (kind === "f") new Float32Array(range)[0] = value; else new Int32Array(range)[0] = value;
      buf.unmap();
      this.uniforms.set(key, buf);
    }
    return buf;
  }

  /**
   * Record one kernel launch. Arguments: program id, workgroup counts, number of buffers, then the buffer handles,
   * then the integer values of symbolic variables. Binding 0 is tinygrad's `INFINITY` uniform.
   */
  dispatch(pid: number, gx: number, gy: number, gz: number, nbufs: number, ...rest: number[]): void {
    this.check();
    const prg = this.programs.get(pid);
    if (!prg) throw new Error(`invalid program handle ${pid}`);
    if (1 + rest.length !== prg.kinds.length)
      throw new Error(`${prg.name}: kernel declares ${prg.kinds.length - 1} arguments but got ${rest.length}`);
    const key = pid + ":" + rest.join(",") + ":" + nbufs;
    let bg = this.bindGroups.get(key);
    if (bg) this.counters.bindGroupCacheHits++;
    else {
      const entries: GPUBindGroupEntry[] = [{ binding: 0, resource: { buffer: this.uniform("f", Infinity) } }];
      for (let i = 0; i < rest.length; i++) {
        const buffer = i < nbufs ? this.getBuffer(rest[i]) : this.uniform("i", rest[i]);
        entries.push({ binding: i + 1, resource: { buffer } });
      }
      bg = this.device.createBindGroup({ layout: prg.layout, entries });
      if (this.bindGroups.size > 4096) { this.bindGroups.clear(); this.bufferKeys.clear(); }
      this.bindGroups.set(key, bg);
      for (let i = 0; i < nbufs; i++) {
        const keys = this.bufferKeys.get(rest[i]);
        if (keys) keys.push(key); else this.bufferKeys.set(rest[i], [key]);
      }
    }
    if (!this.pass) {
      this.encoder = this.device.createCommandEncoder();
      this.pass = this.encoder.beginComputePass();
    }
    this.pass.setPipeline(prg.pipeline);
    this.pass.setBindGroup(0, bg);
    this.pass.dispatchWorkgroups(gx, gy, gz);
    this.pending++;
    this.counters.kernelsDispatched++;
    if (this.pending >= this.maxBatch) this.flush();
  }

  /** Submit recorded work. `extra` may append commands (e.g. a readback copy) after the compute pass. */
  flush(extra?: (enc: GPUCommandEncoder) => void): void {
    if (!this.pass && !extra) return;
    const enc = this.encoder ?? this.device.createCommandEncoder();
    this.pass?.end();
    extra?.(enc);
    this.device.queue.submit([enc.finish()]);
    this.counters.submits++;
    this.encoder = null; this.pass = null; this.pending = 0;
    for (const b of this.deferredDestroy) b.destroy();
    this.deferredDestroy = [];
  }

  /** Wait for all submitted GPU work. Rejects with any asynchronous validation errors collected since the last check. */
  async sync(): Promise<void> {
    this.check();
    this.flush();
    await this.device.queue.onSubmittedWorkDone();
    this.counters.syncs++;
    this.check();
    const err = this.takeErrors();
    if (err) throw new Error(err);
  }

  /** Free everything. The bridge cannot be used afterwards. */
  destroy(): void {
    try { this.flush(); } catch { /* ignore */ }
    for (const b of this.buffers.values()) b.destroy();
    for (const b of this.uniforms.values()) b.destroy();
    for (const pool of this.staging.values()) for (const b of pool) b.destroy();
    this.buffers.clear(); this.uniforms.clear(); this.staging.clear(); this.bindGroups.clear(); this.programs.clear();
    this.device.destroy();
  }
}
