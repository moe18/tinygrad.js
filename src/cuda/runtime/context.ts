/**
 * The device-independent CUDA runtime: contexts, modules, kernel functions, launches, memory transfers, tensors, streams
 * and events. Backends (WebGPU, WASM, remote) only execute; argument validation, specialization and caching live here.
 */
import { CompileError, DiagnosticSink, type Diagnostic } from "../compiler/diagnostics.js";
import { parseModule, COMPILER_VERSION, type ModuleOptions, type ParsedModule, type KernelSignature } from "../compiler/module.js";
import { printKernel, type Kernel, type KernelParam, type VT } from "../compiler/ir.js";
import type { Backend, BackendCompileOptions, DeviceInfo, Executable, LaunchConfig, ResolvedArg, ScalarValue, SyncReport } from "./backend.js";
import { CudaError } from "./errors.js";
import { DevicePtr, DTYPE_SIZE, bytesOf, typedView, type DType } from "./memory.js";
import { hashString } from "./hash.js";
import { Lowering } from "../compiler/lower.js";

export type Dim3 = number | [number, number?, number?] | { x: number; y?: number; z?: number };

export interface CompileOptions extends ModuleOptions, BackendCompileOptions {
  /** name for messages / the inspector */
  name?: string;
}

export type HostArray = Float32Array | Float64Array | Int32Array | Uint32Array | Int16Array | Uint16Array | Int8Array | Uint8Array | BigInt64Array | BigUint64Array;
export type KernelArg = DevicePtr | DeviceTensor | HostArray | number | bigint | boolean | number[] | Record<string, unknown> | { ptr: DevicePtr };

export interface LaunchParams {
  grid: Dim3;
  block: Dim3;
  args: KernelArg[];
  sharedMem?: number;
  stream?: Stream;
  /** kernel name (template instantiations as "name<args>") when the module defines several kernels */
  kernel?: string;
  /** measure execution time (reported by synchronize() / the returned LaunchResult after sync) */
  timing?: boolean;
}

export interface LaunchInfo { kernel: string; grid: [number, number, number]; block: [number, number, number]; compiled: boolean; compileMs: number }

const toDim3 = (d: Dim3, what: string): [number, number, number] => {
  let r: [number, number, number];
  if (typeof d === "number") r = [d, 1, 1];
  else if (Array.isArray(d)) r = [d[0], d[1] ?? 1, d[2] ?? 1];
  else r = [d.x, d.y ?? 1, d.z ?? 1];
  for (const v of r) if (!Number.isInteger(v) || v < 1) throw new CudaError("cudaErrorInvalidConfiguration", `${what} dimensions must be positive integers (got [${r.join(", ")}])`);
  return r;
};

export class Stream {
  private static next = 1;
  readonly id = Stream.next++;
  constructor(readonly ctx: CudaContext) {}
  /** all streams share one in-order device queue: synchronizing a stream synchronizes the device */
  synchronize(): Promise<SyncReport> { return this.ctx.synchronize(); }
}

export class CudaEvent {
  time: number | null = null;
  private pending: Promise<void> | null = null;
  constructor(readonly ctx: CudaContext) {}
  /** @internal */ record(p: Promise<void>): void { this.time = null; this.pending = p.then(() => { this.time = performance.now(); }); }
  async synchronize(): Promise<void> { if (!this.pending) throw new CudaError("cudaErrorInvalidHandle", "event was never recorded"); await this.pending; }
  query(): boolean { return this.time !== null; }
}

export class CudaContext {
  private modules = new Set<CudaModule>();
  readonly defaultStream: Stream;
  onPrintf: (text: string) => void = (t) => console.log(t.replace(/\n$/, ""));
  private lastError = 0;
  private stdout: string[] = [];

  constructor(readonly backend: Backend, readonly selection: DeviceSelection) {
    this.defaultStream = new Stream(this);
  }

  info(): DeviceInfo { return this.backend.info(); }

  // ------------------------------------------------------------------ modules
  /** Parse and check a CUDA source. Kernels are translated on first launch for each block size (and cached). */
  compile(source: string, options: CompileOptions = {}): CudaModule {
    const m = new CudaModule(this, source, options);
    this.modules.add(m);
    return m;
  }

  // ------------------------------------------------------------------ memory
  malloc(bytes: number, label?: string): DevicePtr { return this.backend.malloc(bytes, label); }
  free(p: DevicePtr | DeviceTensor): void { this.backend.free(p instanceof DeviceTensor ? p.ptr : p); }

  /** cudaMemcpy: host->device returns immediately (data captured), device->host resolves when the data arrived. */
  async memcpy(dst: DevicePtr | HostArray, src: DevicePtr | HostArray | ArrayBufferView, bytes?: number): Promise<void> {
    if (dst instanceof DevicePtr && src instanceof DevicePtr) {
      this.backend.copy(dst, src, bytes ?? Math.min(dst.remaining, src.remaining));
      return;
    }
    if (dst instanceof DevicePtr) {
      const b = bytesOf(src as ArrayBufferView);
      const n = bytes ?? b.byteLength;
      if (n > b.byteLength) throw new CudaError("cudaErrorInvalidValue", `cudaMemcpy: ${n} bytes requested from a ${b.byteLength}-byte host array`);
      this.backend.upload(dst, b.subarray(0, n));
      return;
    }
    if (src instanceof DevicePtr) {
      const d = bytesOf(dst as ArrayBufferView);
      const n = bytes ?? Math.min(d.byteLength, src.remaining);
      const data = await this.backend.download(src, n);
      d.set(data.subarray(0, n));
      return;
    }
    throw new CudaError("cudaErrorInvalidMemcpyDirection", "cudaMemcpy needs at least one device pointer");
  }

  memset(p: DevicePtr | DeviceTensor, byte: number, bytes?: number): void {
    const ptr = p instanceof DeviceTensor ? p.ptr : p;
    this.backend.fill(ptr, byte, bytes ?? ptr.remaining);
  }

  /** Wait for all work; device printf output is delivered here, device-side errors are thrown here (like CUDA). */
  async synchronize(): Promise<SyncReport> {
    try {
      const r = await this.backend.synchronize();
      for (const line of r.printf) { this.stdout.push(line); this.onPrintf(line); }
      return r;
    } catch (e) {
      if (e instanceof CudaError) this.lastError = e.code;
      throw e;
    }
  }
  /** printf output collected since the last call */
  takeOutput(): string { const s = this.stdout.join(""); this.stdout = []; return s; }
  getLastError(): number { const e = this.lastError; this.lastError = 0; return e; }
  peekAtLastError(): number { return this.lastError; }
  /** @internal */ setLastError(code: number): void { this.lastError = code; }

  // ------------------------------------------------------------------ tensors
  tensor(data: ArrayLike<number> | HostArray, shape?: number[], dtype?: DType): DeviceTensor {
    const dt: DType = dtype ?? dtypeOfArray(data);
    const host = data instanceof Float32Array || ArrayBuffer.isView(data) ? (data as HostArray) : toHost(dt, data as ArrayLike<number>);
    const n = (host as HostArray).length;
    const sh = shape ?? [n];
    if (sh.reduce((a, b) => a * b, 1) !== n) throw new CudaError("cudaErrorInvalidValue", `shape [${sh.join(", ")}] does not match ${n} elements`);
    const t = this.empty(sh, dt);
    this.backend.upload(t.ptr, bytesOf(host));
    return t;
  }
  empty(shape: number[], dtype: DType = "float32"): DeviceTensor {
    const n = shape.reduce((a, b) => a * b, 1);
    return new DeviceTensor(this, this.malloc(Math.max(1, n * DTYPE_SIZE[dtype]), `tensor[${shape.join("x")}]`), shape, dtype);
  }
  zeros(shape: number[], dtype: DType = "float32"): DeviceTensor { const t = this.empty(shape, dtype); this.memset(t, 0); return t; }
  full(shape: number[], value: number, dtype: DType = "float32"): DeviceTensor {
    const n = shape.reduce((a, b) => a * b, 1);
    return this.tensor(toHost(dtype, new Array(n).fill(value)), shape, dtype);
  }
  ones(shape: number[], dtype: DType = "float32"): DeviceTensor { return this.full(shape, 1, dtype); }
  arange(n: number, dtype: DType = "float32"): DeviceTensor { return this.tensor(toHost(dtype, Array.from({ length: n }, (_, i) => i)), [n], dtype); }
  /** uniform [0, 1) float32 values from a seeded generator (host side, reproducible across backends) */
  rand(shape: number[], seed = 0): DeviceTensor {
    const n = shape.reduce((a, b) => a * b, 1);
    const a = new Float32Array(n);
    let s = (seed >>> 0) || 0x9e3779b9;
    for (let i = 0; i < n; i++) { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; a[i] = (s >>> 8) / 16777216; }
    return this.tensor(a, shape, "float32");
  }
  randn(shape: number[], seed = 0): DeviceTensor {
    const n = shape.reduce((a, b) => a * b, 1);
    const a = new Float32Array(n);
    let s = (seed >>> 0) || 0x9e3779b9;
    const u = () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return ((s >>> 8) + 0.5) / 16777216; };
    for (let i = 0; i < n; i += 2) {
      const r = Math.sqrt(-2 * Math.log(u())), th = 2 * Math.PI * u();
      a[i] = r * Math.cos(th);
      if (i + 1 < n) a[i + 1] = r * Math.sin(th);
    }
    return this.tensor(a, shape, "float32");
  }

  // ------------------------------------------------------------------ streams & events
  streamCreate(): Stream { return new Stream(this); }
  eventCreate(): CudaEvent { return new CudaEvent(this); }
  eventRecord(ev: CudaEvent, _stream?: Stream): void { ev.record(this.backend.synchronize().then((r) => { for (const l of r.printf) this.onPrintf(l); })); }
  async eventElapsedTime(start: CudaEvent, end: CudaEvent): Promise<number> {
    await start.synchronize(); await end.synchronize();
    return end.time! - start.time!;
  }

  getDeviceProperties(): CudaDeviceProp { return deviceProp(this.backend.info()); }

  destroy(): void {
    this.modules.clear();
    this.backend.destroy();
  }
}

export interface DeviceSelection {
  requested: string;
  selected: string;
  reasons: string[];
}

/** A typed, shaped view of device memory (the JS-side tensor for the standalone CUDA API). */
export class DeviceTensor {
  readonly strides: number[];
  constructor(readonly ctx: CudaContext, readonly ptr: DevicePtr, readonly shape: number[], readonly dtype: DType) {
    this.strides = [];
    let s = 1;
    for (let i = shape.length - 1; i >= 0; i--) { this.strides[i] = s; s *= shape[i]; }
  }
  get size(): number { return this.shape.reduce((a, b) => a * b, 1); }
  get nbytes(): number { return this.size * DTYPE_SIZE[this.dtype]; }
  /** copy to the host (waits for preceding kernels) */
  async toArray(): Promise<HostArray> {
    const bytes = await this.ctx.backend.download(this.ptr, this.nbytes);
    return typedView(this.dtype, bytes.buffer, 0, this.size) as HostArray;
  }
  async toList(): Promise<number[]> { return Array.from(await this.toArray() as ArrayLike<number>, Number); }
  async write(data: ArrayLike<number> | HostArray): Promise<void> {
    const host = ArrayBuffer.isView(data) ? (data as HostArray) : toHost(this.dtype, data);
    this.ctx.backend.upload(this.ptr, bytesOf(host));
  }
  reshape(shape: number[]): DeviceTensor {
    if (shape.reduce((a, b) => a * b, 1) !== this.size) throw new CudaError("cudaErrorInvalidValue", "reshape changes the number of elements");
    return new DeviceTensor(this.ctx, this.ptr, shape, this.dtype);
  }
  free(): void { this.ctx.free(this); }
}

// ------------------------------------------------------------------ modules / functions

export class CudaModule {
  private static next = 1;
  readonly id = CudaModule.next++;
  readonly parsed: ParsedModule;
  private functions = new Map<string, CudaFunction>();
  private symbolPtrs = new Map<string, DevicePtr>();
  readonly hash: string;

  constructor(readonly ctx: CudaContext, readonly source: string, readonly options: CompileOptions) {
    this.parsed = parseModule(source, { ...options, backend: ctx.backend.kind === "webgpu" ? "wgsl" : ctx.backend.kind === "wasm" ? "wasm" : "cuda" });
    this.hash = hashString(`${COMPILER_VERSION}\n${JSON.stringify({ ...options, headers: undefined })}\n${source}`);
  }

  get kernels(): KernelSignature[] { return this.parsed.kernels; }
  get warnings(): Diagnostic[] { return this.parsed.warnings; }

  getFunction(ref?: string): CudaFunction {
    if (!ref) {
      const ks = this.parsed.kernels;
      if (ks.length !== 1) throw new CudaError("cudaErrorInvalidValue", ks.length ? `module defines ${ks.length} kernels (${ks.map((k) => k.name).join(", ")}): pass the kernel name` : "module defines no __global__ kernels");
      ref = ks[0].name;
    }
    let f = this.functions.get(ref);
    if (!f) { f = new CudaFunction(this, ref); this.functions.set(ref, f); }
    return f;
  }

  /** launch (uses the only kernel, or `params.kernel`) */
  launch(params: LaunchParams): Promise<LaunchInfo> { return this.getFunction(params.kernel).launch(params); }

  /**
   * Run a CUDA launch statement, e.g. `vector_add<<<4, 256>>>(a, b, c, n)`, binding identifiers to `vars`.
   * Grid and block may be expressions of numbers or dim3(...).
   */
  exec(statement: string, vars: Record<string, KernelArg>): Promise<LaunchInfo> {
    const e = this.parsed.parseLaunch(statement);
    const evalArg = (x: import("../compiler/ast.js").Expr): unknown => {
      switch (x.k) {
        case "num": return Number(x.text.replace(/[uUlLfF]+$/, ""));
        case "ident": {
          if (!(x.name in vars)) throw new CudaError("cudaErrorInvalidValue", `exec: '${x.name}' is not bound`);
          return vars[x.name];
        }
        case "unary": if (x.op === "-") return -(evalArg(x.e) as number); break;
        case "binary": {
          const a = evalArg(x.l) as number, b = evalArg(x.r) as number;
          switch (x.op) { case "+": return a + b; case "-": return a - b; case "*": return a * b; case "/": return Math.trunc(a / b); case "%": return a % b; }
          break;
        }
        case "call": case "construct":
          if (x.k === "call" && x.callee.k === "ident" && x.callee.name === "dim3") return x.args.map((a) => evalArg(a));
          if (x.k === "construct") return x.args.map((a) => evalArg(a));
          break;
        case "cast": return evalArg(x.e);
      }
      throw new CudaError("cudaErrorInvalidValue", "exec: unsupported expression in launch statement");
    };
    const name = e.callee.k === "ident" ? e.callee.name + (e.callee.targs ? `<${e.callee.targs.map((a) => (a.k === "expr" ? String(evalArg(a.expr)) : typeStr(a.type))).join(", ")}>` : "") : "";
    const cfg = e.config.map(evalArg);
    return this.getFunction(name).launch({ grid: cfg[0] as Dim3, block: cfg[1] as Dim3, sharedMem: (cfg[2] as number) ?? 0, args: e.args.map(evalArg) as KernelArg[] });
  }

  /** device memory of a __constant__ / __device__ variable (cudaMemcpyToSymbol / cudaGetSymbolAddress) */
  symbol(name: string): DevicePtr {
    let p = this.symbolPtrs.get(name);
    if (!p) {
      const g = this.parsed.program.globals.get(name);
      if (!g || !(g.decl.space === "constant" || g.decl.space === "device" || g.decl.space === "managed"))
        throw new CudaError("cudaErrorInvalidSymbol", `'${name}' is not a __constant__ or __device__ variable of this module`);
      // lowering a kernel creates the root with its size and initializer; do it for a throwaway instance if needed
      let root = this.parsed.program.symbolRoots.get(name);
      if (!root) {
        const L = new Lowering(this.parsed.program, null, {});
        L.withoutEffects(() => (L as unknown as { expr(e: unknown): unknown }).expr({ k: "ident", name, loc: g.decl.loc }));
        root = this.parsed.program.symbolRoots.get(name)!;
      }
      p = this.ctx.backend.symbol(this.id, name, root.bytes, root.init);
      this.symbolPtrs.set(name, p);
    }
    return p;
  }
  async memcpyToSymbol(name: string, data: HostArray | ArrayBufferView, offset = 0): Promise<void> {
    const p = this.symbol(name);
    this.ctx.backend.upload(p.add(offset), bytesOf(data as ArrayBufferView));
  }
  async memcpyFromSymbol(dst: HostArray, name: string, offset = 0): Promise<void> {
    const p = this.symbol(name);
    const b = bytesOf(dst);
    b.set(await this.ctx.backend.download(p.add(offset), b.byteLength));
  }
  /** @internal */
  symbolsFor(k: Kernel): Map<string, DevicePtr> {
    const m = new Map<string, DevicePtr>();
    for (const r of k.roots) if (r.symbol) m.set(r.symbol, this.symbol(r.symbol));
    return m;
  }

  /** IR / generated code of a kernel for a launch configuration (for the inspector and debugging). */
  async inspect(ref: string | undefined, block: Dim3, what: "ir" | "code" | "all" = "all", sharedMem = 0): Promise<{ ir: string; code: string; notes: string[]; warnings: Diagnostic[]; passes: unknown }> {
    const f = this.getFunction(ref);
    const b = toDim3(block, "block");
    const k = f.instantiate(b, sharedMem);
    const sink = new DiagnosticSink();
    let code = "";
    let notes: string[] = [];
    if (what !== "ir") {
      const spec = { key: "inspect", aliasGroups: k.params.map((_, i) => i), zeroBase: k.params.map(() => true) };
      const e2 = await this.ctx.backend.prepare({ kernel: k, spec, options: this.options, sink });
      code = e2.code; notes = e2.notes;
    }
    return { ir: printKernel(k), code, notes, warnings: [...this.parsed.warnings, ...sink.warnings], passes: (k as Kernel & { passes?: unknown }).passes };
  }
}


export class CudaFunction {
  private irCache = new Map<string, Kernel>();
  private exeCache = new Map<string, Promise<Executable>>();
  readonly signature: KernelSignature;
  launches = 0;

  constructor(readonly module: CudaModule, readonly ref: string) {
    const name = ref.includes("<") ? ref.slice(0, ref.indexOf("<")).trim() : ref;
    const sig = module.parsed.kernels.find((k) => k.name === name);
    if (!sig) throw new CudaError("cudaErrorInvalidValue", `no kernel named '${name}' (kernels: ${module.parsed.kernels.map((k) => k.name).join(", ") || "none"})`);
    if (sig.template.length && !ref.includes("<")) throw new CudaError("cudaErrorInvalidValue", `kernel '${name}' is a template: give its arguments, e.g. "${name}<${sig.template.map((t) => (t.kind === "type" ? "float" : "256")).join(", ")}>"`);
    this.signature = sig;
  }

  /** @internal lower to IR for a block size (cached) */
  instantiate(block: [number, number, number], sharedMem: number): Kernel {
    const key = `${block.join("x")}/${sharedMem}`;
    let k = this.irCache.get(key);
    if (!k) {
      k = this.module.parsed.instantiate(this.ref, { block, dynamicShared: sharedMem }, { mode: this.module.options.mode });
      this.irCache.set(key, k);
    }
    return k;
  }

  /** @internal */
  executable(k: Kernel, spec: ReturnType<Backend["specKey"]>): Promise<Executable> {
    const key = `${k.block.join("x")}/${this.module.hash}/${spec.key}`;
    let p = this.exeCache.get(key);
    if (!p) {
      const sink = new DiagnosticSink();
      p = this.module.ctx.backend.prepare({ kernel: k, spec, options: this.module.options, sink }).then((exe) => {
        for (const w of sink.warnings) this.module.parsed.sink.warn(w.code, w.message, w.loc);
        return exe;
      });
      this.exeCache.set(key, p);
      p.catch(() => this.exeCache.delete(key));
    }
    return p;
  }

  async launch(params: LaunchParams): Promise<LaunchInfo> {
    const ctx = this.module.ctx;
    const grid = toDim3(params.grid, "grid"), block = toDim3(params.block, "block");
    const smem = params.sharedMem ?? 0;
    const info = ctx.backend.info();
    const threads = block[0] * block[1] * block[2];
    if (threads > info.limits.maxThreadsPerBlock) {
      ctx.setLastError(9);
      throw new CudaError("cudaErrorInvalidConfiguration", `block of ${threads} threads exceeds this device's maximum of ${info.limits.maxThreadsPerBlock} threads per block (${info.backend})`);
    }
    for (let i = 0; i < 3; i++) if (block[i] > info.limits.maxBlockDim[i]) throw new CudaError("cudaErrorInvalidConfiguration", `blockDim.${"xyz"[i]} = ${block[i]} exceeds the device limit ${info.limits.maxBlockDim[i]}`);
    for (let i = 0; i < 3; i++) if (grid[i] > info.limits.maxGridDim[i]) throw new CudaError("cudaErrorInvalidConfiguration", `gridDim.${"xyz"[i]} = ${grid[i]} exceeds the device limit ${info.limits.maxGridDim[i]}`);
    if (smem < 0 || !Number.isInteger(smem)) throw new CudaError("cudaErrorInvalidValue", "sharedMem must be a non-negative integer");
    const sig = this.signature;
    if (params.args.length !== sig.params.length)
      throw new CudaError("cudaErrorInvalidValue", `${this.ref} expects ${sig.params.length} arguments (${sig.params.map((p) => `${p.type} ${p.name}`).join(", ")}), got ${params.args.length}`);
    const t0 = performance.now();
    let compiled = false;
    let k: Kernel;
    try { k = this.instantiate(block, smem); }
    catch (e) { if (e instanceof CompileError) throw e; throw e; }
    // host arrays are copied to the device for the launch and back afterwards (explicit transfer)
    const transfers: { host: HostArray; ptr: DevicePtr; writeBack: boolean }[] = [];
    const args = resolveArgs(ctx, k, sig, params.args, transfers);
    const spec = ctx.backend.specKey(args);
    const key = `${k.block.join("x")}/${this.module.hash}/${spec.key}`;
    const had = (this as unknown as { exeCache: Map<string, unknown> }).exeCache.has(key);
    const exe = await this.executable(k, spec);
    compiled = !had;
    const compileMs = compiled ? performance.now() - t0 : 0;
    ctx.backend.launch(exe, { grid, block, sharedMem: smem }, args, { timing: params.timing, symbols: this.module.symbolsFor(k) });
    this.launches++;
    if (transfers.length) {
      for (const t of transfers) if (t.writeBack) {
        const data = await ctx.backend.download(t.ptr, t.host.byteLength);
        bytesOf(t.host).set(data);
      }
      for (const t of transfers) ctx.backend.free(t.ptr);
    }
    return { kernel: k.displayName, grid, block, compiled, compileMs };
  }
}

/** Map user arguments onto the kernel's IR parameters (struct parameters are flattened). */
function resolveArgs(ctx: CudaContext, k: Kernel, sig: KernelSignature, args: KernelArg[], transfers: { host: HostArray; ptr: DevicePtr; writeBack: boolean }[]): ResolvedArg[] {
  const out: ResolvedArg[] = [];
  let ip = 0;
  sig.params.forEach((sp, i) => {
    const a = args[i];
    const p = k.params[ip];
    if (!p) throw new CudaError("cudaErrorInvalidValue", `internal: parameter mapping for ${sp.name}`);
    if (p.kind === "ptr") {
      out.push({ kind: "ptr", ptr: toPtr(ctx, a, p, i, transfers) });
      ip++;
      return;
    }
    // scalar, vector, or flattened struct
    const group: KernelParam[] = [];
    while (ip < k.params.length && (k.params[ip].name === sp.name || k.params[ip].origin === sp.name)) group.push(k.params[ip++]);
    if (!group.length) { ip++; group.push(p); }
    if (group.length === 1 && group[0].name === sp.name) { out.push({ kind: "scalar", value: toScalar(a, group[0], i), t: group[0].t! }); return; }
    for (const g of group) {
      const path = g.name.slice(sp.name.length);
      out.push({ kind: "scalar", value: toScalar(pick(a, path, i, sp.name), g, i), t: g.t! });
    }
  });
  return out;
}

function pick(obj: unknown, path: string, i: number, name: string): unknown {
  let cur: unknown = obj;
  const parts = path.match(/\.\w+|\[\d+\]/g) ?? [];
  for (const part of parts) {
    if (cur === null || typeof cur !== "object") throw new CudaError("cudaErrorInvalidValue", `argument ${i + 1} (${name}): expected an object with field ${part}`);
    const key = part.startsWith(".") ? part.slice(1) : Number(part.slice(1, -1));
    cur = (cur as Record<string | number, unknown>)[key];
  }
  return cur;
}

function toPtr(ctx: CudaContext, a: KernelArg, p: KernelParam, i: number, transfers: { host: HostArray; ptr: DevicePtr; writeBack: boolean }[]): DevicePtr {
  if (a instanceof DevicePtr) { checkOwner(ctx, a, p, i); return a; }
  if (a instanceof DeviceTensor) { checkOwner(ctx, a.ptr, p, i); return a.ptr; }
  if (a && typeof a === "object" && "ptr" in (a as object) && (a as { ptr: unknown }).ptr instanceof DevicePtr) return (a as { ptr: DevicePtr }).ptr;
  if (ArrayBuffer.isView(a)) {
    const host = a as HostArray;
    const ptr = ctx.malloc(Math.max(4, host.byteLength), `host arg ${p.name}`);
    ctx.backend.upload(ptr, bytesOf(host));
    transfers.push({ host, ptr, writeBack: !p.isConst });
    return ptr;
  }
  throw new CudaError("cudaErrorInvalidValue", `argument ${i + 1} (${p.ctype} ${p.name}) must be a device pointer, DeviceTensor or typed array, got ${describe(a)}`);
}

function checkOwner(ctx: CudaContext, p: DevicePtr, kp: KernelParam, i: number): void {
  p.alloc.check();
  if (p.alloc.owner.id !== ctx.backend.id || p.alloc.owner.kind !== ctx.backend.kind)
    throw new CudaError("cudaErrorInvalidDevicePointer", `argument ${i + 1} (${kp.name}) belongs to a different device (${p.alloc.owner.kind} #${p.alloc.owner.id})`);
}

function toScalar(a: unknown, p: KernelParam, i: number): ScalarValue {
  const t = p.t as VT;
  const bad = () => new CudaError("cudaErrorInvalidValue", `argument ${i + 1} (${p.ctype} ${p.name}) expects ${t.n > 1 ? `an array of ${t.n} numbers` : "a number"}, got ${describe(a)}`);
  if (t.n > 1) {
    let arr: unknown = a;
    if (a && typeof a === "object" && !Array.isArray(a)) arr = ["x", "y", "z", "w"].slice(0, t.n).map((c) => (a as Record<string, unknown>)[c]);
    if (!Array.isArray(arr) || arr.length !== t.n || arr.some((x) => typeof x !== "number" && typeof x !== "bigint")) throw bad();
    return arr as number[];
  }
  if (typeof a === "boolean") return a;
  if (typeof a === "bigint") return a;
  if (typeof a !== "number" || Number.isNaN(a) && !t.s.startsWith("f")) throw bad();
  if (!t.s.startsWith("f") && t.s !== "bool" && t.s !== "bf16" && !Number.isInteger(a)) throw new CudaError("cudaErrorInvalidValue", `argument ${i + 1} (${p.ctype} ${p.name}) expects an integer, got ${a}`);
  return a;
}

function describe(a: unknown): string {
  if (a === null) return "null";
  if (Array.isArray(a)) return `an array of ${a.length}`;
  if (ArrayBuffer.isView(a)) return a.constructor.name;
  return typeof a === "object" ? (a as object).constructor?.name ?? "object" : `${typeof a} ${String(a)}`;
}

function dtypeOfArray(a: unknown): DType {
  if (a instanceof Float64Array) return "float64";
  if (a instanceof Int32Array) return "int32";
  if (a instanceof Uint32Array) return "uint32";
  if (a instanceof Int16Array) return "int16";
  if (a instanceof Uint16Array) return "uint16";
  if (a instanceof Int8Array) return "int8";
  if (a instanceof Uint8Array) return "uint8";
  if (a instanceof BigInt64Array) return "int64";
  if (a instanceof BigUint64Array) return "uint64";
  return "float32";
}

export function toHost(dtype: DType, data: ArrayLike<number>): HostArray {
  switch (dtype) {
    case "float32": return Float32Array.from(data);
    case "float64": return Float64Array.from(data);
    case "int32": return Int32Array.from(data);
    case "uint32": return Uint32Array.from(data);
    case "int16": return Int16Array.from(data);
    case "uint16": return Uint16Array.from(data);
    case "int8": return Int8Array.from(data);
    case "uint8": case "bool": return Uint8Array.from(data);
    case "int64": return BigInt64Array.from(Array.from(data, (x) => BigInt(Math.trunc(x))));
    case "uint64": return BigUint64Array.from(Array.from(data, (x) => BigInt(Math.trunc(x))));
    case "float16": { const { halfBits } = HALF; return Uint16Array.from(Array.from(data, (x) => halfBits(x))); }
    case "bfloat16": { const { bf16Bits } = HALF; return Uint16Array.from(Array.from(data, (x) => bf16Bits(x))); }
  }
}
import { halfBits, bf16Bits } from "../compiler/fold.js";
const HALF = { halfBits, bf16Bits };

function typeStr(t: import("../compiler/ast.js").TypeExpr): string {
  return t.k === "prim" || t.k === "named" ? t.name : t.k === "ptr" ? typeStr(t.to) + "*" : "?";
}

export interface CudaDeviceProp {
  name: string;
  totalGlobalMem: number;
  sharedMemPerBlock: number;
  warpSize: number;
  maxThreadsPerBlock: number;
  maxThreadsDim: [number, number, number];
  maxGridSize: [number, number, number];
  major: number;
  minor: number;
  multiProcessorCount: number;
  /** tinygrad.js extensions */
  backend: string;
  emulated: boolean;
  features: DeviceInfo["features"];
  notes: string[];
}

export function deviceProp(i: DeviceInfo): CudaDeviceProp {
  return {
    name: `${i.name} (tinygrad.js ${i.backend})`, totalGlobalMem: i.limits.memoryLimit, sharedMemPerBlock: i.limits.sharedMemPerBlock, warpSize: 32,
    maxThreadsPerBlock: i.limits.maxThreadsPerBlock, maxThreadsDim: i.limits.maxBlockDim, maxGridSize: i.limits.maxGridDim,
    // a nominal compute capability matching __CUDA_ARCH__ (700) used when compiling
    major: 7, minor: 0, multiProcessorCount: i.threads ?? 1, backend: i.backend, emulated: i.emulated, features: i.features, notes: i.notes,
  };
}
