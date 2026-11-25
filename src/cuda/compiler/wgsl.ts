/**
 * Kernel IR -> WGSL compute shader.
 *
 * Mapping: CUDA thread = shader invocation, block = workgroup (@workgroup_size is specialized to the launch's blockDim),
 * grid = dispatch (split into several dispatches when it exceeds maxComputeWorkgroupsPerDimension; blockIdx gets the
 * chunk's base from the parameter buffer). __shared__ = var<workgroup>, global pointers = storage buffer bindings,
 * __syncthreads = workgroupBarrier (+ storageBarrier when global memory is written).
 *
 * Exactness: WGSL leaves f32->f16 rounding, NaN handling of min/max and fused-ness of fma implementation-defined, so half
 * and bfloat16 rounding, isnan/isinf, fminf/fmaxf, round(), copysign and fmodf are implemented with integer operations.
 * 64-bit integers are emulated with vec2<u32>. double is rejected unless it can be demoted without changing results.
 */
import { fail, type DiagnosticSink } from "./diagnostics.js";
import {
  bin, cast, exprChildren, someExpr, stmtExprs, vt, vtEq, vtSize, vtStr, walkStmts,
  type BinOp, type Expr, type Kernel, type Root, type Stmt, type Var, type VT, type WarpFn,
} from "./ir.js";
import { divExact, foldExpr } from "./fold.js";
import { isFloat, SCALAR_SIZE, type Scalar } from "./types.js";

export interface WGSLCaps {
  shaderF16: boolean;
  subgroups: boolean;
  subgroupMinSize: number;
  subgroupMaxSize: number;
  maxComputeInvocationsPerWorkgroup: number;
  maxComputeWorkgroupSizeX: number;
  maxComputeWorkgroupSizeY: number;
  maxComputeWorkgroupSizeZ: number;
  maxComputeWorkgroupStorageSize: number;
  maxStorageBuffersPerShaderStage: number;
  maxComputeWorkgroupsPerDimension: number;
}

export const DEFAULT_WGSL_CAPS: WGSLCaps = {
  shaderF16: false, subgroups: false, subgroupMinSize: 0, subgroupMaxSize: 0, maxComputeInvocationsPerWorkgroup: 256,
  maxComputeWorkgroupSizeX: 256, maxComputeWorkgroupSizeY: 256, maxComputeWorkgroupSizeZ: 64, maxComputeWorkgroupStorageSize: 16384,
  maxStorageBuffersPerShaderStage: 8, maxComputeWorkgroupsPerDimension: 65535,
};

export interface WGSLOptions {
  /** report out-of-bounds accesses of global buffers through the diagnostics buffer (WebGPU clamps them otherwise) */
  boundsCheck?: boolean;
  /** "error" (default): reject double arithmetic that can't be demoted exactly; "demote": compute double in float (warns) */
  fp64?: "error" | "demote";
  /** warp collectives: "auto" uses subgroups when the subgroup size is exactly 32, else workgroup-memory emulation */
  warps?: "auto" | "emulate" | "subgroups";
  /** pointer parameters that are bound to the same buffer at launch: param index -> alias group (default: own group) */
  aliasGroups?: number[];
  /** pointer parameters whose base offset is known to be 0 (bit i = param i) */
  zeroBase?: boolean[];
}

export type BindingInfo =
  | { binding: number; kind: "params" }
  | { binding: number; kind: "diag" }
  | { binding: number; kind: "buffer"; params: number[]; elemSize: number; access: "read" | "read_write"; layout: string }
  | { binding: number; kind: "symbol"; symbol: string; bytes: number; elemSize: number; access: "read" | "read_write"; init?: Uint8Array };

export interface ParamSlot { param: number; word: number; t: VT; words: number }

export interface DiagInfo {
  printf: { fmt: string; types: string[] }[];
  sites: { kind: "oob-load" | "oob-store" | "oob-atomic" | "assert" | "trap"; root?: string; msg?: string }[];
  /** 32-bit words reserved per launch */
  words: number;
}

export interface WGSLProgram {
  code: string;
  entry: string;
  workgroupSize: [number, number, number];
  bindings: BindingInfo[];
  /** words of the parameter uniform buffer */
  paramWords: number;
  params: ParamSlot[];
  /** per pointer param: word holding its base (in elements of its binding), and word holding its limit (elements) */
  pointerWords: { param: number; baseWord: number; limitWord: number; elemSize: number }[];
  diag: DiagInfo | null;
  sharedBytes: number;
  warpMode: "subgroups" | "emulated" | "none";
  notes: string[];
}

const W_GRID = 0, W_BASE = 4, W_DIAGCAP = 7, W_FIRST = 8;
export const DIAG_HEADER_WORDS = 16;
export const DIAG_DEFAULT_WORDS = 64 * 1024;

type Layout =
  | { k: "typed"; t: VT; elemSize: number; atomic: false }
  | { k: "atomic"; elem: "i32" | "u32"; elemSize: 4 }
  | { k: "words"; atomic: boolean; elemSize: 4 };

interface RootGen { root: Root; name: string; layout: Layout; binding?: number; group?: number; elems: number }

// ------------------------------------------------------------------ register representation of IR types
function regScalar(s: Scalar): string {
  switch (s) {
    case "bool": return "bool";
    case "i8": case "i16": case "i32": return "i32";
    case "u8": case "u16": case "u32": return "u32";
    case "f16": case "bf16": case "f32": return "f32";
    case "i64": case "u64": return "vec2<u32>";
    case "f64": return "f32"; // only after demotion
  }
}
function regType(t: VT): string {
  const s = regScalar(t.s);
  if (t.n === 1) return s;
  if (s.startsWith("vec")) fail("backend", `vectors of 64-bit integers (${vtStr(t)}) are not supported on WebGPU`);
  return `vec${t.n}<${s}>`;
}
const is64i = (s: Scalar) => s === "i64" || s === "u64";

export function generateWGSL(k: Kernel, caps: WGSLCaps, opts: WGSLOptions, sink: DiagnosticSink): WGSLProgram {
  return new WGSLGen(k, caps, opts, sink).run();
}

class WGSLGen {
  private helpers = new Map<string, string>();
  private out: string[] = [];
  private roots = new Map<Root, RootGen>();
  private paramSlots: ParamSlot[] = [];
  private pointerWords: WGSLProgram["pointerWords"] = [];
  private nextWord = W_FIRST;
  private bindings: BindingInfo[] = [];
  private diag: DiagInfo | null = null;
  private globalWrites = false;
  private sharedUsed = false;
  private warpMode: WGSLProgram["warpMode"] = "none";
  private warpScratch: string | null = null;
  private notes: string[] = [];
  private tmpCounter = 0;
  private bounds: boolean;

  constructor(private k: Kernel, private caps: WGSLCaps, private opts: WGSLOptions, private sink: DiagnosticSink) {
    this.bounds = opts.boundsCheck ?? false;
  }

  run(): WGSLProgram {
    const k = this.k;
    const [bx, by, bz] = k.block;
    if (bx <= 0) fail("internal", "WGSL generation needs a block-size specialized kernel");
    const total = bx * by * bz;
    if (total > this.caps.maxComputeInvocationsPerWorkgroup)
      fail("backend", `block of ${bx}x${by}x${bz} = ${total} threads exceeds this GPU's maxComputeInvocationsPerWorkgroup (${this.caps.maxComputeInvocationsPerWorkgroup}); use smaller blocks or the CPU backend`);
    if (bx > this.caps.maxComputeWorkgroupSizeX || by > this.caps.maxComputeWorkgroupSizeY || bz > this.caps.maxComputeWorkgroupSizeZ)
      fail("backend", `block ${bx}x${by}x${bz} exceeds this GPU's workgroup size limits (${this.caps.maxComputeWorkgroupSizeX}x${this.caps.maxComputeWorkgroupSizeY}x${this.caps.maxComputeWorkgroupSizeZ})`);
    this.demoteDoubles();
    this.uniformReturns();
    this.analyzeRoots();
    this.layoutParams();
    const body = this.genBody();
    const decls = this.genDecls();
    const header: string[] = [];
    if (this.helpers.has("__enable_f16")) header.push("enable f16;");
    if (this.warpMode === "subgroups") header.push("enable subgroups;", "diagnostic(off, subgroup_uniformity);");
    const code = [
      `// generated by tinygrad.js CUDA compiler from kernel ${k.displayName} (source line ${k.sourceLine})`,
      ...header,
      ...decls,
      ...[...this.helpers.entries()].filter(([n]) => !n.startsWith("__enable")).map(([, c]) => c),
      `@compute @workgroup_size(${bx}, ${by}, ${bz})`,
      `fn ${this.entryName()}(@builtin(local_invocation_id) lid_: vec3<u32>, @builtin(workgroup_id) wid_: vec3<u32>, @builtin(local_invocation_index) lidx_: u32${this.warpMode === "subgroups" ? ", @builtin(subgroup_invocation_id) sglane_: u32" : ""}) {`,
      `  g_lid = lid_; g_bid = wid_ + vec3<u32>(P[1].x, P[1].y, P[1].z); g_lidx = lidx_;`,
      ...this.k.vars.map((v) => `  var ${v.name}: ${regType(v.t)};`),
      body,
      "}",
    ].join("\n");
    let shared = 0;
    for (const r of this.roots.values()) if (r.root.space === "shared") shared += r.elems * r.layout.elemSize;
    if (this.warpScratch) shared += Math.ceil(total / 32) * 32 * 4 * 2;
    if (shared > this.caps.maxComputeWorkgroupStorageSize)
      fail("backend", `kernel needs ${shared} bytes of shared memory; this GPU allows ${this.caps.maxComputeWorkgroupStorageSize} bytes per workgroup`);
    const storage = this.bindings.filter((b) => b.kind !== "params").length;
    if (storage > this.caps.maxStorageBuffersPerShaderStage)
      fail("backend", `kernel needs ${storage} storage buffers (pointer arguments${this.diag ? " + diagnostics" : ""}); this GPU allows ${this.caps.maxStorageBuffersPerShaderStage} per kernel`);
    return {
      code, entry: this.entryName(), workgroupSize: [bx, by, bz], bindings: this.bindings, paramWords: Math.ceil(this.nextWord / 4) * 4, params: this.paramSlots,
      pointerWords: this.pointerWords, diag: this.diag, sharedBytes: shared, warpMode: this.warpMode, notes: this.notes,
    };
  }

  private entryName(): string { return `cuda_${this.k.name}`; }

  // ------------------------------------------------------------------ early returns before barriers
  /**
   * WGSL requires barriers in uniform control flow, so a thread may not \`return\` before a later __syncthreads(). CUDA
   * kernels often do (\`if (i >= n) return;\` followed by barriers; exited threads don't block them on NVIDIA GPUs). Such
   * returns become \`alive = false\` and the following statements are predicated on \`alive\`, with barriers kept outside
   * the predicates. Threads that "returned" keep executing loop conditions so they reach the barriers.
   */
  private uniformReturns(): void {
    const k = this.k;
    const isBar = (s: Stmt) => s.s === "barrier" || s.s === "warp";
    const hasBar = (l: Stmt[]) => containsStmtLocal(l, isBar);
    const hasRet = (l: Stmt[]) => containsStmtLocal(l, (s) => s.s === "return");
    // needed only if some return can be followed by a barrier
    let needed = false;
    const scan = (l: Stmt[], barrierAfter: boolean): void => {
      for (let i = 0; i < l.length; i++) {
        const after = barrierAfter || hasBar(l.slice(i + 1));
        const s = l[i];
        if (s.s === "return" && after) needed = true;
        if (s.s === "if") { scan(s.then, after); scan(s.else, after); }
        if (s.s === "loop") { const inLoop = after || hasBar([s]); scan(s.body, inLoop); scan(s.cont, inLoop); }
      }
    };
    scan(k.body, false);
    if (!needed) return;
    const alive: Var = { id: 1 + Math.max(0, ...k.vars.map((v) => v.id)), name: "alive_", t: vt("bool") };
    k.vars.push(alive);
    const aliveE: Expr = { op: "var", t: vt("bool"), v: alive };
    const pred = (list: Stmt[], guard: boolean, barrierFreeLoop: boolean): Stmt[] => {
      const out: Stmt[] = [];
      let dead = guard;
      for (const s of list) {
        if (s.s === "return") {
          out.push({ s: "let", v: alive, e: { op: "const", t: vt("bool"), v: false } });
          if (barrierFreeLoop) out.push({ s: "break" });
          return out;
        }
        let t: Stmt = s;
        if (s.s === "if") t = { ...s, then: pred(s.then, dead && hasBar(s.then), barrierFreeLoop), else: pred(s.else, dead && hasBar(s.else), barrierFreeLoop) };
        else if (s.s === "loop") {
          const bf = !hasBar([s]);
          t = { ...s, body: pred(s.body, dead && !bf, bf || barrierFreeLoop && bf), cont: pred(s.cont, dead && !bf, bf) };
          if (bf && hasRet(s.body)) { out.push(dead ? { s: "if", c: aliveE, then: [t], else: [] } : t); if (barrierFreeLoop) out.push({ s: "if", c: { op: "unop", fn: "!", t: vt("bool"), a: aliveE }, then: [{ s: "break" }], else: [] }); dead = true; continue; }
        }
        if (dead && !hasBar([s])) t = { s: "if", c: aliveE, then: [t], else: [] };
        out.push(t);
        if (hasRet([s])) dead = true;
      }
      return out;
    };
    k.body = [{ s: "let", v: alive, e: { op: "const", t: vt("bool"), v: true } }, ...pred(k.body, false, false)];
    this.notes.push("early returns before barriers became per-thread predication (WGSL needs barriers in uniform control flow)");
  }

  // ------------------------------------------------------------------ double precision
  /**
   * WGSL has no f64. A double expression is computed in float only when that is provably bit-identical: one +,-,*,/ on
   * operands that are floats (or double constants exactly representable as float), whose result is immediately converted
   * back to float (double rounding is innocuous because 53 >= 2*24+2); comparisons with exact constants; conversions.
   */
  private demoteDoubles(): void {
    const k = this.k;
    const demoteAll = this.opts.fp64 === "demote";
    let changed = false;
    const f32exact = (e: Expr): Expr | null => {
      if (e.op === "const" && e.t.s === "f64") { const v = e.v as number; return Math.fround(v) === v || Number.isNaN(v) ? { op: "const", t: vt("f32"), v } : null; }
      if (e.op === "cast" && e.t.s === "f64" && e.a.t.n === 1 && (e.a.t.s === "f32" || e.a.t.s === "f16" || e.a.t.s === "bf16")) return cast(vt("f32"), e.a);
      if (e.op === "cast" && e.t.s === "f64" && e.a.t.n === 1 && (e.a.t.s === "i8" || e.a.t.s === "u8" || e.a.t.s === "i16" || e.a.t.s === "u16")) return cast(vt("f32"), e.a);
      return null;
    };
    const rewrite = (e: Expr): Expr => {
      if (e.op === "cast" && e.t.s === "f32" && e.a.t.s === "f64") {
        const a = e.a;
        const ex = f32exact(a);
        if (ex) { changed = true; return ex; }
        if (a.op === "cast" && a.a.t.n === 1 && (a.a.t.s === "i32" || a.a.t.s === "u32")) { changed = true; return cast(vt("f32"), a.a); } // int -> double -> float == int -> float
        if (a.op === "binop" && ["+", "-", "*", "/"].includes(a.fn)) {
          const x = f32exact(a.a), y = f32exact(a.b);
          if (x && y) { changed = true; return bin(a.fn, x, y, vt("f32")); }
        }
        if (a.op === "call" && a.fn === "sqrt") { const x = f32exact(a.args[0]); if (x) { changed = true; return { ...a, t: vt("f32"), args: [x] }; } }
      }
      if (e.op === "binop" && ["<", "<=", ">", ">=", "==", "!="].includes(e.fn) && e.a.t.s === "f64") {
        const x = f32exact(e.a), y = f32exact(e.b);
        if (x && y) { changed = true; return { ...e, a: x, b: y }; }
      }
      // (int)(x * 0.5) where x is float: the double product of a float and an exact constant is exact, int conversion truncates the same
      if (e.op === "cast" && !isFloat(e.t.s) && e.a.t.s === "f64" && e.a.op === "binop" && e.a.fn === "*") {
        const x = f32exact(e.a.a), y = f32exact(e.a.b);
        if (x && y && x.op !== "const" && y.op === "const" && isPow2Abs(y.v as number)) { changed = true; return cast(e.t, bin("*", x, y, vt("f32"))); }
      }
      return e;
    };
    const mapE = (e: Expr): Expr => {
      // bottom-up then top-down to fixpoint
      let cur = e;
      for (let i = 0; i < 4; i++) {
        const next = mapTopDown(cur, rewrite);
        if (next === cur) break;
        cur = foldExpr(next, true);
      }
      return cur;
    };
    const visit = (l: Stmt[]): Stmt[] => l.map((s) => {
      let t = mapStmt(s, mapE);
      if (t.s === "if") t = { ...t, then: visit(t.then), else: visit(t.else) };
      if (t.s === "loop") t = { ...t, body: visit(t.body), cont: visit(t.cont) };
      return t;
    });
    k.body = visit(k.body);
    void changed;
    // anything left in f64?
    let bad: string | null = null;
    const check = (e: Expr) => someExpr(e, (x) => { if (x.t.s === "f64" && !bad) bad = describeExpr(x); return false; });
    walkStmts(k.body, (s) => { for (const e of stmtExprs(s)) check(e); if ((s.s === "store" || s.s === "atomic") && s.t.s === "f64") bad = bad ?? `a double store to ${s.root.name}`; });
    for (const v of k.vars) if (v.t.s === "f64" && !bad) bad = `variable '${v.name.replace(/_\d+$/, "")}' of type double`;
    for (const p of k.params) if (p.t?.s === "f64" && !bad) bad = `parameter '${p.name}' of type double`;
    if (bad) {
      if (!demoteAll) fail("backend", `double precision is not supported by WebGPU (WGSL has no f64): ${bad}. Use float (e.g. write 0.5f instead of 0.5), compile with { fp64: "demote" } to compute doubles in float (changes results), or run on the CPU backend`);
      this.sink.warn("backend", `double precision demoted to float because of { fp64: "demote" }: results differ from native CUDA`);
      this.notes.push("fp64 demoted to f32 (fp64: 'demote')");
      const toF = (t: VT): VT => (t.s === "f64" ? { s: "f32", n: t.n } : t);
      const mapT = (e: Expr): Expr => mapTopDown(e, (x) => (x.t.s === "f64" ? ({ ...x, t: toF(x.t) } as Expr) : x));
      walkStmtsMut(k, mapT);
      for (const v of k.vars) v.t = toF(v.t);
      for (const r of k.roots) if (r.elem?.s === "f64") fail("backend", `buffer '${r.name}' holds doubles; fp64 demotion cannot change the memory layout of a double buffer`);
      for (const p of k.params) if (p.t?.s === "f64") p.t = toF(p.t);
      this.k.features.add("fp64-demoted");
    }
  }

  // ------------------------------------------------------------------ memory layouts
  private analyzeRoots(): void {
    const k = this.k;
    const access = new Map<Root, { types: VT[]; atomics: { t: VT; fn: string }[]; stores: VT[] }>();
    const get = (r: Root) => { let a = access.get(r); if (!a) access.set(r, (a = { types: [], atomics: [], stores: [] })); return a; };
    walkStmts(k.body, (s) => {
      if (s.s === "store") { get(s.root).types.push(s.t); get(s.root).stores.push(s.t); if (s.root.space === "global") this.globalWrites = true; }
      if (s.s === "atomic") { get(s.root).atomics.push({ t: s.t, fn: s.fn }); if (s.root.space === "global") this.globalWrites = true; }
      for (const e of stmtExprs(s)) someExpr(e, (x) => { if (x.op === "load") get(x.root).types.push(x.t); return false; });
    });
    // alias groups: pointer params bound to one buffer share a binding (WebGPU binds whole buffers)
    const groups = new Map<number, Root[]>();
    for (const r of k.roots) {
      if (r.param === undefined) continue;
      const g = this.opts.aliasGroups?.[r.param] ?? r.param;
      groups.set(g, [...(groups.get(g) ?? []), r]);
    }
    const chooseLayout = (rs: Root[]): Layout => {
      const types: VT[] = [], atomics: { t: VT; fn: string }[] = [], stores: VT[] = [];
      for (const r of rs) { const a = access.get(r); if (a) { types.push(...a.types); atomics.push(...a.atomics); stores.push(...a.stores); } }
      for (const a of atomics) {
        if (SCALAR_SIZE[a.t.s] !== 4 && a.t.s !== "u16") fail("backend", `${vtStr(a.t)} atomics are not supported by WebGPU (only 32-bit atomics exist): ${rs[0].name}`);
        if (a.t.s === "u16") fail("backend", "16-bit atomicCAS is not supported by WebGPU");
      }
      const all = [...types, ...atomics.map((a) => a.t)];
      if (all.some((t) => SCALAR_SIZE[t.s] === 8 && t.s === "f64")) fail("backend", `double values in memory are not supported by WebGPU (${rs[0].name})`);
      if (atomics.length) {
        if (all.every((t) => SCALAR_SIZE[t.s] === 4 && t.n === 1)) {
          const signedMinMax = atomics.some((a) => a.t.s === "i32" && (a.fn === "min" || a.fn === "max"));
          return { k: "atomic", elem: signedMinMax ? "i32" : "u32", elemSize: 4 };
        }
        return { k: "words", atomic: true, elemSize: 4 };
      }
      const distinct: VT[] = [];
      for (const t of all) if (!distinct.some((d) => vtEq(d, t))) distinct.push(t);
      if (distinct.length === 1) {
        const t = distinct[0];
        const s = t.s;
        if ((s === "f32" || s === "i32" || s === "u32") && (t.n === 1 || t.n === 2 || t.n === 4)) return { k: "typed", t, elemSize: vtSize(t), atomic: false };
        if (s === "f16" && this.caps.shaderF16 && (t.n === 1 || t.n === 2 || t.n === 4)) { this.helpers.set("__enable_f16", ""); return { k: "typed", t, elemSize: vtSize(t), atomic: false }; }
      }
      if (distinct.length === 0 && rs[0].elem) {
        const t = rs[0].elem;
        if ((t.s === "f32" || t.s === "i32" || t.s === "u32") && t.n === 1) return { k: "typed", t, elemSize: 4, atomic: false };
      }
      // sub-word stores race with neighbours unless done atomically
      const subword = stores.some((t) => SCALAR_SIZE[t.s] < 4 && vtSize(t) % 4 !== 0);
      if (subword) this.notes.push(`buffer '${rs[0].name}': 8/16-bit stores use atomic read-modify-write (WebGPU has no 8/16-bit storage)`);
      return { k: "words", atomic: subword, elemSize: 4 };
    };
    let binding = 0;
    this.bindings.push({ binding: binding++, kind: "params" });
    const needsDiag = this.bounds || k.features.has("printf") || k.features.has("assert") || containsTrap(k.body);
    if (needsDiag) {
      this.diag = { printf: [], sites: [], words: DIAG_DEFAULT_WORDS };
      this.bindings.push({ binding: binding++, kind: "diag" });
    }
    for (const [g, rs] of [...groups.entries()].sort((a, b) => a[0] - b[0])) {
      const layout = chooseLayout(rs);
      const readOnly = rs.every((r) => !(access.get(r)?.stores.length || access.get(r)?.atomics.length));
      const name = `buf${g}_${rs[0].name}`;
      const b = binding++;
      for (const r of rs) this.roots.set(r, { root: r, name, layout, binding: b, group: g, elems: -1 });
      this.bindings.push({ binding: b, kind: "buffer", params: rs.map((r) => r.param!), elemSize: layout.elemSize, access: readOnly ? "read" : "read_write", layout: layoutStr(layout) });
    }
    for (const r of k.roots) {
      if (r.param !== undefined || this.roots.has(r)) continue;
      const layout = chooseLayout([r]);
      if (r.symbol) {
        const readOnly = !(access.get(r)?.stores.length || access.get(r)?.atomics.length);
        const b = binding++;
        const elems = Math.ceil(r.bytes / layout.elemSize);
        this.roots.set(r, { root: r, name: `sym_${r.name}`, layout, binding: b, elems });
        this.bindings.push({ binding: b, kind: "symbol", symbol: r.symbol, bytes: r.bytes, elemSize: layout.elemSize, access: readOnly ? "read" : "read_write", init: r.init });
        continue;
      }
      if (r.space === "shared") this.sharedUsed = true;
      let bytes = r.bytes;
      if (r.dynamic) bytes = Math.max(4, bytes);
      this.roots.set(r, { root: r, name: `${r.space === "shared" ? "sh" : "lc"}_${r.name}`, layout, elems: Math.max(1, Math.ceil(bytes / layout.elemSize)) });
    }
  }

  private layoutParams(): void {
    // scalar parameters
    this.k.params.forEach((p, i) => {
      if (p.kind !== "scalar") return;
      const words = Math.ceil(vtSize(p.t!) / 4);
      if (words > 1 && p.t!.n > 1 && SCALAR_SIZE[p.t!.s] === 8) fail("backend", `parameter ${p.name}: vectors of 64-bit values are not supported on WebGPU`);
      this.paramSlots.push({ param: i, word: this.nextWord, t: p.t!, words });
      this.nextWord += words;
    });
    // pointer bases / limits
    for (const [r, g] of this.roots) {
      if (r.param === undefined) continue;
      this.pointerWords.push({ param: r.param, baseWord: this.nextWord, limitWord: this.nextWord + 1, elemSize: g.layout.elemSize });
      this.nextWord += 2;
    }
  }

  private word(w: number): string { return `P[${w >> 2}].${"xyzw"[w & 3]}`; }

  // ------------------------------------------------------------------ declarations
  private genDecls(): string[] {
    const d: string[] = [];
    const nvec = Math.max(1, Math.ceil(this.nextWord / 4));
    d.push(`@group(0) @binding(0) var<uniform> P: array<vec4<u32>, ${nvec}>;`);
    if (this.diag) d.push(`@group(0) @binding(1) var<storage, read_write> DIAG: array<atomic<u32>>;`);
    const declared = new Set<string>();
    for (const g of this.roots.values()) {
      if (declared.has(g.name)) continue;
      declared.add(g.name);
      const el = layoutElemType(g.layout);
      if (g.binding !== undefined) {
        const b = this.bindings.find((x) => x.binding === g.binding) as { access: string };
        const acc = b.access === "read" && !layoutIsAtomic(g.layout) ? "read" : "read_write";
        d.push(`@group(0) @binding(${g.binding}) var<storage, ${acc}> ${g.name}: array<${el}>;`);
      } else if (g.root.space === "shared") d.push(`var<workgroup> ${g.name}: array<${el}, ${g.elems}>;`);
      else d.push(`var<private> ${g.name}: array<${el}, ${g.elems}>;`);
    }
    d.push("var<private> g_lid: vec3<u32>;", "var<private> g_bid: vec3<u32>;", "var<private> g_lidx: u32;");
    if (this.warpScratch) {
      const n = Math.ceil(this.k.block[0] * this.k.block[1] * this.k.block[2] / 32) * 32;
      d.push(`var<workgroup> warp_lo: array<u32, ${n}>;`, `var<workgroup> warp_hi: array<u32, ${n}>;`);
    }
    return d;
  }

  // ------------------------------------------------------------------ statements
  private genBody(): string {
    this.out = [];
    this.stmts(this.k.body, "  ");
    return this.out.join("\n");
  }

  private emit(line: string): void { this.out.push(line); }

  private stmts(list: Stmt[], ind: string): void { for (const s of list) this.stmt(s, ind); }

  private stmt(s: Stmt, ind: string): void {
    switch (s.s) {
      case "let": this.emit(`${ind}${s.v.name} = ${this.expr(s.e)};`); return;
      case "store": this.store(s.root, s.off, s.t, s.e, ind); return;
      case "atomic": this.atomic(s, ind); return;
      case "if": {
        this.emit(`${ind}if (${this.expr(s.c)}) {`);
        this.stmts(s.then, ind + "  ");
        if (s.else.length) { this.emit(`${ind}} else {`); this.stmts(s.else, ind + "  "); }
        this.emit(`${ind}}`);
        return;
      }
      case "loop": {
        this.emit(`${ind}loop {`);
        this.stmts(s.body, ind + "  ");
        if (s.cont.length || s.breakIf) {
          this.emit(`${ind}  continuing {`);
          this.stmts(s.cont, ind + "    ");
          if (s.breakIf) this.emit(`${ind}    break if ${this.expr(s.breakIf)};`);
          this.emit(`${ind}  }`);
        }
        this.emit(`${ind}}`);
        return;
      }
      case "break": this.emit(`${ind}break;`); return;
      case "continue": this.emit(`${ind}continue;`); return;
      case "return": this.emit(`${ind}return;`); return;
      case "barrier":
        if (s.scope === "warp" && this.warpMode === "subgroups") { this.emit(`${ind}workgroupBarrier();`); return; }
        if (this.globalWrites) this.emit(`${ind}storageBarrier();`);
        this.emit(`${ind}workgroupBarrier();`);
        return;
      case "fence":
        if (s.scope === "block") { this.sink.warn("backend", "__threadfence_block() has no WGSL equivalent outside a barrier; it is ignored (use __syncthreads to order memory)"); return; }
        fail("backend", `__threadfence${s.scope === "system" ? "_system" : ""}(): WebGPU has no device-scope memory fence, so cross-block communication through global memory cannot be ordered; use separate kernel launches`);
      case "warp": this.warp(s, ind); return;
      case "printf": this.printf(s.fmt, s.args, ind); return;
      case "assert": {
        const site = this.site("assert", undefined, s.msg);
        this.emit(`${ind}if (!(${this.expr(s.c)})) { diag_error(${site}u, 0u); }`);
        return;
      }
      case "trap": {
        const site = this.site("trap", undefined, s.msg);
        this.emit(`${ind}diag_error(${site}u, 0u);`);
        this.emit(`${ind}return;`);
        return;
      }
      case "comment": if (s.text) this.emit(`${ind}// ${s.text}`); return;
    }
  }

  private site(kind: DiagInfo["sites"][number]["kind"], root?: string, msg?: string): number {
    const d = this.diag!;
    d.sites.push({ kind, root, msg });
    this.ensureDiagHelpers();
    return d.sites.length - 1;
  }

  private ensureDiagHelpers(): void {
    if (this.helpers.has("diag_error")) return;
    // DIAG[0]: next free word for printf records; DIAG[1]: error count; DIAG[2..9]: first error (site, value, thread, block)
    this.helpers.set("diag_error", `fn diag_error(site: u32, value: u32) {
  if (atomicAdd(&DIAG[1], 1u) == 0u) {
    atomicStore(&DIAG[2], site); atomicStore(&DIAG[3], value);
    atomicStore(&DIAG[4], g_lid.x); atomicStore(&DIAG[5], g_lid.y); atomicStore(&DIAG[6], g_lid.z);
    atomicStore(&DIAG[7], g_bid.x); atomicStore(&DIAG[8], g_bid.y); atomicStore(&DIAG[9], g_bid.z);
  }
}`);
  }

  private printf(fmt: string, args: Expr[], ind: string): void {
    const d = this.diag!;
    this.ensureDiagHelpers();
    const id = d.printf.length;
    d.printf.push({ fmt, types: args.map((a) => a.t.s) });
    const words: string[] = [];
    for (const a of args) {
      const e = this.expr(a);
      const s = a.t.s;
      if (is64i(s)) { const t = this.tmpName(); this.emit(`${ind}let ${t} = ${e};`); words.push(`${t}.x`, `${t}.y`); }
      else if (isFloat(s)) words.push(`bitcast<u32>(f32(${e}))`);
      else if (s === "bool") words.push(`select(0u, 1u, ${e})`);
      else words.push(`bitcast<u32>(${e})`);
    }
    const n = words.length + 2;
    const t = this.tmpName();
    this.emit(`${ind}{ let ${t} = atomicAdd(&DIAG[0], ${n}u) + ${DIAG_HEADER_WORDS}u;`);
    this.emit(`${ind}  if (${t} + ${n}u <= ${this.word(W_DIAGCAP)}) {`);
    this.emit(`${ind}    atomicStore(&DIAG[${t}], ${n}u); atomicStore(&DIAG[${t} + 1u], ${id}u);`);
    words.forEach((w, i) => this.emit(`${ind}    atomicStore(&DIAG[${t} + ${i + 2}u], ${w});`));
    this.emit(`${ind}  } }`);
  }

  private tmpName(): string { return `tmp${this.tmpCounter++}`; }

  // ------------------------------------------------------------------ memory access
  /** element index expression for a typed/atomic root and the bounds limit word */
  private index(g: RootGen, off: Expr): { idx: string; limit: string | null } {
    const es = g.layout.elemSize;
    const q = divExact(foldExpr(off, true), es);
    let idx = q ? `u32(${this.expr(q)})` : `(u32(${this.expr(off)}) / ${es}u)`;
    let limit: string | null = null;
    if (g.root.param !== undefined) {
      const pw = this.pointerWords.find((p) => p.param === g.root.param)!;
      const zero = this.opts.zeroBase?.[g.root.param];
      if (!zero) idx = `(${this.word(pw.baseWord)} + ${idx})`;
      limit = this.word(pw.limitWord);
    }
    return { idx, limit };
  }

  private load(root: Root, off: Expr, t: VT): string {
    const g = this.roots.get(root);
    if (!g) fail("internal", `no layout for root ${root.name}`);
    const L = g.layout;
    if (L.k === "typed" && vtEq(L.t, t)) {
      const { idx, limit } = this.index(g, off);
      const raw = `${g.name}[${idx}]`;
      const conv = (x: string) => (t.s === "f16" ? (t.n === 1 ? `f32(${x})` : `vec${t.n}<f32>(${x})`) : x);
      if (this.bounds && limit) return conv(this.boundedLoad(g, t, idx, limit, `${layoutElemType(L)}`));
      return conv(raw);
    }
    // word-based access (also atomic layouts): helper per (root, type)
    return this.wordLoad(g, off, t);
  }

  private boundedLoad(g: RootGen, t: VT, idx: string, limit: string, wgslT: string): string {
    const name = `ld_${g.name}_${vtStr(t)}`;
    if (!this.helpers.has(name)) {
      const site = this.site("oob-load", g.root.name);
      this.helpers.set(name, `fn ${name}(i: u32, lim: u32) -> ${wgslT} {
  if (i >= lim) { diag_error(${site}u, i); return ${wgslT}(); }
  return ${g.name}[i];
}`);
    }
    return `${name}(${idx}, ${limit})`;
  }

  private store(root: Root, off: Expr, t: VT, value: Expr, ind: string): void {
    const g = this.roots.get(root)!;
    const L = g.layout;
    const v = this.expr(value);
    if (L.k === "typed" && vtEq(L.t, t)) {
      const { idx, limit } = this.index(g, off);
      const val = t.s === "f16" ? (t.n === 1 ? `f16(${v})` : `vec${t.n}<f16>(${v})`) : v;
      if (this.bounds && limit) {
        const site = this.site("oob-store", root.name);
        const ti = this.tmpName();
        this.emit(`${ind}{ let ${ti} = ${idx}; if (${ti} < ${limit}) { ${g.name}[${ti}] = ${val}; } else { diag_error(${site}u, ${ti}); } }`);
      } else this.emit(`${ind}${g.name}[${idx}] = ${val};`);
      return;
    }
    this.wordStore(g, off, t, v, ind);
  }

  /** byte offset (absolute within the binding) as a u32 expression */
  private byteOff(g: RootGen, off: Expr): string {
    let o = `u32(${this.expr(off)})`;
    if (g.root.param !== undefined && !this.opts.zeroBase?.[g.root.param]) {
      const pw = this.pointerWords.find((p) => p.param === g.root.param)!;
      o = `(${this.word(pw.baseWord)} * 4u + ${o})`;
    }
    return o;
  }

  private limitBytes(g: RootGen): string | null {
    if (g.root.param === undefined) return null;
    const pw = this.pointerWords.find((p) => p.param === g.root.param)!;
    // limit word holds elements of 4 bytes for word layouts
    return `(${this.word(pw.limitWord)} * 4u)`;
  }

  private wordRead(g: RootGen, wi: string): string {
    return layoutIsAtomic(g.layout) ? `bitcast<u32>(atomicLoad(&${g.name}[${wi}]))` : `bitcast<u32>(${g.name}[${wi}])`;
  }

  private wordLoad(g: RootGen, off: Expr, t: VT): string {
    const name = `ldw_${g.name}_${vtStr(t)}`;
    if (!this.helpers.has(name)) {
      const ret = regType(t);
      const lines: string[] = [];
      const bc = this.bounds && g.root.param !== undefined;
      const site = bc ? this.site("oob-load", g.root.name) : -1;
      lines.push(`fn ${name}(b: u32${bc ? ", lim: u32" : ""}) -> ${ret} {`);
      if (bc) lines.push(`  if (b + ${vtSize(t)}u > lim) { diag_error(${site}u, b); return ${ret}(); }`);
      const es = SCALAR_SIZE[t.s];
      const lane = (k: number) => {
        const bo = k === 0 ? "b" : `b + ${k * es}u`;
        if (es === 4) {
          const w = this.wordRead(g, `(${bo}) >> 2u`);
          return t.s === "f32" ? `bitcast<f32>(${w})` : t.s === "i32" ? `bitcast<i32>(${w})` : w;
        }
        if (es === 8) {
          return `vec2<u32>(${this.wordRead(g, `(${bo}) >> 2u`)}, ${this.wordRead(g, `((${bo}) >> 2u) + 1u`)})`;
        }
        const w = this.wordRead(g, `(${bo}) >> 2u`);
        const sh = `(((${bo}) & 3u) * 8u)`;
        if (es === 2) {
          const bits = `((${w} >> ${sh}) & 0xffffu)`;
          if (t.s === "f16") return `f16_from_bits(${bits})`;
          if (t.s === "bf16") return `bitcast<f32>(${bits} << 16u)`;
          if (t.s === "i16") return `(bitcast<i32>(${bits} << 16u) >> 16u)`;
          return bits;
        }
        const bits = `((${w} >> ${sh}) & 0xffu)`;
        if (t.s === "i8") return `(bitcast<i32>(${bits} << 24u) >> 24u)`;
        if (t.s === "bool") return `(${bits} != 0u)`;
        return bits;
      };
      if (t.s === "f16") this.halfHelpers();
      if (t.n === 1) lines.push(`  return ${lane(0)};`);
      else lines.push(`  return ${ret}(${Array.from({ length: t.n }, (_, k) => lane(k)).join(", ")});`);
      lines.push("}");
      this.helpers.set(name, lines.join("\n"));
    }
    const lim = this.bounds && g.root.param !== undefined ? `, ${this.limitBytes(g)}` : "";
    return `${name}(${this.byteOff(g, off)}${lim})`;
  }

  private wordStore(g: RootGen, off: Expr, t: VT, v: string, ind: string): void {
    const name = `stw_${g.name}_${vtStr(t)}`;
    const es = SCALAR_SIZE[t.s];
    if (!this.helpers.has(name)) {
      const lines: string[] = [];
      const bc = this.bounds && g.root.param !== undefined;
      const site = bc ? this.site("oob-store", g.root.name) : -1;
      lines.push(`fn ${name}(b: u32, v: ${regType(t)}${bc ? ", lim: u32" : ""}) {`);
      if (bc) lines.push(`  if (b + ${vtSize(t)}u > lim) { diag_error(${site}u, b); return; }`);
      const atomic = layoutIsAtomic(g.layout);
      const put = (wi: string, val: string) => atomic ? `atomicStore(&${g.name}[${wi}], ${g.layout.k === "atomic" && g.layout.elem === "i32" ? `bitcast<i32>(${val})` : val});` : `${g.name}[${wi}] = ${val};`;
      if (t.s === "f16") this.halfHelpers();
      if (t.s === "bf16") this.bf16Helpers();
      for (let k = 0; k < t.n; k++) {
        const x = t.n === 1 ? "v" : `v[${k}]`;
        const bo = k === 0 ? "b" : `b + ${k * es}u`;
        if (es === 4) lines.push(`  ${put(`(${bo}) >> 2u`, t.s === "u32" ? x : `bitcast<u32>(${x})`)}`);
        else if (es === 8) { lines.push(`  ${put(`(${bo}) >> 2u`, `${x}.x`)}`, `  ${put(`((${bo}) >> 2u) + 1u`, `${x}.y`)}`); }
        else {
          const bits = es === 2 ? (t.s === "f16" ? `f16_to_bits(${x})` : t.s === "bf16" ? `bf16_to_bits(${x})` : `(bitcast<u32>(${x}) & 0xffffu)`) : t.s === "bool" ? `select(0u, 1u, ${x})` : `(bitcast<u32>(${x}) & 0xffu)`;
          const mask = es === 2 ? "0xffffu" : "0xffu";
          if (t.n * es === 4 && k === 0 && false) { /* full-word packing handled below */ }
          if (!atomic) fail("internal", "sub-word store into a non-atomic layout");
          const sh = `(((${bo}) & 3u) * 8u)`;
          const wi = `(${bo}) >> 2u`;
          // atomic read-modify-write of the containing word
          lines.push(`  { let sh = ${sh}; let wi = ${wi}; let nb = (${bits}) << sh; let m = ~(${mask} << sh);`,
            `    ${g.layout.k === "atomic" && g.layout.elem === "i32" ? `atomicAnd(&${g.name}[wi], bitcast<i32>(m)); atomicOr(&${g.name}[wi], bitcast<i32>(nb));` : `atomicAnd(&${g.name}[wi], m); atomicOr(&${g.name}[wi], nb);`} }`);
        }
      }
      // whole-word sub-word vectors (char4, half2, short2): one plain word store is enough and race-free
      if (es < 4 && vtSize(t) % 4 === 0) {
        lines.length = bc ? 2 : 1;
        const parts: string[] = [];
        for (let k = 0; k < t.n; k++) {
          const x = `v[${k}]`;
          const bits = es === 2 ? (t.s === "f16" ? `f16_to_bits(${x})` : t.s === "bf16" ? `bf16_to_bits(${x})` : `(bitcast<u32>(${x}) & 0xffffu)`) : `(bitcast<u32>(${x}) & 0xffu)`;
          parts.push(`(${bits} << ${((k * es) % 4) * 8}u)`);
        }
        const perWord = 4 / es;
        for (let w = 0; w < (t.n * es) / 4; w++) {
          const val = parts.slice(w * perWord, (w + 1) * perWord).join(" | ");
          lines.push(`  ${put(`(b >> 2u) + ${w}u`, val)}`);
        }
      }
      lines.push("}");
      this.helpers.set(name, lines.join("\n"));
    }
    const lim = this.bounds && g.root.param !== undefined ? `, ${this.limitBytes(g)}` : "";
    this.emit(`${ind}${name}(${this.byteOff(g, off)}, ${v}${lim});`);
  }

  private atomic(s: Stmt & { s: "atomic" }, ind: string): void {
    const g = this.roots.get(s.root)!;
    const L = g.layout;
    if (L.k !== "atomic" && !(L.k === "words" && L.atomic)) fail("internal", `atomics on ${s.root.name} without an atomic layout`);
    let { idx, limit } = this.index(g, s.off);
    if (L.k === "words") { idx = `(${this.byteOff(g, s.off)} >> 2u)`; limit = g.root.param !== undefined ? this.word(this.pointerWords.find((p) => p.param === g.root.param)!.limitWord) : null; }
    const ref = `&${g.name}[ai]`;
    const res = s.result ? s.result.name : null;
    const ae = L.k === "atomic" ? L.elem : "u32"; // atomic element type
    const toA = (x: string, from: Scalar) => (from === "f32" ? (ae === "u32" ? `bitcast<u32>(${x})` : `bitcast<i32>(${x})`) : from === ae ? x : `bitcast<${ae}>(${x})`);
    const fromA = (x: string) => (s.t.s === "f32" ? `bitcast<f32>(${x})` : s.t.s === ae ? x : `bitcast<${s.t.s}>(${x})`);
    const args = s.args.map((a) => this.expr(a));
    const lines: string[] = [];
    const assign = (x: string) => (res ? `${res} = ${fromA(x)};` : `_ = ${x};`);
    const sameSign = s.t.s === ae;
    switch (s.fn) {
      case "add": case "sub": case "and": case "or": case "xor": case "xchg": {
        if (s.t.s === "f32" && s.fn === "add") {
          lines.push(`loop { let o = atomicLoad(${ref}); let r = atomicCompareExchangeWeak(${ref}, o, ${toA(`(bitcast<f32>(o) + ${args[0]})`, "f32")}); if (r.exchanged) { ${res ? `${res} = bitcast<f32>(o);` : ""} break; } }`);
          break;
        }
        const fn = { add: "atomicAdd", sub: "atomicSub", and: "atomicAnd", or: "atomicOr", xor: "atomicXor", xchg: "atomicExchange" }[s.fn];
        lines.push(assign(`${fn}(${ref}, ${toA(args[0], s.t.s)})`));
        break;
      }
      case "min": case "max": {
        const fn = s.fn === "min" ? "atomicMin" : "atomicMax";
        if (sameSign) { lines.push(assign(`${fn}(${ref}, ${args[0]})`)); break; }
        // signedness differs from the binding's atomic type: compare-exchange loop
        lines.push(`loop { let o = atomicLoad(${ref}); let nv = ${s.fn}(${fromA("o")}, ${args[0]}); let r = atomicCompareExchangeWeak(${ref}, o, ${toA("nv", s.t.s)}); if (r.exchanged) { ${res ? `${res} = ${fromA("o")};` : ""} break; } }`);
        break;
      }
      case "inc": case "dec": {
        const nv = s.fn === "inc" ? `select(ov + 1u, 0u, ov >= ${args[0]})` : `select(ov - 1u, ${args[0]}, ov == 0u || ov > ${args[0]})`;
        lines.push(`loop { let o = atomicLoad(${ref}); let ov = ${fromA("o")}; let r = atomicCompareExchangeWeak(${ref}, o, ${toA(nv, "u32")}); if (r.exchanged) { ${res ? `${res} = ov;` : ""} break; } }`);
        break;
      }
      case "cas": {
        const cmp = toA(args[0], s.t.s), val = toA(args[1], s.t.s);
        lines.push(`loop { let r = atomicCompareExchangeWeak(${ref}, ${cmp}, ${val}); if (r.exchanged || r.old_value != ${cmp}) { ${res ? `${res} = ${fromA("r.old_value")};` : ""} break; } }`);
        break;
      }
    }
    if (this.bounds && limit) {
      const site = this.site("oob-atomic", s.root.name);
      this.emit(`${ind}{ let ai = ${idx}; if (ai < ${limit}) { ${lines.join(" ")} } else { diag_error(${site}u, ai); } }`);
    } else this.emit(`${ind}{ let ai = ${idx}; ${lines.join(" ")} }`);
  }

  // ------------------------------------------------------------------ warp collectives
  private chooseWarpMode(): void {
    if (this.warpMode !== "none") return;
    const total = this.k.block[0] * this.k.block[1] * this.k.block[2];
    const mode = this.opts.warps ?? "auto";
    const native = this.caps.subgroups && this.caps.subgroupMinSize === 32 && this.caps.subgroupMaxSize === 32 && total % 32 === 0;
    if (mode === "subgroups" && !native) fail("backend", `warps: "subgroups" needs WebGPU subgroups of exactly 32 lanes (this GPU: ${this.caps.subgroups ? `${this.caps.subgroupMinSize}-${this.caps.subgroupMaxSize}` : "no subgroups"}) and a block size that is a multiple of 32`);
    if (mode !== "emulate" && native) { this.warpMode = "subgroups"; this.notes.push("warp operations: native WebGPU subgroups (size 32)"); }
    else {
      this.warpMode = "emulated"; this.warpScratch = "warp";
      this.notes.push(`warp operations: emulated with workgroup memory and barriers (${this.caps.subgroups ? `subgroup size ${this.caps.subgroupMinSize}-${this.caps.subgroupMaxSize} != 32` : "no subgroups"}); they must be reached by all threads of the block`);
    }
  }

  private warp(s: Stmt & { s: "warp" }, ind: string): void {
    this.chooseWarpMode();
    const r = s.result.name;
    const v = s.args.length ? this.expr(s.args[0]) : "";
    const t = s.result.t;
    const arg1 = s.args[1] ? this.expr(s.args[1]) : "0";
    const W = s.width;
    if (this.warpMode === "subgroups") {
      const lane = "sglane_";
      switch (s.fn) {
        case "shfl": this.emit(`${ind}${r} = ${this.sgShuffle(`subgroupShuffle`, v, `((${lane} & ~${W - 1}u) | (u32(${arg1}) & ${W - 1}u))`, t)};`); return;
        case "shfl_xor": this.emit(`${ind}${r} = ${this.sgShuffle(`subgroupShuffle`, v, `select(${lane}, ${lane} ^ u32(${arg1}), ((${lane} ^ u32(${arg1})) & ~${W - 1}u) == (${lane} & ~${W - 1}u))`, t)};`); return;
        case "shfl_up": this.emit(`${ind}${r} = select(${this.sgShuffle("subgroupShuffle", v, `${lane} - u32(${arg1})`, t)}, ${v}, (${lane} & ${W - 1}u) < u32(${arg1}));`); return;
        case "shfl_down": this.emit(`${ind}${r} = select(${this.sgShuffle("subgroupShuffle", v, `${lane} + u32(${arg1})`, t)}, ${v}, (${lane} & ${W - 1}u) + u32(${arg1}) >= ${W}u);`); return;
        case "ballot": this.emit(`${ind}${r} = subgroupBallot(${v}).x;`); return;
        case "any": this.emit(`${ind}${r} = select(0i, 1i, subgroupAny(${v}));`); return;
        case "all": this.emit(`${ind}${r} = select(0i, 1i, subgroupAll(${v}));`); return;
        case "reduce_add": this.emit(`${ind}${r} = subgroupAdd(${v});`); return;
        case "reduce_min": this.emit(`${ind}${r} = subgroupMin(${v});`); return;
        case "reduce_max": this.emit(`${ind}${r} = subgroupMax(${v});`); return;
      }
    }
    // emulation through workgroup memory: write own value, barrier, read the source lane, barrier
    const self = "lidx_";
    const base = `(${self} & ~31u)`;
    const lane = `(${self} & 31u)`;
    const words = (x: string): [string, string | null] => {
      if (is64i(t.s)) return [`${x}.x`, `${x}.y`];
      if (t.n > 1) fail("backend", `warp shuffles of vector values (${vtStr(t)}) are not supported; shuffle each component`);
      if (t.s === "bool") return [`select(0u, 1u, ${x})`, null];
      if (t.s === "u32" || t.s === "u8" || t.s === "u16") return [x, null];
      return [`bitcast<u32>(${x})`, null];
    };
    const fromWords = (lo: string, hi: string) => {
      if (is64i(t.s)) return `vec2<u32>(${lo}, ${hi})`;
      if (t.s === "bool") return `(${lo} != 0u)`;
      if (t.s === "u32" || t.s === "u8" || t.s === "u16") return lo;
      return `bitcast<${regScalar(t.s)}>(${lo})`;
    };
    const bar = `${ind}workgroupBarrier();`;
    if (s.fn === "ballot" || s.fn === "any" || s.fn === "all" || s.fn.startsWith("reduce")) {
      const [lo] = s.fn.startsWith("reduce") ? words(v) : [`select(0u, 1u, ${v})`, null];
      this.emit(`${ind}warp_lo[${self}] = ${lo};`, );
      this.emit(bar);
      const acc = this.tmpName();
      if (s.fn === "ballot" || s.fn === "any" || s.fn === "all") {
        this.emit(`${ind}var ${acc}: u32 = 0u; for (var j = 0u; j < 32u; j++) { ${acc} = ${acc} | (warp_lo[${base} + j] << j); }`);
        if (s.fn === "ballot") this.emit(`${ind}${r} = ${acc};`);
        else if (s.fn === "any") this.emit(`${ind}${r} = select(0i, 1i, ${acc} != 0u);`);
        else {
          const n = this.k.block[0] * this.k.block[1] * this.k.block[2];
          // lanes beyond the block size (a partial last warp) count as true
          this.emit(`${ind}${r} = select(0i, 1i, (${acc} | select(0u, ~((1u << (${n}u - ${base})) - 1u), ${n}u - ${base} < 32u)) == 0xffffffffu);`);
        }
      } else {
        const op = s.fn === "reduce_add" ? "+" : s.fn === "reduce_min" ? "min" : "max";
        const st = regScalar(t.s);
        const elem = (j: string) => (st === "u32" ? `warp_lo[${base} + ${j}]` : `bitcast<${st}>(warp_lo[${base} + ${j}])`);
        const n = this.k.block[0] * this.k.block[1] * this.k.block[2];
        this.emit(`${ind}var ${acc}: ${st} = ${elem("0u")}; for (var j = 1u; j < min(32u, ${n}u - ${base}); j++) { ${acc} = ${op === "+" ? `${acc} + ${elem("j")}` : `${op}(${acc}, ${elem("j")})`}; }`);
        this.emit(`${ind}${r} = ${acc};`);
      }
      this.emit(bar);
      return;
    }
    const [lo, hi] = words(v);
    this.emit(`${ind}warp_lo[${self}] = ${lo};`);
    if (hi) this.emit(`${ind}warp_hi[${self}] = ${hi};`);
    this.emit(bar);
    let src: string, keep: string | null = null;
    const seg = `(${lane} & ~${W - 1}u)`;
    switch (s.fn) {
      case "shfl": src = `${seg} | (u32(${arg1}) & ${W - 1}u)`; break;
      case "shfl_xor": src = `${lane} ^ u32(${arg1})`; keep = `((${lane} ^ u32(${arg1})) & ~${W - 1}u) != ${seg}`; break;
      case "shfl_up": src = `${lane} - u32(${arg1})`; keep = `(${lane} & ${W - 1}u) < u32(${arg1})`; break;
      default: src = `${lane} + u32(${arg1})`; keep = `(${lane} & ${W - 1}u) + u32(${arg1}) >= ${W}u`; break;
    }
    const sl = this.tmpName();
    this.emit(`${ind}let ${sl} = ${base} + ((${src}) & 31u);`);
    const got = fromWords(`warp_lo[${sl}]`, hi ? `warp_hi[${sl}]` : "0u");
    this.emit(`${ind}${r} = ${keep ? `select(${got}, ${v}, ${keep})` : got};`);
    this.emit(bar);
  }

  private sgShuffle(fn: string, v: string, lane: string, t: VT): string {
    if (is64i(t.s)) return `vec2<u32>(${fn}((${v}).x, ${lane}), ${fn}((${v}).y, ${lane}))`;
    if (t.s === "bool") return `(${fn}(select(0u, 1u, ${v}), ${lane}) != 0u)`;
    return `${fn}(${v}, ${lane})`;
  }

  // ------------------------------------------------------------------ expressions
  expr(e: Expr): string {
    switch (e.op) {
      case "const": return this.constant(e.t, e.v);
      case "var": return e.v.name;
      case "param": {
        const slot = this.paramSlots.find((p) => p.param === e.index);
        if (!slot) fail("internal", `parameter ${e.name} has no slot`);
        return this.paramRead(slot, e.t);
      }
      case "builtin": return this.builtin(e.name);
      case "unop": {
        const a = this.expr(e.a);
        if (is64i(e.t.s)) {
          if (e.fn === "~") return `(~${a})`;
          if (e.fn === "-") return this.i64("neg", [a]);
        }
        if (e.fn === "!") return `(!${a})`;
        if (e.fn === "-" && (e.t.s === "u32" || e.t.s === "u8" || e.t.s === "u16")) return `(0u - ${a})`;
        return `(${e.fn}${a})`;
      }
      case "binop": return this.binop(e);
      case "select": return `select(${this.expr(e.b)}, ${this.expr(e.a)}, ${this.expr(e.c)})`;
      case "cast": return this.convert(e.a, e.t);
      case "bitcast": return this.bitcast(e.a, e.t);
      case "call": return this.call(e);
      case "load": return this.load(e.root, e.off, e.t);
      case "vec": {
        if (is64i(e.t.s)) fail("backend", "vectors of 64-bit integers are not supported on WebGPU");
        return `${regType(e.t)}(${e.items.map((x) => this.expr(x)).join(", ")})`;
      }
      case "extract": {
        if (e.a.op === "vec") return this.expr(e.a.items[e.lane]);
        return `(${this.expr(e.a)}).${"xyzw"[e.lane]}`;
      }
    }
  }

  private paramRead(slot: ParamSlot, t: VT): string {
    const w = (i: number) => this.word(slot.word + i);
    const s = t.s;
    if (t.n === 1) {
      switch (s) {
        case "f32": return `bitcast<f32>(${w(0)})`;
        case "i32": return `bitcast<i32>(${w(0)})`;
        case "u32": return w(0);
        case "i64": case "u64": return `vec2<u32>(${w(0)}, ${w(1)})`;
        case "bool": return `(${w(0)} != 0u)`;
        case "i8": return `(bitcast<i32>(${w(0)} << 24u) >> 24u)`;
        case "i16": return `(bitcast<i32>(${w(0)} << 16u) >> 16u)`;
        case "u8": return `(${w(0)} & 0xffu)`;
        case "u16": return `(${w(0)} & 0xffffu)`;
        case "f16": this.halfHelpers(); return `f16_from_bits(${w(0)} & 0xffffu)`;
        case "bf16": return `bitcast<f32>((${w(0)} & 0xffffu) << 16u)`;
        case "f64": return `bitcast<f32>(${w(0)})`; // demoted: the runtime packs a float
      }
    }
    // vectors: one word per component for 4-byte types; packed for 1/2-byte types
    const es = SCALAR_SIZE[s];
    const comps: string[] = [];
    for (let i = 0; i < t.n; i++) {
      const bit = i * es * 8;
      const wi = Math.floor(bit / 32), sh = bit % 32;
      const raw = es === 4 ? w(i) : `((${w(wi)} >> ${sh}u) & ${es === 1 ? "0xffu" : "0xffffu"})`;
      if (s === "f32") comps.push(`bitcast<f32>(${raw})`);
      else if (s === "i32") comps.push(`bitcast<i32>(${raw})`);
      else if (s === "f16") { this.halfHelpers(); comps.push(`f16_from_bits(${raw})`); }
      else if (s === "bf16") comps.push(`bitcast<f32>(${raw} << 16u)`);
      else if (s === "i8") comps.push(`(bitcast<i32>(${raw} << 24u) >> 24u)`);
      else if (s === "i16") comps.push(`(bitcast<i32>(${raw} << 16u) >> 16u)`);
      else comps.push(raw);
    }
    return `${regType(t)}(${comps.join(", ")})`;
  }

  private builtin(n: string): string {
    switch (n) {
      // entry-point builtins, not the var<private> copies: WGSL's uniformity analysis treats private variables as
      // non-uniform, which would reject barriers in loops bounded by blockIdx
      case "tid.x": return "lid_.x"; case "tid.y": return "lid_.y"; case "tid.z": return "lid_.z";
      case "bid.x": return "(wid_.x + P[1].x)"; case "bid.y": return "(wid_.y + P[1].y)"; case "bid.z": return "(wid_.z + P[1].z)";
      case "bdim.x": return `${this.k.block[0]}u`; case "bdim.y": return `${this.k.block[1]}u`; case "bdim.z": return `${this.k.block[2]}u`;
      case "gdim.x": return this.word(W_GRID); case "gdim.y": return this.word(W_GRID + 1); case "gdim.z": return this.word(W_GRID + 2);
      case "tid.linear": return "lidx_";
      case "lane": return "(lidx_ & 31u)";
      case "warp": return "(lidx_ >> 5u)";
    }
    fail("internal", `unknown builtin ${n}`);
  }

  private constant(t: VT, v: number | bigint | boolean): string {
    if (t.n > 1) fail("internal", "vector constant");
    const s = t.s;
    if (s === "bool") return v ? "true" : "false";
    if (is64i(s)) {
      const b = BigInt.asUintN(64, BigInt(v as number | bigint));
      return `vec2<u32>(${Number(b & 0xffffffffn)}u, ${Number(b >> 32n)}u)`;
    }
    if (isFloat(s)) {
      const n = Number(v);
      // WGSL rejects constant expressions that evaluate to inf/NaN: build them from a parameter word (always 0x7f800000)
      if (Number.isNaN(n)) return "bitcast<f32>(P[0].w | 0x400000u)";
      if (!Number.isFinite(n)) return n > 0 ? "bitcast<f32>(P[0].w)" : "bitcast<f32>(P[0].w | 0x80000000u)";
      const f = Math.fround(n);
      if (Object.is(f, -0)) return "-0.0f";
      if (f !== 0 && Math.abs(f) < 1.1754943508222875e-38) { const b = new Uint32Array(new Float32Array([f]).buffer)[0]; return `bitcast<f32>(${b}u)`; }
      let str = String(f);
      if (!/[.e]/.test(str)) str += ".0";
      return `${str}f`;
    }
    const n = Number(v);
    if (s === "i32" || s === "i8" || s === "i16") return n === -2147483648 ? "i32(-2147483647 - 1)" : n < 0 ? `(${n}i)` : `${n}i`;
    return `${n >>> 0}u`;
  }

  private binop(e: Expr & { op: "binop" }): string {
    const a = this.expr(e.a), b = this.expr(e.b);
    const s = e.a.t.s;
    if (is64i(s)) {
      const signed = s === "i64";
      switch (e.fn) {
        case "+": return this.i64("add", [a, b]);
        case "-": return this.i64("sub", [a, b]);
        case "*": return this.i64("mul", [a, b]);
        case "/": return this.i64(signed ? "sdiv" : "udiv", [a, b]);
        case "%": return this.i64(signed ? "srem" : "urem", [a, b]);
        case "&": case "|": case "^": return `(${a} ${e.fn} ${b})`;
        case "<<": return this.i64("shl", [a, `${b}.x`]);
        case ">>": return this.i64(signed ? "sar" : "shr", [a, `${b}.x`]);
        case "==": return `all(${a} == ${b})`;
        case "!=": return `any(${a} != ${b})`;
        case "<": case "<=": case ">": case ">=": {
          const lt = (x: string, y: string) => this.i64(signed ? "slt" : "ult", [x, y]);
          if (e.fn === "<") return lt(a, b);
          if (e.fn === ">") return lt(b, a);
          if (e.fn === "<=") return `(!${lt(b, a)})`;
          return `(!${lt(a, b)})`;
        }
        default: fail("internal", `64-bit ${e.fn}`);
      }
    }
    if (e.fn === "<<" || e.fn === ">>") {
      const bu = e.b.t.n === 1 ? `u32(${b})` : `vec${e.b.t.n}<u32>(${b})`;
      // C: shifting by >= width is undefined; WGSL masks the amount: same result as the hardware does
      return `(${a} ${e.fn} ${bu})`;
    }
    if (e.fn === "%" && isFloat(s)) fail("internal", "float %");
    return `(${a} ${e.fn} ${b})`;
  }

  private convert(a: Expr, to: VT): string {
    const from = a.t;
    const x = this.expr(a);
    if (from.n !== to.n) {
      if (from.n === 1) return `${regType(to)}(${this.convert(a, vt(to.s))})`;
      fail("internal", `vector size cast ${vtStr(from)} -> ${vtStr(to)}`);
    }
    if (from.n > 1) {
      // component-wise
      const parts: string[] = [];
      for (let i = 0; i < from.n; i++) parts.push(this.convert({ op: "extract", t: vt(from.s), a, lane: i }, vt(to.s)));
      return `${regType(to)}(${parts.join(", ")})`;
    }
    const fs = from.s, ts = to.s;
    if (fs === ts) return x;
    // 64-bit integers
    if (is64i(ts)) {
      if (is64i(fs)) return x;
      if (fs === "bool") return `vec2<u32>(select(0u, 1u, ${x}), 0u)`;
      if (fs === "i32" || fs === "i8" || fs === "i16") return this.i64("from_i32", [x]);
      if (fs === "u32" || fs === "u8" || fs === "u16") return `vec2<u32>(${x}, 0u)`;
      if (isFloat(fs)) return this.i64(ts === "i64" ? "from_f32" : "from_f32u", [x]);
    }
    if (is64i(fs)) {
      if (ts === "bool") return `any(${x} != vec2<u32>(0u))`;
      if (ts === "u32") return `${x}.x`;
      if (ts === "i32") return `bitcast<i32>(${x}.x)`;
      if (ts === "u8") return `(${x}.x & 0xffu)`; if (ts === "u16") return `(${x}.x & 0xffffu)`;
      if (ts === "i8") return `(bitcast<i32>(${x}.x << 24u) >> 24u)`; if (ts === "i16") return `(bitcast<i32>(${x}.x << 16u) >> 16u)`;
      if (isFloat(ts)) { const f = this.i64(fs === "i64" ? "to_f32" : "to_f32u", [x]); return ts === "f16" ? (this.halfHelpers(), `f16_round(${f})`) : ts === "bf16" ? (this.bf16Helpers(), `bf16_round(${f})`) : f; }
    }
    if (ts === "bool") return isFloat(fs) ? `(${x} != 0.0f)` : `(${x} != ${fs === "u32" || fs === "u8" || fs === "u16" ? "0u" : "0i"})`;
    const rf = regScalar(fs), rt = regScalar(ts);
    let base = x;
    if (rf !== rt) base = fs === "bool" ? `select(${rt}(0), ${rt}(1), ${x})` : `${rt}(${x})`;
    // float -> int: C truncates; out of range is undefined, WGSL saturates
    switch (ts) {
      case "i8": return `(bitcast<i32>(bitcast<u32>(${base}) << 24u) >> 24u)`;
      case "i16": return `(bitcast<i32>(bitcast<u32>(${base}) << 16u) >> 16u)`;
      case "u8": return `(${base} & 0xffu)`;
      case "u16": return `(${base} & 0xffffu)`;
      case "f16": this.halfHelpers(); return `f16_round(${base})`;
      case "bf16": this.bf16Helpers(); return `bf16_round(${base})`;
    }
    if (isFloat(fs) && (ts === "i32" || ts === "u32") && (fs === "f16" || fs === "bf16")) return `${rt}(${x})`;
    return base;
  }

  private bitcast(a: Expr, to: VT): string {
    const x = this.expr(a);
    const fs = a.t.s, ts = to.s;
    if (fs === ts) return x;
    if (SCALAR_SIZE[fs] === 4 && SCALAR_SIZE[ts] === 4) return `bitcast<${regScalar(ts)}>(${x})`;
    if (SCALAR_SIZE[fs] === 2 && SCALAR_SIZE[ts] === 2) {
      // f16 <-> u16/i16 bits
      if (fs === "f16") { this.halfHelpers(); const b = `f16_to_bits(${x})`; return ts === "i16" ? `(bitcast<i32>(${b} << 16u) >> 16u)` : b; }
      if (fs === "bf16") { this.bf16Helpers(); const b = `bf16_to_bits(${x})`; return ts === "i16" ? `(bitcast<i32>(${b} << 16u) >> 16u)` : b; }
      const bits = `(bitcast<u32>(${x}) & 0xffffu)`;
      if (ts === "f16") { this.halfHelpers(); return `f16_from_bits(${bits})`; }
      if (ts === "bf16") return `bitcast<f32>(${bits} << 16u)`;
      return ts === "i16" ? `(bitcast<i32>(${bits} << 16u) >> 16u)` : bits;
    }
    if (SCALAR_SIZE[fs] === 8 && SCALAR_SIZE[ts] === 8) {
      if (fs === "f64" || ts === "f64") fail("backend", "double reinterpretation (__double_as_longlong) is not supported on WebGPU");
      return x;
    }
    fail("internal", `bitcast ${fs} -> ${ts}`);
  }

  private call(e: Expr & { op: "call" }): string {
    const args = e.args.map((x) => this.expr(x));
    const s = e.args[0]?.t.s ?? e.t.s;
    if (is64i(s) || is64i(e.t.s)) {
      switch (e.fn) {
        case "min": case "max": { const lt = this.i64(s === "i64" ? "slt" : "ult", [args[0], args[1]]); return e.fn === "min" ? `select(${args[1]}, ${args[0]}, ${lt})` : `select(${args[0]}, ${args[1]}, ${lt})`; }
        case "abs": return `select(${args[0]}, ${this.i64("neg", [args[0]])}, (${args[0]}.y & 0x80000000u) != 0u)`;
        case "popc": return `(countOneBits(${args[0]}.x) + countOneBits(${args[0]}.y))`;
        case "clz": return `select(countLeadingZeros(bitcast<u32>(${args[0]}.y)), 32u + countLeadingZeros(${args[0]}.x), ${args[0]}.y == 0u)`;
        case "ffs": return `select(select(33u + countTrailingZeros(${args[0]}.y), 0u, ${args[0]}.y == 0u), countTrailingZeros(${args[0]}.x) + 1u, ${args[0]}.x != 0u)`;
        case "brev": return `vec2<u32>(reverseBits(${args[0]}.y), reverseBits(${args[0]}.x))`;
        case "mulhi": return this.i64(s === "i64" ? "smulhi" : "umulhi", args);
        default: fail("backend", `${e.fn} on 64-bit integers is not supported on WebGPU`);
      }
    }
    const f = (n: string) => `${n}(${args.join(", ")})`;
    const isF = isFloat(s);
    switch (e.fn) {
      case "sqrt": case "exp": case "exp2": case "log": case "log2": case "sin": case "cos": case "tan": case "asin": case "acos": case "atan":
      case "sinh": case "cosh": case "asinh": case "acosh": case "atanh": case "floor": case "ceil": case "trunc": case "ldexp": case "fma":
        return f(e.fn);
      case "atan2": return f("atan2");
      case "tanh": return this.helper("cuda_tanh", `fn cuda_tanh(x: f32) -> f32 {
  // tanh via exp(-2|x|): finite for large |x| (sinh/cosh would overflow), series near 0 for accuracy
  let a = abs(x);
  if (a < 0.0625f) { let x2 = x * x; return x * (1.0f + x2 * (-0.33333334f + x2 * (0.13333334f - x2 * 0.053968254f))); }
  let e = exp(-2.0f * a);
  return sign(x) * (1.0f - e) / (1.0f + e);
}`, args);
      case "rsqrt": return f("inverseSqrt");
      case "fabs": return f("abs");
      case "rint": return f("round"); // WGSL round() is round-half-to-even, like rint
      case "round": return this.helper("cuda_round", `fn cuda_round(x: f32) -> f32 { let t = trunc(x); return select(t, t + sign(x), abs(x - t) >= 0.5f); }`, args);
      case "exp10": return `exp2(${args[0]} * 3.321928094887362f)`;
      case "log10": return `(log2(${args[0]}) * 0.30102999566398120f)`;
      case "expm1": return this.helper("cuda_expm1", `fn cuda_expm1(x: f32) -> f32 {
  if (abs(x) < 1e-5f) { return x + 0.5f * x * x; }
  let u = exp(x);
  if (u == 1.0f) { return x; }
  let um1 = u - 1.0f;
  if (um1 == -1.0f) { return -1.0f; }
  return um1 * x / log(u);
}`, args);
      case "log1p": return this.helper("cuda_log1p", `fn cuda_log1p(x: f32) -> f32 { let u = 1.0f + x; if (u == 1.0f) { return x; } return log(u) * x / (u - 1.0f); }`, args);
      case "cbrt": return this.helper("cuda_cbrt", `fn cuda_cbrt(x: f32) -> f32 {
  if (x == 0.0f || cuda_isinf(x) || cuda_isnan(x)) { return x; }
  var y = sign(x) * exp2(log2(abs(x)) / 3.0f);
  y = y - (y * y * y - x) / (3.0f * y * y);
  return y;
}`, args, ["isnan", "isinf"]);
      case "hypot": return this.helper("cuda_hypot", `fn cuda_hypot(x: f32, y: f32) -> f32 { let a = abs(x); let b = abs(y); let m = max(a, b); if (m == 0.0f) { return 0.0f; } let n = min(a, b) / m; return m * sqrt(1.0f + n * n); }`, args);
      case "pow": return this.helper("cuda_pow", `fn cuda_pow(x: f32, y: f32) -> f32 {
  // C semantics: negative bases with integer exponents, x^0 == 1, 1^y == 1
  if (y == 0.0f || x == 1.0f) { return 1.0f; }
  if (x == 0.0f) { return select(0.0f, bitcast<f32>(P[0].w), y < 0.0f); }
  let r = exp2(y * log2(abs(x)));
  if (x < 0.0f) {
    if (trunc(y) != y) { return bitcast<f32>(P[0].w | 0x400000u); }
    let odd = abs(y) < 16777216.0f && (i32(abs(y)) & 1) == 1;
    return select(r, -r, odd);
  }
  return r;
}`, args);
      case "fmin": case "fmax": {
        const nm = e.fn === "fmin" ? "cuda_fmin" : "cuda_fmax";
        return this.helper(nm, `fn ${nm}(a: f32, b: f32) -> f32 { if (cuda_isnan(a)) { return b; } if (cuda_isnan(b)) { return a; } return ${e.fn === "fmin" ? "min" : "max"}(a, b); }`, args, ["isnan"]);
      }
      case "isnan": return this.floatClass("isnan", args[0]);
      case "isinf": return this.floatClass("isinf", args[0]);
      case "isfinite": return `((bitcast<u32>(f32(${args[0]})) & 0x7f800000u) != 0x7f800000u)`;
      case "signbit": return `((bitcast<u32>(f32(${args[0]})) & 0x80000000u) != 0u)`;
      case "copysign": return `bitcast<f32>((bitcast<u32>(${args[0]}) & 0x7fffffffu) | (bitcast<u32>(${args[1]}) & 0x80000000u))`;
      case "fdim": return `select(0.0f, ${args[0]} - ${args[1]}, ${args[0]} > ${args[1]})`;
      case "saturate": return this.helper("cuda_saturate", `fn cuda_saturate(x: f32) -> f32 { if (cuda_isnan(x)) { return 0.0f; } return clamp(x, 0.0f, 1.0f); }`, args, ["isnan"]);
      case "fmod": return this.fmodHelper(args, false);
      case "remainder": return this.fmodHelper(args, true);
      case "erf": case "erfc": case "normcdf": {
        this.helper("cuda_erf", `fn cuda_erf(x: f32) -> f32 {
  // max abs error ~1e-7 (W. J. Cody-style rational approximations, two ranges)
  let a = abs(x);
  if (a < 0.84375f) {
    let z = x * x;
    let r = 1.28379167e-1f + z * (-3.25042108e-1f + z * (-2.84817495e-2f + z * (-5.77027029e-3f + z * -2.37630166e-5f)));
    let s = 1.0f + z * (3.97917223e-1f + z * (6.50222499e-2f + z * (5.08130628e-3f + z * (1.32494738e-4f + z * -3.96022827e-6f))));
    return x + x * (r / s);
  }
  if (a < 1.25f) {
    let s = a - 1.0f;
    let p = -2.36211857e-3f + s * (4.14856106e-1f + s * (-3.72207481e-1f + s * (3.18346620e-1f + s * (-1.10894694e-1f + s * (3.54783043e-2f + s * -2.16637559e-3f)))));
    let q = 1.0f + s * (1.06420877e-1f + s * (5.40397918e-1f + s * (7.18286547e-2f + s * (1.26171219e-1f + s * (1.36370839e-2f + s * 1.19844998e-2f)))));
    return sign(x) * (8.45062911e-1f + p / q);
  }
  if (a >= 6.0f) { return sign(x); }
  return sign(x) * (1.0f - cuda_erfc_pos(a));
}
fn cuda_erfc_pos(a: f32) -> f32 {
  // erfc for a >= 1.25: exp(-a^2) * R(1/a^2) / a
  let s = 1.0f / (a * a);
  var r: f32; var q: f32;
  if (a < 2.857143f) {
    r = -9.86494403e-3f + s * (-6.93858572e-1f + s * (-1.05586262e1f + s * (-6.23753324e1f + s * (-1.62396669e2f + s * (-1.84605088e2f + s * (-8.12874355e1f + s * -9.81432934e0f))))));
    q = 1.0f + s * (1.96512716e1f + s * (1.37657754e2f + s * (4.34565877e2f + s * (6.45387271e2f + s * (4.29008140e2f + s * (1.08635005e2f + s * (6.57024977e0f + s * -6.04244152e-2f)))))));
  } else {
    r = -9.86494292e-3f + s * (-7.99283237e-1f + s * (-1.77579549e1f + s * (-1.60636384e2f + s * (-6.37566443e2f + s * (-1.02509513e3f + s * -4.83519191e2f)))));
    q = 1.0f + s * (3.03380607e1f + s * (3.25792512e2f + s * (1.53672958e3f + s * (3.19985821e3f + s * (2.55305040e3f + s * (4.74528541e2f + s * -2.24409524e1f))))));
  }
  let z = bitcast<f32>(bitcast<u32>(a) & 0xfffff000u);
  return exp(-z * z - 0.5625f) * exp((z - a) * (z + a) + r / q) / a;
}`, []);
        if (e.fn === "erf") return `cuda_erf(${args[0]})`;
        if (e.fn === "erfc") return this.helper("cuda_erfc", `fn cuda_erfc(x: f32) -> f32 { if (x >= 1.25f) { if (x >= 10.0f) { return 0.0f; } return cuda_erfc_pos(x); } return 1.0f - cuda_erf(x); }`, args);
        return this.helper("cuda_normcdf", `fn cuda_normcdf(x: f32) -> f32 { let t = -x * 0.70710678118654752f; if (t >= 1.25f) { if (t >= 10.0f) { return 0.0f; } return 0.5f * cuda_erfc_pos(t); } return 0.5f * (1.0f - cuda_erf(t)); }`, args);
      }
      case "lgamma": case "tgamma": {
        this.helper("cuda_lgamma", `fn cuda_lgamma(x0: f32) -> f32 {
  // Lanczos approximation (g=7, n=9) with reflection for x < 0.5
  var x = x0;
  var refl = 0.0f;
  if (x < 0.5f) {
    let s = sin(3.14159265358979f * x);
    if (s == 0.0f) { return bitcast<f32>(P[0].w); }
    refl = log(3.14159265358979f / abs(s));
    x = 1.0f - x;
  }
  x = x - 1.0f;
  var a = 0.99999999999980993f;
  let c = array<f32, 8>(676.5203681218851f, -1259.1392167224028f, 771.32342877765313f, -176.61502916214059f, 12.507343278686905f, -0.13857109526572012f, 9.9843695780195716e-6f, 1.5056327351493116e-7f);
  for (var i = 0; i < 8; i++) { a = a + c[i] / (x + f32(i + 1)); }
  let t = x + 7.5f;
  let lg = 0.91893853320467274f + (x + 0.5f) * log(t) - t + log(a);
  return select(lg, refl - lg, x0 < 0.5f);
}`, []);
        if (e.fn === "lgamma") return `cuda_lgamma(${args[0]})`;
        return this.helper("cuda_tgamma", `fn cuda_tgamma(x: f32) -> f32 {
  if (x == trunc(x) && x <= 0.0f) { return bitcast<f32>(P[0].w | 0x400000u); }
  let r = exp(cuda_lgamma(x));
  if (x > 0.0f) { return r; }
  // sign of gamma for negative non-integers: (-1)^ceil(-x)
  let k = i32(floor(x));
  return select(r, -r, (k & 1) != 0);
}`, args);
      }
      case "min": case "max": return isF ? f(e.fn) : f(e.fn);
      case "abs": return f("abs");
      case "popc": return `countOneBits(${args[0]})`;
      case "clz": return `countLeadingZeros(${args[0]})`;
      case "ffs": return `select(0i, i32(countTrailingZeros(${args[0]})) + 1i, ${args[0]} != 0i)`;
      case "brev": return `reverseBits(${args[0]})`;
      case "mulhi": {
        if (s === "u32") return this.helper("cuda_umulhi", `fn cuda_umulhi(a: u32, b: u32) -> u32 {
  let al = a & 0xffffu; let ah = a >> 16u; let bl = b & 0xffffu; let bh = b >> 16u;
  let ll = al * bl; let lh = al * bh; let hl = ah * bl; let hh = ah * bh;
  let mid = (ll >> 16u) + (lh & 0xffffu) + (hl & 0xffffu);
  return hh + (lh >> 16u) + (hl >> 16u) + (mid >> 16u);
}`, args);
        return this.helper("cuda_mulhi", `fn cuda_mulhi(a: i32, b: i32) -> i32 {
  var h = cuda_umulhi(bitcast<u32>(a), bitcast<u32>(b));
  if (a < 0i) { h = h - bitcast<u32>(b); }
  if (b < 0i) { h = h - bitcast<u32>(a); }
  return bitcast<i32>(h);
}`, args, ["umulhi"]);
      }
    }
    fail("backend", `math function ${e.fn} is not available on WebGPU`);
  }

  private floatClass(kind: "isnan" | "isinf", x: string): string {
    this.helper("cuda_isnan", `fn cuda_isnan(x: f32) -> bool { return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u; }`, []);
    this.helper("cuda_isinf", `fn cuda_isinf(x: f32) -> bool { return (bitcast<u32>(x) & 0x7fffffffu) == 0x7f800000u; }`, []);
    return `cuda_${kind}(f32(${x}))`;
  }

  private helper(name: string, code: string, args: string[], deps: string[] = []): string {
    for (const d of deps) {
      if (d === "isnan" || d === "isinf") this.floatClass("isnan", "0.0f");
      if (d === "umulhi") this.call({ op: "call", fn: "mulhi", t: vt("u32"), args: [{ op: "const", t: vt("u32"), v: 0 }, { op: "const", t: vt("u32"), v: 0 }] });
    }
    if (!this.helpers.has(name)) this.helpers.set(name, code);
    return `${name}(${args.join(", ")})`;
  }

  private fmodHelper(args: string[], ieee: boolean): string {
    this.floatClass("isnan", "0.0f");
    this.helper("cuda_fmodf", `fn cuda_fmodf(x: f32, y: f32) -> f32 {
  // exact fmod (as in C): long division on the significands
  var ux = bitcast<u32>(x); var uy = bitcast<u32>(y);
  let sx = ux & 0x80000000u;
  ux = ux & 0x7fffffffu; uy = uy & 0x7fffffffu;
  if (uy == 0u || ux >= 0x7f800000u || uy > 0x7f800000u) { return bitcast<f32>(P[0].w | 0x400000u); }
  if (ux < uy) { return x; }
  var ex = i32(ux >> 23u); var ey = i32(uy >> 23u);
  var mx: u32; var my: u32;
  if (ex == 0) { mx = ux << 1u; var t = mx; loop { if ((t & 0x00800000u) != 0u) { break; } t = t << 1u; ex = ex - 1; } mx = t; } else { mx = (ux & 0x007fffffu) | 0x00800000u; }
  if (ey == 0) { my = uy << 1u; var t = my; loop { if ((t & 0x00800000u) != 0u) { break; } t = t << 1u; ey = ey - 1; } my = t; } else { my = (uy & 0x007fffffu) | 0x00800000u; }
  loop {
    if (ex <= ey) { break; }
    if (mx >= my) { mx = mx - my; }
    if (mx == 0u) { return bitcast<f32>(sx); }
    mx = mx << 1u;
    ex = ex - 1;
  }
  if (mx >= my) { mx = mx - my; }
  if (mx == 0u) { return bitcast<f32>(sx); }
  loop { if ((mx & 0x00800000u) != 0u) { break; } mx = mx << 1u; ex = ex - 1; }
  var r: u32;
  if (ex > 0) { r = (mx - 0x00800000u) | (u32(ex) << 23u); } else { r = mx >> u32(1 - ex); }
  return bitcast<f32>(r | sx);
}`, []);
    if (!ieee) return `cuda_fmodf(${args.join(", ")})`;
    return this.helper("cuda_remainderf", `fn cuda_remainderf(x: f32, y: f32) -> f32 {
  // IEEE remainder x - n*y, n = x/y rounded half to even; t - ay and d - ay are exact (Sterbenz)
  let ay = abs(y);
  if (cuda_isnan(x) || cuda_isnan(y) || ay == 0.0f || cuda_isinf(x)) { return bitcast<f32>(P[0].w | 0x400000u); }
  let t = abs(cuda_fmodf(x, 2.0f * ay));
  var res = t;
  if (t > 0.5f * ay) {
    let d = t - ay;
    res = select(d, d - ay, d >= 0.5f * ay);
  }
  return select(res, -res, x < 0.0f || (x == 0.0f && (bitcast<u32>(x) & 0x80000000u) != 0u));
}`, args);
  }

  // ------------------------------------------------------------------ half / bfloat16 helpers (exact RNE in integers)
  private halfHelpers(): void {
    if (this.helpers.has("f16_to_bits")) return;
    this.helpers.set("f16_to_bits", `fn f16_to_bits(f: f32) -> u32 {
  // IEEE binary32 -> binary16, round to nearest even (WGSL leaves f32->f16 rounding implementation-defined)
  let x = bitcast<u32>(f);
  let sign = (x >> 16u) & 0x8000u;
  let ax = x & 0x7fffffffu;
  if (ax > 0x7f800000u) { return sign | 0x7e00u; }
  if (ax >= 0x477ff000u) { return sign | 0x7c00u; }
  if (ax < 0x38800000u) {
    if (ax < 0x33000000u) { return sign; }
    let e = ax >> 23u;
    let m = (ax & 0x7fffffu) | 0x800000u;
    let shift = 126u - e;
    let h = m >> shift;
    let rem = m & ((1u << shift) - 1u);
    let half = 1u << (shift - 1u);
    let up = select(0u, 1u, rem > half || (rem == half && (h & 1u) == 1u));
    return sign | (h + up);
  }
  let h = ((ax >> 13u) - 0x1c000u);
  let rem = ax & 0x1fffu;
  let up = select(0u, 1u, rem > 0x1000u || (rem == 0x1000u && (h & 1u) == 1u));
  return sign | (h + up);
}
fn f16_from_bits(h: u32) -> f32 { return unpack2x16float(h & 0xffffu).x; }
fn f16_round(f: f32) -> f32 { return f16_from_bits(f16_to_bits(f)); }`);
  }
  private bf16Helpers(): void {
    if (this.helpers.has("bf16_to_bits")) return;
    this.helpers.set("bf16_to_bits", `fn bf16_to_bits(f: f32) -> u32 {
  let x = bitcast<u32>(f);
  if ((x & 0x7fffffffu) > 0x7f800000u) { return (x >> 16u) | 0x40u; }
  return (x + 0x7fffu + ((x >> 16u) & 1u)) >> 16u;
}
fn bf16_round(f: f32) -> f32 { return bitcast<f32>(bf16_to_bits(f) << 16u); }`);
  }

  // ------------------------------------------------------------------ 64-bit integer emulation
  private i64(op: string, args: string[]): string {
    const H: Record<string, string> = {
      add: `fn i64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> { let lo = a.x + b.x; return vec2<u32>(lo, a.y + b.y + select(0u, 1u, lo < a.x)); }`,
      sub: `fn i64_sub(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> { return vec2<u32>(a.x - b.x, a.y - b.y - select(0u, 1u, a.x < b.x)); }`,
      neg: `fn i64_neg(a: vec2<u32>) -> vec2<u32> { return i64_sub(vec2<u32>(0u, 0u), a); }`,
      mul: `fn i64_mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let al = a.x & 0xffffu; let ah = a.x >> 16u; let bl = b.x & 0xffffu; let bh = b.x >> 16u;
  let ll = al * bl; let lh = al * bh; let hl = ah * bl; let hh = ah * bh;
  let mid = (ll >> 16u) + (lh & 0xffffu) + (hl & 0xffffu);
  let lo = (ll & 0xffffu) | (mid << 16u);
  let hi = hh + (lh >> 16u) + (hl >> 16u) + (mid >> 16u) + a.x * b.y + a.y * b.x;
  return vec2<u32>(lo, hi);
}`,
      shl: `fn i64_shl(a: vec2<u32>, n0: u32) -> vec2<u32> { let n = n0 & 63u; if (n == 0u) { return a; } if (n >= 32u) { return vec2<u32>(0u, a.x << (n - 32u)); } return vec2<u32>(a.x << n, (a.y << n) | (a.x >> (32u - n))); }`,
      shr: `fn i64_shr(a: vec2<u32>, n0: u32) -> vec2<u32> { let n = n0 & 63u; if (n == 0u) { return a; } if (n >= 32u) { return vec2<u32>(a.y >> (n - 32u), 0u); } return vec2<u32>((a.x >> n) | (a.y << (32u - n)), a.y >> n); }`,
      sar: `fn i64_sar(a: vec2<u32>, n0: u32) -> vec2<u32> { let n = n0 & 63u; let s = bitcast<i32>(a.y); if (n == 0u) { return a; } if (n >= 32u) { return vec2<u32>(bitcast<u32>(s >> (n - 32u)), bitcast<u32>(s >> 31u)); } return vec2<u32>((a.x >> n) | (a.y << (32u - n)), bitcast<u32>(s >> n)); }`,
      ult: `fn i64_ult(a: vec2<u32>, b: vec2<u32>) -> bool { return a.y < b.y || (a.y == b.y && a.x < b.x); }`,
      slt: `fn i64_slt(a: vec2<u32>, b: vec2<u32>) -> bool { let ah = bitcast<i32>(a.y); let bh = bitcast<i32>(b.y); return ah < bh || (ah == bh && a.x < b.x); }`,
      from_i32: `fn i64_from_i32(x: i32) -> vec2<u32> { return vec2<u32>(bitcast<u32>(x), select(0u, 0xffffffffu, x < 0i)); }`,
      from_f32: `fn i64_from_f32(f: f32) -> vec2<u32> { let a = trunc(abs(f)); let hi = floor(a / 4294967296.0f); let lo = a - hi * 4294967296.0f; let r = vec2<u32>(u32(lo), u32(hi)); return select(r, i64_neg(r), f < 0.0f); }`,
      from_f32u: `fn i64_from_f32u(f: f32) -> vec2<u32> { let a = trunc(max(f, 0.0f)); let hi = floor(a / 4294967296.0f); let lo = a - hi * 4294967296.0f; return vec2<u32>(u32(lo), u32(hi)); }`,
      to_f32u: `fn i64_to_f32u(a: vec2<u32>) -> f32 {
  // round once: the top 32 significant bits with a sticky bit for the rest, then one u32 -> f32 conversion
  if (a.y == 0u) { return f32(a.x); }
  let lz = countLeadingZeros(a.y);
  let top = i64_shl(a, lz);
  let m = top.y | select(0u, 1u, top.x != 0u);
  return f32(m) * exp2(f32(32u - lz));
}`,
      to_f32: `fn i64_to_f32(a: vec2<u32>) -> f32 { if ((a.y & 0x80000000u) != 0u) { return -i64_to_f32u(i64_neg(a)); } return i64_to_f32u(a); }`,
      udivrem: `fn i64_udivrem(n: vec2<u32>, d: vec2<u32>) -> array<vec2<u32>, 2> {
  var q = vec2<u32>(0u, 0u); var r = vec2<u32>(0u, 0u);
  if (all(d == vec2<u32>(0u, 0u))) { return array<vec2<u32>, 2>(vec2<u32>(0xffffffffu, 0xffffffffu), n); }
  for (var i = 63i; i >= 0i; i--) {
    r = i64_shl(r, 1u);
    let bit = select((n.x >> u32(i)) & 1u, (n.y >> u32(i - 32i)) & 1u, i >= 32i);
    r.x = r.x | bit;
    if (!i64_ult(r, d)) { r = i64_sub(r, d); if (i >= 32i) { q.y = q.y | (1u << u32(i - 32i)); } else { q.x = q.x | (1u << u32(i)); } }
  }
  return array<vec2<u32>, 2>(q, r);
}`,
      udiv: `fn i64_udiv(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> { return i64_udivrem(a, b)[0]; }`,
      urem: `fn i64_urem(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> { return i64_udivrem(a, b)[1]; }`,
      sdiv: `fn i64_sdiv(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> { let na = (a.y & 0x80000000u) != 0u; let nb = (b.y & 0x80000000u) != 0u; let q = i64_udiv(select(a, i64_neg(a), na), select(b, i64_neg(b), nb)); return select(q, i64_neg(q), na != nb); }`,
      srem: `fn i64_srem(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> { let na = (a.y & 0x80000000u) != 0u; let nb = (b.y & 0x80000000u) != 0u; let r = i64_urem(select(a, i64_neg(a), na), select(b, i64_neg(b), nb)); return select(r, i64_neg(r), na); }`,
      umulhi: `fn i64_umulhi(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  // high 64 bits of a 64x64 product, schoolbook on 32-bit halves
  let p0 = i64_mul(vec2<u32>(a.x, 0u), vec2<u32>(b.x, 0u));
  let p1 = i64_mul(vec2<u32>(a.x, 0u), vec2<u32>(b.y, 0u));
  let p2 = i64_mul(vec2<u32>(a.y, 0u), vec2<u32>(b.x, 0u));
  let p3 = i64_mul(vec2<u32>(a.y, 0u), vec2<u32>(b.y, 0u));
  let mid = i64_add(i64_add(vec2<u32>(p0.y, 0u), vec2<u32>(p1.x, 0u)), vec2<u32>(p2.x, 0u));
  return i64_add(i64_add(i64_add(p3, vec2<u32>(p1.y, 0u)), vec2<u32>(p2.y, 0u)), vec2<u32>(mid.y, 0u));
}`,
      smulhi: `fn i64_smulhi(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  var h = i64_umulhi(a, b);
  if ((a.y & 0x80000000u) != 0u) { h = i64_sub(h, b); }
  if ((b.y & 0x80000000u) != 0u) { h = i64_sub(h, a); }
  return h;
}`,
    };
    const deps: Record<string, string[]> = {
      neg: ["sub"], from_f32: ["neg"], to_f32: ["to_f32u", "neg"], to_f32u: ["shl"], udivrem: ["shl", "ult", "sub"], udiv: ["udivrem"], urem: ["udivrem"],
      sdiv: ["udiv", "neg"], srem: ["urem", "neg"], umulhi: ["mul", "add"], smulhi: ["umulhi", "sub"],
    };
    const need = (o: string) => {
      if (this.helpers.has(`i64_${o}`)) return;
      for (const d of deps[o] ?? []) need(d);
      this.helpers.set(`i64_${o}`, H[o]);
    };
    need(op);
    this.k.features.add("int64-emulated");
    return `i64_${op}(${args.join(", ")})`;
  }
}

// ------------------------------------------------------------------ helpers
function layoutElemType(L: Layout): string {
  if (L.k === "atomic") return `atomic<${L.elem}>`;
  if (L.k === "words") return L.atomic ? "atomic<u32>" : "u32";
  const t = L.t;
  const s = t.s === "f16" ? "f16" : t.s;
  return t.n === 1 ? s : `vec${t.n}<${s}>`;
}
const layoutIsAtomic = (L: Layout) => L.k === "atomic" || (L.k === "words" && L.atomic);
function layoutStr(L: Layout): string { return L.k === "typed" ? `array<${layoutElemType(L)}>` : `array<${layoutElemType(L)}>`; }

function containsStmtLocal(body: Stmt[], pred: (s: Stmt) => boolean): boolean {
  let f = false;
  walkStmts(body, (s) => { if (!f && pred(s)) f = true; });
  return f;
}

function containsTrap(body: Stmt[]): boolean {
  let f = false;
  walkStmts(body, (s) => { if (s.s === "trap") f = true; });
  return f;
}

function isPow2Abs(v: number): boolean { const a = Math.abs(v); return a > 0 && Math.log2(a) === Math.round(Math.log2(a)); }

function mapTopDown(e: Expr, f: (e: Expr) => Expr): Expr {
  const r = f(e);
  if (r !== e) return r;
  switch (e.op) {
    case "unop": case "cast": case "bitcast": case "extract": { const a = mapTopDown(e.a, f); return a === e.a ? e : ({ ...e, a } as Expr); }
    case "binop": { const a = mapTopDown(e.a, f), b = mapTopDown(e.b, f); return a === e.a && b === e.b ? e : { ...e, a, b }; }
    case "select": { const c = mapTopDown(e.c, f), a = mapTopDown(e.a, f), b = mapTopDown(e.b, f); return c === e.c && a === e.a && b === e.b ? e : { ...e, c, a, b }; }
    case "call": { const args = e.args.map((x) => mapTopDown(x, f)); return args.every((x, i) => x === e.args[i]) ? e : { ...e, args }; }
    case "load": { const off = mapTopDown(e.off, f); return off === e.off ? e : { ...e, off }; }
    case "vec": { const items = e.items.map((x) => mapTopDown(x, f)); return items.every((x, i) => x === e.items[i]) ? e : { ...e, items }; }
    default: return e;
  }
}

function mapStmt(s: Stmt, f: (e: Expr) => Expr): Stmt {
  switch (s.s) {
    case "let": return { ...s, e: f(s.e) };
    case "store": return { ...s, off: f(s.off), e: f(s.e) };
    case "atomic": return { ...s, off: f(s.off), args: s.args.map(f) };
    case "if": return { ...s, c: f(s.c) };
    case "loop": return s.breakIf ? { ...s, breakIf: f(s.breakIf) } : s;
    case "warp": return { ...s, args: s.args.map(f) };
    case "printf": return { ...s, args: s.args.map(f) };
    case "assert": return { ...s, c: f(s.c) };
    default: return s;
  }
}

function walkStmtsMut(k: Kernel, f: (e: Expr) => Expr): void {
  const visit = (l: Stmt[]): Stmt[] => l.map((s) => {
    let t = mapStmt(s, f);
    if (t.s === "store" || t.s === "atomic") t = { ...t, t: t.t.s === "f64" ? { s: "f32", n: t.t.n } : t.t } as Stmt;
    if (t.s === "if") t = { ...t, then: visit(t.then), else: visit(t.else) };
    if (t.s === "loop") t = { ...t, body: visit(t.body), cont: visit(t.cont) };
    return t;
  });
  k.body = visit(k.body);
}

function describeExpr(e: Expr): string {
  switch (e.op) {
    case "const": return `the double constant ${e.v}`;
    case "var": return `variable '${e.v.name.replace(/_\d+$/, "")}' of type double`;
    case "call": return `${e.fn}() on a double argument`;
    case "binop": return `double arithmetic '${e.fn}'`;
    case "cast": return "a conversion to double";
    default: return "a double expression";
  }
}

export type { Var, WarpFn, BinOp };
export { exprChildren };
