/**
 * Kernel IR -> WebAssembly for CPU execution.
 *
 * Execution model: one call of the exported `run(args, scratch, worker)` executes whole thread blocks, fetching block
 * indices from an atomic counter shared by all workers (dynamic load balancing across CPU cores). Inside a block the CUDA
 * threads are simulated by loops, never by OS threads:
 *
 *  - Code between barriers ("runs") executes as `for each thread { ... }`. A kernel without __syncthreads is one run.
 *  - __syncthreads() ends a run: every thread finishes the run before any starts the next one, which is exactly the
 *    barrier's guarantee. Warp collectives are a write run + read run.
 *  - Variables live across runs are spilled to per-thread slots in the worker's scratch memory.
 *  - Control flow that contains barriers becomes masked execution (like SIMT reconvergence): `if` conditions are stored
 *    per thread, both branches run with the corresponding thread masks, loops repeat while any thread is still active,
 *    and break/continue/return set per-thread flags. Every well-defined CUDA program keeps its meaning.
 *
 * Memory safety: every global/shared/local access is bounds checked against its buffer; violations call env.fail and stop
 * the launch (the linear memory also holds other allocations, so unchecked accesses could read unrelated buffers).
 */
import { fail as cfail, type DiagnosticSink } from "../diagnostics.js";
import {
  containsStmt, someExpr, stmtExprs, vt, vtSize, walkStmts,
  type Expr, type Kernel, type Root, type Stmt, type Var, type VT, type WarpFn,
} from "../ir.js";
import { isFloat, SCALAR_SIZE, type Scalar } from "../types.js";
import { AT, Func, ModuleBuilder, OP, SAT, T, type ValType } from "./encoder.js";
import { halfFromBits } from "../fold.js";

export const WASM_MATH_IMPORTS = [
  "exp", "exp2", "log", "log2", "log10", "expm1", "log1p", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
  "asinh", "acosh", "atanh", "cbrt", "erf", "erfc", "lgamma", "tgamma", "exp10", "normcdf",
] as const;
export const WASM_MATH2_IMPORTS = ["pow", "atan2", "hypot", "fmod", "remainder"] as const;

export interface WasmProgram {
  bytes: Uint8Array;
  /** byte layout of the per-launch argument block */
  args: {
    size: number;
    grid: number; counter: number; stop: number; chunk: number;
    pointers: { param: number; base: number; limit: number }[];
    scalars: { param: number; off: number; t: VT }[];
    symbols: { symbol: string; base: number; limit: number }[];
  };
  /** bytes of scratch memory one worker needs (shared memory + per-thread slots) */
  scratchBytes: number;
  printf: { fmt: string; types: string[] }[];
  sites: { kind: "oob-load" | "oob-store" | "oob-atomic" | "assert" | "trap" | "divergence"; root?: string; msg?: string }[];
  phases: number;
  notes: string[];
  /** textual summary of the module (functions, phases, layout) for the inspector */
  text: string;
}

type Mask = { kind: "cond"; slot: number; negate: boolean } | { kind: "brk"; loop: number } | { kind: "cont"; loop: number };
type Phase =
  | { k: "run"; id: number; stmts: Stmt[]; masks: Mask[]; inLoop: boolean; loops: number[]; allThreads?: boolean }
  | { k: "loop"; id: number; body: Phase[]; cont: Phase[]; breakIf: Expr | null; masks: Mask[] };

// internal statements for warp collectives
interface WWrite { s: "_wwrite"; fn: WarpFn; value: Expr; slot: number }
interface WRead { s: "_wread"; fn: WarpFn; args: Expr[]; width: number; result: Var; slot: number; t: VT }
interface CondSet { s: "_cond"; slot: number; c: Expr }
interface BrkSet { s: "_brkif"; loop: number; c: Expr }
interface PartClear { s: "_wclear" }
type XStmt = Stmt | WWrite | WRead | CondSet | BrkSet | PartClear;

const wt = (s: Scalar): ValType => (s === "i64" || s === "u64" ? T.i64 : s === "f64" ? T.f64 : s === "f32" || s === "f16" || s === "bf16" ? T.f32 : T.i32);
const isBarrierStmt = (s: Stmt) => s.s === "barrier" || s.s === "warp";

export function generateWasm(k: Kernel, opts: { shared: boolean; maxMemoryPages: number }, sink: DiagnosticSink): WasmProgram {
  return new WasmGen(k, opts, sink).run();
}

class WasmGen {
  private mb: ModuleBuilder;
  private f!: Func;
  private imp = new Map<string, number>();
  private helperFuncs = new Map<string, number>();
  private phases: Phase[] = [];
  private nRuns = 0;
  private nLoops = 0;
  private condSlots = 0;
  private warpSlots = 0;
  private hasBarriers = false;
  private printf: WasmProgram["printf"] = [];
  private sites: WasmProgram["sites"] = [];
  private notes: string[] = [];
  // layout
  private args: WasmProgram["args"];
  private scratch = 0;
  private rootOff = new Map<Root, { off: number; perThread: number }>();
  private slotOff = new Map<Var, number>();
  private condOff = 0; private brkOff = 0; private contOff = 0; private aliveOff = 0; private warpOff = 0; private warpPartOff = 0; private printfOff = 0;
  private threads: number;
  private bx: number; private by: number; private bz: number;
  // codegen state
  private locals = new Map<Var, number[]>();
  private argsLocal = 0; private scratchLocal = 0; private workerLocal = 0;
  private bidx = 0; private bidy = 0; private bidz = 0; private blin = 0;
  private tx = 0; private ty = 0; private tz = 0; private tlin = 0;
  private ptrBase = new Map<number, number>(); private ptrLimit = new Map<number, number>();
  private symBase = new Map<Root, number>(); private symLimit = new Map<Root, number>();
  private gdim = [0, 0, 0];
  private runExit = -1;
  private runLoopStack: { phaseLoop: number | null; label: { brk: number; cont: number } | null }[] = [];
  private currentRun: Phase & { k: "run" } | null = null;
  private tmpI32: number[] = []; private tmpI64: number[] = []; private tmpF32: number[] = []; private tmpF64: number[] = [];

  constructor(private k: Kernel, private opts: { shared: boolean; maxMemoryPages: number }, private sink: DiagnosticSink) {
    [this.bx, this.by, this.bz] = k.block;
    if (this.bx <= 0) cfail("internal", "WASM generation needs a block-size specialized kernel");
    this.threads = this.bx * this.by * this.bz;
    this.mb = new ModuleBuilder({ initial: 1, maximum: opts.maxMemoryPages, shared: opts.shared });
    this.args = { size: 0, grid: 0, counter: 12, stop: 16, chunk: 20, pointers: [], scalars: [], symbols: [] };
  }

  run(): WasmProgram {
    this.buildPhases();
    this.layout();
    this.imports();
    const helpers = this.buildHelpers();
    void helpers;
    const fn = new Func([T.i32, T.i32, T.i32], [], "run");
    this.f = fn;
    this.genRun();
    const idx = this.mb.addFunc(fn);
    this.mb.export("run", idx);
    const bytes = this.mb.encode();
    return {
      bytes, args: this.args, scratchBytes: this.scratch, printf: this.printf, sites: this.sites, phases: this.nRuns, notes: this.notes,
      text: this.describe(bytes.length),
    };
  }

  // ------------------------------------------------------------------ phases
  private buildPhases(): void {
    this.hasBarriers = containsStmt(this.k.body, isBarrierStmt);
    this.phases = this.transform(this.k.body, [], [], false);
    if (this.hasBarriers) this.notes.push(`${this.nRuns} thread phases separated by barriers; ${this.nLoops} loops contain barriers`);
  }

  private newRun(stmts: XStmt[], masks: Mask[], loops: number[], allThreads = false): Phase {
    return { k: "run", id: this.nRuns++, stmts: stmts as Stmt[], masks, inLoop: loops.length > 0, loops, allThreads };
  }

  private transform(list: Stmt[], masks: Mask[], loops: number[], _inLoop: boolean): Phase[] {
    const out: Phase[] = [];
    let buf: XStmt[] = [];
    const flush = () => { if (buf.length) { out.push(this.newRun(buf, masks, loops)); buf = []; } };
    for (const s of list) {
      if (!containsStmt([s], isBarrierStmt)) { buf.push(s); continue; }
      if (s.s === "barrier") { flush(); continue; }
      if (s.s === "warp") {
        const slot = this.warpSlots++;
        if (s.fn === "ballot" || s.fn === "any" || s.fn === "all" || s.fn.startsWith("reduce")) { flush(); out.push(this.newRun([{ s: "_wclear" }], masks, loops, true)); }
        buf.push({ s: "_wwrite", fn: s.fn, value: s.args[0], slot } as WWrite);
        flush();
        buf.push({ s: "_wread", fn: s.fn, args: s.args, width: s.width, result: s.result, slot, t: s.result.t } as WRead);
        continue;
      }
      if (s.s === "if") {
        const slot = this.condSlots++;
        buf.push({ s: "_cond", slot, c: s.c } as CondSet);
        flush();
        out.push(...this.transform(s.then, [...masks, { kind: "cond", slot, negate: false }], loops, true));
        out.push(...this.transform(s.else, [...masks, { kind: "cond", slot, negate: true }], loops, true));
        continue;
      }
      if (s.s === "loop") {
        flush();
        const id = this.nLoops++;
        const inner = [...loops, id];
        const body = this.transform(s.body, [...masks, { kind: "brk", loop: id }, { kind: "cont", loop: id }], inner, true);
        const cont = this.transform(s.cont, [...masks, { kind: "brk", loop: id }], inner, true);
        if (s.breakIf) cont.push(this.newRun([{ s: "_brkif", loop: id, c: s.breakIf } as BrkSet], [...masks, { kind: "brk", loop: id }], inner));
        out.push({ k: "loop", id, body, cont, breakIf: null, masks });
        continue;
      }
      cfail("internal", `unexpected statement with barrier: ${s.s}`);
    }
    flush();
    return out;
  }

  private allRuns(ph: Phase[] = this.phases, acc: (Phase & { k: "run" })[] = []): (Phase & { k: "run" })[] {
    for (const p of ph) { if (p.k === "run") acc.push(p); else { this.allRuns(p.body, acc); this.allRuns(p.cont, acc); } }
    return acc;
  }

  // ------------------------------------------------------------------ memory layout
  private layout(): void {
    const k = this.k;
    // argument block: grid(12) counter(4) stop(4) chunk(4) then pointers (base, limit) and scalars (8-aligned)
    let off = 24;
    const align = (n: number, a: number) => Math.ceil(n / a) * a;
    for (const r of k.roots) {
      if (r.param !== undefined) { this.args.pointers.push({ param: r.param, base: off, limit: off + 4 }); off += 8; }
    }
    for (const r of k.roots) if (r.symbol) { this.args.symbols.push({ symbol: r.symbol, base: off, limit: off + 4 }); off += 8; }
    k.params.forEach((p, i) => {
      if (p.kind !== "scalar") return;
      const sz = vtSize(p.t!);
      off = align(off, Math.min(8, SCALAR_SIZE[p.t!.s]));
      this.args.scalars.push({ param: i, off, t: p.t! });
      off += sz;
    });
    this.args.size = align(off, 16);
    // var classification: per-thread slots for values that live across runs
    const runs = this.allRuns();
    const uses = new Map<Var, Set<number>>();
    const firstIsDef = new Map<Var, boolean>();
    for (const r of runs) {
      const seen = new Set<Var>();
      const note = (v: Var, def: boolean) => {
        if (!uses.has(v)) uses.set(v, new Set());
        uses.get(v)!.add(r.id);
        if (!seen.has(v)) { seen.add(v); if (r.inLoop) firstIsDef.set(v, (firstIsDef.get(v) ?? true) && def && uses.get(v)!.size === 1); }
      };
      this.walkRun(r.stmts as XStmt[], (s, top) => {
        for (const e of xExprs(s)) someExpr(e, (x) => { if (x.op === "var") note(x.v, false); return false; });
        const d = xDef(s);
        if (d) note(d, top && s.s === "let");
      });
    }
    let sOff = 0;
    const alloc = (bytes: number, al = 8) => { sOff = align(sOff, al); const o = sOff; sOff += bytes; return o; };
    // shared memory (per block) and local memory roots
    const rootRuns = new Map<Root, Set<number>>();
    for (const r of runs) this.walkRun(r.stmts as XStmt[], (s) => {
      const add = (root: Root) => { if (!rootRuns.has(root)) rootRuns.set(root, new Set()); rootRuns.get(root)!.add(r.id); };
      if (s.s === "store" || s.s === "atomic") add(s.root);
      for (const e of xExprs(s)) someExpr(e, (x) => { if (x.op === "load") add(x.root); return false; });
    });
    for (const root of k.roots) {
      if (root.space === "shared") this.rootOff.set(root, { off: alloc(Math.max(4, root.bytes), 16), perThread: 0 });
      else if (root.space === "local") {
        const rr = rootRuns.get(root);
        const inLoopRun = rr ? [...rr].some((id) => runs.find((x) => x.id === id)!.inLoop) : false;
        const perThread = (rr && rr.size > 1) || inLoopRun ? align(root.bytes, 8) : 0;
        this.rootOff.set(root, { off: alloc(perThread ? perThread * this.threads : Math.max(8, root.bytes), 16), perThread });
      }
    }
    for (const [v, rs] of uses) {
      const r = [...rs];
      const single = r.length === 1;
      const run = runs.find((x) => x.id === r[0])!;
      if (single && (!run.inLoop || firstIsDef.get(v) === true)) continue;
      this.slotOff.set(v, alloc(vtSize(v.t) <= 4 ? 4 * v.t.n : 8 * v.t.n, 8) );
      // slots are threads x size: reserve the rest
      sOff += (this.threads - 1) * (vtSize(v.t) <= 4 ? 4 * v.t.n : 8 * v.t.n);
    }
    this.condOff = alloc(this.condSlots * this.threads, 4);
    this.brkOff = alloc(this.nLoops * this.threads, 4);
    this.contOff = alloc(this.nLoops * this.threads, 4);
    this.aliveOff = alloc(this.threads, 4);
    this.warpOff = alloc(this.warpSlots ? this.threads * 8 : 0, 8);
    this.warpPartOff = alloc(this.warpSlots ? this.threads : 0, 4);
    this.printfOff = alloc(this.k.features.has("printf") ? 8 * 32 : 0, 8);
    this.scratch = align(sOff, 16);
    if (this.slotOff.size) this.notes.push(`${this.slotOff.size} values spilled to per-thread slots across barriers`);
  }

  private walkRun(list: XStmt[], f: (s: XStmt, top: boolean) => void, top = true): void {
    for (const s of list) {
      f(s, top);
      if (s.s === "if") { this.walkRun(s.then, f, false); this.walkRun(s.else, f, false); }
      else if (s.s === "loop") { this.walkRun(s.body, f, false); this.walkRun(s.cont, f, false); }
    }
  }

  // ------------------------------------------------------------------ imports & helpers
  private imports(): void {
    const imp = (name: string, params: ValType[], results: ValType[]) => this.imp.set(name, this.mb.importFunc("env", name, params, results));
    imp("fail", [T.i32, T.i32, T.i32, T.i32], []);
    if (this.k.features.has("printf")) imp("printf", [T.i32, T.i32], []);
    const needed = new Set<string>();
    walkStmts(this.k.body, (s) => { for (const e of stmtExprs(s)) someExpr(e, (x) => { if (x.op === "call") needed.add(x.fn); return false; }); });
    for (const m of WASM_MATH_IMPORTS) if (needed.has(m)) imp(m, [T.f64], [T.f64]);
    for (const m of WASM_MATH2_IMPORTS) if (needed.has(m)) imp(m, [T.f64, T.f64], [T.f64]);
  }

  private buildHelpers(): void {
    // f32 -> f16 bits (round to nearest even), f16 bits -> f32: integer implementations (exact)
    const uses16 = someTypeUsed(this.k, (s) => s === "f16");
    const usesBf = someTypeUsed(this.k, (s) => s === "bf16");
    if (uses16) {
      const h2f = new Func([T.i32], [T.f32], "h2f");
      const h = 0, s = h2f.local(T.i32), e = h2f.local(T.i32), m = h2f.local(T.i32);
      h2f.get(h).i32c(0x8000).op(OP.i32_and).i32c(16).op(OP.i32_shl).set(s);
      h2f.get(h).i32c(10).op(OP.i32_shr_u).i32c(0x1f).op(OP.i32_and).set(e);
      h2f.get(h).i32c(0x3ff).op(OP.i32_and).set(m);
      // subnormal / zero: m * 2^-24 with sign
      h2f.get(e).op(OP.i32_eqz); h2f.if_(); h2f.get(m).op(OP.f32_convert_i32_u).f32c(5.960464477539063e-8).op(OP.f32_mul)
        .get(s).op(OP.f32_reinterpret_i32).op(OP.f32_copysign).ret().end();
      h2f.get(e).i32c(31).op(OP.i32_eq); h2f.if_(); h2f.get(s).i32c(0x7f800000).op(OP.i32_or).get(m).i32c(13).op(OP.i32_shl).op(OP.i32_or).op(OP.f32_reinterpret_i32).ret().end();
      h2f.get(s).get(e).i32c(112).op(OP.i32_add).i32c(23).op(OP.i32_shl).op(OP.i32_or).get(m).i32c(13).op(OP.i32_shl).op(OP.i32_or).op(OP.f32_reinterpret_i32);
      this.helperFuncs.set("h2f", this.mb.addFunc(h2f));
      const f2h = new Func([T.f32], [T.i32], "f2h");
      const x = f2h.local(T.i32), sign = f2h.local(T.i32), ax = f2h.local(T.i32), hh = f2h.local(T.i32), rem = f2h.local(T.i32), sh = f2h.local(T.i32), mm = f2h.local(T.i32), half = f2h.local(T.i32);
      f2h.get(0).op(OP.i32_reinterpret_f32).set(x);
      f2h.get(x).i32c(16).op(OP.i32_shr_u).i32c(0x8000).op(OP.i32_and).set(sign);
      f2h.get(x).i32c(0x7fffffff).op(OP.i32_and).set(ax);
      f2h.get(ax).i32c(0x7f800000).op(OP.i32_gt_u); f2h.if_(); f2h.get(sign).i32c(0x7e00).op(OP.i32_or).ret().end();
      f2h.get(ax).i32c(0x477ff000).op(OP.i32_ge_u); f2h.if_(); f2h.get(sign).i32c(0x7c00).op(OP.i32_or).ret().end();
      f2h.get(ax).i32c(0x38800000).op(OP.i32_lt_u); f2h.if_();
      {
        f2h.get(ax).i32c(0x33000000).op(OP.i32_lt_u); f2h.if_(); f2h.get(sign).ret().end();
        f2h.get(ax).i32c(0x7fffff).op(OP.i32_and).i32c(0x800000).op(OP.i32_or).set(mm);
        f2h.i32c(126).get(ax).i32c(23).op(OP.i32_shr_u).op(OP.i32_sub).set(sh);
        f2h.get(mm).get(sh).op(OP.i32_shr_u).set(hh);
        f2h.get(mm).i32c(1).get(sh).op(OP.i32_shl).i32c(1).op(OP.i32_sub).op(OP.i32_and).set(rem);
        f2h.i32c(1).get(sh).i32c(1).op(OP.i32_sub).op(OP.i32_shl).set(half);
        // up = rem > half || (rem == half && h odd)
        f2h.get(sign).get(hh).get(rem).get(half).op(OP.i32_gt_u).get(rem).get(half).op(OP.i32_eq).get(hh).i32c(1).op(OP.i32_and).op(OP.i32_and).op(OP.i32_or)
          .op(OP.i32_add).op(OP.i32_or).ret();
      }
      f2h.end();
      f2h.get(ax).i32c(13).op(OP.i32_shr_u).i32c(0x1c000).op(OP.i32_sub).set(hh);
      f2h.get(ax).i32c(0x1fff).op(OP.i32_and).set(rem);
      f2h.get(sign).get(hh).get(rem).i32c(0x1000).op(OP.i32_gt_u).get(rem).i32c(0x1000).op(OP.i32_eq).get(hh).i32c(1).op(OP.i32_and).op(OP.i32_and).op(OP.i32_or)
        .op(OP.i32_add).op(OP.i32_or);
      this.helperFuncs.set("f2h", this.mb.addFunc(f2h));
    }
    if (usesBf) {
      const f2b = new Func([T.f32], [T.i32], "f2b");
      const x = f2b.local(T.i32);
      f2b.get(0).op(OP.i32_reinterpret_f32).set(x);
      f2b.get(x).i32c(0x7fffffff).op(OP.i32_and).i32c(0x7f800000).op(OP.i32_gt_u); f2b.if_(); f2b.get(x).i32c(16).op(OP.i32_shr_u).i32c(0x40).op(OP.i32_or).ret().end();
      f2b.get(x).i32c(0x7fff).op(OP.i32_add).get(x).i32c(16).op(OP.i32_shr_u).i32c(1).op(OP.i32_and).op(OP.i32_add).i32c(16).op(OP.i32_shr_u);
      this.helperFuncs.set("f2b", this.mb.addFunc(f2b));
    }
  }

  // ------------------------------------------------------------------ the run() function
  private genRun(): void {
    const f = this.f;
    this.argsLocal = 0; this.scratchLocal = 1; this.workerLocal = 2;
    const total = f.local(T.i32), b0 = f.local(T.i32), b1 = f.local(T.i32), chunk = f.local(T.i32);
    this.blin = f.local(T.i32); this.bidx = f.local(T.i32); this.bidy = f.local(T.i32); this.bidz = f.local(T.i32);
    this.tx = f.local(T.i32); this.ty = f.local(T.i32); this.tz = f.local(T.i32); this.tlin = f.local(T.i32);
    for (let i = 0; i < 3; i++) { this.gdim[i] = f.local(T.i32); f.get(this.argsLocal).mem(OP.i32_load, 2, this.args.grid + 4 * i).set(this.gdim[i]); }
    for (const p of this.args.pointers) {
      const b = f.local(T.i32), l = f.local(T.i64);
      f.get(this.argsLocal).mem(OP.i32_load, 2, p.base).set(b);
      f.get(this.argsLocal).mem(OP.i32_load, 2, p.limit).op(OP.i64_extend_i32_u).set(l);
      this.ptrBase.set(p.param, b); this.ptrLimit.set(p.param, l);
    }
    for (const s of this.args.symbols) {
      const root = this.k.roots.find((r) => r.symbol === s.symbol)!;
      const b = f.local(T.i32), l = f.local(T.i64);
      f.get(this.argsLocal).mem(OP.i32_load, 2, s.base).set(b);
      f.get(this.argsLocal).mem(OP.i32_load, 2, s.limit).op(OP.i64_extend_i32_u).set(l);
      this.symBase.set(root, b); this.symLimit.set(root, l);
    }
    // all variables used outside slots get locals now (one per lane)
    for (const v of this.k.vars) if (!this.slotOff.has(v)) this.locals.set(v, Array.from({ length: v.t.n }, () => f.local(wt(v.t.s))));
    f.get(this.gdim[0]).get(this.gdim[1]).op(OP.i32_mul).get(this.gdim[2]).op(OP.i32_mul).set(total);
    f.get(this.argsLocal).mem(OP.i32_load, 2, this.args.chunk).set(chunk);
    // chunk loop
    const outer = f.block();
    const chunkLoop = f.loop();
    {
      // stop early when another worker failed
      f.get(this.argsLocal).atomic(AT.i32_load, 2, this.args.stop).brIf(outer);
      f.get(this.argsLocal).get(chunk).atomic(AT.i32_add, 2, this.args.counter).tee(b0).get(total).op(OP.i32_ge_u).brIf(outer);
      f.get(b0).get(chunk).op(OP.i32_add).tee(b1).get(total).get(b1).get(total).op(OP.i32_lt_u).select().set(b1);
      f.get(b0).set(this.blin);
      const blockEnd = f.block();
      const blockLoop = f.loop();
      {
        f.get(this.blin).get(b1).op(OP.i32_ge_u).brIf(blockEnd);
        f.get(this.blin).get(this.gdim[0]).op(OP.i32_rem_u).set(this.bidx);
        f.get(this.blin).get(this.gdim[0]).op(OP.i32_div_u).get(this.gdim[1]).op(OP.i32_rem_u).set(this.bidy);
        f.get(this.blin).get(this.gdim[0]).get(this.gdim[1]).op(OP.i32_mul).op(OP.i32_div_u).set(this.bidz);
        if (this.hasBarriers) this.fillBytes(this.aliveOff, this.threads, 1);
        this.genPhases(this.phases);
        f.get(this.blin).i32c(1).op(OP.i32_add).set(this.blin);
        f.br(blockLoop);
      }
      f.end(); // loop
      f.end(); // block
      f.br(chunkLoop);
    }
    f.end(); f.end();
  }

  private fillBytes(off: number, n: number, value: number): void {
    if (n <= 0) return;
    const f = this.f;
    // memory.fill(dest, value, n)
    f.get(this.scratchLocal).i32c(off).op(OP.i32_add).i32c(value).i32c(n).fc(0x0b, 0x00);
  }

  private genPhases(ph: Phase[]): void {
    for (const p of ph) {
      if (p.k === "run") this.genThreadLoop(p);
      else this.genPhaseLoop(p);
    }
  }

  private genPhaseLoop(p: Phase & { k: "loop" }): void {
    const f = this.f;
    this.fillBytes(this.brkOff + p.id * this.threads, this.threads, 0);
    const exit = f.block();
    const lp = f.loop();
    this.fillBytes(this.contOff + p.id * this.threads, this.threads, 0);
    // any thread still active?
    const any = this.tmp(T.i32);
    f.i32c(0).set(any);
    this.forEachThread(() => {
      this.maskCheck([...p.masks, { kind: "brk", loop: p.id }], () => { f.i32c(1).set(any); });
    });
    f.get(any).op(OP.i32_eqz).brIf(exit);
    this.release(T.i32, any);
    this.genPhases(p.body);
    this.fillBytes(this.contOff + p.id * this.threads, this.threads, 0);
    this.genPhases(p.cont);
    f.br(lp);
    f.end(); f.end();
  }

  /** for tz, ty, tx: body() with tx/ty/tz/tlin set */
  private forEachThread(body: () => void): void {
    const f = this.f;
    const dims: [number, number, number][] = [[this.tz, this.bz, 2], [this.ty, this.by, 1], [this.tx, this.bx, 0]];
    const opened: number[] = [];
    for (const [loc, n] of dims) {
      if (n === 1) { f.i32c(0).set(loc); continue; }
      f.i32c(0).set(loc);
      opened.push(f.block(), f.loop());
      f.get(loc).i32c(n).op(OP.i32_ge_u).brIf(opened[opened.length - 2]);
    }
    f.get(this.tx).get(this.ty).i32c(this.bx).op(OP.i32_mul).op(OP.i32_add).get(this.tz).i32c(this.bx * this.by).op(OP.i32_mul).op(OP.i32_add).set(this.tlin);
    const cont = f.block();
    body();
    f.end();
    void cont;
    for (const [loc, n] of [...dims].reverse()) {
      if (n === 1) continue;
      const lp = opened.pop()!, blk = opened.pop()!;
      void blk;
      f.get(loc).i32c(1).op(OP.i32_add).set(loc);
      f.br(lp);
      f.end(); f.end();
    }
  }

  /** run `then` only for threads that pass all masks */
  private maskCheck(masks: Mask[], then: () => void): void {
    const f = this.f;
    if (!masks.length && !this.hasBarriers) { then(); return; }
    const conds: (() => void)[] = [];
    if (this.hasBarriers) conds.push(() => this.byteAt(this.aliveOff));
    for (const m of masks) {
      if (m.kind === "cond") conds.push(() => { this.byteAt(this.condOff + m.slot * this.threads); if (m.negate) f.op(OP.i32_eqz); });
      else if (m.kind === "brk") conds.push(() => { this.byteAt(this.brkOff + m.loop * this.threads); f.op(OP.i32_eqz); });
      else conds.push(() => { this.byteAt(this.contOff + m.loop * this.threads); f.op(OP.i32_eqz); });
    }
    conds.forEach((c, i) => { c(); if (i) f.op(OP.i32_and); });
    f.if_(); then(); f.end();
  }

  /** push scratch byte [base + tlin] */
  private byteAt(base: number): void { this.f.get(this.scratchLocal).get(this.tlin).op(OP.i32_add).mem(OP.i32_load8_u, 0, base); }
  private setByte(base: number, v: number): void { this.f.get(this.scratchLocal).get(this.tlin).op(OP.i32_add).i32c(v).mem(OP.i32_store8, 0, base); }

  private genThreadLoop(r: Phase & { k: "run" }): void {
    this.currentRun = r;
    this.forEachThread(() => {
      const run = () => {
        const f = this.f;
        this.runExit = f.block();
        this.runLoopStack = [{ phaseLoop: r.loops.length ? r.loops[r.loops.length - 1] : null, label: null }];
        this.stmts(r.stmts as XStmt[]);
        f.end();
      };
      if (r.allThreads) run(); else this.maskCheck(r.masks, run);
    });
    this.currentRun = null;
  }

  // ------------------------------------------------------------------ statements
  private stmts(list: XStmt[]): void { for (const s of list) this.stmt(s); }

  private stmt(s: XStmt): void {
    const f = this.f;
    switch (s.s) {
      case "let": {
        const v = s.v;
        for (let i = 0; i < v.t.n; i++) {
          if (this.slotOff.has(v)) { this.slotAddr(v, i); this.lane(s.e, i); this.storeSlot(v); }
          else { this.lane(s.e, i); f.set(this.locals.get(v)![i]); }
        }
        // vector assignment from an expression that reads the var itself: lanes were computed one at a time; safe because
        // the IR only builds vectors from lanes (vec/extract), and lane i reads lane i before it is written
        return;
      }
      case "store": return this.store(s.root, s.off, s.t, s.e);
      case "atomic": return this.atomic(s);
      case "if": {
        this.cond(s.c);
        f.if_();
        this.runLoopStack.push({ ...this.runLoopStack[this.runLoopStack.length - 1] });
        this.stmts(s.then);
        if (s.else.length) { f.else_(); this.stmts(s.else); }
        this.runLoopStack.pop();
        f.end();
        return;
      }
      case "loop": {
        const brk = f.block();
        const lp = f.loop();
        const contBlk = f.block();
        this.runLoopStack.push({ phaseLoop: null, label: { brk, cont: contBlk } });
        this.stmts(s.body);
        this.runLoopStack.pop();
        f.end(); // continue target
        this.runLoopStack.push({ phaseLoop: null, label: { brk, cont: contBlk } });
        this.stmts(s.cont);
        this.runLoopStack.pop();
        if (s.breakIf) { this.cond(s.breakIf); f.brIf(brk); }
        f.br(lp);
        f.end(); f.end();
        return;
      }
      case "break": case "continue": {
        const top = this.runLoopStack[this.runLoopStack.length - 1];
        if (top.label) { f.br(s.s === "break" ? top.label.brk : top.label.cont); return; }
        if (top.phaseLoop === null) cfail("internal", `${s.s} outside of a loop`);
        // break/continue of a loop that contains barriers: per-thread flag, leave this run
        this.setByte((s.s === "break" ? this.brkOff : this.contOff) + top.phaseLoop * this.threads, 1);
        f.br(this.runExit);
        return;
      }
      case "return":
        if (this.hasBarriers) this.setByte(this.aliveOff, 0);
        f.br(this.runExit);
        return;
      case "barrier": case "warp": cfail("internal", "barrier inside a run");
      case "fence": f.atomicFence(); return;
      case "printf": return this.printfStmt(s.fmt, s.args);
      case "assert": {
        this.cond(s.c);
        f.op(OP.i32_eqz); f.if_();
        this.failCall(this.site("assert", undefined, s.msg), () => f.i32c(0));
        f.end();
        return;
      }
      case "trap": this.failCall(this.site("trap", undefined, s.msg), () => f.i32c(0)); return;
      case "comment": return;
      case "_cond": {
        this.f.get(this.scratchLocal).get(this.tlin).op(OP.i32_add);
        this.cond(s.c);
        this.f.mem(OP.i32_store8, 0, this.condOff + s.slot * this.threads);
        return;
      }
      case "_brkif": {
        this.cond(s.c);
        f.if_(); this.setByte(this.brkOff + s.loop * this.threads, 1); f.end();
        return;
      }
      case "_wclear": this.setByte(this.warpPartOff, 0); return;
      case "_wwrite": {
        // value bits into the warp scratch slot (8 bytes per thread), participation flag
        const t = s.value.t;
        if (t.n > 1) cfail("backend", "warp shuffles of vector values are not supported; shuffle each component");
        f.get(this.scratchLocal).get(this.tlin).i32c(3).op(OP.i32_shl).op(OP.i32_add);
        this.lane(s.value, 0);
        this.storeOp(slotScalar(t.s), this.warpOff);
        this.setByte(this.warpPartOff, 1);
        return;
      }
      case "_wread": return this.warpRead(s);
    }
  }

  private warpRead(s: WRead): void {
    const f = this.f;
    const t = s.t;
    const vs = slotScalar(s.args[0].t.s); // how the written value is stored in the warp scratch
    const ws = s.width;
    const lane = this.tmp(T.i32), base = this.tmp(T.i32), src = this.tmp(T.i32);
    f.get(this.tlin).i32c(31).op(OP.i32_and).set(lane);
    f.get(this.tlin).i32c(~31).op(OP.i32_and).set(base);
    const writeResult = (pushValue: () => void) => {
      const v = s.result;
      if (this.slotOff.has(v)) { this.slotAddr(v, 0); pushValue(); this.storeSlot(v); }
      else { pushValue(); f.set(this.locals.get(v)![0]); }
    };
    const loadLane = (srcLocal: number) => {
      f.get(this.scratchLocal).get(srcLocal).i32c(3).op(OP.i32_shl).op(OP.i32_add);
      this.loadOp(vs, this.warpOff);
    };
    const arg = (i: number) => this.lane(s.args[i], 0);
    switch (s.fn) {
      case "shfl": case "shfl_up": case "shfl_down": case "shfl_xor": {
        // source lane and "keep own value" condition (CUDA semantics with width segments)
        const seg = this.tmp(T.i32);
        f.get(lane).i32c(~(ws - 1)).op(OP.i32_and).set(seg);
        const keep = this.tmp(T.i32);
        f.i32c(0).set(keep);
        if (s.fn === "shfl") { f.get(seg); arg(1); f.i32c(ws - 1).op(OP.i32_and).op(OP.i32_or).set(src); }
        else if (s.fn === "shfl_xor") {
          f.get(lane); arg(1); f.op(OP.i32_xor).set(src);
          f.get(src).i32c(~(ws - 1)).op(OP.i32_and).get(seg).op(OP.i32_ne).set(keep);
        } else if (s.fn === "shfl_up") {
          f.get(lane); arg(1); f.op(OP.i32_sub).set(src);
          f.get(lane).i32c(ws - 1).op(OP.i32_and); arg(1); f.op(OP.i32_lt_u).set(keep);
        } else {
          f.get(lane); arg(1); f.op(OP.i32_add).set(src);
          f.get(lane).i32c(ws - 1).op(OP.i32_and); arg(1); f.op(OP.i32_add).i32c(ws).op(OP.i32_ge_u).set(keep);
        }
        f.get(base).get(src).i32c(31).op(OP.i32_and).op(OP.i32_add).set(src);
        // reading beyond the block (partial last warp) keeps the own value
        f.get(keep).get(src).i32c(this.threads).op(OP.i32_ge_u).op(OP.i32_or).set(keep);
        writeResult(() => {
          f.get(keep);
          f.if_(wt(t.s));
          f.get(this.scratchLocal).get(this.tlin).i32c(3).op(OP.i32_shl).op(OP.i32_add); this.loadOp(vs, this.warpOff);
          f.else_();
          loadLane(src);
          f.end();
        });
        this.release(T.i32, seg); this.release(T.i32, keep);
        break;
      }
      case "ballot": case "any": case "all": case "reduce_add": case "reduce_min": case "reduce_max": {
        const j = this.tmp(T.i32), acc = this.tmp(s.fn.startsWith("reduce") ? wt(t.s) : T.i32), cnt = this.tmp(T.i32);
        const isRed = s.fn.startsWith("reduce");
        if (s.fn === "reduce_min") f.i32c(t.s === "i32" ? 0x7fffffff : -1).set(acc);
        else if (s.fn === "reduce_max") f.i32c(t.s === "i32" ? -0x80000000 : 0).set(acc);
        else f.i32c(0).set(acc);
        f.i32c(0).set(cnt);
        f.i32c(0).set(j);
        const end = f.block(); const lp = f.loop();
        f.get(j).i32c(32).op(OP.i32_ge_u).brIf(end);
        f.get(base).get(j).op(OP.i32_add).set(src);
        f.get(src).i32c(this.threads).op(OP.i32_lt_u);
        f.if_();
        // participating lanes only
        f.get(this.scratchLocal).get(src).op(OP.i32_add).mem(OP.i32_load8_u, 0, this.warpPartOff);
        f.if_();
        f.get(cnt).i32c(1).op(OP.i32_add).set(cnt);
        if (isRed) {
          const sig = t.s === "i32";
          if (s.fn === "reduce_add") { f.get(acc); loadLane(src); f.op(OP.i32_add).set(acc); }
          else {
            const val = this.tmp(T.i32);
            loadLane(src); f.set(val);
            f.get(val).get(acc).get(val).get(acc).op(s.fn === "reduce_min" ? (sig ? OP.i32_lt_s : OP.i32_lt_u) : (sig ? OP.i32_gt_s : OP.i32_gt_u)).select().set(acc);
            this.release(T.i32, val);
          }
        } else {
          f.get(acc); loadLane(src); f.i32c(0).op(OP.i32_ne).get(j).op(OP.i32_shl).op(OP.i32_or).set(acc);
        }
        f.end(); f.end();
        f.get(j).i32c(1).op(OP.i32_add).set(j);
        f.br(lp);
        f.end(); f.end();
        writeResult(() => {
          if (s.fn === "ballot" || isRed) f.get(acc);
          else if (s.fn === "any") f.get(acc).i32c(0).op(OP.i32_ne);
          else f.get(acc).op(OP.i32_popcnt).get(cnt).op(OP.i32_eq);
        });
        this.release(T.i32, j); this.release(T.i32, cnt);
        if (isRed) this.release(wt(t.s), acc); else this.release(T.i32, acc);
        break;
      }
    }
    this.release(T.i32, lane); this.release(T.i32, base); this.release(T.i32, src);
  }

  // ------------------------------------------------------------------ slots & temps
  private slotAddr(v: Var, lane: number): void {
    const es = vtSize(vt(v.t.s)) <= 4 ? 4 : 8;
    void lane;
    this.f.get(this.scratchLocal).get(this.tlin).i32c(es * v.t.n).op(OP.i32_mul).op(OP.i32_add);
    (this as unknown as { pendingSlot: { v: Var; lane: number } }).pendingSlot = { v, lane };
  }
  private storeSlot(v: Var): void {
    const { lane } = (this as unknown as { pendingSlot: { v: Var; lane: number } }).pendingSlot;
    const es = vtSize(vt(v.t.s)) <= 4 ? 4 : 8;
    const off = this.slotOff.get(v)! + lane * es;
    const w = wt(v.t.s);
    this.f.mem(w === T.i32 ? OP.i32_store : w === T.i64 ? OP.i64_store : w === T.f32 ? OP.f32_store : OP.f64_store, es === 8 ? 3 : 2, off);
  }
  private loadSlot(v: Var, lane: number): void {
    const es = vtSize(vt(v.t.s)) <= 4 ? 4 : 8;
    const off = this.slotOff.get(v)! + lane * es;
    const w = wt(v.t.s);
    this.f.get(this.scratchLocal).get(this.tlin).i32c(es * v.t.n).op(OP.i32_mul).op(OP.i32_add);
    this.f.mem(w === T.i32 ? OP.i32_load : w === T.i64 ? OP.i64_load : w === T.f32 ? OP.f32_load : OP.f64_load, es === 8 ? 3 : 2, off);
  }

  private tmp(t: ValType): number {
    const pool = t === T.i32 ? this.tmpI32 : t === T.i64 ? this.tmpI64 : t === T.f32 ? this.tmpF32 : this.tmpF64;
    return pool.pop() ?? this.f.local(t);
  }
  private release(t: ValType, l: number): void {
    (t === T.i32 ? this.tmpI32 : t === T.i64 ? this.tmpI64 : t === T.f32 ? this.tmpF32 : this.tmpF64).push(l);
  }

  // ------------------------------------------------------------------ diagnostics
  private site(kind: WasmProgram["sites"][number]["kind"], root?: string, msg?: string): number {
    this.sites.push({ kind, root, msg });
    return this.sites.length - 1;
  }
  /** env.fail(site, value, thread_linear, block_linear): throws in JS */
  private failCall(site: number, pushValue: () => void): void {
    const f = this.f;
    f.i32c(site); pushValue(); f.get(this.tlin).get(this.blin).call(this.imp.get("fail")!);
  }

  private printfStmt(fmt: string, args: Expr[]): void {
    const f = this.f;
    const id = this.printf.length;
    this.printf.push({ fmt, types: args.map((a) => a.t.s) });
    // arguments as 8-byte words in the printf staging area, then env.printf(id, ptr)
    if (args.length > 32) cfail("unsupported", "printf with more than 32 arguments");
    args.forEach((a, i) => {
      f.get(this.scratchLocal);
      this.lane(a, 0);
      const s = a.t.s;
      const w = wt(s);
      if (w === T.i32) { f.mem(OP.i32_store, 2, this.printfOff + i * 8); }
      else if (w === T.i64) f.mem(OP.i64_store, 3, this.printfOff + i * 8);
      else if (w === T.f32) f.mem(OP.f32_store, 2, this.printfOff + i * 8);
      else f.mem(OP.f64_store, 3, this.printfOff + i * 8);
    });
    f.i32c(id).get(this.scratchLocal).i32c(this.printfOff).op(OP.i32_add).call(this.imp.get("printf")!);
  }

  // ------------------------------------------------------------------ memory access
  /** push the absolute address of (root, off) after checking bounds for an access of `size` bytes */
  private address(root: Root, off: Expr, size: number, kind: "oob-load" | "oob-store" | "oob-atomic"): void {
    const f = this.f;
    const o = this.tmp(T.i32);
    this.lane(off, 0);
    f.set(o);
    let limit: () => void;
    let base: () => void;
    if (root.param !== undefined) {
      const lim = this.ptrLimit.get(root.param)!, b = this.ptrBase.get(root.param)!;
      limit = () => f.get(lim);
      base = () => f.get(b);
    } else if (root.symbol) {
      const lim = this.symLimit.get(root)!, b = this.symBase.get(root)!;
      limit = () => f.get(lim);
      base = () => f.get(b);
    } else {
      const ro = this.rootOff.get(root)!;
      const bytes = Math.max(root.bytes, 0);
      limit = () => f.i64c(BigInt(bytes));
      base = () => {
        f.get(this.scratchLocal);
        if (ro.perThread) f.get(this.tlin).i32c(ro.perThread).op(OP.i32_mul).op(OP.i32_add);
        f.i32c(ro.off).op(OP.i32_add);
      };
    }
    // bounds: u64(off) + size > limit -> fail
    f.get(o).op(OP.i64_extend_i32_u).i64c(BigInt(size)).op(OP.i64_add);
    limit();
    f.op(OP.i64_gt_u);
    f.if_();
    this.failCall(this.site(kind, root.name), () => f.get(o));
    f.end();
    base();
    f.get(o).op(OP.i32_add);
    this.release(T.i32, o);
  }

  private loadOp(s: Scalar, offset: number): void {
    const f = this.f;
    switch (s) {
      case "bool": case "u8": f.mem(OP.i32_load8_u, 0, offset); return;
      case "i8": f.mem(OP.i32_load8_s, 0, offset); return;
      case "i16": f.mem(OP.i32_load16_s, 1, offset); return;
      case "u16": f.mem(OP.i32_load16_u, 1, offset); return;
      case "i32": case "u32": f.mem(OP.i32_load, 2, offset); return;
      case "i64": case "u64": f.mem(OP.i64_load, 3, offset); return;
      case "f32": f.mem(OP.f32_load, 2, offset); return;
      case "f64": f.mem(OP.f64_load, 3, offset); return;
      case "f16": f.mem(OP.i32_load16_u, 1, offset); f.call(this.helperFuncs.get("h2f")!); return;
      case "bf16": f.mem(OP.i32_load16_u, 1, offset).i32c(16).op(OP.i32_shl).op(OP.f32_reinterpret_i32); return;
    }
  }
  private storeOp(s: Scalar, offset: number): void {
    const f = this.f;
    switch (s) {
      case "bool": case "u8": case "i8": f.mem(OP.i32_store8, 0, offset); return;
      case "i16": case "u16": f.mem(OP.i32_store16, 1, offset); return;
      case "i32": case "u32": f.mem(OP.i32_store, 2, offset); return;
      case "i64": case "u64": f.mem(OP.i64_store, 3, offset); return;
      case "f32": f.mem(OP.f32_store, 2, offset); return;
      case "f64": f.mem(OP.f64_store, 3, offset); return;
      case "f16": f.call(this.helperFuncs.get("f2h")!); f.mem(OP.i32_store16, 1, offset); return;
      case "bf16": f.call(this.helperFuncs.get("f2b")!); f.mem(OP.i32_store16, 1, offset); return;
    }
  }

  private loadLaneOf(e: Expr & { op: "load" }, lane: number): void {
    const es = SCALAR_SIZE[e.t.s];
    this.address(e.root, e.off, vtSize(e.t), "oob-load");
    this.loadOp(e.t.s, lane * es);
  }

  private store(root: Root, off: Expr, t: VT, value: Expr): void {
    const f = this.f;
    const es = SCALAR_SIZE[t.s];
    const addr = this.tmp(T.i32);
    this.address(root, off, vtSize(t), "oob-store");
    f.set(addr);
    for (let i = 0; i < t.n; i++) {
      f.get(addr);
      this.lane(value, i);
      this.storeOp(t.s, i * es);
    }
    this.release(T.i32, addr);
  }

  private atomic(s: Stmt & { s: "atomic" }): void {
    const f = this.f;
    const st = s.t.s;
    const is64 = st === "i64" || st === "u64" || st === "f64";
    const addr = this.tmp(T.i32);
    this.address(s.root, s.off, SCALAR_SIZE[st], "oob-atomic");
    f.set(addr);
    const W = is64 ? T.i64 : T.i32;
    const old = this.tmp(W), nv = this.tmp(W);
    const al = is64 ? 3 : 2;
    const ld = () => f.get(addr).atomic(is64 ? AT.i64_load : AT.i32_load, al);
    const toBits = (pushVal: () => void) => { pushVal(); if (st === "f32") f.op(OP.i32_reinterpret_f32); if (st === "f64") f.op(OP.i64_reinterpret_f64); };
    const arg = (i: number) => () => this.lane(s.args[i], 0);
    // compare-exchange loop computing new = op(old)
    const casLoop = (compute: () => void) => {
      const done = f.block(); const lp = f.loop();
      ld(); f.set(old);
      compute(); f.set(nv);
      f.get(addr).get(old).get(nv).atomic(is64 ? AT.i64_cmpxchg : AT.i32_cmpxchg, al);
      f.get(old).op(is64 ? OP.i64_eq : OP.i32_eq).brIf(done);
      f.br(lp);
      f.end(); f.end();
    };
    const direct: Partial<Record<string, number>> = {
      add: is64 ? AT.i64_add : AT.i32_add, sub: is64 ? AT.i64_sub : AT.i32_sub, and: is64 ? AT.i64_and : AT.i32_and,
      or: is64 ? AT.i64_or : AT.i32_or, xor: is64 ? AT.i64_xor : AT.i32_xor, xchg: is64 ? AT.i64_xchg : AT.i32_xchg,
    };
    const floatAdd = (st === "f32" || st === "f64") && s.fn === "add";
    if (direct[s.fn] !== undefined && !floatAdd) {
      f.get(addr); toBits(arg(0)); f.atomic(direct[s.fn]!, al); f.set(old);
    } else if (floatAdd) {
      casLoop(() => {
        f.get(old).op(st === "f32" ? OP.f32_reinterpret_i32 : OP.f64_reinterpret_i64); arg(0)(); f.op(st === "f32" ? OP.f32_add : OP.f64_add);
        f.op(st === "f32" ? OP.i32_reinterpret_f32 : OP.i64_reinterpret_f64);
      });
    } else if (s.fn === "min" || s.fn === "max") {
      const signed = st === "i32" || st === "i64";
      const cmp = is64 ? (s.fn === "min" ? (signed ? OP.i64_lt_s : OP.i64_lt_u) : (signed ? OP.i64_gt_s : OP.i64_gt_u)) : (s.fn === "min" ? (signed ? OP.i32_lt_s : OP.i32_lt_u) : (signed ? OP.i32_gt_s : OP.i32_gt_u));
      casLoop(() => { arg(0)(); f.get(old); arg(0)(); f.get(old).op(cmp).select(); });
    } else if (s.fn === "inc" || s.fn === "dec") {
      casLoop(() => {
        if (s.fn === "inc") { f.i32c(0).get(old).i32c(1).op(OP.i32_add).get(old); arg(0)(); f.op(OP.i32_ge_u).select(); }
        else { arg(0)(); f.get(old).i32c(1).op(OP.i32_sub).get(old).op(OP.i32_eqz).get(old); arg(0)(); f.op(OP.i32_gt_u).op(OP.i32_or).select(); }
      });
    } else if (s.fn === "cas") {
      f.get(addr); toBits(arg(0)); toBits(arg(1)); f.atomic(is64 ? AT.i64_cmpxchg : AT.i32_cmpxchg, al); f.set(old);
    }
    if (s.result) {
      const v = s.result;
      const push = () => { f.get(old); if (st === "f32") f.op(OP.f32_reinterpret_i32); if (st === "f64") f.op(OP.f64_reinterpret_i64); };
      if (this.slotOff.has(v)) { this.slotAddr(v, 0); push(); this.storeSlot(v); }
      else { push(); f.set(this.locals.get(v)![0]); }
    }
    this.release(T.i32, addr); this.release(W, old); this.release(W, nv);
  }

  // ------------------------------------------------------------------ expressions (one lane at a time)
  private cond(e: Expr): void { this.lane(e, 0); }

  private lane(e: Expr, lane: number): void {
    const f = this.f;
    const s = e.t.s;
    switch (e.op) {
      case "const": return this.constant(s, e.v);
      case "var": {
        if (this.slotOff.has(e.v)) return this.loadSlot(e.v, lane);
        f.get(this.locals.get(e.v)![e.t.n > 1 ? lane : 0]);
        return;
      }
      case "param": {
        const sl = this.args.scalars.find((x) => x.param === e.index)!;
        const es = SCALAR_SIZE[s];
        f.get(this.argsLocal);
        this.loadOp(s, sl.off + lane * es);
        return;
      }
      case "builtin": return this.builtin(e.name);
      case "vec": return this.lane(e.items[lane], 0);
      case "extract": return this.lane(e.a, e.lane);
      case "load": return this.loadLaneOf(e, lane);
      case "select": {
        this.lane(e.a, lane); this.lane(e.b, lane); this.cond(e.c); f.select();
        return;
      }
      case "unop": {
        if (e.fn === "!") { this.lane(e.a, lane); f.op(OP.i32_eqz); return; }
        if (e.fn === "~") { this.lane(e.a, lane); if (wt(s) === T.i64) f.i64c(-1n).op(OP.i64_xor); else f.i32c(-1).op(OP.i32_xor); this.normalize(s); return; }
        // negation
        const w = wt(s);
        if (w === T.f32) { this.lane(e.a, lane); f.op(OP.f32_neg); return; }
        if (w === T.f64) { this.lane(e.a, lane); f.op(OP.f64_neg); return; }
        if (w === T.i64) { f.i64c(0n); this.lane(e.a, lane); f.op(OP.i64_sub); return; }
        f.i32c(0); this.lane(e.a, lane); f.op(OP.i32_sub); this.normalize(s);
        return;
      }
      case "binop": return this.binop(e, lane);
      case "cast": return this.convert(e.a, e.t, lane);
      case "bitcast": return this.bitcast(e.a, e.t, lane);
      case "call": return this.call(e, lane);
    }
  }

  /** keep small integer types normalized (sign/zero extended) in i32 registers */
  private normalize(s: Scalar): void {
    const f = this.f;
    switch (s) {
      case "i8": f.op(OP.i32_extend8_s); return;
      case "i16": f.op(OP.i32_extend16_s); return;
      case "u8": f.i32c(0xff).op(OP.i32_and); return;
      case "u16": f.i32c(0xffff).op(OP.i32_and); return;
      case "bool": f.i32c(1).op(OP.i32_and); return;
    }
  }

  private constant(s: Scalar, v: number | bigint | boolean): void {
    const f = this.f;
    const w = wt(s);
    if (w === T.i64) { f.i64c(BigInt(v as number | bigint)); return; }
    if (w === T.f64) { f.f64c(Number(v)); return; }
    if (w === T.f32) { f.f32c(Number(v)); return; }
    f.i32c(typeof v === "boolean" ? (v ? 1 : 0) : Number(v) | 0);
  }

  private builtin(n: string): void {
    const f = this.f;
    switch (n) {
      case "tid.x": f.get(this.tx); return; case "tid.y": f.get(this.ty); return; case "tid.z": f.get(this.tz); return;
      case "bid.x": f.get(this.bidx); return; case "bid.y": f.get(this.bidy); return; case "bid.z": f.get(this.bidz); return;
      case "bdim.x": f.i32c(this.bx); return; case "bdim.y": f.i32c(this.by); return; case "bdim.z": f.i32c(this.bz); return;
      case "gdim.x": f.get(this.gdim[0]); return; case "gdim.y": f.get(this.gdim[1]); return; case "gdim.z": f.get(this.gdim[2]); return;
      case "tid.linear": f.get(this.tlin); return;
      case "lane": f.get(this.tlin).i32c(31).op(OP.i32_and); return;
      case "warp": f.get(this.tlin).i32c(5).op(OP.i32_shr_u); return;
    }
    cfail("internal", `builtin ${n}`);
  }

  private binop(e: Expr & { op: "binop" }, lane: number): void {
    const f = this.f;
    const s = e.a.t.s;
    const w = wt(s);
    if (e.fn === "&&" || e.fn === "||") {
      this.cond(e.a);
      f.if_(T.i32);
      if (e.fn === "&&") { this.cond(e.b); f.else_(); f.i32c(0); }
      else { f.i32c(1); f.else_(); this.cond(e.b); }
      f.end();
      return;
    }
    this.lane(e.a, lane);
    this.lane(e.b, lane);
    const signed = s === "i8" || s === "i16" || s === "i32" || s === "i64";
    const I = w === T.i64;
    const ops32: Record<string, number> = {
      "+": OP.i32_add, "-": OP.i32_sub, "*": OP.i32_mul, "/": signed ? OP.i32_div_s : OP.i32_div_u, "%": signed ? OP.i32_rem_s : OP.i32_rem_u,
      "&": OP.i32_and, "|": OP.i32_or, "^": OP.i32_xor, "<<": OP.i32_shl, ">>": signed ? OP.i32_shr_s : OP.i32_shr_u,
      "==": OP.i32_eq, "!=": OP.i32_ne, "<": signed ? OP.i32_lt_s : OP.i32_lt_u, "<=": signed ? OP.i32_le_s : OP.i32_le_u,
      ">": signed ? OP.i32_gt_s : OP.i32_gt_u, ">=": signed ? OP.i32_ge_s : OP.i32_ge_u,
    };
    const ops64: Record<string, number> = {
      "+": OP.i64_add, "-": OP.i64_sub, "*": OP.i64_mul, "/": signed ? OP.i64_div_s : OP.i64_div_u, "%": signed ? OP.i64_rem_s : OP.i64_rem_u,
      "&": OP.i64_and, "|": OP.i64_or, "^": OP.i64_xor, "<<": OP.i64_shl, ">>": signed ? OP.i64_shr_s : OP.i64_shr_u,
      "==": OP.i64_eq, "!=": OP.i64_ne, "<": signed ? OP.i64_lt_s : OP.i64_lt_u, "<=": signed ? OP.i64_le_s : OP.i64_le_u,
      ">": signed ? OP.i64_gt_s : OP.i64_gt_u, ">=": signed ? OP.i64_ge_s : OP.i64_ge_u,
    };
    const opsF32: Record<string, number> = { "+": OP.f32_add, "-": OP.f32_sub, "*": OP.f32_mul, "/": OP.f32_div, "==": OP.f32_eq, "!=": OP.f32_ne, "<": OP.f32_lt, "<=": OP.f32_le, ">": OP.f32_gt, ">=": OP.f32_ge };
    const opsF64: Record<string, number> = { "+": OP.f64_add, "-": OP.f64_sub, "*": OP.f64_mul, "/": OP.f64_div, "==": OP.f64_eq, "!=": OP.f64_ne, "<": OP.f64_lt, "<=": OP.f64_le, ">": OP.f64_gt, ">=": OP.f64_ge };
    const table = w === T.f32 ? opsF32 : w === T.f64 ? opsF64 : I ? ops64 : ops32;
    const op = table[e.fn];
    if (op === undefined) cfail("internal", `wasm binop ${e.fn} on ${s}`);
    // shifts: C semantics for in-range amounts; wasm masks the amount like hardware
    f.op(op);
    if (!["==", "!=", "<", "<=", ">", ">="].includes(e.fn) && w === T.i32) this.normalize(s);
  }

  private convert(a: Expr, to: VT, lane: number): void {
    const f = this.f;
    const fs = a.t.s, ts = to.s;
    if (a.t.n === 1 && to.n > 1) { this.convert(a, vt(ts), 0); return; }
    this.lane(a, a.t.n > 1 ? lane : 0);
    if (fs === ts) return;
    const wf = wt(fs), wtT = wt(ts);
    const sf = fs === "i8" || fs === "i16" || fs === "i32" || fs === "i64";
    if (ts === "bool") {
      if (wf === T.i32) f.i32c(0).op(OP.i32_ne);
      else if (wf === T.i64) f.op(OP.i64_eqz).op(OP.i32_eqz);
      else if (wf === T.f32) f.f32c(0).op(OP.f32_ne);
      else f.f64c(0).op(OP.f64_ne);
      return;
    }
    // to float
    if (ts === "f32" || ts === "f16" || ts === "bf16") {
      if (wf === T.i32) f.op(sf ? OP.f32_convert_i32_s : OP.f32_convert_i32_u);
      else if (wf === T.i64) f.op(sf ? OP.f32_convert_i64_s : OP.f32_convert_i64_u);
      else if (wf === T.f64) f.op(OP.f32_demote_f64);
      if (ts === "f16") f.call(this.helperFuncs.get("f2h")!).call(this.helperFuncs.get("h2f")!);
      if (ts === "bf16") f.call(this.helperFuncs.get("f2b")!).i32c(16).op(OP.i32_shl).op(OP.f32_reinterpret_i32);
      return;
    }
    if (ts === "f64") {
      if (wf === T.i32) f.op(sf ? OP.f64_convert_i32_s : OP.f64_convert_i32_u);
      else if (wf === T.i64) f.op(sf ? OP.f64_convert_i64_s : OP.f64_convert_i64_u);
      else f.op(OP.f64_promote_f32);
      return;
    }
    // to integer
    const st = ts === "i8" || ts === "i16" || ts === "i32" || ts === "i64";
    if (wf === T.f32 || wf === T.f64) {
      // C truncation; saturating (no trap) when out of range, which is undefined in C
      if (wtT === T.i64) f.fc(wf === T.f32 ? (st ? SAT.i64_f32_s : SAT.i64_f32_u) : (st ? SAT.i64_f64_s : SAT.i64_f64_u));
      else f.fc(wf === T.f32 ? (st || ts === "u8" || ts === "u16" ? SAT.i32_f32_s : SAT.i32_f32_u) : (st || ts === "u8" || ts === "u16" ? SAT.i32_f64_s : SAT.i32_f64_u));
      if (wtT === T.i32) this.normalize(ts);
      return;
    }
    if (wf === T.i64 && wtT === T.i32) { f.op(OP.i32_wrap_i64); this.normalize(ts); return; }
    if (wf === T.i32 && wtT === T.i64) { f.op(sf ? OP.i64_extend_i32_s : OP.i64_extend_i32_u); return; }
    this.normalize(ts);
  }

  private bitcast(a: Expr, to: VT, lane: number): void {
    const f = this.f;
    this.lane(a, a.t.n > 1 ? lane : 0);
    const fs = a.t.s, ts = to.s;
    if (fs === ts) return;
    const pair = `${fs}>${ts}`;
    switch (pair) {
      case "f32>i32": case "f32>u32": f.op(OP.i32_reinterpret_f32); return;
      case "i32>f32": case "u32>f32": f.op(OP.f32_reinterpret_i32); return;
      case "f64>i64": case "f64>u64": f.op(OP.i64_reinterpret_f64); return;
      case "i64>f64": case "u64>f64": f.op(OP.f64_reinterpret_i64); return;
      case "f16>u16": case "f16>i16": f.call(this.helperFuncs.get("f2h")!); if (ts === "i16") f.op(OP.i32_extend16_s); return;
      case "u16>f16": case "i16>f16": f.i32c(0xffff).op(OP.i32_and).call(this.helperFuncs.get("h2f")!); return;
      case "bf16>u16": case "bf16>i16": f.op(OP.i32_reinterpret_f32).i32c(16).op(OP.i32_shr_u); if (ts === "i16") f.op(OP.i32_extend16_s); return;
      case "u16>bf16": case "i16>bf16": f.i32c(16).op(OP.i32_shl).op(OP.f32_reinterpret_i32); return;
    }
    if (wt(fs) === wt(ts)) return; // i32 <-> u32, i64 <-> u64
    cfail("internal", `bitcast ${pair}`);
  }

  private call(e: Expr & { op: "call" }, lane: number): void {
    const f = this.f;
    const s = e.args[0]?.t.s ?? e.t.s;
    const w = wt(s);
    const F = w === T.f32, D = w === T.f64;
    const arg = (i: number) => this.lane(e.args[i], e.args[i].t.n > 1 ? lane : 0);
    const viaF64 = (name: string, n: number) => {
      for (let i = 0; i < n; i++) { arg(i); if (F) f.op(OP.f64_promote_f32); }
      f.call(this.imp.get(name)!);
      if (F) f.op(OP.f32_demote_f64);
    };
    switch (e.fn) {
      case "sqrt": arg(0); f.op(F ? OP.f32_sqrt : OP.f64_sqrt); return;
      case "rsqrt":
        if (F) { arg(0); f.op(OP.f64_promote_f32).op(OP.f64_sqrt); f.f64c(1).op(OP.f64_div); return; }
        f.f64c(1); arg(0); f.op(OP.f64_sqrt).op(OP.f64_div); return;
      case "floor": arg(0); f.op(F ? OP.f32_floor : OP.f64_floor); return;
      case "ceil": arg(0); f.op(F ? OP.f32_ceil : OP.f64_ceil); return;
      case "trunc": arg(0); f.op(F ? OP.f32_trunc : OP.f64_trunc); return;
      case "rint": arg(0); f.op(F ? OP.f32_nearest : OP.f64_nearest); return;
      case "round": {
        // half away from zero: t = trunc(x); |x - t| >= 0.5 ? t + copysign(1, x) : t
        const x = this.tmp(w), t = this.tmp(w);
        arg(0); f.set(x);
        f.get(x).op(F ? OP.f32_trunc : OP.f64_trunc).set(t);
        f.get(t); if (F) f.f32c(1); else f.f64c(1); f.get(x).op(F ? OP.f32_copysign : OP.f64_copysign).op(F ? OP.f32_add : OP.f64_add);
        f.get(t);
        f.get(x).get(t).op(F ? OP.f32_sub : OP.f64_sub).op(F ? OP.f32_abs : OP.f64_abs); if (F) f.f32c(0.5); else f.f64c(0.5); f.op(F ? OP.f32_ge : OP.f64_ge);
        f.select();
        this.release(w, x); this.release(w, t);
        return;
      }
      case "fabs": arg(0); f.op(F ? OP.f32_abs : OP.f64_abs); return;
      case "copysign": arg(0); arg(1); f.op(F ? OP.f32_copysign : OP.f64_copysign); return;
      case "fma": {
        // exact fused multiply-add for float through double (the product is exact in double, one more rounding is
        // innocuous for + since 53 >= 2*24+2); for double use the product and sum (not fused: documented)
        if (F) { arg(0); f.op(OP.f64_promote_f32); arg(1); f.op(OP.f64_promote_f32).op(OP.f64_mul); arg(2); f.op(OP.f64_promote_f32).op(OP.f64_add).op(OP.f32_demote_f64); return; }
        arg(0); arg(1); f.op(OP.f64_mul); arg(2); f.op(OP.f64_add); return;
      }
      case "fmin": case "fmax": {
        // C: NaN operands are ignored
        const a = this.tmp(w), b = this.tmp(w);
        arg(0); f.set(a); arg(1); f.set(b);
        // isnan(a) ? b : isnan(b) ? a : min(a, b)
        f.get(a).get(b).op(e.fn === "fmin" ? (F ? OP.f32_min : OP.f64_min) : (F ? OP.f32_max : OP.f64_max));
        f.get(a);
        f.get(b).get(b).op(F ? OP.f32_eq : OP.f64_eq); f.select();
        f.get(b);
        f.get(a).get(a).op(F ? OP.f32_eq : OP.f64_eq); f.select();
        this.release(w, a); this.release(w, b);
        return;
      }
      case "fdim": {
        const a = this.tmp(w), b = this.tmp(w);
        arg(0); f.set(a); arg(1); f.set(b);
        f.get(a).get(b).op(F ? OP.f32_sub : OP.f64_sub); if (F) f.f32c(0); else f.f64c(0); f.get(a).get(b).op(F ? OP.f32_gt : OP.f64_gt); f.select();
        // NaN propagates
        this.release(w, a); this.release(w, b);
        return;
      }
      case "ldexp": {
        // x * 2^n as two exact power-of-two multiplications in double (n clamped to +-2000: beyond that the result is 0/inf anyway)
        const n = this.tmp(T.i32), h = this.tmp(T.i32);
        arg(1); f.set(n);
        f.get(n).i32c(-2000).get(n).i32c(-2000).op(OP.i32_gt_s).select().set(n);
        f.get(n).i32c(2000).get(n).i32c(2000).op(OP.i32_lt_s).select().set(n);
        f.get(n).i32c(1).op(OP.i32_shr_s).set(h);
        const pow2 = (push: () => void) => { push(); f.op(OP.i64_extend_i32_s).i64c(1023n).op(OP.i64_add).i64c(52n).op(OP.i64_shl).op(OP.f64_reinterpret_i64); };
        arg(0); if (F) f.op(OP.f64_promote_f32);
        pow2(() => f.get(h)); f.op(OP.f64_mul);
        pow2(() => f.get(n).get(h).op(OP.i32_sub)); f.op(OP.f64_mul);
        if (F) f.op(OP.f32_demote_f64);
        this.release(T.i32, n); this.release(T.i32, h);
        return;
      }
      case "saturate": {
        const x = this.tmp(T.f32);
        arg(0); f.set(x);
        f.f32c(0); f.f32c(1); f.get(x); f.op(OP.f32_min).op(OP.f32_max); // NaN -> max(0, NaN)... handle explicitly
        f.f32c(0); f.get(x).get(x).op(OP.f32_eq); f.select();
        this.release(T.f32, x);
        return;
      }
      case "isnan": { const x = this.tmp(w); arg(0); f.set(x); f.get(x).get(x).op(F ? OP.f32_ne : OP.f64_ne); this.release(w, x); return; }
      case "isinf": { arg(0); f.op(F ? OP.f32_abs : OP.f64_abs); if (F) f.f32c(Infinity); else f.f64c(Infinity); f.op(F ? OP.f32_eq : OP.f64_eq); return; }
      case "isfinite": { arg(0); f.op(F ? OP.f32_abs : OP.f64_abs); if (F) f.f32c(Infinity); else f.f64c(Infinity); f.op(F ? OP.f32_lt : OP.f64_lt); return; }
      case "signbit": { arg(0); if (F) f.op(OP.i32_reinterpret_f32).i32c(31).op(OP.i32_shr_u); else f.op(OP.i64_reinterpret_f64).i64c(63n).op(OP.i64_shr_u).op(OP.i32_wrap_i64); return; }
      case "min": case "max": {
        const a = this.tmp(w), b = this.tmp(w);
        arg(0); f.set(a); arg(1); f.set(b);
        const signed = s === "i32" || s === "i64" || s === "i8" || s === "i16";
        const I = w === T.i64;
        f.get(a).get(b).get(a).get(b).op(e.fn === "min" ? (I ? (signed ? OP.i64_lt_s : OP.i64_lt_u) : (signed ? OP.i32_lt_s : OP.i32_lt_u)) : (I ? (signed ? OP.i64_gt_s : OP.i64_gt_u) : (signed ? OP.i32_gt_s : OP.i32_gt_u))).select();
        this.release(w, a); this.release(w, b);
        return;
      }
      case "abs": {
        const a = this.tmp(w);
        arg(0); f.set(a);
        if (w === T.i64) { f.i64c(0n).get(a).op(OP.i64_sub).get(a).get(a).i64c(0n).op(OP.i64_lt_s).select(); }
        else { f.i32c(0).get(a).op(OP.i32_sub).get(a).get(a).i32c(0).op(OP.i32_lt_s).select(); }
        this.release(w, a);
        return;
      }
      case "popc": arg(0); f.op(w === T.i64 ? OP.i64_popcnt : OP.i32_popcnt); if (w === T.i64) f.op(OP.i32_wrap_i64); return;
      case "clz": arg(0); f.op(w === T.i64 ? OP.i64_clz : OP.i32_clz); if (w === T.i64) f.op(OP.i32_wrap_i64); return;
      case "ffs": {
        const a = this.tmp(w);
        arg(0); f.set(a);
        if (w === T.i64) { f.get(a).op(OP.i64_ctz).op(OP.i32_wrap_i64).i32c(1).op(OP.i32_add).i32c(0).get(a).op(OP.i64_eqz).select(); }
        else { f.get(a).op(OP.i32_ctz).i32c(1).op(OP.i32_add).i32c(0).get(a).op(OP.i32_eqz).select(); }
        this.release(w, a);
        return;
      }
      case "brev": {
        // bit reversal by swapping progressively smaller groups
        const a = this.tmp(w);
        arg(0); f.set(a);
        const I = w === T.i64;
        const masks32: [number, number][] = [[0x55555555, 1], [0x33333333, 2], [0x0f0f0f0f, 4], [0x00ff00ff, 8], [0x0000ffff, 16]];
        const masks64: [bigint, bigint][] = [[0x5555555555555555n, 1n], [0x3333333333333333n, 2n], [0x0f0f0f0f0f0f0f0fn, 4n], [0x00ff00ff00ff00ffn, 8n], [0x0000ffff0000ffffn, 16n], [0x00000000ffffffffn, 32n]];
        if (I) for (const [m, sh] of masks64) {
          f.get(a).i64c(sh).op(OP.i64_shr_u).i64c(m).op(OP.i64_and).get(a).i64c(m).op(OP.i64_and).i64c(sh).op(OP.i64_shl).op(OP.i64_or).set(a);
        }
        else for (const [m, sh] of masks32) {
          f.get(a).i32c(sh).op(OP.i32_shr_u).i32c(m).op(OP.i32_and).get(a).i32c(m).op(OP.i32_and).i32c(sh).op(OP.i32_shl).op(OP.i32_or).set(a);
        }
        f.get(a);
        this.release(w, a);
        return;
      }
      case "mulhi": {
        if (w === T.i32) {
          const signed = s === "i32";
          arg(0); f.op(signed ? OP.i64_extend_i32_s : OP.i64_extend_i32_u);
          arg(1); f.op(signed ? OP.i64_extend_i32_s : OP.i64_extend_i32_u);
          f.op(OP.i64_mul).i64c(32n).op(OP.i64_shr_u).op(OP.i32_wrap_i64);
          return;
        }
        // 64x64 -> high 64 via 32-bit halves
        const a = this.tmp(T.i64), b = this.tmp(T.i64), lo = this.tmp(T.i64), mid = this.tmp(T.i64);
        arg(0); f.set(a); arg(1); f.set(b);
        const M = 0xffffffffn;
        // p0 = alo*blo, p1 = alo*bhi, p2 = ahi*blo, p3 = ahi*bhi
        f.get(a).i64c(M).op(OP.i64_and).get(b).i64c(M).op(OP.i64_and).op(OP.i64_mul).i64c(32n).op(OP.i64_shr_u).set(lo);
        f.get(a).i64c(M).op(OP.i64_and).get(b).i64c(32n).op(OP.i64_shr_u).op(OP.i64_mul).set(mid);
        // mid2 = lo + (mid & M) + ((ahi*blo) & M)
        f.get(lo).get(mid).i64c(M).op(OP.i64_and).op(OP.i64_add).get(a).i64c(32n).op(OP.i64_shr_u).get(b).i64c(M).op(OP.i64_and).op(OP.i64_mul).tee(lo).i64c(M).op(OP.i64_and).op(OP.i64_add);
        f.i64c(32n).op(OP.i64_shr_u); // carry
        f.get(mid).i64c(32n).op(OP.i64_shr_u).op(OP.i64_add);
        f.get(lo).i64c(32n).op(OP.i64_shr_u).op(OP.i64_add);
        f.get(a).i64c(32n).op(OP.i64_shr_u).get(b).i64c(32n).op(OP.i64_shr_u).op(OP.i64_mul).op(OP.i64_add);
        if (s === "i64") {
          // signed correction: hi -= (a<0 ? b : 0) + (b<0 ? a : 0)
          f.get(b).i64c(0n).get(a).i64c(0n).op(OP.i64_lt_s).select().op(OP.i64_sub);
          f.get(a).i64c(0n).get(b).i64c(0n).op(OP.i64_lt_s).select().op(OP.i64_sub);
        }
        this.release(T.i64, a); this.release(T.i64, b); this.release(T.i64, lo); this.release(T.i64, mid);
        return;
      }
    }
    const one = (WASM_MATH_IMPORTS as readonly string[]).includes(e.fn);
    const two = (WASM_MATH2_IMPORTS as readonly string[]).includes(e.fn);
    if (one || two) { viaF64(e.fn, two ? 2 : 1); return; }
    cfail("backend", `math function ${e.fn} is not available on the CPU backend`);
  }

  private importOr(name: string, n: number): number {
    const i = this.imp.get(name);
    if (i === undefined) cfail("internal", `missing import ${name}/${n}`);
    return i;
  }

  private describe(size: number): string {
    const lines = [`;; WebAssembly module for kernel ${this.k.displayName}: ${size} bytes, block ${this.k.block.join("x")} (${this.threads} logical threads)`];
    lines.push(`;; export run(args: i32, scratch: i32, worker: i32): executes blocks fetched from the shared counter at args+${this.args.counter}`);
    lines.push(`;; argument block (${this.args.size} bytes): grid@0 counter@12 stop@16 chunk@20 ` + this.args.pointers.map((p) => `${this.k.params[p.param].name}: base@${p.base} limit@${p.limit}`).join(" ") + " " + this.args.scalars.map((s) => `${this.k.params[s.param].name}@${s.off}`).join(" "));
    lines.push(`;; scratch per worker: ${this.scratch} bytes (shared memory ${[...this.rootOff].filter(([r]) => r.space === "shared").map(([r, o]) => `${r.name}@${o.off}:${r.bytes}B`).join(", ") || "none"}; spilled slots ${this.slotOff.size})`);
    const ph = (list: Phase[], ind: string): void => {
      for (const p of list) {
        if (p.k === "run") lines.push(`${ind}(thread-phase ${p.id}${p.masks.length ? ` masked by ${p.masks.map((m) => m.kind === "cond" ? `${m.negate ? "!" : ""}cond${m.slot}` : `${m.kind}${m.loop}`).join(",")}` : ""}${p.allThreads ? " all-threads" : ""}: ${countStmts(p.stmts)} statements) ;; barrier after`);
        else { lines.push(`${ind}(uniform-loop ${p.id}: repeats while any thread is active`); ph(p.body, ind + "  "); if (p.cont.length) { lines.push(`${ind}  ;; continuing`); ph(p.cont, ind + "  "); } lines.push(`${ind})`); }
      }
    };
    ph(this.phases, "");
    lines.push(...this.notes.map((n) => `;; ${n}`));
    return lines.join("\n");
  }
}

/** warp scratch holds every value as a full register: small integers and bool as 32-bit words */
function slotScalar(s: Scalar): Scalar {
  const w = wt(s);
  if (w === T.i32) return s === "i8" || s === "i16" || s === "i32" ? "i32" : "u32";
  if (s === "f16" || s === "bf16") return "f32";
  return s;
}

function countStmts(l: Stmt[]): number { let n = 0; walkStmts(l, () => n++); return n; }

function xExprs(s: XStmt): Expr[] {
  switch (s.s) {
    case "_cond": return [s.c];
    case "_brkif": return [s.c];
    case "_wwrite": return [s.value];
    case "_wread": return s.args;
    case "_wclear": return [];
    default: return stmtExprs(s);
  }
}
function xDef(s: XStmt): Var | null {
  if (s.s === "let") return s.v;
  if (s.s === "atomic") return s.result;
  if (s.s === "_wread") return s.result;
  return null;
}

function someTypeUsed(k: Kernel, pred: (s: Scalar) => boolean): boolean {
  if (k.vars.some((v) => pred(v.t.s))) return true;
  if (k.params.some((p) => p.t && pred(p.t.s))) return true;
  let found = false;
  walkStmts(k.body, (s) => {
    if ((s.s === "store" || s.s === "atomic") && pred(s.t.s)) found = true;
    for (const e of stmtExprs(s)) someExpr(e, (x) => { if (pred(x.t.s)) found = true; return found; });
  });
  return found;
}

export { isFloat, halfFromBits };
