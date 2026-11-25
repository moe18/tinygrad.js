/**
 * Semantic analysis and lowering from the AST to the kernel IR.
 *
 * One kernel instance (a __global__ function, its template arguments and a block-size specialization) is lowered at a
 * time. Device functions, methods, constructors and operators are inlined at their call sites (recursion is rejected), so
 * every pointer's target buffer ("root") is known statically. Structs are scalar-replaced into per-field variables unless
 * they live in memory. Side effects inside expressions are hoisted into statements in C's evaluation order.
 */
import { CompileError, fail, type DiagnosticSink, type SourceLoc } from "./diagnostics.js";
import type { Expr as AExpr, FuncDecl, Param, StructDecl, Stmt as AStmt, TemplateArg, TopDecl, TranslationUnit, TypeExpr, VarDecl } from "./ast.js";
import {
  bin, bool, cast, cnst, i32, readVar, T_BOOL, T_I32, T_U32, vt, vtEq,
  type Builtin, type Expr, type Kernel, type KernelParam, type Root, type Space, type Stmt, type Var, type VT,
} from "./ir.js";
import { foldExpr, normalize } from "./fold.js";
import { parseNumber } from "./lexer.js";
import {
  alignOf, BOOL, commonScalar, F32, F64, I32, I64, isFloat, isInt, isSigned, layoutStruct, NAMED_SCALARS, PRIM_SCALAR, promote, sameType,
  sc, sizeOf, typeName, U32, U64, VECTOR_TYPES, VOID, vecType,
  type CType, type Scalar, type StructDef,
} from "./types.js";
import { lowerBuiltinCall, BUILTIN_FUNCTIONS } from "./builtins.js";

// ------------------------------------------------------------------ values and lvalues
export type PtrT = CType & { k: "ptr" };
export type StructT = CType & { k: "struct" };

export type LV =
  | { k: "var"; v: Var; t: CType }
  | { k: "mem"; root: Root; off: Expr; t: CType }
  | { k: "lane"; base: LV & { k: "var" }; lane: number; t: CType }
  | { k: "agg"; t: StructT; fields: Map<string, LV> }
  | { k: "ptrvar"; b: PtrBinding }
  | { k: "builtin"; name: "threadIdx" | "blockIdx" | "blockDim" | "gridDim"; t: CType };

export type Val =
  | { k: "rv"; e: Expr; t: CType }
  | { k: "ptr"; root: Root | null; off: Expr; t: PtrT; reg?: LV }
  | { k: "agg"; t: StructT; fields: Map<string, Val> }
  | { k: "lv"; lv: LV }
  | { k: "void" }
  | { k: "str"; s: string }
  | { k: "fn"; name: string; targs?: TemplateArg[]; recv?: LV; struct?: StructDef; loc: SourceLoc };

export interface PtrBinding { root: Root | null; offVar: Var; t: PtrT; name: string; reg?: LV; isParam?: boolean }

type Binding =
  | { k: "lv"; lv: LV; constVal?: Expr }
  | { k: "const"; e: Expr; t: CType }
  | { k: "type"; t: CType };

export interface CompileOptions {
  /** "optimized" (default) applies IR optimization passes, "direct" keeps the source structure */
  mode?: "optimized" | "direct";
  defines?: Record<string, string>;
  headers?: Record<string, string>;
  /** maximum inlining depth / IR statements, guarding against compile-time resource exhaustion */
  maxInlineDepth?: number;
  maxStatements?: number;
}

const MAX_INLINE_DEPTH = 64;
const MAX_STATEMENTS = 400_000;

// ------------------------------------------------------------------ program-level symbol tables
export interface GlobalSym { decl: VarDecl; t: CType; root?: Root; constVal?: Expr; init?: Uint8Array }

export class Program {
  funcs = new Map<string, FuncDecl[]>();
  structs = new Map<string, StructDecl>();
  structInst = new Map<string, StructDef>();
  typedefs = new Map<string, { type: TypeExpr; template?: TopDecl & { k: "typedef" } }>();
  globals = new Map<string, GlobalSym>();
  enums = new Map<string, { value: number; t: CType }>();
  hostNames = new Set<string>();
  kernels: FuncDecl[] = [];
  /** module-scope memory (__constant__ / __device__ variables): one Root per symbol, shared by all kernels */
  symbolRoots = new Map<string, Root>();
  private nextRootId = 1_000_000;

  constructor(readonly tu: TranslationUnit, readonly sink: DiagnosticSink) {
    for (const d of tu.decls) {
      switch (d.k) {
        case "func": {
          const list = this.funcs.get(d.name) ?? [];
          // a prototype followed by its definition: keep the definition
          const proto = list.findIndex((f) => !f.body && sameSignature(f, d));
          if (proto >= 0 && d.body) list.splice(proto, 1);
          if (d.body || !list.some((f) => sameSignature(f, d))) list.push(d);
          this.funcs.set(d.name, list);
          if (d.global && d.body) this.kernels.push(d);
          break;
        }
        case "struct": this.structs.set(d.name, d); break;
        case "typedef": this.typedefs.set(d.name, { type: d.type, template: d.template ? d : undefined }); break;
        case "enum": {
          let v = 0;
          const ctx = new Lowering(this, null, {});
          for (const it of d.items) {
            if (it.value) v = Number(ctx.constInt(it.value, "enumerator value"));
            this.enums.set(it.name, { value: v, t: I32 });
            v++;
          }
          break;
        }
        case "var": this.globals.set(d.decl.name, { decl: d.decl, t: VOID }); break;
        case "host": this.hostNames.add(d.name); break;
        case "static_assert": break;
      }
    }
  }
  newRootId(): number { return this.nextRootId++; }
}

function sameSignature(a: FuncDecl, b: FuncDecl): boolean {
  return a.params.length === b.params.length && a.memberOf === b.memberOf && JSON.stringify(a.params.map((p) => stripLoc(p.type))) === JSON.stringify(b.params.map((p) => stripLoc(p.type)));
}
function stripLoc(t: unknown): unknown {
  return JSON.parse(JSON.stringify(t, (k, v) => (k === "loc" ? undefined : v)));
}

// ------------------------------------------------------------------ control-flow frames
interface Frame {
  kind: "loop" | "switch" | "func";
  /** jump codes that must be re-dispatched after this frame's loop closes */
  exits: Map<number, () => void>;
  flag: Var | null;
  /** func frames */
  retVar?: Var | null;
  retLV?: LV | null;
  retType?: CType;
  hasWrapper?: boolean;
  isKernel?: boolean;
}

interface FuncCtx {
  decl: FuncDecl;
  env: Map<string, Binding>;
  thisLV?: LV;
  struct?: StructDef;
  frame: Frame;
}

export interface KernelSpec { block: [number, number, number]; dynamicShared: number }

// ------------------------------------------------------------------ the lowering context
export class Lowering {
  vars: Var[] = [];
  roots: Root[] = [];
  params: KernelParam[] = [];
  features = new Set<string>();
  private out: Stmt[][] = [[]];
  private scopes: Map<string, Binding>[] = [new Map()];
  private frames: Frame[] = [];
  private funcStack: FuncCtx[] = [];
  private inlineDepth = 0;
  private stmtCount = 0;
  private nextVar = 1;
  private nextRoot = 1;
  private jumpCodes = 0;
  sharedBytes = 0;
  readonly maxInlineDepth: number;
  readonly maxStatements: number;

  constructor(readonly prog: Program, readonly spec: KernelSpec | null, readonly opts: CompileOptions) {
    this.maxInlineDepth = opts.maxInlineDepth ?? MAX_INLINE_DEPTH;
    this.maxStatements = opts.maxStatements ?? MAX_STATEMENTS;
  }

  get sink(): DiagnosticSink { return this.prog.sink; }

  // ------------------------------------------------ emission
  emit(s: Stmt): void {
    if (++this.stmtCount > this.maxStatements) fail("limit", `kernel exceeds ${this.maxStatements} IR statements after inlining/unrolling`);
    this.out[this.out.length - 1].push(s);
  }
  block(f: () => void): Stmt[] {
    this.out.push([]);
    this.scopes.push(new Map());
    try { f(); } finally { this.scopes.pop(); }
    return this.out.pop()!;
  }
  newVar(name: string, t: VT): Var {
    const v: Var = { id: this.nextVar++, name: `${sanitize(name)}_${this.nextVar - 1}`, t };
    this.vars.push(v);
    return v;
  }
  tmp(e: Expr, name = "t"): Expr {
    if (e.op === "const" || e.op === "var" || e.op === "param" || e.op === "builtin") return e;
    const v = this.newVar(name, e.t);
    this.emit({ s: "let", v, e });
    return readVar(v);
  }
  newRoot(name: string, space: Space, bytes: number, elem: VT | null, extra: Partial<Root> = {}): Root {
    const r: Root = { id: this.nextRoot++, name: `${sanitize(name)}${space === "global" ? "" : "_" + space}`, space, bytes, elem, align: 16, ...extra };
    this.roots.push(r);
    return r;
  }

  // ------------------------------------------------ scopes
  bind(name: string, b: Binding): void { this.scopes[this.scopes.length - 1].set(name, b); }
  pushScope(): void { this.scopes.push(new Map()); }
  popScope(): void { this.scopes.pop(); }
  private lookup(name: string): Binding | undefined {
    for (let i = this.scopes.length - 1; i >= 0; i--) {
      const b = this.scopes[i].get(name);
      if (b) return b;
      // function boundaries: inlined bodies must not see the caller's locals
      if (this.scopes[i].has("\0func")) break;
    }
    const fc = this.funcStack[this.funcStack.length - 1];
    if (fc) {
      const b = fc.env.get(name);
      if (b) return b;
    }
    return undefined;
  }

  // ------------------------------------------------ types
  resolveType(t: TypeExpr, env?: Map<string, Binding>): CType {
    switch (t.k) {
      case "prim": return t.name === "void" ? VOID : sc(PRIM_SCALAR[t.name], t.name);
      case "ptr": return { k: "ptr", to: this.resolveType(t.to, env), isConst: Boolean((t.to as { isConst?: boolean }).isConst) };
      case "ref": return this.resolveType(t.to, env); // references are handled by callers (params / locals)
      case "array": {
        const of = this.resolveType(t.of, env);
        const n = t.size === null ? -1 : Number(this.constInt(t.size, "array size", env));
        if (t.size !== null && n <= 0) fail("type", `array size must be positive (got ${n})`, t.loc);
        return { k: "array", of, n };
      }
      case "auto": fail("type", "'auto' is only allowed with an initializer", t.loc);
      case "decltype": return this.withoutEffects(() => this.typeOfVal(this.rvalue(this.expr(t.expr))));
      case "named": {
        const name = t.name;
        const b = env?.get(name) ?? this.lookup(name);
        if (b?.k === "type") return b.t;
        if (name in NAMED_SCALARS) return sc(NAMED_SCALARS[name], name);
        const v = VECTOR_TYPES.get(name);
        if (v) return v.n === 1 ? { k: "vec", s: v.s, n: 1, name } : { k: "vec", s: v.s, n: v.n, name };
        if (name === "curandState" || name === "curandState_t" || name === "curandStateXORWOW" || name === "curandStateXORWOW_t")
          this.sink.warn("unsupported", `${name} (NVIDIA's XORWOW generator) is mapped to curandStatePhilox4_32_10_t: same API and statistical quality, but a different random sequence than on NVIDIA GPUs`, t.loc);
        const td = this.prog.typedefs.get(name);
        if (td) {
          if (td.template?.template) {
            const tenv = this.templateEnv(td.template.template, t.args ?? [], t.loc, env);
            return this.resolveType(td.type, tenv);
          }
          return this.resolveType(td.type, env);
        }
        const sd = this.prog.structs.get(name);
        if (sd) return { k: "struct", def: this.instantiateStruct(sd, t.args ?? [], t.loc, env) };
        if (name.includes("::")) {
          const [outer, inner] = name.split("::");
          const osd = this.prog.structs.get(outer);
          if (osd) fail("unsupported", `nested type ${inner} of ${outer} is not supported`, t.loc);
        }
        fail("name", `unknown type '${name}'`, t.loc);
      }
    }
  }

  templateEnv(params: { kind: "type" | "value"; name: string; valueType?: TypeExpr; def?: TemplateArg }[], args: TemplateArg[], loc: SourceLoc, outer?: Map<string, Binding>): Map<string, Binding> {
    const env = new Map<string, Binding>();
    if (args.length > params.length) fail("type", `too many template arguments (${args.length} for ${params.length})`, loc);
    params.forEach((p, i) => {
      const a = args[i] ?? p.def;
      if (!a) fail("type", `missing template argument for '${p.name}'`, loc);
      if (p.kind === "type") {
        if (a.k !== "type") {
          // an identifier that names a type in the caller's scope was parsed as an expression
          if (a.expr.k === "ident") { env.set(p.name, { k: "type", t: this.resolveType({ k: "named", name: a.expr.name, loc }, outer) }); return; }
          fail("type", `template parameter '${p.name}' expects a type`, loc);
        }
        env.set(p.name, { k: "type", t: this.resolveType(a.type, i < args.length ? outer : env) });
      } else {
        const vtT = p.valueType ? this.resolveType(p.valueType, env) : I32;
        if (a.k !== "expr") fail("type", `template parameter '${p.name}' expects a value`, loc);
        const saved = this.funcStack;
        const e = this.constValue(a.expr, `template argument '${p.name}'`, i < args.length ? undefined : env);
        this.funcStack = saved;
        env.set(p.name, { k: "const", e: this.convertConstTo(e, vtT, loc), t: vtT });
      }
    });
    return env;
  }

  private instantiateStruct(sd: StructDecl, args: TemplateArg[], loc: SourceLoc, outer?: Map<string, Binding>): StructDef {
    const env = sd.template ? this.templateEnv(sd.template, args, loc, outer) : new Map<string, Binding>();
    const key = sd.name + "<" + [...env.entries()].map(([k, b]) => `${k}=${b.k === "type" ? typeName(b.t) : b.k === "const" ? String((b.e as { v: unknown }).v) : "?"}`).join(",") + ">";
    const have = this.prog.structInst.get(key);
    if (have) return have;
    // placeholder for self-referential pointers
    const def: StructDef = layoutStruct(sd.template ? key : sd.name, [], Boolean(sd.isUnion), sd.align);
    this.prog.structInst.set(key, def);
    const fields: { name: string; type: CType }[] = [];
    for (const f of sd.fields) {
      if ((f as { isStatic?: boolean }).isStatic) {
        const t = this.resolveType(f.type, env);
        if (!f.init) fail("unsupported", `static member '${f.name}' needs a constant initializer`, f.loc);
        const saved = this.funcStack;
        this.funcStack = [{ decl: sd as unknown as FuncDecl, env, frame: { kind: "func", exits: new Map(), flag: null } }];
        const e = this.constValue(f.init, `static member ${f.name}`);
        this.funcStack = saved;
        def.statics.set(f.name, { type: t, value: (this.convertConstTo(e, t, f.loc) as { v: number | bigint }).v });
        continue;
      }
      fields.push({ name: f.name, type: this.resolveType(f.type, env) });
    }
    const laid = layoutStruct(def.name, fields, Boolean(sd.isUnion), sd.align);
    Object.assign(def, { fields: laid.fields, size: laid.size, align: laid.align });
    for (const m of sd.methods) {
      const list = (def.methods.get(m.name) ?? []) as FuncDecl[];
      list.push(m);
      def.methods.set(m.name, list);
    }
    def.env = { env, decl: sd };
    return def;
  }

  // ------------------------------------------------ constant evaluation
  /** Evaluate an expression that must be a compile-time constant (array sizes, template arguments, case labels). */
  constValue(e: AExpr, what: string, env?: Map<string, Binding>): Expr {
    const res = this.withoutEffects(() => {
      if (env) {
        this.funcStack.push({ decl: { name: "<const>" } as FuncDecl, env, frame: { kind: "func", exits: new Map(), flag: null } });
        try { return this.rvalue(this.expr(e)); } finally { this.funcStack.pop(); }
      }
      return this.rvalue(this.expr(e));
    });
    if (res.k !== "rv") fail("type", `${what} must be an arithmetic constant`, e.loc);
    const f = foldExpr(res.e, true);
    if (f.op !== "const") fail("type", `${what} must be a compile-time constant`, e.loc);
    return f;
  }
  constInt(e: AExpr, what: string, env?: Map<string, Binding>): bigint {
    const c = this.constValue(e, what, env) as Expr & { op: "const" };
    if (isFloat(c.t.s)) fail("type", `${what} must be an integer`, e.loc);
    return BigInt(c.v as number | bigint);
  }
  private convertConstTo(e: Expr, t: CType, loc: SourceLoc): Expr {
    if (t.k !== "scalar") fail("type", "non-scalar template value parameters are not supported", loc);
    return foldExpr(cast(vt(t.s), e), true);
  }
  /** Run f without emitting any statements; fail if it would have side effects. */
  withoutEffects<T>(f: () => T): T {
    this.out.push([]);
    try {
      const r = f();
      const emitted = this.out[this.out.length - 1];
      if (emitted.some((s) => s.s !== "let")) fail("type", "constant expression has side effects");
      // allow pure temporaries by substituting them
      return r;
    } finally { this.out.pop(); }
  }

  // ------------------------------------------------ value helpers
  rvalue(v: Val): Val {
    if (v.k !== "lv") return v;
    const lv = v.lv;
    switch (lv.k) {
      case "var": return { k: "rv", e: readVar(lv.v), t: lv.t };
      case "lane": return { k: "rv", e: { op: "extract", t: vt((lv.base.t as { s: Scalar }).s), a: readVar(lv.base.v), lane: lv.lane }, t: lv.t };
      case "mem": return this.loadMem(lv.root, lv.off, lv.t);
      case "agg": {
        const fields = new Map<string, Val>();
        for (const [n, f] of lv.fields) fields.set(n, this.rvalue({ k: "lv", lv: f }));
        return { k: "agg", t: lv.t, fields };
      }
      case "ptrvar": {
        if (lv.b.reg) return { k: "ptr", root: null, off: i32(0), t: lv.b.t, reg: lv.b.reg };
        if (!lv.b.root && !lv.b.isParam) fail("unsupported", `pointer '${lv.b.name}' is used before it is assigned (or is null); every pointer must point into one statically known buffer`);
        return { k: "ptr", root: lv.b.root, off: readVar(lv.b.offVar), t: lv.b.t };
      }
      case "builtin": {
        const n = lv.name;
        return { k: "rv", e: { op: "vec", t: vt("u32", 3), items: ["x", "y", "z"].map((c) => this.builtinComponent(n, c)) }, t: lv.t };
      }
    }
  }

  builtinComponent(name: string, c: string): Expr {
    const map: Record<string, string> = { threadIdx: "tid", blockIdx: "bid", blockDim: "bdim", gridDim: "gdim" };
    const idx = "xyz".indexOf(c);
    if (name === "blockDim" && this.spec && this.spec.block[0] > 0) return cnst(T_U32, this.spec.block[idx]);
    if (name === "threadIdx" && this.spec && this.spec.block[idx] === 1) return cnst(T_U32, 0);
    this.features.add(`builtin:${name}`);
    return { op: "builtin", t: T_U32, name: `${map[name]}.${c}` as Builtin };
  }

  loadMem(root: Root, off: Expr, t: CType): Val {
    switch (t.k) {
      case "scalar": case "vec": return { k: "rv", e: { op: "load", t: this.vtOf(t), root, off: foldExpr(off, true) }, t };
      case "struct": {
        const fields = new Map<string, Val>();
        for (const f of t.def.fields) fields.set(f.name, this.loadMem(root, this.addOff(off, f.offset), f.type));
        return { k: "agg", t, fields };
      }
      case "array": return { k: "ptr", root, off, t: { k: "ptr", to: t.of } };
      case "ptr": fail("unsupported", `pointers stored in memory (here: a ${typeName(t)} in ${root.name}) are not supported; pass buffers as kernel arguments`);
      default: fail("type", `cannot load a value of type ${typeName(t)}`);
    }
  }

  addOff(off: Expr, n: number): Expr { return n === 0 ? off : foldExpr(bin("+", off, i32(n)), true); }

  vtOf(t: CType): VT {
    if (t.k === "scalar") return vt(t.s);
    if (t.k === "vec") return vt(t.s, t.n);
    fail("type", `expected a scalar or vector type, got ${typeName(t)}`);
  }

  /** Assign a value to an lvalue (with implicit conversion to the lvalue's type). */
  assign(lv: LV, v: Val, loc: SourceLoc): void {
    switch (lv.k) {
      case "var": this.emit({ s: "let", v: lv.v, e: this.convertExpr(v, lv.t, loc) }); return;
      case "lane": {
        const base = lv.base;
        const bt = base.t as CType & { k: "vec" };
        const items: Expr[] = [];
        for (let i = 0; i < bt.n; i++) items.push(i === lv.lane ? this.convertExpr(v, lv.t, loc) : { op: "extract", t: vt(bt.s), a: readVar(base.v), lane: i });
        this.emit({ s: "let", v: base.v, e: { op: "vec", t: base.v.t, items } });
        return;
      }
      case "mem": {
        const t = lv.t;
        if (t.k === "scalar" || t.k === "vec") {
          this.emit({ s: "store", root: lv.root, off: foldExpr(lv.off, true), t: this.vtOf(t), e: this.convertExpr(v, t, loc) });
          return;
        }
        if (t.k === "struct") {
          const src = this.convertVal(v, t, loc) as Val & { k: "agg" };
          for (const f of t.def.fields) this.assign({ k: "mem", root: lv.root, off: this.addOff(lv.off, f.offset), t: f.type }, src.fields.get(f.name)!, loc);
          return;
        }
        if (t.k === "array") {
          // array assignment only arises from struct copies with array members
          const src = this.rvalue(v);
          if (src.k !== "ptr" || !src.root) fail("type", "cannot assign to an array", loc);
          const es = sizeOf(t.of);
          for (let i = 0; i < t.n; i++)
            this.assign({ k: "mem", root: lv.root, off: this.addOff(lv.off, i * es), t: t.of }, this.loadMem(src.root, this.addOff(src.off, i * es), t.of), loc);
          return;
        }
        fail("unsupported", `storing a ${typeName(t)} in memory is not supported`, loc);
      }
      case "agg": {
        const src = this.convertVal(v, lv.t, loc) as Val & { k: "agg" };
        for (const [n, f] of lv.fields) this.assign(f, src.fields.get(n)!, loc);
        return;
      }
      case "ptrvar": {
        const p = this.convertVal(v, lv.b.t, loc) as Val & { k: "ptr" };
        if (p.reg) {
          if (lv.b.root || (lv.b.reg && lv.b.reg !== p.reg)) fail("unsupported", `pointer '${lv.b.name}' would point to different objects`, loc);
          lv.b.reg = p.reg; return;
        }
        if (p.root === null) { if (!lv.b.root) fail("unsupported", `null pointer assigned to '${lv.b.name}': pointers must point into one statically known buffer`, loc); }
        else if (lv.b.root && lv.b.root !== p.root)
          fail("unsupported", `pointer '${lv.b.name}' may point into two different buffers ('${lv.b.root.name}' and '${p.root.name}'); this compiler needs each pointer's target buffer to be known at compile time`, loc);
        else lv.b.root = p.root;
        this.emit({ s: "let", v: lv.b.offVar, e: p.off });
        return;
      }
      case "builtin": fail("type", `${lv.name} is read-only`, loc);
    }
  }

  typeOfVal(v: Val): CType {
    switch (v.k) {
      case "rv": case "ptr": case "agg": return v.t;
      case "lv": {
        const lv = v.lv;
        return lv.k === "ptrvar" ? lv.b.t : lv.t;
      }
      case "str": return { k: "ptr", to: sc("i8", "char"), isConst: true };
      default: return VOID;
    }
  }

  /** Implicit conversion to t, returning a Val. */
  convertVal(v0: Val, t: CType, loc: SourceLoc, explicit = false): Val {
    let v = this.rvalue(v0);
    if (v.k === "void") fail("type", `cannot use a void value as ${typeName(t)}`, loc);
    if (t.k === "void") return { k: "void" };
    if (t.k === "struct") {
      if (v.k === "agg" && v.t.def === t.def) return v;
      fail("type", `cannot convert ${typeName(this.typeOfVal(v))} to ${typeName(t)}`, loc);
    }
    if (t.k === "ptr") {
      if (v.k === "ptr") {
        if (!explicit && !sameType(v.t.to, t.to) && v.t.to.k !== "void" && t.to.k !== "void" && v.root)
          fail("type", `cannot implicitly convert ${typeName(v.t)} to ${typeName(t)} (use a cast)`, loc);
        return this.repoint(v, t);
      }
      if (v.k === "rv" && v.e.op === "const" && !isFloat(v.t.k === "scalar" ? v.t.s : "f32") && Number(v.e.v) === 0) return { k: "ptr", root: null, off: i32(0), t };
      if (v.k === "rv" && explicit) fail("unsupported", "casting an integer to a pointer is not supported", loc);
      fail("type", `cannot convert ${typeName(this.typeOfVal(v))} to ${typeName(t)}`, loc);
    }
    if (t.k === "array") fail("type", "cannot convert to an array type", loc);
    if (v.k === "ptr") {
      if (t.k === "scalar" && t.s === "bool") return { k: "rv", e: bool(v.root !== null || v.reg !== undefined), t: BOOL };
      fail("unsupported", `converting a pointer to ${typeName(t)} is not supported`, loc);
    }
    if (v.k !== "rv") fail("type", `cannot convert ${typeName(this.typeOfVal(v))} to ${typeName(t)}`, loc);
    return { k: "rv", e: this.convertScalarExpr(v.e, v.t, t, loc, explicit), t };
  }

  /** pointer cast: off stays a byte offset, only the pointee type changes */
  private repoint(v: Val & { k: "ptr" }, t: PtrT): Val {
    return { ...v, t };
  }

  convertExpr(v: Val, t: CType, loc: SourceLoc): Expr {
    const r = this.convertVal(v, t, loc);
    if (r.k !== "rv") fail("type", `expected a value of type ${typeName(t)}`, loc);
    return r.e;
  }

  convertScalarExpr(e: Expr, from: CType, to: CType, loc: SourceLoc, explicit = false): Expr {
    if (to.k === "scalar" && from.k === "scalar") {
      if (to.s === "bool" && from.s !== "bool") return this.toBool(e, from);
      return cast(vt(to.s), e);
    }
    if (to.k === "vec" && from.k === "vec") {
      if (from.s === to.s && from.n === to.n) return e;
      if (explicit || from.n === to.n) {
        if (from.n !== to.n) fail("type", `cannot convert ${typeName(from)} to ${typeName(to)}`, loc);
        return { op: "vec", t: vt(to.s, to.n), items: Array.from({ length: to.n }, (_, i) => cast(vt(to.s), { op: "extract", t: vt(from.s), a: e, lane: i })) };
      }
    }
    if (to.k === "vec" && from.k === "scalar" && to.n === 1) return { op: "vec", t: vt(to.s, 1), items: [cast(vt(to.s), e)] };
    if (to.k === "scalar" && from.k === "vec" && from.n === 1) return cast(vt(to.s), { op: "extract", t: vt(from.s), a: e, lane: 0 });
    if (to.k === "vec" && from.k === "scalar" && explicit) {
      fail("type", `cannot convert ${typeName(from)} to ${typeName(to)}; use make_${to.name}(...)`, loc);
    }
    fail("type", `cannot convert ${typeName(from)} to ${typeName(to)}`, loc);
  }

  toBool(e: Expr, t: CType): Expr {
    if (t.k === "scalar" && t.s === "bool") return e;
    if (t.k !== "scalar") fail("type", `a ${typeName(t)} cannot be used as a condition`);
    if (t.s === "f16" || t.s === "bf16") return bin("!=", cast(vt("f32"), e), cnst(vt("f32"), 0));
    return bin("!=", e, cnst(e.t, isFloat(t.s) ? 0 : t.s.endsWith("64") ? 0n : 0));
  }

  cond(e: AExpr): Expr {
    const v = this.rvalue(this.expr(e));
    if (v.k === "ptr") return bool(v.root !== null || v.reg !== undefined);
    if (v.k !== "rv") fail("type", "condition must be a scalar", e.loc);
    return foldExpr(this.toBool(v.e, v.t), this.optimizing);
  }

  get optimizing(): boolean { return (this.opts.mode ?? "optimized") === "optimized"; }

  // ------------------------------------------------ expressions
  expr(e: AExpr): Val {
    switch (e.k) {
      case "num": {
        const n = parseNumber(e.text, e.loc);
        const s = n.type === "float" ? "f32" : n.type === "double" ? "f64" : PRIM_SCALAR[n.type];
        return { k: "rv", e: cnst(vt(s), normalize(s, n.value)), t: sc(s, n.type) };
      }
      case "char": return { k: "rv", e: cnst(vt("i8"), normalize("i8", e.value)), t: sc("i8", "char") };
      case "bool": return { k: "rv", e: bool(e.value), t: BOOL };
      case "str": return { k: "str", s: e.value };
      case "nullptr": return { k: "ptr", root: null, off: i32(0), t: { k: "ptr", to: VOID } };
      case "this": {
        const fc = this.funcStack[this.funcStack.length - 1];
        if (!fc?.thisLV) fail("type", "'this' outside of a member function", e.loc);
        return { k: "ptr", root: null, off: i32(0), t: { k: "ptr", to: fc.thisLV.k === "ptrvar" ? fc.thisLV.b.t : (fc.thisLV as { t: CType }).t }, reg: fc.thisLV };
      }
      case "ident": return this.ident(e);
      case "binary": return this.binary(e.op, e.l, e.r, e.loc);
      case "assign": return this.assignExpr(e.op, e.l, e.r, e.loc);
      case "unary": return this.unary(e);
      case "postfix": return this.incdec(e.e, e.op, false, e.loc);
      case "cond": return this.conditional(e);
      case "comma": { this.discard(this.expr(e.l)); return this.expr(e.r); }
      case "call": return this.call(e);
      case "index": return this.index(e.base, e.index, e.loc);
      case "member": return this.member(e);
      case "cast": return this.castExpr(e);
      case "construct": return this.construct(this.resolveTypeLocal(e.type), e.args, e.loc, e.braces);
      case "sizeof": {
        let t: CType;
        if (e.type) t = this.resolveTypeLocal(e.type);
        else {
          const v = this.withoutEffects(() => this.expr(e.e!));
          t = v.k === "lv" && v.lv.k === "mem" ? v.lv.t : this.typeOfVal(v);
          if (v.k === "lv" && v.lv.k === "var") t = v.lv.t;
        }
        if (t.k === "array" && t.n < 0) fail("type", "sizeof of an unsized array", e.loc);
        return { k: "rv", e: cnst(vt("u64"), BigInt(sizeOf(t))), t: sc("u64", "size_t") };
      }
      case "alignof": return { k: "rv", e: cnst(vt("u64"), BigInt(alignOf(this.resolveTypeLocal(e.type)))), t: sc("u64", "size_t") };
      case "init": fail("type", "braced initializer list is only allowed in a declaration or as a function argument", e.loc);
      case "launch": fail("unsupported", "launching kernels from device code (dynamic parallelism) is not supported", e.loc);
    }
  }

  resolveTypeLocal(t: TypeExpr): CType { return this.resolveType(t); }

  discard(v: Val): void {
    // evaluating for side effects only: loads are dropped (no statement emitted)
    void v;
  }

  private ident(e: AExpr & { k: "ident" }): Val {
    const name = e.name;
    const b = this.lookup(name);
    if (b) {
      if (b.k === "lv") return { k: "lv", lv: b.lv };
      if (b.k === "const") return { k: "rv", e: b.e, t: b.t };
      fail("type", `'${name}' is a type, not a value`, e.loc);
    }
    // struct members inside methods
    const fc = this.funcStack[this.funcStack.length - 1];
    if (fc?.struct) {
      const st = fc.struct.statics.get(name);
      if (st) return { k: "rv", e: cnst(this.vtOf(st.type), st.value), t: st.type };
      if (fc.thisLV && fc.struct.fields.some((f) => f.name === name)) return this.fieldOf({ k: "lv", lv: fc.thisLV }, name, e.loc);
      if (fc.struct.methods.has(name)) return { k: "fn", name, targs: e.targs, recv: fc.thisLV, struct: fc.struct, loc: e.loc };
    }
    switch (name) {
      case "threadIdx": case "blockIdx": case "blockDim": case "gridDim":
        return { k: "lv", lv: { k: "builtin", name, t: { k: "vec", s: "u32", n: 3, name: name === "threadIdx" || name === "blockIdx" ? "uint3" : "dim3" } } };
      case "warpSize": return { k: "rv", e: i32(32), t: I32 };
    }
    const en = this.prog.enums.get(name);
    if (en) return { k: "rv", e: i32(en.value), t: en.t };
    if (name.includes("::")) {
      // Struct<args>::member: static constexpr members and static methods
      const [sname, member] = name.split("::");
      const sd = this.prog.structs.get(sname);
      if (sd) {
        const def = this.instantiateStruct(sd, e.targs ?? [], e.loc);
        const st = def.statics.get(member);
        if (st) return { k: "rv", e: cnst(this.vtOf(st.type), st.value), t: st.type };
        if (def.methods.has(member)) return { k: "fn", name: member, struct: def, loc: e.loc };
        fail("name", `${sname} has no static member '${member}'`, e.loc);
      }
    }
    const g = this.prog.globals.get(name);
    if (g) return this.globalVar(name, g, e.loc);
    if (this.prog.funcs.has(name) || name in BUILTIN_FUNCTIONS || name.startsWith("__builtin_")) return { k: "fn", name, targs: e.targs, loc: e.loc };
    if (this.prog.hostNames.has(name)) fail("type", `'${name}' is a host function or variable and cannot be used in device code`, e.loc);
    fail("name", `use of undeclared identifier '${name}'`, e.loc);
  }

  private globalVar(name: string, g: GlobalSym, loc: SourceLoc): Val {
    const d = g.decl;
    if (g.t.k === "void") g.t = this.resolveType(d.type);
    const t = g.t;
    if (d.space === "shared") {
      // file-scope __shared__: one array per block, like a kernel-local __shared__
      const key = "\0shared:" + name;
      let b = this.scopes[0].get(key);
      if (!b) {
        b = this.declareShared(d, t);
        this.scopes[0].set(key, b);
      }
      return { k: "lv", lv: (b as { lv: LV }).lv };
    }
    if (d.space === "constant" || d.space === "device" || d.space === "managed") {
      let root = this.prog.symbolRoots.get(name);
      if (!root) {
        if (t.k === "array" && t.n < 0) fail("type", `__${d.space}__ array '${name}' needs a size`, loc);
        root = {
          id: this.prog.newRootId(), name: sanitize(name), space: d.space === "constant" ? "constant" : "global", bytes: sizeOf(t),
          elem: elemVT(t), structured: !elemVT(t), symbol: name, align: Math.max(16, alignOf(t)), readonly: d.space === "constant",
        } as Root & { symbol: string };
        if (d.init) root.init = this.constInitBytes(d.init, t, loc);
        this.prog.symbolRoots.set(name, root);
      }
      if (!this.roots.includes(root)) this.roots.push(root);
      this.features.add(d.space === "constant" ? "constant-memory" : "device-variables");
      return { k: "lv", lv: { k: "mem", root, off: i32(0), t } };
    }
    // const / constexpr at file scope
    if (!d.init) fail("type", `file-scope variable '${name}' is host memory; mark it __device__ or __constant__ to use it in device code`, loc);
    if (t.k === "scalar") {
      if (g.constVal === undefined) g.constVal = this.convertConstTo(this.constValue(d.init, `initializer of '${name}'`), t, loc);
      return { k: "rv", e: g.constVal, t };
    }
    // const arrays/structs at file scope: read-only module constant memory
    let root = this.prog.symbolRoots.get(name);
    if (!root) {
      root = { id: this.prog.newRootId(), name: sanitize(name), space: "constant", bytes: sizeOf(t), elem: elemVT(t), structured: !elemVT(t), align: 16, readonly: true } as Root;
      (root as Root & { symbol: string }).symbol = name;
      root.init = this.constInitBytes(d.init, t, loc);
      this.prog.symbolRoots.set(name, root);
    }
    if (!this.roots.includes(root)) this.roots.push(root);
    return { k: "lv", lv: { k: "mem", root, off: i32(0), t } };
  }

  /** Evaluate a constant initializer into bytes (little endian, C layout). */
  constInitBytes(init: AExpr, t: CType, loc: SourceLoc): Uint8Array {
    const buf = new Uint8Array(sizeOf(t));
    const dv = new DataView(buf.buffer);
    const put = (off: number, ty: CType, e: AExpr) => {
      if (ty.k === "array") {
        const items = e.k === "init" ? e.items : null;
        if (!items) fail("type", "array initializer must be a braced list", e.loc);
        if (ty.n >= 0 && items.length > ty.n) fail("type", "too many initializers", e.loc);
        items.forEach((it, i) => put(off + i * sizeOf(ty.of), ty.of, it));
        return;
      }
      if (ty.k === "struct") {
        const items = e.k === "init" ? e.items : null;
        if (!items) fail("type", "struct initializer must be a braced list", e.loc);
        items.forEach((it, i) => { const f = ty.def.fields[i]; if (!f) fail("type", "too many initializers", it.loc); put(off + f.offset, f.type, it); });
        return;
      }
      if (ty.k === "vec") {
        const items = e.k === "init" ? e.items : null;
        if (!items) fail("type", "vector initializer must be a braced list", e.loc);
        items.forEach((it, i) => put(off + i * sizeOf(sc(ty.s)), sc(ty.s), it));
        return;
      }
      if (ty.k !== "scalar") fail("type", `cannot initialize ${typeName(ty)} at file scope`, loc);
      const c = this.convertConstTo(this.constValue(e, "initializer"), ty, e.loc) as Expr & { op: "const" };
      writeScalar(dv, off, ty.s, c.v);
    };
    put(0, t, init);
    return buf;
  }

  fieldOf(base: Val, name: string, loc: SourceLoc): Val {
    if (base.k === "lv") {
      const lv = base.lv;
      if (lv.k === "builtin") {
        if (!"xyz".includes(name) || name.length !== 1) fail("name", `${lv.name} has no member '${name}'`, loc);
        return { k: "rv", e: this.builtinComponent(lv.name, name), t: U32 };
      }
      const t = lv.k === "ptrvar" ? lv.b.t : lv.t;
      if (t.k === "vec") {
        const lane = laneIndex(name, t.n, loc);
        if (lv.k === "var") return { k: "lv", lv: { k: "lane", base: lv, lane, t: sc(t.s) } };
        if (lv.k === "mem") return { k: "lv", lv: { k: "mem", root: lv.root, off: this.addOff(lv.off, lane * sizeOf(sc(t.s))), t: sc(t.s) } };
      }
      if (t.k === "struct") {
        const st = t.def.statics.get(name);
        if (st) return { k: "rv", e: cnst(this.vtOf(st.type), st.value), t: st.type };
        if (t.def.methods.has(name)) return { k: "fn", name, recv: lv, struct: t.def, loc };
        if (lv.k === "agg") {
          const f = lv.fields.get(name);
          if (!f) fail("name", `${t.def.name} has no member '${name}'`, loc);
          return { k: "lv", lv: f };
        }
        if (lv.k === "mem") {
          const f = t.def.fields.find((x) => x.name === name);
          if (!f) fail("name", `${t.def.name} has no member '${name}'`, loc);
          return { k: "lv", lv: { k: "mem", root: lv.root, off: this.addOff(lv.off, f.offset), t: f.type } };
        }
      }
      return this.fieldOf(this.rvalue(base), name, loc);
    }
    if (base.k === "rv" && base.t.k === "vec") {
      const lane = laneIndex(name, base.t.n, loc);
      return { k: "rv", e: { op: "extract", t: vt(base.t.s), a: base.e, lane }, t: sc(base.t.s) };
    }
    if (base.k === "agg") {
      const st = base.t.def.statics.get(name);
      if (st) return { k: "rv", e: cnst(this.vtOf(st.type), st.value), t: st.type };
      const f = base.fields.get(name);
      if (f) return f;
      if (base.t.def.methods.has(name)) {
        // method on an rvalue: materialize it
        const lv = this.materialize(base);
        return { k: "fn", name, recv: lv, struct: base.t.def, loc };
      }
      fail("name", `${base.t.def.name} has no member '${name}'`, loc);
    }
    fail("type", `member access '.${name}' on a non-struct value of type ${typeName(this.typeOfVal(base))}`, loc);
  }

  /** Store an rvalue into fresh variables so it can be used as an lvalue. */
  materialize(v: Val, name = "tmp"): LV {
    const t = this.typeOfVal(v);
    const lv = this.allocLocal(name, t, false);
    this.assign(lv, v, { file: "", line: 0, col: 0 });
    return lv;
  }

  /** Allocate storage for a local object of type t: registers for scalars/vectors/structs, memory for arrays. */
  allocLocal(name: string, t: CType, inMemory: boolean, loc?: SourceLoc): LV {
    if (t.k === "void") fail("type", `variable '${name}' has type void`, loc);
    if (inMemory || t.k === "array" || (t.k === "struct" && structNeedsMemory(t.def))) {
      if (t.k === "array" && t.n < 0) fail("type", `local array '${name}' needs a size`, loc);
      const root = this.newRoot(name, "local", sizeOf(t), elemVT(t), { structured: !elemVT(t), align: Math.max(4, alignOf(t)) });
      this.features.add("local-memory");
      return { k: "mem", root, off: i32(0), t };
    }
    if (t.k === "scalar" || t.k === "vec") return { k: "var", v: this.newVar(name, this.vtOf(t)), t };
    if (t.k === "struct") {
      const fields = new Map<string, LV>();
      for (const f of t.def.fields) fields.set(f.name, this.allocLocal(`${name}_${f.name}`, f.type, false, loc));
      return { k: "agg", t, fields };
    }
    if (t.k === "ptr") return { k: "ptrvar", b: { root: null, offVar: this.newVar(name, T_I32), t, name } };
    fail("type", `cannot declare a local of type ${typeName(t)}`, loc);
  }

  declareShared(d: VarDecl, t: CType): Binding {
    if (d.init) fail("type", `__shared__ variable '${d.name}' cannot have an initializer`, d.loc);
    let root: Root;
    if (d.isExtern || (t.k === "array" && t.n < 0)) {
      if (!(t.k === "array" && t.n < 0)) fail("type", "extern __shared__ must be an unsized array", d.loc);
      // all extern __shared__ arrays alias the same dynamic shared memory
      const existing = this.roots.find((r) => r.dynamic);
      const bytes = this.spec ? this.spec.dynamicShared : 0;
      root = existing ?? this.newRoot("dyn_smem", "shared", bytes, elemVT(t.of), { dynamic: true, structured: !elemVT(t.of) });
      if (existing && existing.elem && (!elemVT(t.of) || !vtEq(existing.elem, elemVT(t.of)!))) existing.elem = null, existing.structured = true;
      this.features.add("dynamic-shared-memory");
      return { k: "lv", lv: { k: "mem", root, off: i32(0), t: { k: "array", of: t.of, n: Math.floor(bytes / Math.max(1, sizeOf(t.of))) } } };
    }
    root = this.newRoot(d.name, "shared", sizeOf(t), elemVT(t), { structured: !elemVT(t), align: Math.max(4, d.align ?? 0, alignOf(t)) });
    this.sharedBytes += sizeOf(t);
    this.features.add("shared-memory");
    return { k: "lv", lv: { k: "mem", root, off: i32(0), t } };
  }

  // ------------------------------------------------ operators
  arithType(a: CType, b: CType, op: string, loc: SourceLoc): CType {
    if (a.k === "scalar" && b.k === "scalar") {
      const s = commonScalar(a.s, b.s);
      if (!s) fail("type", `ambiguous operands for '${op}': ${typeName(a)} and ${typeName(b)} (convert one of them explicitly, e.g. with __half2float)`, loc);
      return sc(s);
    }
    fail("type", `invalid operands to '${op}': ${typeName(a)} and ${typeName(b)}`, loc);
  }

  /** Apply a binary arithmetic/bitwise/comparison operator to two rvalues. */
  binop(op: string, l: Val, r: Val, loc: SourceLoc): Val {
    l = this.rvalue(l); r = this.rvalue(r);
    // pointer arithmetic
    if (l.k === "ptr" || r.k === "ptr") return this.ptrBinop(op, l, r, loc);
    if (l.k === "agg" || r.k === "agg") {
      const ov = this.operatorOverload(op, [l, r], loc);
      if (ov) return ov;
      fail("type", `no operator${op} for ${typeName(this.typeOfVal(l))} and ${typeName(this.typeOfVal(r))}`, loc);
    }
    if (l.k !== "rv" || r.k !== "rv") fail("type", `invalid operands to '${op}'`, loc);
    // vector arithmetic (helper_math.h-style component-wise operators)
    if (l.t.k === "vec" || r.t.k === "vec") return this.vecBinop(op, l as Val & { k: "rv" }, r as Val & { k: "rv" }, loc);
    const lt = l.t as CType & { k: "scalar" }, rt = r.t as CType & { k: "scalar" };
    if (op === "&&" || op === "||") return { k: "rv", e: bin(op, this.toBool(l.e, lt), this.toBool(r.e, rt)), t: BOOL };
    if (op === "<<" || op === ">>") {
      if (!isInt(promote(lt.s)) || !isInt(promote(rt.s)) || isFloat(lt.s) || isFloat(rt.s)) fail("type", `shift of a non-integer (${typeName(lt)} ${op} ${typeName(rt)})`, loc);
      const ps = promote(lt.s);
      return { k: "rv", e: bin(op as "<<", cast(vt(ps), l.e), cast(vt(ps), r.e)), t: sc(ps) };
    }
    const ct = this.arithType(lt, rt, op, loc) as CType & { k: "scalar" };
    if (op === "%" && isFloat(ct.s)) fail("type", "invalid operands to '%' for floating point (use fmodf)", loc);
    if ((op === "&" || op === "|" || op === "^") && isFloat(ct.s)) fail("type", `invalid operands to '${op}' for floating point`, loc);
    // half / bfloat16 arithmetic: compute in float and round once (exactly IEEE half semantics for + - * /)
    const narrow = ct.s === "f16" || ct.s === "bf16";
    const opT = narrow ? vt("f32") : vt(ct.s);
    const a = cast(opT, l.e), b = cast(opT, r.e);
    const isCmp = ["==", "!=", "<", "<=", ">", ">="].includes(op);
    if (narrow) this.features.add(ct.s === "f16" ? "fp16" : "bf16");
    if (isCmp) return { k: "rv", e: bin(op as "==", a, b), t: BOOL };
    let res = bin(op as "+", a, b, opT);
    if (narrow) res = cast(vt(ct.s), res);
    return { k: "rv", e: res, t: ct };
  }

  private vecBinop(op: string, l: Val & { k: "rv" }, r: Val & { k: "rv" }, loc: SourceLoc): Val {
    const vtT = (l.t.k === "vec" ? l.t : r.t) as CType & { k: "vec" };
    if (l.t.k === "vec" && r.t.k === "vec" && (l.t.s !== r.t.s || l.t.n !== r.t.n)) fail("type", `invalid operands to '${op}': ${typeName(l.t)} and ${typeName(r.t)}`, loc);
    if (["==", "!=", "<", "<=", ">", ">=", "&&", "||"].includes(op)) fail("type", `operator '${op}' is not defined for vector type ${vtT.name}`, loc);
    const lane = (v: Val & { k: "rv" }, i: number): Val => v.t.k === "vec" ? { k: "rv", e: { op: "extract", t: vt(v.t.s), a: v.e, lane: i }, t: sc(v.t.s) } : v;
    const items: Expr[] = [];
    for (let i = 0; i < vtT.n; i++) {
      const x = this.binop(op, lane(l, i), lane(r, i), loc) as Val & { k: "rv" };
      items.push(cast(vt(vtT.s), x.e));
    }
    this.features.add("vector-arithmetic");
    return { k: "rv", e: { op: "vec", t: vt(vtT.s, vtT.n), items }, t: vtT };
  }

  private ptrBinop(op: string, l: Val, r: Val, loc: SourceLoc): Val {
    const isCmp = ["==", "!=", "<", "<=", ">", ">="].includes(op);
    if (l.k === "ptr" && r.k === "ptr") {
      if (l.reg || r.reg) {
        if (op === "==" || op === "!=") return { k: "rv", e: bool((l.reg === r.reg) === (op === "==")), t: BOOL };
        fail("unsupported", "arithmetic on pointers to registers is not supported", loc);
      }
      if (op === "-") {
        if (l.root !== r.root) fail("unsupported", "subtracting pointers into different buffers", loc);
        const es = Math.max(1, sizeOf(l.t.to));
        return { k: "rv", e: bin("/", cast(vt("i64"), bin("-", l.off, r.off)), cnst(vt("i64"), BigInt(es))), t: sc("i64", "ptrdiff_t") };
      }
      if (isCmp) {
        if (l.root === null || r.root === null || l.root !== r.root) {
          if (op === "==" || op === "!=") return { k: "rv", e: bool((l.root === r.root && l.root === null) === (op === "==")), t: BOOL };
          fail("unsupported", "comparing pointers into different buffers", loc);
        }
        return { k: "rv", e: bin(op as "<", l.off, r.off), t: BOOL };
      }
      if (op === "&&" || op === "||") return { k: "rv", e: bin(op, bool(l.root !== null), bool(r.root !== null)), t: BOOL };
      fail("type", `invalid operands to '${op}' (two pointers)`, loc);
    }
    const p = (l.k === "ptr" ? l : r) as Val & { k: "ptr" };
    const n = (l.k === "ptr" ? r : l) as Val;
    if (op === "&&" || op === "||") {
      const nb = this.toBool((n as Val & { k: "rv" }).e, (n as Val & { k: "rv" }).t);
      return { k: "rv", e: bin(op, bool(p.root !== null), nb), t: BOOL };
    }
    if (isCmp && n.k === "rv" && n.e.op === "const" && Number(n.e.v) === 0) {
      // p == 0 / p != NULL
      const isNull = p.root === null && !p.reg;
      return { k: "rv", e: bool((op === "==") === isNull), t: BOOL };
    }
    if (n.k !== "rv" || n.t.k !== "scalar" || !isInt(n.t.s)) fail("type", `invalid operands to '${op}' (pointer and ${typeName(this.typeOfVal(n))})`, loc);
    if (op !== "+" && !(op === "-" && l.k === "ptr")) fail("type", `invalid pointer arithmetic '${op}'`, loc);
    if (p.reg) fail("unsupported", "arithmetic on a pointer to a register variable (take the address of an array element instead)", loc);
    if (p.t.to.k === "void") fail("type", "arithmetic on void*", loc);
    const es = sizeOf(p.t.to);
    let delta = foldExpr(bin("*", cast(T_I32, n.e), i32(es)), true);
    if (op === "-") delta = bin("-", i32(0), delta);
    return { k: "ptr", root: p.root, off: foldExpr(bin("+", p.off, delta), true), t: p.t };
  }

  operatorOverload(op: string, args: Val[], loc: SourceLoc): Val | null {
    const name = "operator" + op;
    const first = args[0];
    const t0 = this.typeOfVal(first);
    if (t0.k === "struct" && t0.def.methods.has(name)) {
      const recv = first.k === "lv" ? first.lv : this.materialize(first);
      return this.callFunction({ k: "fn", name, recv, struct: t0.def, loc }, args.slice(1), loc);
    }
    if (this.prog.funcs.has(name)) return this.callFunction({ k: "fn", name, loc }, args, loc);
    return null;
  }

  private binary(op: string, le: AExpr, re: AExpr, loc: SourceLoc): Val {
    if (op === "&&" || op === "||") {
      const l = this.cond(le);
      // the right operand is evaluated only if needed: lower it in a nested block, and use real control flow when it has
      // side effects or memory accesses
      let r: Expr | null = null;
      const body = this.block(() => { r = this.cond(re); });
      if (body.length === 0 && !someLoad(r!)) return { k: "rv", e: foldExpr(bin(op, l, r!), this.optimizing), t: BOOL };
      if (body.length === 0) return { k: "rv", e: bin(op, l, r!), t: BOOL }; // backends short-circuit && and || (loads stay guarded)
      const res = this.newVar(op === "&&" ? "and" : "or", T_BOOL);
      this.emit({ s: "let", v: res, e: l });
      const guarded = [...body, { s: "let", v: res, e: r! } as Stmt];
      this.emit(op === "&&" ? { s: "if", c: readVar(res), then: guarded, else: [] } : { s: "if", c: readVar(res), then: [], else: guarded });
      return { k: "rv", e: readVar(res), t: BOOL };
    }
    const l = this.rvalue(this.expr(le));
    const lSnap = this.snapshot(l);
    const r = this.rvalue(this.expr(re));
    return this.binop(op, lSnap, r, loc);
  }

  /** If evaluating the next operand may emit statements, freeze this operand's value into a temporary first. */
  snapshot(v: Val): Val {
    if (v.k === "rv" && v.e.op !== "const" && v.e.op !== "param" && v.e.op !== "builtin") return { k: "rv", e: this.tmp(v.e), t: v.t };
    if (v.k === "ptr" && v.off.op !== "const") return { ...v, off: this.tmp(v.off, "off") };
    return v;
  }

  private assignExpr(op: string, le: AExpr, re: AExpr, loc: SourceLoc): Val {
    const target = this.expr(le);
    if (target.k !== "lv") fail("type", "expression is not assignable", loc);
    const lv = this.stableLV(target.lv);
    if (lv.k === "agg" || (lv.k === "mem" && lv.t.k === "struct")) {
      if (op !== "=") {
        const ov = this.operatorOverload(op, [{ k: "lv", lv }, this.expr(re)], loc);
        if (ov) return ov;
        fail("type", `no operator${op} for ${typeName((lv as { t: CType }).t)}`, loc);
      }
      const rv = re.k === "init" ? this.initValue((lv as { t: CType }).t, re) : this.expr(re);
      this.assign(lv, rv, loc);
      return { k: "lv", lv };
    }
    let v: Val = re.k === "init" ? this.initValue(lvType(lv), re) : this.expr(re);
    if (op !== "=") {
      const bop = op.slice(0, -1);
      const cur = this.rvalue({ k: "lv", lv });
      v = this.binop(bop, cur, v, loc);
    }
    this.assign(lv, v, loc);
    return { k: "lv", lv };
  }

  /** Freeze an lvalue's address so later side effects cannot change which object it designates. */
  stableLV(lv: LV): LV {
    if (lv.k === "mem" && lv.off.op !== "const" && lv.off.op !== "var") return { ...lv, off: this.tmp(lv.off, "addr") };
    return lv;
  }

  private unary(e: AExpr & { k: "unary" }): Val {
    switch (e.op) {
      case "++": case "--": return this.incdec(e.e, e.op, true, e.loc);
      case "&": return this.addressOf(e.e, e.loc);
      case "*": {
        const p = this.rvalue(this.expr(e.e));
        if (p.k !== "ptr") fail("type", `indirection requires a pointer (got ${typeName(this.typeOfVal(p))})`, e.loc);
        return this.deref(p, e.loc);
      }
    }
    const v = this.rvalue(this.expr(e.e));
    if (v.k === "agg") {
      const ov = this.operatorOverload(e.op, [v], e.loc);
      if (ov) return ov;
    }
    if (v.k === "ptr" && e.op === "!") return { k: "rv", e: bool(v.root === null && !v.reg), t: BOOL };
    if (v.k !== "rv") fail("type", `invalid operand to unary '${e.op}'`, e.loc);
    if (v.t.k === "vec") {
      if (e.op !== "-" && e.op !== "+") fail("type", `unary '${e.op}' is not defined for ${typeName(v.t)}`, e.loc);
      if (e.op === "+") return v;
      const t = v.t;
      return { k: "rv", e: { op: "vec", t: vt(t.s, t.n), items: Array.from({ length: t.n }, (_, i) => this.negate({ op: "extract", t: vt(t.s), a: v.e, lane: i }, t.s)) }, t };
    }
    const t = v.t as CType & { k: "scalar" };
    if (e.op === "!") return { k: "rv", e: { op: "unop", fn: "!", t: T_BOOL, a: this.toBool(v.e, t) }, t: BOOL };
    if (e.op === "+") { const ps = isFloat(t.s) ? t.s : promote(t.s); return { k: "rv", e: cast(vt(ps), v.e), t: sc(ps) }; }
    if (e.op === "-") { const ps = isFloat(t.s) ? t.s : promote(t.s); return { k: "rv", e: this.negate(cast(vt(ps), v.e), ps), t: sc(ps) }; }
    if (isFloat(t.s)) fail("type", "'~' applied to a floating point value", e.loc);
    const ps = promote(t.s);
    return { k: "rv", e: { op: "unop", fn: "~", t: vt(ps), a: cast(vt(ps), v.e) }, t: sc(ps) };
  }

  negate(e: Expr, s: Scalar): Expr {
    if (s === "f16" || s === "bf16") return cast(vt(s), { op: "unop", fn: "-", t: vt("f32"), a: cast(vt("f32"), e) });
    return foldExpr({ op: "unop", fn: "-", t: e.t, a: e }, true);
  }

  private incdec(target: AExpr, op: "++" | "--", prefix: boolean, loc: SourceLoc): Val {
    const v = this.expr(target);
    if (v.k !== "lv") fail("type", `operand of '${op}' is not assignable`, loc);
    const lv = this.stableLV(v.lv);
    const old = this.rvalue({ k: "lv", lv });
    const oldSnap = prefix ? old : this.snapshot(old);
    if (oldSnap.k === "agg") {
      const ov = this.operatorOverload(op, [{ k: "lv", lv }], loc);
      if (ov) return ov;
      fail("type", `no operator${op} for this type`, loc);
    }
    const one: Val = { k: "rv", e: i32(1), t: I32 };
    const nv = this.binop(op === "++" ? "+" : "-", oldSnap, one, loc);
    this.assign(lv, nv, loc);
    return prefix ? { k: "lv", lv } : oldSnap;
  }

  private addressOf(e: AExpr, loc: SourceLoc): Val {
    const v = this.expr(e);
    if (v.k === "fn") fail("unsupported", "function pointers are not supported", loc);
    if (v.k !== "lv") fail("type", "cannot take the address of an rvalue", loc);
    const lv = v.lv;
    const t = lvType(lv);
    if (lv.k === "mem") return { k: "ptr", root: lv.root, off: lv.off, t: { k: "ptr", to: t } };
    if (lv.k === "ptrvar") fail("unsupported", "pointers to pointers are not supported", loc);
    if (lv.k === "builtin") fail("type", `cannot take the address of ${lv.name}`, loc);
    // address of a register variable / struct: usable for passing to inlined functions (out-parameters) and *p
    return { k: "ptr", root: null, off: i32(0), t: { k: "ptr", to: t }, reg: lv };
  }

  deref(p: Val & { k: "ptr" }, loc: SourceLoc, extra?: Expr): Val {
    if (p.t.to.k === "void") fail("type", "dereferencing a void pointer", loc);
    if (p.reg) {
      if (extra && !(extra.op === "const" && Number(extra.v) === 0)) fail("unsupported", "indexing a pointer to a register variable", loc);
      return { k: "lv", lv: p.reg };
    }
    if (!p.root) fail("type", "dereferencing a null pointer", loc);
    const off = extra ? foldExpr(bin("+", p.off, extra), true) : p.off;
    return { k: "lv", lv: { k: "mem", root: p.root, off, t: p.t.to } };
  }

  private index(be: AExpr, ie: AExpr, loc: SourceLoc): Val {
    const base = this.expr(be);
    // struct operator[]
    if (base.k === "lv" && (base.lv.k === "agg" || (base.lv.k === "mem" && base.lv.t.k === "struct")) || base.k === "agg") {
      const ov = this.operatorOverload("[]", [base, this.expr(ie)], loc);
      if (ov) return ov;
      fail("type", "subscripted value is not an array or pointer", loc);
    }
    // vector component by index (v[i] with constant i)
    const bt = base.k === "lv" ? lvType(base.lv) : this.typeOfVal(base);
    if (bt.k === "vec") {
      const c = this.constValue(ie, "vector component index");
      return this.fieldOf(base, "xyzw"[Number((c as { v: number }).v)], loc);
    }
    let p = this.rvalue(base);
    const bSnap = this.snapshot(p);
    let idx = this.rvalue(this.expr(ie));
    // i[a] is a[i] in C
    if (bSnap.k !== "ptr" && idx.k === "ptr") { const t = idx; idx = bSnap; p = t; } else p = bSnap;
    if (p.k !== "ptr") fail("type", `subscripted value is not an array or pointer (got ${typeName(this.typeOfVal(p))})`, loc);
    if (idx.k !== "rv" || idx.t.k !== "scalar" || (!isInt(idx.t.s) && idx.t.s !== "bool")) fail("type", "array subscript is not an integer", loc);
    if (p.reg) return this.deref(p, loc, foldExpr(cast(T_I32, idx.e), true));
    const es = sizeOf(p.t.to);
    return this.deref(p, loc, foldExpr(bin("*", cast(T_I32, idx.e), i32(es)), true));
  }

  private member(e: AExpr & { k: "member" }): Val {
    let base = this.expr(e.base);
    if (e.arrow) {
      const p = this.rvalue(base);
      if (p.k !== "ptr") fail("type", `'->' applied to a non-pointer (${typeName(this.typeOfVal(p))})`, e.loc);
      base = this.deref(p, e.loc);
    }
    return this.fieldOf(base, e.name, e.loc);
  }

  private castExpr(e: AExpr & { k: "cast" }): Val {
    const t = this.resolveTypeLocal(e.type);
    if (e.e.k === "init") return this.initValue(t, e.e);
    const v = this.expr(e.e);
    if (t.k === "struct") {
      const r = this.rvalue(v);
      if (r.k === "agg" && r.t.def === t.def) return r;
      return this.construct(t, [e.e], e.loc, false);
    }
    if (t.k === "vec" && this.rvalue(v).k === "rv" && (this.typeOfVal(v).k === "scalar")) {
      fail("type", `cannot convert ${typeName(this.typeOfVal(v))} to ${typeName(t)} (use make_${t.name})`, e.loc);
    }
    if (t.k === "ptr" && e.style === "static") {
      const r = this.rvalue(v);
      if (r.k === "ptr" && !sameType(r.t.to, t.to) && r.t.to.k !== "void" && t.to.k !== "void") fail("type", `static_cast from ${typeName(r.t)} to ${typeName(t)} is not allowed (use reinterpret_cast)`, e.loc);
    }
    return this.convertVal(v, t, e.loc, true);
  }

  private conditional(e: AExpr & { k: "cond" }): Val {
    const c = this.cond(e.c);
    if (c.op === "const") return this.rvalue(this.expr(c.v ? e.t : e.f));
    // lower both arms in their own blocks; use select only when both are pure, load-free and of one value type
    let tv: Val | null = null, fv: Val | null = null;
    const tb = this.block(() => { tv = this.rvalue(this.expr(e.t)); });
    const fb = this.block(() => { fv = this.rvalue(this.expr(e.f)); });
    const T = tv! as Val, F = fv! as Val;
    if (T.k === "void" && F.k === "void") {
      this.emit({ s: "if", c, then: tb, else: fb });
      return { k: "void" };
    }
    if (T.k === "ptr" || F.k === "ptr") {
      const tp = T as Val & { k: "ptr" }, fp = F as Val & { k: "ptr" };
      if (T.k !== "ptr" || F.k !== "ptr") fail("type", "conditional operator with a pointer and a non-pointer", e.loc);
      if (tp.reg || fp.reg) {
        if (tp.reg !== fp.reg) fail("unsupported", "conditional choice between pointers to different variables", e.loc);
        return tp;
      }
      if (tp.root !== fp.root && tp.root && fp.root) fail("unsupported", `conditional choice between pointers into different buffers ('${tp.root.name}' and '${fp.root.name}') is not supported`, e.loc);
      const root = tp.root ?? fp.root;
      const off = this.newVar("poff", T_I32);
      this.emit({ s: "if", c, then: [...tb, { s: "let", v: off, e: tp.off }], else: [...fb, { s: "let", v: off, e: fp.off }] });
      return { k: "ptr", root, off: readVar(off), t: tp.t };
    }
    let rt: CType;
    const tt = this.typeOfVal(T), ft = this.typeOfVal(F);
    if (tt.k === "scalar" && ft.k === "scalar") rt = tt.s === ft.s ? tt : this.arithType(tt, ft, "?:", e.loc);
    else if (sameType(tt, ft)) rt = tt;
    else fail("type", `incompatible operand types in conditional: ${typeName(tt)} and ${typeName(ft)}`, e.loc);
    if (rt.k === "struct") {
      const lv = this.allocLocal("cond", rt, false);
      this.emit({ s: "if", c, then: this.block(() => { this.out[this.out.length - 1].push(...tb); this.assign(lv, T, e.loc); }), else: this.block(() => { this.out[this.out.length - 1].push(...fb); this.assign(lv, F, e.loc); }) });
      return this.rvalue({ k: "lv", lv });
    }
    const te = this.convertExpr(T, rt, e.loc), fe = this.convertExpr(F, rt, e.loc);
    if (tb.length === 0 && fb.length === 0 && !someLoad(te) && !someLoad(fe)) return { k: "rv", e: foldExpr({ op: "select", t: te.t, c, a: te, b: fe }, this.optimizing), t: rt };
    const res = this.newVar("sel", te.t);
    this.emit({ s: "if", c, then: [...tb, { s: "let", v: res, e: te }], else: [...fb, { s: "let", v: res, e: fe }] });
    return { k: "rv", e: readVar(res), t: rt };
  }

  /** Value of a braced initializer for type t. */
  initValue(t: CType, init: AExpr & { k: "init" }): Val {
    if (t.k === "scalar") {
      if (init.items.length > 1) fail("type", "too many initializers for a scalar", init.loc);
      return init.items.length ? this.convertVal(this.expr(init.items[0]), t, init.loc) : { k: "rv", e: cnst(vt(t.s), normalize(t.s, 0)), t };
    }
    if (t.k === "vec") {
      if (init.items.length > t.n) fail("type", `too many initializers for ${t.name}`, init.loc);
      const items = Array.from({ length: t.n }, (_, i) => (init.items[i] ? this.convertExpr(this.expr(init.items[i]), sc(t.s), init.loc) : cnst(vt(t.s), normalize(t.s, 0))));
      return { k: "rv", e: { op: "vec", t: vt(t.s, t.n), items }, t };
    }
    if (t.k === "struct") {
      if (t.def.methods.has("__ctor") && !init.designators) return this.construct(t, init.items, init.loc, true);
      const fields = new Map<string, Val>();
      let pos = 0;
      init.items.forEach((it, i) => {
        const dn = init.designators?.[i];
        const f = dn ? t.def.fields.find((x) => x.name === dn) : t.def.fields[pos];
        if (!f) fail("type", dn ? `${t.def.name} has no member '${dn}'` : `too many initializers for ${t.def.name}`, it.loc);
        pos = t.def.fields.indexOf(f) + 1;
        fields.set(f.name, it.k === "init" ? this.initValue(f.type, it) : this.convertVal(this.expr(it), f.type, it.loc));
      });
      for (const f of t.def.fields) if (!fields.has(f.name)) fields.set(f.name, this.zeroValue(f.type, init.loc));
      return { k: "agg", t, fields };
    }
    fail("type", `cannot initialize ${typeName(t)} with a braced list here`, init.loc);
  }

  zeroValue(t: CType, loc: SourceLoc): Val {
    if (t.k === "scalar") return { k: "rv", e: cnst(vt(t.s), normalize(t.s, 0)), t };
    if (t.k === "vec") return { k: "rv", e: { op: "vec", t: vt(t.s, t.n), items: Array(t.n).fill(cnst(vt(t.s), normalize(t.s, 0))) }, t };
    if (t.k === "struct") {
      const fields = new Map<string, Val>();
      for (const f of t.def.fields) fields.set(f.name, this.zeroValue(f.type, loc));
      return { k: "agg", t, fields };
    }
    if (t.k === "ptr") return { k: "ptr", root: null, off: i32(0), t };
    fail("unsupported", `cannot zero-initialize ${typeName(t)} here`, loc);
  }

  /** Object construction: T(args), T{args}: vectors, scalars, dim3, structs (constructors or aggregates). */
  construct(t: CType, argsE: AExpr[], loc: SourceLoc, braces: boolean): Val {
    if (t.k === "scalar") {
      if (argsE.length === 0) return this.zeroValue(t, loc);
      if (argsE.length !== 1) fail("type", `too many arguments to construct ${typeName(t)}`, loc);
      return this.convertVal(this.expr(argsE[0]), t, loc, true);
    }
    if (t.k === "vec") {
      if (t.name === "dim3") {
        if (argsE.length > 3) fail("type", "dim3 takes at most 3 arguments", loc);
        const items = [0, 1, 2].map((i) => (argsE[i] ? this.convertExpr(this.expr(argsE[i]), U32, loc) : cnst(T_U32, 1)));
        return { k: "rv", e: { op: "vec", t: vt("u32", 3), items }, t };
      }
      if (argsE.length === 0) return this.zeroValue(t, loc);
      if (argsE.length === 1 && !braces) {
        const v = this.rvalue(this.expr(argsE[0]));
        if (v.k === "rv" && v.t.k === "vec") return this.convertVal(v, t, loc, true);
      }
      return this.initValue(t, { k: "init", items: argsE, loc });
    }
    if (t.k === "struct") {
      const ctors = t.def.methods.get("__ctor") as FuncDecl[] | undefined;
      if (!ctors) {
        if (argsE.length === 1 && !braces) {
          const v = this.rvalue(this.expr(argsE[0]));
          if (v.k === "agg" && v.t.def === t.def) return v;
        }
        return this.initValue(t, { k: "init", items: argsE, loc });
      }
      const lv = this.allocLocal("obj", t, false, loc);
      // default member initializers
      const sd = (t.def.env as { decl: StructDecl }).decl;
      for (const f of sd.fields) if (f.init && !(f as { isStatic?: boolean }).isStatic) {
        const flv = this.fieldOf({ k: "lv", lv }, f.name, loc) as Val & { k: "lv" };
        this.assign(flv.lv, f.init.k === "init" ? this.initValue(lvType(flv.lv), f.init) : this.expr(f.init), loc);
      }
      const args = argsE.map((a) => this.snapshot(this.rvalueKeepLV(this.expr(a))));
      this.callFunction({ k: "fn", name: "__ctor", recv: lv, struct: t.def, loc }, args, loc);
      return this.rvalue({ k: "lv", lv });
    }
    fail("type", `cannot construct a value of type ${typeName(t)}`, loc);
  }

  rvalueKeepLV(v: Val): Val { return v; }

  // ------------------------------------------------ calls
  private call(e: AExpr & { k: "call" }): Val {
    const callee = e.callee;
    let fnv: Val;
    if (callee.k === "ident") {
      // builtins take precedence unless the user defined a function with the same name
      if ((callee.name in BUILTIN_FUNCTIONS || callee.name.startsWith("__builtin_")) && !this.prog.funcs.has(callee.name) && !this.lookup(callee.name)) {
        const fc = this.funcStack[this.funcStack.length - 1];
        if (!(fc?.struct?.methods.has(callee.name))) return lowerBuiltinCall(this, callee.name, e.args, e.loc, callee.targs);
      }
      fnv = this.ident(callee);
    } else fnv = this.expr(callee);
    if (fnv.k !== "fn") {
      // functor: obj(args)
      const t = fnv.k === "lv" ? lvType(fnv.lv) : this.typeOfVal(fnv);
      if (t.k === "struct" && t.def.methods.has("operator()")) {
        const recv = fnv.k === "lv" ? fnv.lv : this.materialize(fnv);
        return this.callFunction({ k: "fn", name: "operator()", recv, struct: t.def, loc: e.loc }, this.callArgs(e.args), e.loc);
      }
      fail("type", "called object is not a function", e.loc);
    }
    if (!fnv.recv && !fnv.struct && (fnv.name in BUILTIN_FUNCTIONS || fnv.name.startsWith("__builtin_")) && !this.prog.funcs.has(fnv.name))
      return lowerBuiltinCall(this, fnv.name, e.args, e.loc, fnv.targs);
    return this.callFunction(fnv, this.callArgs(e.args), e.loc);
  }

  private callArgs(args: AExpr[]): Val[] {
    const out: Val[] = [];
    for (const a of args) {
      if (a.k === "init") { out.push({ k: "str", s: "\0init" } as Val); (out[out.length - 1] as unknown as { init: AExpr }).init = a; continue; }
      const v = this.expr(a);
      out.push(v.k === "lv" ? { k: "lv", lv: this.stableLV(v.lv) } : this.snapshot(v));
    }
    return out;
  }

  /** Resolve an overload / template instantiation and inline the call. */
  callFunction(fnv: Val & { k: "fn" }, args: Val[], loc: SourceLoc): Val {
    let cands: FuncDecl[];
    if (fnv.struct) cands = (fnv.struct.methods.get(fnv.name) ?? []) as FuncDecl[];
    else cands = this.prog.funcs.get(fnv.name) ?? [];
    if (!cands.length) {
      if (this.prog.hostNames.has(fnv.name)) fail("type", `calling host function '${fnv.name}' from device code (mark it __device__)`, loc);
      fail("name", `no function named '${fnv.name}'`, loc);
    }
    const argTypes = args.map((a) => (isInitPlaceholder(a) ? null : a.k === "lv" ? lvType(a.lv) : this.typeOfVal(a)));
    const argIsLV = args.map((a) => a.k === "lv");
    type Cand = { fd: FuncDecl; env: Map<string, Binding>; score: number };
    const viable: Cand[] = [];
    const errors: string[] = [];
    for (const fd of cands) {
      if (!fd.body && !fd.template) { /* prototype without body */ }
      const nReq = fd.params.filter((p) => !p.def).length;
      if (args.length < nReq || args.length > fd.params.length) { errors.push(`${fd.name}: expects ${fd.params.length} arguments`); continue; }
      try {
        const env = this.deduce(fd, fnv.targs, argTypes, loc);
        let score = 0;
        for (let i = 0; i < args.length; i++) {
          if (argTypes[i] === null) { score += 1; continue; }
          const pt = this.resolveType(fd.params[i].type, env);
          const s = conversionScore(argTypes[i]!, pt, argIsLV[i], fd.params[i].type);
          if (s < 0) throw new Error(`argument ${i + 1}: cannot convert ${typeName(argTypes[i]!)} to ${typeName(pt)}`);
          score += s;
        }
        if (fd.template && !fnv.targs) score += 0.5; // prefer non-templates
        viable.push({ fd, env, score });
      } catch (err) { errors.push(`${fd.name}: ${(err as Error).message}`); }
    }
    if (!viable.length) fail("type", `no matching function for call to '${fnv.name}' with (${argTypes.map((t) => (t ? typeName(t) : "{...}")).join(", ")})${errors.length ? ": " + errors.slice(0, 3).join("; ") : ""}`, loc);
    viable.sort((a, b) => a.score - b.score);
    if (viable.length > 1 && viable[0].score === viable[1].score && viable[0].fd !== viable[1].fd) {
      if (viable.filter((v) => v.score === viable[0].score && v.fd.body).length > 1)
        fail("type", `call to '${fnv.name}' is ambiguous`, loc);
    }
    const best = viable.find((v) => v.fd.body) ?? viable[0];
    if (!best.fd.body) fail("name", `function '${fnv.name}' is declared but not defined in device code`, loc);
    if (!best.fd.device && !best.fd.global) fail("type", `calling host function '${fnv.name}' from device code`, loc);
    if (best.fd.global) fail("unsupported", `calling a __global__ function from device code (dynamic parallelism) is not supported`, loc);
    return this.inline(best.fd, best.env, args, fnv.recv, fnv.struct, loc);
  }

  private deduce(fd: FuncDecl, targs: TemplateArg[] | undefined, argTypes: (CType | null)[], loc: SourceLoc): Map<string, Binding> {
    // struct template environment for methods
    const base = new Map<string, Binding>();
    if (fd.memberOf) {
      const fc = this.funcStack[this.funcStack.length - 1];
      void fc;
    }
    if (!fd.template) return base;
    const explicit = targs ?? [];
    const env = new Map<string, Binding>();
    const tparams = fd.template;
    tparams.forEach((p, i) => {
      if (i < explicit.length) {
        const sub = this.templateEnv([p], [explicit[i]], loc);
        env.set(p.name, sub.get(p.name)!);
      }
    });
    // deduce remaining type parameters from arguments
    const unify = (pt: TypeExpr, at: CType): void => {
      if (pt.k === "ref") return unify(pt.to, at);
      if (pt.k === "named" && tparams.some((p) => p.name === pt.name && p.kind === "type")) {
        const prev = env.get(pt.name);
        const decayed = at.k === "array" ? { k: "ptr" as const, to: at.of } : at;
        if (prev && prev.k === "type" && !sameType(prev.t, decayed)) {
          // allow a later argument to agree after promotion only when exact: C++ deduction fails on conflicts
          throw new Error(`conflicting deductions for '${pt.name}': ${typeName(prev.t)} vs ${typeName(decayed)}`);
        }
        if (!prev) env.set(pt.name, { k: "type", t: decayed });
        return;
      }
      if (pt.k === "ptr") {
        if (at.k === "ptr") return unify(pt.to, at.to);
        if (at.k === "array") return unify(pt.to, at.of);
      }
      if (pt.k === "array" && at.k === "ptr") return unify(pt.of, at.to);
    };
    fd.params.forEach((p, i) => { if (argTypes[i]) unify(p.type, argTypes[i]!); });
    for (const p of tparams) {
      if (env.has(p.name)) continue;
      if (p.def) { const sub = this.templateEnv([p], [], loc, env); env.set(p.name, sub.get(p.name)!); continue; }
      throw new Error(`cannot deduce template parameter '${p.name}'`);
    }
    return env;
  }

  /** Inline a device function body at the call site. */
  inline(fd: FuncDecl, tenv: Map<string, Binding>, args: Val[], recv: LV | undefined, struct: StructDef | undefined, loc: SourceLoc): Val {
    if (this.inlineDepth >= this.maxInlineDepth) fail("limit", `inlining depth exceeds ${this.maxInlineDepth} (recursive device functions are not supported)`, loc);
    if (this.funcStack.some((f) => f.decl === fd)) fail("unsupported", `recursive call to '${fd.name}': recursion is not supported (all device functions are inlined)`, loc);
    // environment: struct template params + function template params
    const env = new Map<string, Binding>();
    if (struct?.env) for (const [k, b] of (struct.env as { env: Map<string, Binding> }).env) env.set(k, b);
    for (const [k, b] of tenv) env.set(k, b);
    // parameters
    const paramBinds = new Map<string, Binding>();
    const retT = fd.name === "__ctor" ? VOID : fd.ret.k === "auto" ? null : this.resolveType(fd.ret.k === "ref" ? fd.ret.to : fd.ret, env);
    const retIsRef = fd.ret.k === "ref";
    fd.params.forEach((p, i) => {
      const isRef = p.type.k === "ref";
      const pt = this.resolveType(isRef ? (p.type as { to: TypeExpr }).to : p.type, env);
      let a: Val;
      if (i < args.length) {
        a = args[i];
        if (isInitPlaceholder(a)) a = this.initValue(pt, (a as unknown as { init: AExpr & { k: "init" } }).init);
      } else {
        // default argument, evaluated in the callee's environment
        this.funcStack.push({ decl: fd, env, frame: { kind: "func", exits: new Map(), flag: null }, struct, thisLV: recv });
        try { a = this.expr(p.def!); } finally { this.funcStack.pop(); }
      }
      if (isRef) {
        const constRef = Boolean((p.type as { to: { isConst?: boolean } }).to.isConst);
        if (a.k === "lv" && (sameType(lvType(a.lv), pt) || (a.lv.k === "ptrvar"))) { paramBinds.set(p.name, { k: "lv", lv: a.lv }); return; }
        if (!constRef && !(p.type as { rvalue?: boolean }).rvalue) fail("type", `cannot bind a non-const reference parameter '${p.name}' to ${a.k === "lv" ? "an lvalue of a different type" : "a temporary"}`, loc);
        const lv = this.allocLocal(p.name, pt, false, p.loc);
        this.assign(lv, a, loc);
        paramBinds.set(p.name, { k: "lv", lv });
        return;
      }
      const lv = this.allocLocal(p.name, pt.k === "array" ? { k: "ptr", to: pt.of } : pt, false, p.loc);
      this.assign(lv, a, loc);
      paramBinds.set(p.name, { k: "lv", lv });
    });
    // return slot
    const hasWrapper = !returnsOnlyAtEnd(fd.body!);
    const frame: Frame = { kind: "func", exits: new Map(), flag: null, hasWrapper, retType: retT ?? undefined };
    let retLV: LV | null = null;
    const isVoid = retT?.k === "void";
    if (!isVoid && !retIsRef && retT) { retLV = this.allocLocal(`${fd.name}_ret`, retT, false); frame.retLV = retLV; }
    if (retIsRef) frame.retLV = null;
    (frame as Frame & { retIsRef?: boolean; refResult?: LV | null }).retIsRef = retIsRef;
    if (retT === null) (frame as Frame & { autoRet?: boolean }).autoRet = true;
    const fctx: FuncCtx = { decl: fd, env, frame, thisLV: recv, struct };
    this.funcStack.push(fctx);
    this.inlineDepth++;
    const boundary = new Map<string, Binding>();
    boundary.set("\0func", { k: "const", e: i32(0), t: I32 });
    for (const [k, b] of paramBinds) boundary.set(k, b);
    this.scopes.push(boundary);
    try {
      // constructor member initializers
      if (fd.ctorInits && recv) {
        for (const ini of fd.ctorInits) {
          const flv = this.fieldOf({ k: "lv", lv: recv }, ini.name, fd.loc);
          if (flv.k !== "lv") fail("type", `'${ini.name}' is not a member`, fd.loc);
          const ft = lvType(flv.lv);
          const v = ini.args.length === 1 && ft.k !== "struct" ? this.expr(ini.args[0]) : this.construct(ft, ini.args, fd.loc, false);
          this.assign(flv.lv, v, fd.loc);
        }
      }
      if (hasWrapper) {
        this.frames.push(frame);
        const body = this.block(() => { this.stmts(fd.body!.body); this.emit({ s: "break" }); });
        this.frames.pop();
        this.closeFrameLoop(frame, { s: "loop", body, cont: [], breakIf: null });
      } else {
        this.frames.push(frame);
        this.stmts(fd.body!.body);
        this.frames.pop();
      }
    } finally {
      this.scopes.pop();
      this.inlineDepth--;
      this.funcStack.pop();
    }
    const fx = frame as Frame & { refResult?: LV | null; autoLV?: LV };
    if (retIsRef) {
      if (!fx.refResult) fail("type", `function '${fd.name}' returning a reference did not return`, loc);
      return { k: "lv", lv: fx.refResult };
    }
    if (retT === null) return fx.autoLV ? this.rvalue({ k: "lv", lv: fx.autoLV }) : { k: "void" };
    if (isVoid) return { k: "void" };
    return this.rvalue({ k: "lv", lv: retLV! });
  }

  // ------------------------------------------------ jumps through wrapper loops
  /** Emit a loop that represents frame f, then re-dispatch jumps that left through it. */
  private closeFrameLoop(f: Frame, loop: Stmt & { s: "loop" }): void {
    if (f.flag) this.emit({ s: "let", v: f.flag, e: i32(0) });
    // the flag must be reset before the loop starts: insert the reset first
    if (f.flag) {
      const list = this.out[this.out.length - 1];
      const reset = list.pop()!;
      list.push(reset);
    }
    this.emit(loop);
    for (const [code, redo] of f.exits) {
      const then = this.block(redo);
      this.emit({ s: "if", c: bin("==", readVar(f.flag!), i32(code)), then, else: [] });
    }
  }

  /** Jump to frame index ti: break (kind 'break'), continue, or leave a function (kind 'return'). */
  private jump(kind: "break" | "continue" | "return", ti: number): void {
    const top = this.frames.length - 1;
    const target = this.frames[ti];
    if (ti === top) {
      if (kind === "continue") this.emit({ s: "continue" });
      else if (target.kind === "func") { if (target.hasWrapper) this.emit({ s: "break" }); }
      else this.emit({ s: "break" });
      return;
    }
    // leave the innermost frame first, then repeat the jump after it closes
    const inner = this.frames[top];
    if (!inner.flag) inner.flag = this.newVar("jump", T_I32);
    const code = ++this.jumpCodes;
    inner.exits.set(code, () => this.jump(kind, ti));
    this.emit({ s: "let", v: inner.flag, e: i32(code) });
    this.emit({ s: "break" });
  }

  private findFrame(pred: (f: Frame) => boolean): number {
    for (let i = this.frames.length - 1; i >= 0; i--) {
      if (pred(this.frames[i])) return i;
      if (this.frames[i].kind === "func") return -1;
    }
    return -1;
  }

  // ------------------------------------------------ statements
  stmts(list: AStmt[]): void {
    for (let i = 0; i < list.length; i++) {
      const s = list[i];
      this.stmt(s);
      // unreachable code after a jump is dropped
      if (s.k === "return" || s.k === "break" || s.k === "continue") {
        if (list.slice(i + 1).some((x) => x.k === "case")) continue;
        break;
      }
    }
  }

  stmt(s: AStmt): void {
    switch (s.k) {
      case "empty": return;
      case "block": this.pushScope(); try { this.stmts(s.body); } finally { this.popScope(); } return;
      case "expr": {
        const v = this.expr(s.e);
        // a bare load (e.g. `a[i];`) has no effect
        void v;
        return;
      }
      case "decl": for (const d of s.decls) this.localDecl(d); return;
      case "typedef": this.bind(s.name, { k: "type", t: this.resolveType(s.type) }); return;
      case "struct": this.prog.structs.set(s.decl.name, s.decl); return;
      case "if": {
        const c = this.cond(s.c);
        if (c.op === "const") {
          const taken = c.v ? s.t : s.f;
          if (taken) { this.pushScope(); try { this.stmt(taken); } finally { this.popScope(); } }
          return;
        }
        if (s.constexpr) fail("type", "if constexpr condition is not a constant", s.loc);
        const then = this.block(() => this.stmt(s.t));
        const els = s.f ? this.block(() => this.stmt(s.f!)) : [];
        this.emit({ s: "if", c, then, else: els });
        return;
      }
      case "for": case "while": case "do": return this.loop(s);
      case "switch": return this.switchStmt(s);
      case "case": fail("syntax", "case label outside of a switch", s.loc);
      case "break": {
        const ti = this.findFrame((f) => f.kind === "loop" || f.kind === "switch");
        if (ti < 0) fail("syntax", "break outside of a loop or switch", s.loc);
        this.jump("break", ti);
        return;
      }
      case "continue": {
        const ti = this.findFrame((f) => f.kind === "loop");
        if (ti < 0) fail("syntax", "continue outside of a loop", s.loc);
        this.jump("continue", ti);
        return;
      }
      case "return": return this.returnStmt(s);
    }
  }

  private returnStmt(s: AStmt & { k: "return" }): void {
    const fc = this.funcStack[this.funcStack.length - 1];
    if (!fc || fc.frame.isKernel) {
      if (s.e) fail("type", "a __global__ function must return void", s.loc);
      this.emit({ s: "return" });
      return;
    }
    const f = fc.frame as Frame & { retIsRef?: boolean; refResult?: LV | null; autoRet?: boolean; autoLV?: LV };
    if (f.retIsRef) {
      if (!s.e) fail("type", "missing return value", s.loc);
      const v = this.expr(s.e);
      if (v.k !== "lv") fail("type", "returning a temporary from a function returning a reference", s.loc);
      if (f.refResult && f.refResult !== v.lv) fail("unsupported", "functions returning references must return the same object on every path", s.loc);
      f.refResult = this.stableLV(v.lv);
    } else if (s.e) {
      if (f.retType?.k === "void") fail("type", "void function returns a value", s.loc);
      if (f.autoRet) {
        const v = this.rvalue(this.expr(s.e));
        if (!f.autoLV) f.autoLV = this.allocLocal("ret", this.typeOfVal(v), false);
        this.assign(f.autoLV, v, s.loc);
      } else {
        const v = s.e.k === "init" ? this.initValue(f.retType!, s.e) : this.expr(s.e);
        this.assign(f.retLV!, v, s.loc);
      }
    } else if (f.retType && f.retType.k !== "void") fail("type", "non-void function must return a value", s.loc);
    const ti = this.frames.lastIndexOf(f);
    this.jump("return", ti);
  }

  private loop(s: AStmt & { k: "for" | "while" | "do" }): void {
    this.pushScope();
    try {
      if (s.k === "for" && s.init) this.stmt(s.init);
      const frame: Frame = { kind: "loop", exits: new Map(), flag: null };
      this.frames.push(frame);
      let body: Stmt[], cont: Stmt[] = [], breakIf: Expr | null = null;
      try {
        body = this.block(() => {
          if (s.k !== "do" && (s.k === "while" || s.c)) {
            const c = this.cond(s.c!);
            if (!(c.op === "const" && c.v)) this.emit({ s: "if", c: foldExpr({ op: "unop", fn: "!", t: T_BOOL, a: c }, true), then: [{ s: "break" }], else: [] });
          }
          this.stmt(s.body);
        });
        // a step or do-while condition that needs statements goes into the continuing block
        if (s.k === "for" && s.step) cont = this.block(() => { this.expr(s.step!); });
        if (s.k === "do") {
          let c: Expr | null = null;
          const pre = this.block(() => { c = this.cond(s.c); });
          cont = pre;
          breakIf = foldExpr({ op: "unop", fn: "!", t: T_BOOL, a: c! }, true);
        }
      } finally { this.frames.pop(); }
      if (breakIf && breakIf.op === "const" && breakIf.v === true) {
        // do { } while (0): a block that `break` leaves; keep it as a loop that runs once
      }
      this.closeFrameLoop(frame, { s: "loop", body: body!, cont, breakIf, unroll: s.unroll });
    } finally { this.popScope(); }
  }

  private switchStmt(s: AStmt & { k: "switch" }): void {
    const cv = this.rvalue(this.expr(s.c));
    if (cv.k !== "rv" || cv.t.k !== "scalar" || !(isInt(cv.t.s) || cv.t.s === "bool")) fail("type", "switch condition must be an integer", s.loc);
    const ps = promote(cv.t.s);
    const sel = this.tmp(cast(vt(ps), cv.e), "switch");
    const body = s.body.k === "block" ? s.body.body : [s.body];
    // split into sections at case labels
    const sections: { labels: (Expr | null)[]; stmts: AStmt[] }[] = [];
    for (const st of body) {
      if (st.k === "case") {
        const last = sections[sections.length - 1];
        const label = st.value === null ? null : foldExpr(cast(vt(ps), this.constValue(st.value, "case label")), true);
        if (last && last.stmts.length === 0) last.labels.push(label);
        else sections.push({ labels: [label], stmts: [] });
        continue;
      }
      if (!sections.length) { this.sink.warn("syntax", "statement before the first case label is unreachable", st.loc); continue; }
      sections[sections.length - 1].stmts.push(st);
    }
    const seen = new Set<string>();
    for (const sec of sections) for (const l of sec.labels) {
      const key = l === null ? "default" : String((l as { v: unknown }).v);
      if (seen.has(key)) fail("syntax", `duplicate case label ${key}`, s.loc);
      seen.add(key);
    }
    // entry section index
    const entry = this.newVar("case", T_I32);
    const defaultIdx = sections.findIndex((x) => x.labels.includes(null));
    let entryExpr: Expr = i32(defaultIdx >= 0 ? defaultIdx : sections.length);
    for (let i = sections.length - 1; i >= 0; i--)
      for (const l of sections[i].labels) if (l) entryExpr = { op: "select", t: T_I32, c: bin("==", sel, l), a: i32(i), b: entryExpr };
    this.emit({ s: "let", v: entry, e: foldExpr(entryExpr, true) });
    const frame: Frame = { kind: "switch", exits: new Map(), flag: null };
    this.frames.push(frame);
    let lbody: Stmt[];
    this.pushScope();
    try {
      lbody = this.block(() => {
        sections.forEach((sec, i) => {
          const then = this.block(() => this.stmts(sec.stmts));
          this.emit({ s: "if", c: bin("<=", readVar(entry), i32(i)), then, else: [] });
        });
        this.emit({ s: "break" });
      });
    } finally { this.popScope(); this.frames.pop(); }
    this.closeFrameLoop(frame, { s: "loop", body: lbody!, cont: [], breakIf: null });
  }

  localDecl(d: VarDecl): void {
    if (d.space === "shared") {
      const t = this.resolveType(d.type);
      this.bind(d.name, this.declareShared(d, t));
      return;
    }
    if (d.space === "constant" || d.space === "device") fail("type", `__${d.space}__ variables must be declared at file scope`, d.loc);
    if (d.isStatic && !d.isConstexpr) fail("unsupported", `static local variable '${d.name}' is not supported in device code`, d.loc);
    if (d.isExtern) fail("unsupported", "extern local declarations are not supported", d.loc);
    const isRef = d.type.k === "ref";
    if (isRef) {
      if (!d.init) fail("type", `reference '${d.name}' must be initialized`, d.loc);
      const v = this.expr(d.init);
      const constRef = Boolean(((d.type as { to: { isConst?: boolean } }).to).isConst);
      if (v.k !== "lv") {
        if (!constRef) fail("type", `non-const reference '${d.name}' bound to a temporary`, d.loc);
        const t = this.resolveType((d.type as { to: TypeExpr }).to);
        const lv = this.allocLocal(d.name, t, false, d.loc);
        this.assign(lv, v, d.loc);
        this.bind(d.name, { k: "lv", lv });
        return;
      }
      this.bind(d.name, { k: "lv", lv: this.stableLV(v.lv) });
      return;
    }
    let t: CType;
    let initVal: Val | null = null;
    if (d.type.k === "auto" || (d.type.k === "ptr" && d.type.to.k === "auto")) {
      if (!d.init || d.init.k === "init") fail("type", `'auto' variable '${d.name}' needs an initializer expression`, d.loc);
      initVal = this.rvalue(this.expr(d.init));
      t = this.typeOfVal(initVal);
      if (t.k === "void") fail("type", `'auto' variable '${d.name}' initialized with void`, d.loc);
    } else t = this.resolveType(d.type);
    // unsized array with initializer: int a[] = {1,2,3}
    if (t.k === "array" && t.n < 0) {
      if (d.init?.k !== "init") fail("type", `array '${d.name}' needs a size`, d.loc);
      t = { k: "array", of: t.of, n: d.init.items.length };
    }
    const isConstScalar = (d.isConstexpr || (d.type as { isConst?: boolean }).isConst) && t.k === "scalar";
    if (isConstScalar && d.init && d.init.k !== "init") {
      // const int N = 16; is a constant expression when its initializer is
      const v = initVal ?? this.convertVal(this.expr(d.init), t, d.loc);
      const e = foldExpr(this.convertExpr(v, t, d.loc), true);
      if (e.op === "const") { this.bind(d.name, { k: "const", e, t }); return; }
      if (d.isConstexpr) fail("type", `constexpr variable '${d.name}' is not initialized with a constant`, d.loc);
      initVal = { k: "rv", e, t };
    }
    const lv = this.allocLocal(d.name, t, false, d.loc);
    this.bind(d.name, { k: "lv", lv });
    if (lv.k === "ptrvar") lv.b.name = d.name;
    if (!d.init && !initVal) return;
    if (t.k === "array" && lv.k === "mem") {
      if (d.init!.k !== "init") fail("type", "array initializer must be a braced list", d.loc);
      this.initArray(lv, t, d.init as AExpr & { k: "init" });
      return;
    }
    if (t.k === "struct" && lv.k === "mem" && d.init?.k === "init" && !t.def.methods.has("__ctor")) {
      this.initMemory(lv, t, d.init);
      return;
    }
    if (d.init?.k === "construct" && !initVal) {
      initVal = this.construct(t, d.init.args, d.init.loc, d.init.braces);
    }
    const v = initVal ?? (d.init!.k === "init" ? this.initValue(t, d.init as AExpr & { k: "init" }) : this.expr(d.init!));
    this.assign(lv, v, d.loc);
  }

  /** brace-initialize an aggregate that lives in memory (structs with array members, arrays of structs) */
  private initMemory(lv: LV & { k: "mem" }, t: CType, init: AExpr): void {
    if (t.k === "array") {
      if (init.k !== "init") fail("type", "array initializer must be a braced list", init.loc);
      this.initArray(lv, t, init);
      return;
    }
    if (t.k === "struct" && init.k === "init") {
      let pos = 0;
      const done = new Set<string>();
      init.items.forEach((it, i) => {
        const dn = init.designators?.[i];
        const f = dn ? t.def.fields.find((x) => x.name === dn) : t.def.fields[pos];
        if (!f) fail("type", dn ? `${t.def.name} has no member '${dn}'` : `too many initializers for ${t.def.name}`, it.loc);
        pos = t.def.fields.indexOf(f) + 1;
        done.add(f.name);
        this.initMemory({ k: "mem", root: lv.root, off: this.addOff(lv.off, f.offset), t: f.type }, f.type, it);
      });
      for (const f of t.def.fields) if (!done.has(f.name)) {
        const target: LV & { k: "mem" } = { k: "mem", root: lv.root, off: this.addOff(lv.off, f.offset), t: f.type };
        this.initMemory(target, f.type, { k: "init", items: [], loc: init.loc });
      }
      return;
    }
    const v = init.k === "init" ? (t.k === "struct" ? this.initValue(t, init) : this.initValue(t, init)) : this.expr(init);
    this.assign(lv, v, init.loc);
  }

  private initArray(lv: LV & { k: "mem" }, t: CType & { k: "array" }, init: AExpr & { k: "init" }): void {
    if (init.items.length > t.n) fail("type", "too many initializers for array", init.loc);
    const es = sizeOf(t.of);
    for (let i = 0; i < t.n; i++) {
      const target: LV = { k: "mem", root: lv.root, off: this.addOff(lv.off, i * es), t: t.of };
      const it = init.items[i];
      if (t.of.k === "array") {
        if (it && it.k !== "init") fail("type", "nested array initializer must be a braced list", init.loc);
        this.initArray(target as LV & { k: "mem" }, t.of, (it ?? { k: "init", items: [], loc: init.loc }) as AExpr & { k: "init" });
        continue;
      }
      if (t.of.k === "struct" && (!it || it.k === "init")) { this.initMemory(target as LV & { k: "mem" }, t.of, it ?? { k: "init", items: [], loc: init.loc }); continue; }
      const v = it ? (it.k === "init" ? this.initValue(t.of, it) : this.expr(it)) : this.zeroValue(t.of, init.loc);
      this.assign(target, v, init.loc);
    }
  }

  // ------------------------------------------------ kernel entry
  lowerKernel(fd: FuncDecl, tenv: Map<string, Binding>, displayName: string): Kernel {
    const frame: Frame = { kind: "func", exits: new Map(), flag: null, isKernel: true };
    const fctx: FuncCtx = { decl: fd, env: tenv, frame };
    this.funcStack.push(fctx);
    const boundary = new Map<string, Binding>();
    this.scopes.push(boundary);
    fd.params.forEach((p) => this.kernelParam(p, boundary, tenv));
    this.frames.push(frame);
    try { this.stmts(fd.body!.body); } finally { this.frames.pop(); this.funcStack.pop(); this.scopes.pop(); }
    const name = sanitize(displayName.replace(/[<>, ]+/g, "_").replace(/_+$/, ""));
    return {
      name, displayName, params: this.params, roots: this.roots, body: this.out[0], vars: this.vars, features: this.features,
      launchBounds: fd.launchBounds, sharedBytes: this.sharedBytes, block: this.spec ? this.spec.block : [0, 0, 0], sourceLine: fd.loc.line,
    };
  }

  private kernelParam(p: Param, scope: Map<string, Binding>, env: Map<string, Binding>): void {
    if (p.type.k === "ref") fail("type", `kernel parameter '${p.name}' cannot be a reference`, p.loc);
    let t = this.resolveType(p.type, env);
    if (t.k === "array") t = { k: "ptr", to: t.of };
    if (!p.name) fail("syntax", "kernel parameters must be named", p.loc);
    if (t.k === "ptr") {
      if (t.to.k === "ptr") fail("unsupported", `kernel parameter '${p.name}': pointers to pointers are not supported`, p.loc);
      const elem = elemVT(t.to);
      const root = this.newRoot(p.name, "global", -1, elem, { param: this.params.length, structured: !elem && t.to.k !== "void", readonly: false });
      const b: PtrBinding = { root, offVar: this.newVar(p.name + "_off", T_I32), t, name: p.name, isParam: true };
      this.emit({ s: "let", v: b.offVar, e: i32(0) });
      this.params.push({ name: p.name, kind: "ptr", ctype: typeName(t), t: elem, root, isConst: Boolean(t.isConst) });
      scope.set(p.name, { k: "lv", lv: { k: "ptrvar", b } });
      return;
    }
    // by-value: scalars, vectors and structs of scalars (flattened into several parameters)
    const flat: { path: string; t: CType }[] = [];
    const walk = (ty: CType, path: string) => {
      if (ty.k === "scalar" || ty.k === "vec") { flat.push({ path, t: ty }); return; }
      if (ty.k === "struct") { for (const f of ty.def.fields) walk(f.type, `${path}.${f.name}`); return; }
      if (ty.k === "array") { for (let i = 0; i < ty.n; i++) walk(ty.of, `${path}[${i}]`); return; }
      if (ty.k === "ptr") fail("unsupported", `kernel parameter '${p.name}': pointers inside by-value structs are not supported; pass each buffer as its own parameter`, p.loc);
      fail("type", `kernel parameter '${p.name}' has unsupported type ${typeName(ty)}`, p.loc);
    };
    walk(t, p.name);
    const paramExpr = new Map<string, Expr>();
    for (const f of flat) {
      const idx = this.params.length;
      const vtT = this.vtOf(f.t);
      this.params.push({ name: f.path, kind: "scalar", ctype: typeName(f.t), t: vtT, ...(f.path !== p.name ? { origin: p.name } : {}) });
      paramExpr.set(f.path, { op: "param", t: vtT, index: idx, name: sanitize(f.path) });
    }
    // bind: scalar params are copied into variables (kernels may assign to their parameters)
    const lv = this.allocLocal(p.name, t, false, p.loc);
    const fill = (target: LV, ty: CType, path: string) => {
      if (ty.k === "scalar" || ty.k === "vec") { this.assign(target, { k: "rv", e: paramExpr.get(path)!, t: ty }, p.loc); return; }
      if (ty.k === "struct") {
        for (const f of ty.def.fields) {
          const flv = this.fieldOf({ k: "lv", lv: target }, f.name, p.loc) as Val & { k: "lv" };
          fill(flv.lv, f.type, `${path}.${f.name}`);
        }
        return;
      }
      if (ty.k === "array" && target.k === "mem") {
        for (let i = 0; i < ty.n; i++) fill({ k: "mem", root: target.root, off: this.addOff(target.off, i * sizeOf(ty.of)), t: ty.of }, ty.of, `${path}[${i}]`);
      }
    };
    fill(lv, t, p.name);
    scope.set(p.name, { k: "lv", lv });
  }
}

// ------------------------------------------------------------------ helpers
export function lvType(lv: LV): CType { return lv.k === "ptrvar" ? lv.b.t : lv.t; }

function laneIndex(name: string, n: number, loc: SourceLoc): number {
  const i = "xyzw".indexOf(name);
  if (name.length !== 1 || i < 0 || i >= n) fail("name", `vector type with ${n} components has no member '${name}'`, loc);
  return i;
}

export function elemVT(t: CType): VT | null {
  if (t.k === "scalar") return vt(t.s);
  if (t.k === "vec") return vt(t.s, t.n);
  if (t.k === "array") return elemVT(t.of);
  return null;
}

function structNeedsMemory(def: StructDef): boolean {
  return def.isUnion || def.fields.some((f) => f.type.k === "array" || (f.type.k === "struct" && structNeedsMemory(f.type.def)));
}

function isInitPlaceholder(v: Val): boolean { return v.k === "str" && v.s === "\0init"; }

function someLoad(e: Expr): boolean {
  if (e.op === "load") return true;
  switch (e.op) {
    case "unop": case "cast": case "bitcast": case "extract": return someLoad(e.a);
    case "binop": return someLoad(e.a) || someLoad(e.b);
    case "select": return someLoad(e.c) || someLoad(e.a) || someLoad(e.b);
    case "call": return e.args.some(someLoad);
    case "vec": return e.items.some(someLoad);
    default: return false;
  }
}

/** Implicit conversion cost: 0 exact, 1 promotion, 2 conversion, -1 impossible. */
function conversionScore(from: CType, to: CType, isLV: boolean, pte: TypeExpr): number {
  if (pte.k === "ref" && !(pte.to as { isConst?: boolean }).isConst && !pte.rvalue) {
    // non-const reference needs an lvalue of exactly that type
    if (!isLV) return -1;
    return sameType(from, to) ? 0 : -1;
  }
  if (sameType(from, to)) return 0;
  if (from.k === "array" && to.k === "ptr") return sameType(from.of, to.to) ? 0 : -1;
  if (from.k === "scalar" && to.k === "scalar") {
    if (from.s === to.s) return 0;
    if (promote(from.s) === to.s || (from.s === "f32" && to.s === "f64")) return 1;
    if ((from.s === "f16" || from.s === "bf16") !== (to.s === "f16" || to.s === "bf16") && from.s !== "bool" && to.s !== "bool") {
      // half <-> float/int conversions are user-defined in CUDA: allowed but ranked after standard conversions
      return 3;
    }
    return 2;
  }
  if (from.k === "vec" && to.k === "vec") return from.s === to.s && from.n === to.n ? 0 : -1;
  if (from.k === "ptr" && to.k === "ptr") {
    if (sameType(from.to, to.to)) return 0;
    if (to.to.k === "void") return 2;
    return -1;
  }
  if (from.k === "nullptr" && to.k === "ptr") return 1;
  if (from.k === "ptr" && to.k === "scalar" && to.s === "bool") return 2;
  if (from.k === "struct" && to.k === "struct") return from.def === to.def ? 0 : -1;
  return -1;
}

/** true if the function's only return statement (if any) is the last top-level statement */
function returnsOnlyAtEnd(body: AStmt & { k: "block" }): boolean {
  let count = 0;
  const visit = (s: AStmt): void => {
    switch (s.k) {
      case "return": count++; break;
      case "block": s.body.forEach(visit); break;
      case "if": visit(s.t); if (s.f) visit(s.f); break;
      case "for": case "while": case "do": visit(s.body); break;
      case "switch": visit(s.body); break;
    }
  };
  body.body.forEach(visit);
  if (count === 0) return true;
  return count === 1 && body.body[body.body.length - 1]?.k === "return";
}

export function sanitize(s: string): string {
  const r = s.replace(/[^A-Za-z0-9_]/g, "_");
  return /^[0-9]/.test(r) ? "_" + r : r.replace(/^__+/, "u_");
}

function writeScalar(dv: DataView, off: number, s: Scalar, v: number | bigint | boolean): void {
  const n = typeof v === "boolean" ? (v ? 1 : 0) : v;
  switch (s) {
    case "bool": case "u8": dv.setUint8(off, Number(n)); break;
    case "i8": dv.setInt8(off, Number(n)); break;
    case "i16": dv.setInt16(off, Number(n), true); break;
    case "u16": dv.setUint16(off, Number(n), true); break;
    case "i32": dv.setInt32(off, Number(n), true); break;
    case "u32": dv.setUint32(off, Number(n), true); break;
    case "i64": dv.setBigInt64(off, BigInt(n), true); break;
    case "u64": dv.setBigUint64(off, BigInt(n), true); break;
    case "f32": dv.setFloat32(off, Number(n), true); break;
    case "f64": dv.setFloat64(off, Number(n), true); break;
    case "f16": dv.setUint16(off, halfBitsLocal(Number(n)), true); break;
    case "bf16": { const f = new Float32Array([Number(n)]); dv.setUint16(off, new Uint32Array(f.buffer)[0] >>> 16, true); break; }
  }
}
import { halfBits as halfBitsLocal } from "./fold.js";

export { CompileError, sameType, F32, F64, I32, I64, U32, U64, BOOL, VOID, vecType, isSigned };
