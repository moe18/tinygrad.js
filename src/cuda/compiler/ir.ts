/**
 * Kernel IR: a structured, typed intermediate representation of one CUDA kernel after semantic analysis and inlining.
 *
 * Why not tinygrad's UOps? UOps describe *what* to compute over shaped tensors and let tinygrad pick the parallelization
 * (ranges, locals, upcasts). A CUDA kernel fixes the parallelization: explicit thread indices, shared memory, barriers,
 * atomics, warp operations, pointer arithmetic and arbitrary control flow. Those don't map onto UOps without losing the
 * kernel's meaning, so CUDA kernels get this IR, and are handed to tinygrad as precompiled programs (python/tinygrad_js/cuda.py).
 *
 * - Expressions are side-effect free trees (loads included: they are evaluated at the statement that uses them).
 * - Statements carry all effects: variable assignment, stores, atomics, barriers, warp collectives, printf.
 * - Control flow is structured (if / loop with a continuing block / break / continue / return), which every backend needs
 *   (WGSL has no goto; WebAssembly only has structured branches).
 * - Memory is addressed by (root, byte offset): every pointer's target buffer is known statically. Roots are kernel pointer
 *   parameters, __shared__ arrays, local arrays, __constant__ and __device__ variables.
 */
import { SCALAR_SIZE, type Scalar } from "./types.js";

export interface VT { s: Scalar; n: 1 | 2 | 3 | 4 }
export const vt = (s: Scalar, n: 1 | 2 | 3 | 4 = 1): VT => ({ s, n });
export const vtSize = (t: VT) => SCALAR_SIZE[t.s] * t.n;
export const vtEq = (a: VT, b: VT) => a.s === b.s && a.n === b.n;
export const vtStr = (t: VT) => (t.n === 1 ? t.s : `${t.s}x${t.n}`);
export const T_BOOL = vt("bool"), T_I32 = vt("i32"), T_U32 = vt("u32"), T_F32 = vt("f32"), T_I64 = vt("i64"), T_U64 = vt("u64");

export type Space = "global" | "shared" | "local" | "constant";

export interface Root {
  id: number;
  name: string;
  space: Space;
  /** for global roots that are kernel parameters: index of the pointer parameter */
  param?: number;
  /** size in bytes for fixed-size roots; -1 for kernel parameters and dynamic shared memory */
  bytes: number;
  /** extern __shared__ (dynamic shared memory, size given at launch) */
  dynamic?: boolean;
  /** declared element type (pointee type of the parameter / array element) used for typed buffer layouts */
  elem: VT | null;
  /** element type is a struct or otherwise not a single value type */
  structured?: boolean;
  readonly?: boolean;
  /** initial contents (constant / device / shared arrays with initializers are not allowed for shared) */
  init?: Uint8Array;
  align: number;
  /** module-scope variable (__constant__ / __device__): runtime-managed buffer shared by all kernels of the module */
  symbol?: string;
}

export interface Var { id: number; name: string; t: VT }

export type MathFn =
  | "sqrt" | "rsqrt" | "cbrt" | "exp" | "exp2" | "exp10" | "expm1" | "log" | "log2" | "log10" | "log1p" | "pow"
  | "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "atan2" | "sinh" | "cosh" | "tanh" | "asinh" | "acosh" | "atanh"
  | "erf" | "erfc" | "lgamma" | "tgamma" | "hypot" | "floor" | "ceil" | "round" | "rint" | "trunc" | "fabs" | "fmin" | "fmax"
  | "fmod" | "remainder" | "copysign" | "fma" | "fdim" | "ldexp" | "isnan" | "isinf" | "isfinite" | "signbit" | "saturate"
  | "min" | "max" | "abs" | "popc" | "clz" | "ffs" | "brev" | "mulhi" | "normcdf";

export type BinOp = "+" | "-" | "*" | "/" | "%" | "&" | "|" | "^" | "<<" | ">>" | "==" | "!=" | "<" | "<=" | ">" | ">=" | "&&" | "||";

export type Builtin =
  | "tid.x" | "tid.y" | "tid.z" | "bid.x" | "bid.y" | "bid.z" | "bdim.x" | "bdim.y" | "bdim.z" | "gdim.x" | "gdim.y" | "gdim.z"
  | "tid.linear" | "lane" | "warp";

export type Expr =
  | { op: "const"; t: VT; v: number | bigint | boolean }
  | { op: "var"; t: VT; v: Var }
  | { op: "param"; t: VT; index: number; name: string }
  | { op: "builtin"; t: VT; name: Builtin }
  | { op: "unop"; t: VT; fn: "-" | "!" | "~"; a: Expr }
  | { op: "binop"; t: VT; fn: BinOp; a: Expr; b: Expr }
  | { op: "select"; t: VT; c: Expr; a: Expr; b: Expr }       // c ? a : b, both evaluated (pure, load-free arms only)
  | { op: "cast"; t: VT; a: Expr }                           // C conversion
  | { op: "bitcast"; t: VT; a: Expr }
  | { op: "call"; t: VT; fn: MathFn; args: Expr[] }
  | { op: "load"; t: VT; root: Root; off: Expr }             // byte offset into the root
  | { op: "vec"; t: VT; items: Expr[] }
  | { op: "extract"; t: VT; a: Expr; lane: number };

export type AtomicFn = "add" | "sub" | "xchg" | "min" | "max" | "and" | "or" | "xor" | "inc" | "dec" | "cas";
export type WarpFn = "shfl" | "shfl_up" | "shfl_down" | "shfl_xor" | "ballot" | "any" | "all" | "reduce_add" | "reduce_min" | "reduce_max";

export type Stmt =
  | { s: "let"; v: Var; e: Expr }
  | { s: "store"; root: Root; off: Expr; t: VT; e: Expr }
  | { s: "atomic"; fn: AtomicFn; root: Root; off: Expr; t: VT; args: Expr[]; result: Var | null }
  | { s: "if"; c: Expr; then: Stmt[]; else: Stmt[] }
  /** loop { body } continuing { cont; break if breakIf } */
  | { s: "loop"; body: Stmt[]; cont: Stmt[]; breakIf: Expr | null; unroll?: number | "full" | "none"; label?: string }
  | { s: "break" }
  | { s: "continue" }
  | { s: "return" }
  /** __syncthreads (block) / __syncwarp (warp). `count` variants are lowered before backends see them. */
  | { s: "barrier"; scope: "block" | "warp" }
  | { s: "fence"; scope: "block" | "device" | "system" }
  | { s: "warp"; fn: WarpFn; args: Expr[]; width: number; result: Var }
  | { s: "printf"; fmt: string; args: Expr[] }
  | { s: "assert"; c: Expr; msg: string; line: number }
  | { s: "trap"; msg: string }
  | { s: "comment"; text: string };

export interface KernelParam {
  name: string;
  /** "ptr": a device pointer (a buffer binding); "scalar": a by-value argument */
  kind: "ptr" | "scalar";
  /** C type spelling, for messages and the JS argument validator */
  ctype: string;
  /** scalar params: value type; pointer params: pointee element type if it's a value type */
  t: VT | null;
  /** pointer params */
  root?: Root;
  /** for flattened struct params: the original param and field path */
  origin?: string;
  isConst?: boolean;
}

export interface Kernel {
  name: string;
  /** the kernel name as written in the source (template instantiations: `name<args>`) */
  displayName: string;
  params: KernelParam[];
  roots: Root[];
  body: Stmt[];
  vars: Var[];
  /** features used, for capability checks and device selection */
  features: Set<string>;
  launchBounds?: number[];
  /** static shared memory bytes */
  sharedBytes: number;
  /** block size this IR was specialized for ([0,0,0] = unspecialized) */
  block: [number, number, number];
  sourceLine: number;
}

// ------------------------------------------------------------------ constructors
export const cnst = (t: VT, v: number | bigint | boolean): Expr => ({ op: "const", t, v });
export const i32 = (v: number): Expr => ({ op: "const", t: T_I32, v: v | 0 });
export const u32 = (v: number): Expr => ({ op: "const", t: T_U32, v: v >>> 0 });
export const bool = (v: boolean): Expr => ({ op: "const", t: T_BOOL, v });
export const readVar = (v: Var): Expr => ({ op: "var", t: v.t, v });
export const bin = (fn: BinOp, a: Expr, b: Expr, t?: VT): Expr => ({
  op: "binop", fn, a, b, t: t ?? (["==", "!=", "<", "<=", ">", ">=", "&&", "||"].includes(fn) ? T_BOOL : a.t),
});
export const cast = (t: VT, a: Expr): Expr => (vtEq(a.t, t) ? a : { op: "cast", t, a });

// ------------------------------------------------------------------ traversal helpers
export function exprChildren(e: Expr): Expr[] {
  switch (e.op) {
    case "unop": case "cast": case "bitcast": case "extract": return [e.a];
    case "binop": return [e.a, e.b];
    case "select": return [e.c, e.a, e.b];
    case "call": return e.args;
    case "load": return [e.off];
    case "vec": return e.items;
    default: return [];
  }
}

export function mapExpr(e: Expr, f: (e: Expr) => Expr): Expr {
  switch (e.op) {
    case "unop": return f({ ...e, a: mapExpr(e.a, f) });
    case "cast": case "bitcast": case "extract": return f({ ...e, a: mapExpr(e.a, f) } as Expr);
    case "binop": return f({ ...e, a: mapExpr(e.a, f), b: mapExpr(e.b, f) });
    case "select": return f({ ...e, c: mapExpr(e.c, f), a: mapExpr(e.a, f), b: mapExpr(e.b, f) });
    case "call": return f({ ...e, args: e.args.map((x) => mapExpr(x, f)) });
    case "load": return f({ ...e, off: mapExpr(e.off, f) });
    case "vec": return f({ ...e, items: e.items.map((x) => mapExpr(x, f)) });
    default: return f(e);
  }
}

export function someExpr(e: Expr, pred: (e: Expr) => boolean): boolean {
  if (pred(e)) return true;
  for (const c of exprChildren(e)) if (someExpr(c, pred)) return true;
  return false;
}

export const hasLoad = (e: Expr) => someExpr(e, (x) => x.op === "load");
export const readsVar = (e: Expr, v: Var) => someExpr(e, (x) => x.op === "var" && x.v === v);

/** Expressions of a statement (not of nested statements). */
export function stmtExprs(s: Stmt): Expr[] {
  switch (s.s) {
    case "let": return [s.e];
    case "store": return [s.off, s.e];
    case "atomic": return [s.off, ...s.args];
    case "if": return [s.c];
    case "loop": return s.breakIf ? [s.breakIf] : [];
    case "warp": return s.args;
    case "printf": return s.args;
    case "assert": return [s.c];
    default: return [];
  }
}

export function mapStmtExprs(s: Stmt, f: (e: Expr) => Expr): Stmt {
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

/** Visit every statement, depth first. */
export function walkStmts(body: Stmt[], f: (s: Stmt, depthInLoop: number) => void, loopDepth = 0): void {
  for (const s of body) {
    f(s, loopDepth);
    if (s.s === "if") { walkStmts(s.then, f, loopDepth); walkStmts(s.else, f, loopDepth); }
    else if (s.s === "loop") { walkStmts(s.body, f, loopDepth + 1); walkStmts(s.cont, f, loopDepth + 1); }
  }
}

export function containsStmt(body: Stmt[], pred: (s: Stmt) => boolean): boolean {
  let found = false;
  walkStmts(body, (s) => { if (!found && pred(s)) found = true; });
  return found;
}

export const isBarrierLike = (s: Stmt) => s.s === "barrier" || s.s === "warp";

// ------------------------------------------------------------------ printer
export function exprStr(e: Expr): string {
  switch (e.op) {
    case "const": return typeof e.v === "boolean" ? String(e.v) : e.t.s.startsWith("f") ? fmtFloat(e.v as number) : `${e.v}${e.t.s === "u32" ? "u" : e.t.s === "i64" ? "ll" : e.t.s === "u64" ? "ull" : ""}`;
    case "var": return e.v.name;
    case "param": return `param.${e.name}`;
    case "builtin": return e.name;
    case "unop": return `${e.fn}(${exprStr(e.a)})`;
    case "binop": return `(${exprStr(e.a)} ${e.fn} ${exprStr(e.b)})`;
    case "select": return `select(${exprStr(e.c)}, ${exprStr(e.a)}, ${exprStr(e.b)})`;
    case "cast": return `${vtStr(e.t)}(${exprStr(e.a)})`;
    case "bitcast": return `bitcast<${vtStr(e.t)}>(${exprStr(e.a)})`;
    case "call": return `${e.fn}.${vtStr(e.t)}(${e.args.map(exprStr).join(", ")})`;
    case "load": return `load.${vtStr(e.t)} ${e.root.name}[${exprStr(e.off)}]`;
    case "vec": return `${vtStr(e.t)}(${e.items.map(exprStr).join(", ")})`;
    case "extract": return `${exprStr(e.a)}.${"xyzw"[e.lane]}`;
  }
}
function fmtFloat(v: number): string {
  if (Number.isNaN(v)) return "nan"; if (!Number.isFinite(v)) return v > 0 ? "inf" : "-inf";
  const s = String(v); return /[.e]/.test(s) ? s : s + ".0";
}

export function printStmts(body: Stmt[], ind = "  "): string {
  const out: string[] = [];
  for (const s of body) {
    switch (s.s) {
      case "let": out.push(`${ind}${s.v.name}: ${vtStr(s.v.t)} = ${exprStr(s.e)}`); break;
      case "store": out.push(`${ind}store.${vtStr(s.t)} ${s.root.name}[${exprStr(s.off)}] = ${exprStr(s.e)}`); break;
      case "atomic": out.push(`${ind}${s.result ? s.result.name + " = " : ""}atomic_${s.fn}.${vtStr(s.t)} ${s.root.name}[${exprStr(s.off)}], ${s.args.map(exprStr).join(", ")}`); break;
      case "if":
        out.push(`${ind}if ${exprStr(s.c)} {`, printStmts(s.then, ind + "  "));
        if (s.else.length) out.push(`${ind}} else {`, printStmts(s.else, ind + "  "));
        out.push(`${ind}}`); break;
      case "loop":
        out.push(`${ind}loop${s.unroll ? ` [unroll ${s.unroll}]` : ""} {`, printStmts(s.body, ind + "  "));
        if (s.cont.length || s.breakIf) {
          out.push(`${ind}} continuing {`, printStmts(s.cont, ind + "  "));
          if (s.breakIf) out.push(`${ind}  break if ${exprStr(s.breakIf)}`);
        }
        out.push(`${ind}}`); break;
      case "break": case "continue": case "return": out.push(ind + s.s); break;
      case "barrier": out.push(`${ind}barrier.${s.scope}`); break;
      case "fence": out.push(`${ind}fence.${s.scope}`); break;
      case "warp": out.push(`${ind}${s.result.name} = warp_${s.fn}(${s.args.map(exprStr).join(", ")}) width=${s.width}`); break;
      case "printf": out.push(`${ind}printf(${JSON.stringify(s.fmt)}${s.args.map((a) => ", " + exprStr(a)).join("")})`); break;
      case "assert": out.push(`${ind}assert ${exprStr(s.c)}  // ${s.msg}`); break;
      case "trap": out.push(`${ind}trap  // ${s.msg}`); break;
      case "comment": out.push(`${ind}// ${s.text}`); break;
    }
  }
  return out.join("\n");
}

export function printKernel(k: Kernel): string {
  const params = k.params.map((p) => `${p.kind === "ptr" ? `${p.ctype} ${p.name} -> root ${p.root?.name}` : `${p.ctype} ${p.name}`}`).join(", ");
  const roots = k.roots.map((r) => `//   root ${r.name}: ${r.space}${r.param !== undefined ? ` param#${r.param}` : ""}${r.bytes >= 0 ? ` ${r.bytes}B` : ""}${r.dynamic ? " dynamic" : ""} elem=${r.elem ? vtStr(r.elem) : "bytes"}${r.readonly ? " readonly" : ""}`);
  return [`// kernel ${k.displayName}(${params})`, `//   block specialization: ${k.block.join("x")}`, ...roots,
    `kernel ${k.name} {`, printStmts(k.body), "}"].join("\n");
}
