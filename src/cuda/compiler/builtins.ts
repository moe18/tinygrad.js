/**
 * CUDA built-in functions: math library (float/double/half overloads), integer intrinsics, type conversion intrinsics,
 * atomics, warp collectives, barriers and fences, printf/assert, vector constructors and helper_math.h-style helpers.
 */
import { fail, type SourceLoc } from "./diagnostics.js";
import type { Expr as AExpr, TemplateArg } from "./ast.js";
import { bin, bool, cast, cnst, i32, readVar, T_BOOL, T_I32, T_U32, vt, type AtomicFn, type Expr, type MathFn, type VT, type WarpFn } from "./ir.js";
import { foldExpr } from "./fold.js";
import { commonScalar, isFloat, isInt, sc, typeName, VECTOR_TYPES, type CType, type Scalar } from "./types.js";
import { lvType, type Lowering, type LV, type Val } from "./lower.js";

const MATH1: Record<string, MathFn> = {
  sqrt: "sqrt", rsqrt: "rsqrt", cbrt: "cbrt", exp: "exp", exp2: "exp2", exp10: "exp10", expm1: "expm1", log: "log", log2: "log2", log10: "log10",
  log1p: "log1p", sin: "sin", cos: "cos", tan: "tan", asin: "asin", acos: "acos", atan: "atan", sinh: "sinh", cosh: "cosh", tanh: "tanh",
  asinh: "asinh", acosh: "acosh", atanh: "atanh", erf: "erf", erfc: "erfc", lgamma: "lgamma", tgamma: "tgamma", floor: "floor", ceil: "ceil",
  round: "round", rint: "rint", nearbyint: "rint", trunc: "trunc", fabs: "fabs", normcdf: "normcdf",
};
const MATH2: Record<string, MathFn> = { pow: "pow", atan2: "atan2", hypot: "hypot", fmin: "fmin", fmax: "fmax", fmod: "fmod", remainder: "remainder", copysign: "copysign", fdim: "fdim" };
/** fast-math intrinsics map to the same functions (at least as accurate as NVIDIA's approximations) */
const FAST1: Record<string, MathFn> = {
  __expf: "exp", __exp10f: "exp10", __logf: "log", __log2f: "log2", __log10f: "log10", __sinf: "sin", __cosf: "cos", __tanf: "tan",
  __fsqrt_rn: "sqrt", __frsqrt_rn: "rsqrt", __dsqrt_rn: "sqrt", __tanhf: "tanh",
};

const MAKE_VEC = new Map<string, { s: Scalar; n: 1 | 2 | 3 | 4; name: string }>();
for (const [name, v] of VECTOR_TYPES) if (!name.startsWith("__") && name !== "dim3") MAKE_VEC.set(`make_${name}`, { ...v, name });

const ATOMIC_NAMES: Record<string, AtomicFn> = {
  atomicAdd: "add", atomicSub: "sub", atomicExch: "xchg", atomicMin: "min", atomicMax: "max", atomicInc: "inc", atomicDec: "dec",
  atomicAnd: "and", atomicOr: "or", atomicXor: "xor", atomicCAS: "cas",
};

const SIMPLE = [
  "printf", "assert", "__assert_fail", "__trap", "__brkpt", "__nanosleep", "clock", "clock64", "__globaltimer",
  "__syncthreads", "__syncthreads_count", "__syncthreads_and", "__syncthreads_or", "__syncwarp", "__threadfence", "__threadfence_block",
  "__threadfence_system", "__shfl_sync", "__shfl_up_sync", "__shfl_down_sync", "__shfl_xor_sync", "__ballot_sync", "__any_sync", "__all_sync",
  "__activemask", "__reduce_add_sync", "__reduce_min_sync", "__reduce_max_sync", "__shfl", "__shfl_up", "__shfl_down", "__shfl_xor", "__ballot", "__any", "__all",
  "__ldg", "__ldcg", "__ldca", "__ldcs", "__ldlu", "__ldcv", "__stcg", "__stcs", "__stwb", "__stwt",
  "min", "max", "umin", "umax", "llmin", "llmax", "ullmin", "ullmax", "abs", "labs", "llabs", "fma", "fmaf", "__fmaf_rn", "__fma_rn",
  "ldexp", "ldexpf", "isnan", "isinf", "isfinite", "signbit", "__isnanf", "__isinff", "__finitef", "__signbitf", "__isnan", "__isinf",
  "sincos", "sincosf", "__sincosf", "sincospi", "sincospif", "sinpi", "sinpif", "cospi", "cospif", "rcbrt", "rcbrtf", "frexp", "frexpf", "modf", "modff",
  "__fdividef", "__fdiv_rn", "__frcp_rn", "__drcp_rn", "__fadd_rn", "__fsub_rn", "__fmul_rn", "__dadd_rn", "__dmul_rn", "__dsub_rn", "__ddiv_rn", "__saturatef", "__powf", "saturate",
  "__popc", "__popcll", "__clz", "__clzll", "__ffs", "__ffsll", "__brev", "__brevll", "__mulhi", "__umulhi", "__mul64hi", "__umul64hi", "__mul24", "__umul24",
  "__funnelshift_l", "__funnelshift_r", "__funnelshift_lc", "__funnelshift_rc", "__byte_perm", "__sad", "__usad", "__rhadd", "__urhadd", "__uhadd",
  "__float_as_int", "__int_as_float", "__float_as_uint", "__uint_as_float", "__double_as_longlong", "__longlong_as_double", "__double2hiint", "__double2loint", "__hiloint2double",
  "__half_as_short", "__half_as_ushort", "__short_as_half", "__ushort_as_half", "__bfloat16_as_ushort", "__ushort_as_bfloat16",
  "__float2half", "__float2half_rn", "__half2float", "__float2bfloat16", "__float2bfloat16_rn", "__bfloat162float", "__double2half", "__double2bfloat16",
  "__float2half2_rn", "__floats2half2_rn", "__half22float2", "__float22half2_rn", "__low2float", "__high2float", "__low2half", "__high2half",
  "__halves2half2", "__half2half2", "__lows2half2", "__highs2half2", "__floats2bfloat162_rn", "__bfloat1622float2", "__float2bfloat162_rn", "__low2bfloat16", "__high2bfloat16",
  "__hadd", "__hsub", "__hmul", "__hdiv", "__hfma", "__hneg", "__habs", "__hmax", "__hmin", "__heq", "__hne", "__hlt", "__hle", "__hgt", "__hge", "__hisnan", "__hisinf",
  "__hadd_sat", "__hmul_sat", "__hsub_sat", "__hfma_sat", "__hmax_nan", "__hmin_nan", "__hadd2", "__hsub2", "__hmul2", "__h2div", "__hfma2", "__hneg2", "__habs2", "__hmax2", "__hmin2",
  "hsqrt", "hrsqrt", "hexp", "hexp2", "hexp10", "hlog", "hlog2", "hlog10", "hsin", "hcos", "hceil", "hfloor", "htrunc", "hrint", "hrcp", "h2sqrt", "h2exp", "h2log", "h2rcp",
  "dot", "length", "normalize", "cross", "lerp", "clamp", "fracf", "__builtin_expect", "__builtin_assume", "__assume", "__builtin_unreachable",
  "__builtin_inff", "__builtin_inf", "__builtin_nanf", "__builtin_nan", "__builtin_huge_valf", "__builtin_huge_val", "__builtin_isnan", "__builtin_isinf", "__builtin_fabsf",
  "__lanemask_lt", "__lanemask_le", "__lanemask_eq", "__lanemask_gt", "__lanemask_ge", "__laneid", "__isGlobal", "__isShared", "__cvta_generic_to_shared",
  "__dp4a", "__dp2a_lo", "__dp2a_hi", "__vabsdiffu4",
];
const CONV_RE = /^__(float|double|int|uint|ll|ull)2(int|uint|ll|ull|float|double)_(rn|rz|rd|ru)$/;

export const BUILTIN_FUNCTIONS: Record<string, true> = Object.fromEntries([
  ...Object.keys(MATH1).flatMap((n) => [n, n + "f"]), ...Object.keys(MATH2).flatMap((n) => [n, n + "f"]), ...Object.keys(FAST1), ...MAKE_VEC.keys(),
  ...Object.keys(ATOMIC_NAMES).flatMap((n) => [n, n + "_block", n + "_system"]), ...SIMPLE,
  ...["float", "double", "int", "uint", "ll", "ull"].flatMap((a) => ["int", "uint", "ll", "ull", "float", "double"].flatMap((b) => ["rn", "rz", "rd", "ru"].map((m) => `__${a}2${b}_${m}`))),
].map((n) => [n, true]));

type RV = Val & { k: "rv" };

export function lowerBuiltinCall(L: Lowering, name: string, argsE: AExpr[], loc: SourceLoc, targs?: TemplateArg[]): Val {
  void targs;
  const val = (i: number): RV => {
    if (!argsE[i]) fail("type", `${name}: missing argument ${i + 1}`, loc);
    const v = L.rvalue(L.expr(argsE[i]));
    if (v.k !== "rv") fail("type", `${name}: argument ${i + 1} must be a number or vector (got ${typeName(L.typeOfVal(v))})`, loc);
    return v as RV;
  };
  const nargs = (n: number, max = n) => { if (argsE.length < n || argsE.length > max) fail("type", `${name} expects ${n === max ? n : `${n}-${max}`} argument(s), got ${argsE.length}`, loc); };
  const rv = (e: Expr, t: CType): Val => ({ k: "rv", e: foldExpr(e, L.optimizing), t });
  const scalarOf = (v: RV): Scalar => { if (v.t.k !== "scalar") fail("type", `${name}: expected a scalar argument, got ${typeName(v.t)}`, loc); return v.t.s; };
  const callE = (fn: MathFn, s: Scalar, args: Expr[]): Expr => ({ op: "call", fn, t: vt(s), args: args.map((a) => cast(vt(s), a)) });
  /** floating math with C++ overloading: f32 for float args (and the f-suffixed name), f64 for double / integers */
  const floatType = (vals: RV[], forceF32: boolean): Scalar => {
    if (forceF32) return "f32";
    let s: Scalar = "f32";
    for (const v of vals) {
      const vs = scalarOf(v);
      if (vs === "f64" || isInt(vs) || vs === "bool") s = "f64";
    }
    if (vals.every((v) => (v.t as { s: Scalar }).s === "f16" || (v.t as { s: Scalar }).s === "bf16")) s = "f32";
    return s;
  };
  const halfS = (v: RV): Scalar => { const s = v.t.k === "vec" ? v.t.s : scalarOf(v); if (s !== "f16" && s !== "bf16") fail("type", `${name} expects __half or __nv_bfloat16 arguments (got ${typeName(v.t)})`, loc); L.features.add(s === "f16" ? "fp16" : "bf16"); return s; };

  // -------------------------------------------------- math
  let base = name;
  let forceF32 = false;
  if (!(name in MATH1) && !(name in MATH2) && name.endsWith("f") && (name.slice(0, -1) in MATH1 || name.slice(0, -1) in MATH2)) { base = name.slice(0, -1); forceF32 = true; }
  if (base in MATH1) {
    nargs(1);
    const a = val(0);
    if (a.t.k === "vec") return vecMap(L, a, (x) => callE(MATH1[base], "f32", [x]), loc);
    const s = floatType([a], forceF32);
    if (s === "f64") L.features.add("fp64");
    return rv(callE(MATH1[base], s, [a.e]), sc(s));
  }
  if (base in MATH2) {
    nargs(2);
    const a = val(0), b = val(1);
    const s = floatType([a, b], forceF32);
    if (s === "f64") L.features.add("fp64");
    return rv(callE(MATH2[base], s, [a.e, b.e]), sc(s));
  }
  if (name in FAST1) { nargs(1); const a = val(0); const s: Scalar = name.startsWith("__d") ? "f64" : "f32"; return rv(callE(FAST1[name], s, [a.e]), sc(s)); }

  // -------------------------------------------------- vector construction
  const mv = MAKE_VEC.get(name);
  if (mv) {
    nargs(mv.n);
    const items = argsE.map((_, i) => cast(vt(mv.s), val(i).e));
    if (mv.s === "f64") L.features.add("fp64");
    return rv({ op: "vec", t: vt(mv.s, mv.n), items }, mv.n === 1 ? { k: "vec", s: mv.s, n: 1, name: mv.name } : { k: "vec", s: mv.s, n: mv.n, name: mv.name });
  }

  // -------------------------------------------------- conversions with explicit rounding
  const cm = CONV_RE.exec(name);
  if (cm) {
    nargs(1);
    const S: Record<string, Scalar> = { float: "f32", double: "f64", int: "i32", uint: "u32", ll: "i64", ull: "u64" };
    const from = S[cm[1]], to = S[cm[2]], mode = cm[3];
    const a = cast(vt(from), val(0).e);
    if (isFloat(from) && !isFloat(to)) {
      const fn: MathFn = mode === "rn" ? "rint" : mode === "rz" ? "trunc" : mode === "rd" ? "floor" : "ceil";
      return rv(cast(vt(to), callE(fn, from, [a])), sc(to));
    }
    if (mode !== "rn" && from !== to) {
      // e.g. __int2float_rz: directed rounding of int->float conversions is not available on WGSL/WASM
      if (!(isInt(from) && to === "f64" && (from === "i32" || from === "u32"))) fail("unsupported", `${name}: only round-to-nearest (_rn) integer/float conversions are supported`, loc);
    }
    return rv(cast(vt(to), a), sc(to));
  }

  // -------------------------------------------------- atomics
  let an = name;
  if (an.endsWith("_block")) an = an.slice(0, -6);
  else if (an.endsWith("_system")) an = an.slice(0, -7);
  if (an in ATOMIC_NAMES) return atomic(L, name, ATOMIC_NAMES[an], argsE, loc);

  switch (name) {
    // ------------------------------------------------ barriers and fences
    case "__syncthreads": nargs(0); L.emit({ s: "barrier", scope: "block" }); L.features.add("barrier"); return { k: "void" };
    case "__syncthreads_count": case "__syncthreads_and": case "__syncthreads_or": {
      nargs(1);
      const p = L.toBool(val(0).e, val(0).t);
      return syncReduce(L, name, p);
    }
    case "__syncwarp": nargs(0, 1); if (argsE[0]) val(0); L.emit({ s: "barrier", scope: "warp" }); L.features.add("warp-sync"); return { k: "void" };
    case "__threadfence_block": nargs(0); L.emit({ s: "fence", scope: "block" }); return { k: "void" };
    case "__threadfence": nargs(0); L.emit({ s: "fence", scope: "device" }); L.features.add("fence-device"); return { k: "void" };
    case "__threadfence_system": nargs(0); L.emit({ s: "fence", scope: "system" }); L.features.add("fence-device"); return { k: "void" };

    // ------------------------------------------------ warp collectives
    case "__shfl": case "__shfl_up": case "__shfl_down": case "__shfl_xor": case "__ballot": case "__any": case "__all":
      fail("unsupported", `${name} was removed in CUDA 9 for compute capability 7.0+: use ${name}_sync(0xffffffff, ...)`, loc);
    case "__shfl_sync": case "__shfl_up_sync": case "__shfl_down_sync": case "__shfl_xor_sync": {
      nargs(3, 4);
      val(0); // mask: all lanes are assumed to participate (documented)
      const v = val(1);
      const width = argsE[3] ? Number((L.constValue(argsE[3], `${name} width`) as { v: number }).v) : 32;
      if (![1, 2, 4, 8, 16, 32].includes(width)) fail("type", `${name}: width must be a power of two <= 32 (got ${width})`, loc);
      const lane = cast(T_I32, val(2).e);
      const fn = name.slice(2, -5) as WarpFn;
      return warpOp(L, fn, v, [lane], width);
    }
    case "__ballot_sync": case "__any_sync": case "__all_sync": {
      nargs(2);
      val(0);
      const p = L.toBool(val(1).e, val(1).t);
      const fn = name === "__ballot_sync" ? "ballot" : name === "__any_sync" ? "any" : "all";
      const r = L.newVar(fn, fn === "ballot" ? T_U32 : T_I32);
      L.emit({ s: "warp", fn, args: [p], width: 32, result: r });
      L.features.add("warp");
      return { k: "rv", e: readVar(r), t: fn === "ballot" ? sc("u32") : sc("i32") };
    }
    case "__reduce_add_sync": case "__reduce_min_sync": case "__reduce_max_sync": {
      nargs(2);
      val(0);
      const v = val(1);
      const s = scalarOf(v);
      if (s !== "i32" && s !== "u32") fail("type", `${name} takes an int or unsigned value`, loc);
      const fn = name.slice(2, -5) as WarpFn;
      return warpOp(L, fn, v, [], 32);
    }
    case "__activemask": nargs(0); L.sink.warn("unsupported", "__activemask() returns 0xffffffff: lanes are not tracked individually; it is exact only in converged code", loc); return rv(cnst(T_U32, 0xffffffff), sc("u32"));
    case "__lanemask_lt": case "__lanemask_le": case "__lanemask_eq": case "__lanemask_gt": case "__lanemask_ge": case "__laneid": {
      nargs(0);
      const lane: Expr = { op: "builtin", t: T_U32, name: "lane" };
      L.features.add("builtin:lane");
      if (name === "__laneid") return rv(lane, sc("u32"));
      const one = cnst(T_U32, 1);
      const lt = bin("-", bin("<<", one, lane), one);
      const eq = bin("<<", one, lane);
      const m = name === "__lanemask_lt" ? lt : name === "__lanemask_le" ? bin("|", lt, eq) : name === "__lanemask_eq" ? eq : name === "__lanemask_gt" ? { op: "unop", fn: "~", t: T_U32, a: bin("|", lt, eq) } as Expr : { op: "unop", fn: "~", t: T_U32, a: lt } as Expr;
      return rv(m, sc("u32"));
    }

    // ------------------------------------------------ memory loads/stores with cache hints
    case "__ldg": case "__ldcg": case "__ldca": case "__ldcs": case "__ldlu": case "__ldcv": {
      nargs(1);
      const p = L.rvalue(L.expr(argsE[0]));
      if (p.k !== "ptr") fail("type", `${name} expects a pointer`, loc);
      return L.rvalue(L.deref(p, loc));
    }
    case "__stcg": case "__stcs": case "__stwb": case "__stwt": {
      nargs(2);
      const p = L.rvalue(L.expr(argsE[0]));
      if (p.k !== "ptr") fail("type", `${name} expects a pointer`, loc);
      const target = L.deref(p, loc) as Val & { k: "lv" };
      L.assign(target.lv, L.expr(argsE[1]), loc);
      return { k: "void" };
    }

    // ------------------------------------------------ printf / assert / trap
    case "printf": {
      if (!argsE.length) fail("type", "printf needs a format string", loc);
      const f = L.expr(argsE[0]);
      if (f.k !== "str") fail("unsupported", "printf's format must be a string literal", loc);
      const args = argsE.slice(1).map((a, i) => {
        const v = L.rvalue(L.expr(a));
        if (v.k === "str") fail("unsupported", `printf argument ${i + 2}: only string literals as the format are supported, not as %s arguments`, loc);
        if (v.k === "ptr") fail("unsupported", `printf argument ${i + 2}: printing pointers (%p) is not supported`, loc);
        if (v.k !== "rv" || v.t.k !== "scalar") fail("type", `printf argument ${i + 2} must be a scalar`, loc);
        return v.e;
      });
      checkPrintf(f.s, args, loc);
      L.emit({ s: "printf", fmt: f.s, args });
      L.features.add("printf");
      return rv(i32(args.length), sc("i32"));
    }
    case "assert": case "__assert_fail": {
      nargs(1, 4);
      const c = L.toBool(val(0).e, val(0).t);
      const text = exprText(argsE[0]);
      L.emit({ s: "assert", c, msg: `${loc.file}:${loc.line}: Assertion \`${text}\` failed.`, line: loc.line });
      L.features.add("assert");
      return { k: "void" };
    }
    case "__trap": case "__brkpt": nargs(0); L.emit({ s: "trap", msg: `${name}() at ${loc.file}:${loc.line}` }); L.features.add("assert"); return { k: "void" };
    case "__nanosleep": nargs(1); val(0); return { k: "void" };
    case "clock": case "clock64": case "__globaltimer":
      fail("unsupported", `${name}() is not available: browsers expose no per-thread cycle counter (time launches with CUDA events instead)`, loc);
    case "__builtin_expect": nargs(2); return val(0);
    case "__builtin_assume": case "__assume": nargs(1); val(0); return { k: "void" };
    case "__builtin_unreachable": return { k: "void" };
    case "__builtin_inff": case "__builtin_huge_valf": return rv(cnst(vt("f32"), Infinity), sc("f32"));
    case "__builtin_inf": case "__builtin_huge_val": L.features.add("fp64-const"); return rv(cnst(vt("f64"), Infinity), sc("f64"));
    case "__builtin_nanf": return rv(cnst(vt("f32"), NaN), sc("f32"));
    case "__builtin_nan": return rv(cnst(vt("f64"), NaN), sc("f64"));
    case "__builtin_isnan": case "__builtin_isinf": { nargs(1); const a = val(0); return rv(cast(T_I32, callE(name.endsWith("nan") ? "isnan" : "isinf", scalarOf(a) === "f64" ? "f64" : "f32", [a.e])), sc("i32")); }
    case "__builtin_fabsf": nargs(1); return rv(callE("fabs", "f32", [val(0).e]), sc("f32"));
    case "__isGlobal": case "__isShared": case "__cvta_generic_to_shared": fail("unsupported", `${name}: generic address space queries are not supported`, loc);

    // ------------------------------------------------ min / max / abs
    case "min": case "max": case "umin": case "umax": case "llmin": case "llmax": case "ullmin": case "ullmax": {
      nargs(2);
      const a = val(0), b = val(1);
      const fn: MathFn = name.endsWith("min") ? "min" : "max";
      let s: Scalar;
      if (name.startsWith("u") && name.length === 4) s = "u32";
      else if (name.startsWith("ll")) s = "i64";
      else if (name.startsWith("ull")) s = "u64";
      else {
        if (a.t.k === "vec" || b.t.k === "vec") return vecZip(L, a, b, (x, y, es) => callE(isFloat(es) ? (fn === "min" ? "fmin" : "fmax") : fn, es, [x, y]), loc);
        const c = commonScalar(scalarOf(a), scalarOf(b));
        if (!c) fail("type", `${name}: ambiguous argument types ${typeName(a.t)} and ${typeName(b.t)}`, loc);
        s = c;
      }
      if (s === "f16" || s === "bf16") return rv(cast(vt(s), callE(fn === "min" ? "fmin" : "fmax", "f32", [a.e, b.e])), sc(s));
      return rv(callE(isFloat(s) ? (fn === "min" ? "fmin" : "fmax") : fn, s, [a.e, b.e]), sc(s));
    }
    case "abs": case "labs": case "llabs": {
      nargs(1);
      const a = val(0);
      let s = scalarOf(a);
      if (name === "labs" || name === "llabs") s = "i64";
      if (isFloat(s)) return rv(callE("fabs", s, [a.e]), sc(s));
      const ps: Scalar = s === "i64" || s === "u64" ? s : "i32";
      return rv(callE("abs", ps, [a.e]), sc(ps));
    }
    case "fma": case "fmaf": case "__fmaf_rn": case "__fma_rn": {
      nargs(3);
      const a = val(0), b = val(1), c = val(2);
      const s = name === "fma" || name === "__fma_rn" ? floatType([a, b, c], false) : "f32";
      return rv(callE("fma", s, [a.e, b.e, c.e]), sc(s));
    }
    case "ldexp": case "ldexpf": {
      nargs(2);
      const a = val(0);
      const s = name === "ldexpf" ? "f32" : floatType([a], false);
      return rv({ op: "call", fn: "ldexp", t: vt(s), args: [cast(vt(s), a.e), cast(T_I32, val(1).e)] }, sc(s));
    }
    case "isnan": case "isinf": case "isfinite": case "signbit": case "__isnanf": case "__isinff": case "__finitef": case "__signbitf": case "__isnan": case "__isinf": {
      nargs(1);
      const a = val(0);
      const fn = (name.includes("nan") ? "isnan" : name.includes("inf") ? "isinf" : name.includes("finite") ? "isfinite" : "signbit") as MathFn;
      const s = name.startsWith("__") && name.endsWith("f") ? "f32" : floatType([a], false);
      return rv({ op: "call", fn, t: T_BOOL, args: [cast(vt(s), a.e)] }, sc("bool"));
    }
    case "sincos": case "sincosf": case "__sincosf": case "sincospi": case "sincospif": {
      nargs(3);
      const a = val(0);
      const s = name === "sincos" || name === "sincospi" ? floatType([a], false) : "f32";
      let x = cast(vt(s), a.e);
      if (name.includes("pi")) x = bin("*", x, cnst(vt(s), Math.PI));
      x = L.tmp(x, "sc");
      for (const [i, fn] of [[1, "sin"], [2, "cos"]] as const) {
        const p = L.rvalue(L.expr(argsE[i]));
        if (p.k !== "ptr") fail("type", `${name}: argument ${i + 1} must be a pointer`, loc);
        const target = L.deref(p, loc) as Val & { k: "lv" };
        L.assign(target.lv, { k: "rv", e: callE(fn, s, [x]), t: sc(s) }, loc);
      }
      return { k: "void" };
    }
    case "sinpi": case "sinpif": case "cospi": case "cospif": {
      nargs(1);
      const a = val(0);
      const s = name.endsWith("f") ? "f32" : floatType([a], false);
      return rv(callE(name.startsWith("sin") ? "sin" : "cos", s, [bin("*", cast(vt(s), a.e), cnst(vt(s), Math.PI))]), sc(s));
    }
    case "rcbrt": case "rcbrtf": { nargs(1); const a = val(0); const s = name.endsWith("f") ? "f32" : floatType([a], false); return rv(bin("/", cnst(vt(s), 1), callE("cbrt", s, [a.e])), sc(s)); }
    case "frexp": case "frexpf": case "modf": case "modff": {
      nargs(2);
      const a = val(0);
      const s = name.endsWith("f") ? "f32" : floatType([a], false);
      const x = L.tmp(cast(vt(s), a.e), "x");
      const p = L.rvalue(L.expr(argsE[1]));
      if (p.k !== "ptr") fail("type", `${name}: second argument must be a pointer`, loc);
      const target = L.deref(p, loc) as Val & { k: "lv" };
      if (name.startsWith("modf")) {
        const ip = L.tmp(callE("trunc", s, [x]), "ip");
        L.assign(target.lv, { k: "rv", e: ip, t: sc(s) }, loc);
        return rv(callE("copysign", s, [{ op: "select", t: vt(s), c: { op: "call", fn: "isinf", t: T_BOOL, args: [x] }, a: cnst(vt(s), 0), b: bin("-", x, ip) }, x]), sc(s));
      }
      // frexp: x = m * 2^e with 0.5 <= |m| < 1
      const isZero = bin("||", bin("==", x, cnst(vt(s), 0)), bin("||", { op: "call", fn: "isinf", t: T_BOOL, args: [x] }, { op: "call", fn: "isnan", t: T_BOOL, args: [x] }));
      const e = L.tmp({ op: "select", t: T_I32, c: isZero, a: i32(0), b: bin("+", cast(T_I32, callE("floor", s, [callE("log2", s, [callE("fabs", s, [x])])])), i32(1)) }, "e");
      // correct off-by-one from log2 rounding
      const m0 = L.tmp({ op: "call", fn: "ldexp", t: vt(s), args: [x, bin("-", i32(0), e)] }, "m");
      const big = bin(">=", callE("fabs", s, [m0]), cnst(vt(s), 1));
      const small = bin("&&", bin("<", callE("fabs", s, [m0]), cnst(vt(s), 0.5)), bin("!=", m0, cnst(vt(s), 0)));
      const e2 = L.tmp({ op: "select", t: T_I32, c: big, a: bin("+", e, i32(1)), b: { op: "select", t: T_I32, c: small, a: bin("-", e, i32(1)), b: e } }, "e");
      L.assign(target.lv, { k: "rv", e: e2, t: sc("i32") }, loc);
      return rv({ op: "select", t: vt(s), c: isZero, a: x, b: { op: "call", fn: "ldexp", t: vt(s), args: [x, bin("-", i32(0), e2)] } }, sc(s));
    }
    case "__fdividef": case "__fdiv_rn": case "__ddiv_rn": { nargs(2); const s: Scalar = name === "__ddiv_rn" ? "f64" : "f32"; return rv(bin("/", cast(vt(s), val(0).e), cast(vt(s), val(1).e)), sc(s)); }
    case "__frcp_rn": case "__drcp_rn": { nargs(1); const s: Scalar = name === "__drcp_rn" ? "f64" : "f32"; return rv(bin("/", cnst(vt(s), 1), cast(vt(s), val(0).e)), sc(s)); }
    case "__fadd_rn": case "__fsub_rn": case "__fmul_rn": case "__dadd_rn": case "__dsub_rn": case "__dmul_rn": {
      nargs(2);
      const s: Scalar = name.startsWith("__d") ? "f64" : "f32";
      const op = name.includes("add") ? "+" : name.includes("sub") ? "-" : "*";
      return rv(bin(op, cast(vt(s), val(0).e), cast(vt(s), val(1).e)), sc(s));
    }
    case "__saturatef": case "saturate": { nargs(1); return rv(callE("saturate", "f32", [val(0).e]), sc("f32")); }
    case "__powf": { nargs(2); return rv(callE("pow", "f32", [val(0).e, val(1).e]), sc("f32")); }

    // ------------------------------------------------ integer intrinsics
    case "__popc": case "__popcll": { nargs(1); const s: Scalar = name === "__popcll" ? "u64" : "u32"; return rv(cast(T_I32, callE("popc", s, [val(0).e])), sc("i32")); }
    case "__clz": case "__clzll": { nargs(1); const s: Scalar = name === "__clzll" ? "i64" : "i32"; return rv(cast(T_I32, callE("clz", s, [val(0).e])), sc("i32")); }
    case "__ffs": case "__ffsll": { nargs(1); const s: Scalar = name === "__ffsll" ? "i64" : "i32"; return rv(cast(T_I32, callE("ffs", s, [val(0).e])), sc("i32")); }
    case "__brev": case "__brevll": { nargs(1); const s: Scalar = name === "__brevll" ? "u64" : "u32"; return rv(callE("brev", s, [val(0).e]), sc(s)); }
    case "__mulhi": case "__umulhi": case "__mul64hi": case "__umul64hi": {
      nargs(2);
      const s: Scalar = name === "__mulhi" ? "i32" : name === "__umulhi" ? "u32" : name === "__mul64hi" ? "i64" : "u64";
      return rv(callE("mulhi", s, [val(0).e, val(1).e]), sc(s));
    }
    case "__mul24": case "__umul24": {
      nargs(2);
      const s: Scalar = name === "__mul24" ? "i32" : "u32";
      // the low 32 bits of the product of the low 24 bits (sign-extended for __mul24)
      const lo = (e: Expr) => s === "i32" ? bin(">>", bin("<<", cast(T_I32, e), i32(8)), i32(8)) : bin("&", cast(T_U32, e), cnst(T_U32, 0xffffff));
      return rv(bin("*", lo(val(0).e), lo(val(1).e)), sc(s));
    }
    case "__funnelshift_l": case "__funnelshift_r": case "__funnelshift_lc": case "__funnelshift_rc": {
      nargs(3);
      const lo = cast(T_U32, val(0).e), hi = cast(T_U32, val(1).e), sh0 = cast(T_U32, val(2).e);
      const clamp = name.endsWith("c");
      const sh = clamp ? callE("min", "u32", [sh0, cnst(T_U32, 32)]) : bin("&", sh0, cnst(T_U32, 31));
      const w = cast(vt("u64"), bin("|", bin("<<", cast(vt("u64"), hi), cnst(vt("u64"), 32n)), cast(vt("u64"), lo)));
      const shifted = name.startsWith("__funnelshift_l") ? bin(">>", bin("<<", w, cast(vt("u64"), sh)), cnst(vt("u64"), 32n)) : bin(">>", w, cast(vt("u64"), sh));
      L.features.add("int64");
      return rv(cast(T_U32, shifted), sc("u32"));
    }
    case "__byte_perm": {
      nargs(3);
      const x = L.tmp(cast(T_U32, val(0).e)), y = L.tmp(cast(T_U32, val(1).e)), sel = L.tmp(cast(T_U32, val(2).e));
      let r: Expr = cnst(T_U32, 0);
      for (let i = 0; i < 4; i++) {
        const nib = bin("&", bin(">>", sel, cnst(T_U32, 4 * i)), cnst(T_U32, 7));
        const src: Expr = { op: "select", t: T_U32, c: bin("<", nib, cnst(T_U32, 4)), a: x, b: y };
        const byte = bin("&", bin(">>", src, bin("*", bin("&", nib, cnst(T_U32, 3)), cnst(T_U32, 8))), cnst(T_U32, 0xff));
        r = bin("|", r, bin("<<", byte, cnst(T_U32, 8 * i)));
      }
      return rv(r, sc("u32"));
    }
    case "__sad": case "__usad": {
      nargs(3);
      const s: Scalar = name === "__sad" ? "i32" : "u32";
      const a = L.tmp(cast(vt(s), val(0).e)), b = L.tmp(cast(vt(s), val(1).e));
      const d: Expr = { op: "select", t: T_U32, c: bin(">", a, b), a: cast(T_U32, bin("-", a, b)), b: cast(T_U32, bin("-", b, a)) };
      return rv(bin("+", d, cast(T_U32, val(2).e)), sc("u32"));
    }
    case "__rhadd": case "__urhadd": case "__uhadd": {
      nargs(2);
      const s: Scalar = name === "__rhadd" ? "i32" : "u32";
      const a = L.tmp(cast(vt(s), val(0).e)), b = L.tmp(cast(vt(s), val(1).e));
      // (a + b [+1]) >> 1 without overflow
      const one = cnst(vt(s), 1);
      const r = name === "__uhadd" ? bin("+", bin("&", a, b), bin(">>", bin("^", a, b), one)) : bin("-", bin("|", a, b), bin(">>", bin("^", a, b), one));
      return rv(r, sc(s));
    }
    case "__dp4a": case "__dp2a_lo": case "__dp2a_hi": {
      // __dp4a(int a, int b, int c): c + sum of the four signed byte products (CUDA 8.1+, sm_61+)
      nargs(3);
      const av = val(0), bv = val(1);
      const signedA = av.t.k === "scalar" && av.t.s === "i32", signedB = bv.t.k === "scalar" && bv.t.s === "i32";
      const a = L.tmp(cast(T_I32, av.e)), b = L.tmp(cast(T_I32, bv.e));
      const byteOf = (x: Expr, i: number, signed: boolean, bits: number) =>
        signed ? bin(">>", bin("<<", x, i32(32 - bits * (i + 1))), i32(32 - bits)) : bin("&", bin(">>", x, i32(bits * i)), i32((1 << bits) - 1));
      let acc = cast(T_I32, val(2).e);
      if (name === "__dp4a") for (let i = 0; i < 4; i++) acc = bin("+", acc, bin("*", byteOf(a, i, signedA, 8), byteOf(b, i, signedB, 8)));
      else {
        const off = name === "__dp2a_hi" ? 2 : 0;
        for (let i = 0; i < 2; i++) acc = bin("+", acc, bin("*", byteOf(a, i, signedA, 16), byteOf(b, i + off, signedB, 8)));
      }
      L.features.add("dp4a");
      return rv(acc, sc("i32"));
    }
    case "__vabsdiffu4": {
      nargs(2);
      const a = L.tmp(cast(T_U32, val(0).e)), b = L.tmp(cast(T_U32, val(1).e));
      let r: Expr = cnst(T_U32, 0);
      for (let i = 0; i < 4; i++) {
        const x = bin("&", bin(">>", a, cnst(T_U32, 8 * i)), cnst(T_U32, 255)), y = bin("&", bin(">>", b, cnst(T_U32, 8 * i)), cnst(T_U32, 255));
        r = bin("|", r, bin("<<", { op: "select", t: T_U32, c: bin(">", x, y), a: bin("-", x, y), b: bin("-", y, x) }, cnst(T_U32, 8 * i)));
      }
      return rv(r, sc("u32"));
    }

    // ------------------------------------------------ reinterpretation
    case "__float_as_int": nargs(1); return rv({ op: "bitcast", t: T_I32, a: cast(vt("f32"), val(0).e) }, sc("i32"));
    case "__float_as_uint": nargs(1); return rv({ op: "bitcast", t: T_U32, a: cast(vt("f32"), val(0).e) }, sc("u32"));
    case "__int_as_float": nargs(1); return rv({ op: "bitcast", t: vt("f32"), a: cast(T_I32, val(0).e) }, sc("f32"));
    case "__uint_as_float": nargs(1); return rv({ op: "bitcast", t: vt("f32"), a: cast(T_U32, val(0).e) }, sc("f32"));
    case "__double_as_longlong": nargs(1); return rv({ op: "bitcast", t: vt("i64"), a: cast(vt("f64"), val(0).e) }, sc("i64"));
    case "__longlong_as_double": nargs(1); return rv({ op: "bitcast", t: vt("f64"), a: cast(vt("i64"), val(0).e) }, sc("f64"));
    case "__double2hiint": nargs(1); return rv(cast(T_I32, bin(">>", { op: "bitcast", t: vt("i64"), a: cast(vt("f64"), val(0).e) }, cnst(vt("i64"), 32n))), sc("i32"));
    case "__double2loint": nargs(1); return rv(cast(T_I32, { op: "bitcast", t: vt("i64"), a: cast(vt("f64"), val(0).e) }), sc("i32"));
    case "__hiloint2double": {
      nargs(2);
      const w = bin("|", bin("<<", cast(vt("i64"), val(0).e), cnst(vt("i64"), 32n)), cast(vt("i64"), cast(T_U32, val(1).e)));
      return rv({ op: "bitcast", t: vt("f64"), a: w }, sc("f64"));
    }
    case "__half_as_short": case "__half_as_ushort": case "__bfloat16_as_ushort": {
      nargs(1);
      const a = val(0);
      const to: Scalar = name === "__half_as_short" ? "i16" : "u16";
      return rv({ op: "bitcast", t: vt(to), a: a.e }, sc(to));
    }
    case "__short_as_half": case "__ushort_as_half": case "__ushort_as_bfloat16": {
      nargs(1);
      const to: Scalar = name === "__ushort_as_bfloat16" ? "bf16" : "f16";
      const from: Scalar = name === "__short_as_half" ? "i16" : "u16";
      L.features.add(to === "f16" ? "fp16" : "bf16");
      return rv({ op: "bitcast", t: vt(to), a: cast(vt(from), val(0).e) }, sc(to));
    }

    // ------------------------------------------------ half / bfloat16 conversions
    case "__float2half": case "__float2half_rn": case "__double2half": nargs(1); L.features.add("fp16"); return rv(cast(vt("f16"), cast(vt(name === "__double2half" ? "f64" : "f32"), val(0).e)), sc("f16"));
    case "__float2bfloat16": case "__float2bfloat16_rn": case "__double2bfloat16": nargs(1); L.features.add("bf16"); return rv(cast(vt("bf16"), cast(vt(name === "__double2bfloat16" ? "f64" : "f32"), val(0).e)), sc("bf16"));
    case "__half2float": case "__bfloat162float": { nargs(1); const a = val(0); halfS(a); return rv(cast(vt("f32"), a.e), sc("f32")); }
    case "__float2half2_rn": case "__float2bfloat162_rn": {
      nargs(1);
      const s: Scalar = name.includes("bfloat") ? "bf16" : "f16";
      const h = L.tmp(cast(vt(s), cast(vt("f32"), val(0).e)));
      L.features.add(s === "f16" ? "fp16" : "bf16");
      return rv({ op: "vec", t: vt(s, 2), items: [h, h] }, { k: "vec", s, n: 2, name: s === "f16" ? "__half2" : "__nv_bfloat162" });
    }
    case "__floats2half2_rn": case "__floats2bfloat162_rn": case "__float22half2_rn": {
      const s: Scalar = name.includes("bfloat") ? "bf16" : "f16";
      L.features.add(s === "f16" ? "fp16" : "bf16");
      let xs: Expr[];
      if (name === "__float22half2_rn") { nargs(1); const v = L.tmp(val(0).e); xs = [0, 1].map((i) => ({ op: "extract", t: vt("f32"), a: v, lane: i }) as Expr); }
      else { nargs(2); xs = [val(0).e, val(1).e]; }
      return rv({ op: "vec", t: vt(s, 2), items: xs.map((x) => cast(vt(s), cast(vt("f32"), x))) }, { k: "vec", s, n: 2, name: s === "f16" ? "__half2" : "__nv_bfloat162" });
    }
    case "__half22float2": case "__bfloat1622float2": {
      nargs(1);
      const v = L.tmp(val(0).e);
      return rv({ op: "vec", t: vt("f32", 2), items: [0, 1].map((i) => cast(vt("f32"), { op: "extract", t: vt(v.t.s), a: v, lane: i })) }, { k: "vec", s: "f32", n: 2, name: "float2" });
    }
    case "__low2float": case "__high2float": { nargs(1); const v = val(0); return rv(cast(vt("f32"), { op: "extract", t: vt((v.t as { s: Scalar }).s), a: v.e, lane: name === "__low2float" ? 0 : 1 }), sc("f32")); }
    case "__low2half": case "__high2half": case "__low2bfloat16": case "__high2bfloat16": {
      nargs(1);
      const v = val(0); const s = halfS(v);
      return rv({ op: "extract", t: vt(s), a: v.e, lane: name.includes("low") ? 0 : 1 }, sc(s));
    }
    case "__halves2half2": { nargs(2); const a = val(0), b = val(1); const s = halfS(a); halfS(b); return rv({ op: "vec", t: vt(s, 2), items: [a.e, b.e] }, { k: "vec", s, n: 2, name: "__half2" }); }
    case "__half2half2": { nargs(1); const a = val(0); const s = halfS(a); const h = L.tmp(a.e); return rv({ op: "vec", t: vt(s, 2), items: [h, h] }, { k: "vec", s, n: 2, name: "__half2" }); }
    case "__lows2half2": case "__highs2half2": {
      nargs(2);
      const a = L.tmp(val(0).e), b = L.tmp(val(1).e);
      const lane = name === "__lows2half2" ? 0 : 1;
      const s = a.t.s;
      return rv({ op: "vec", t: vt(s, 2), items: [{ op: "extract", t: vt(s), a, lane }, { op: "extract", t: vt(s), a: b, lane }] }, { k: "vec", s, n: 2, name: "__half2" });
    }

    // ------------------------------------------------ half / bfloat16 arithmetic (computed in float, rounded once)
    case "__hadd": case "__hsub": case "__hmul": case "__hdiv": case "__hadd_sat": case "__hsub_sat": case "__hmul_sat": {
      nargs(2);
      const a = val(0), b = val(1);
      if (a.t.k === "scalar" && isInt(a.t.s)) {
        // integer __hadd(int, int): (a + b) >> 1 without overflow
        const s: Scalar = a.t.s === "u32" ? "u32" : "i32";
        const x = L.tmp(cast(vt(s), a.e)), y = L.tmp(cast(vt(s), b.e));
        return rv(bin("+", bin("&", x, y), bin(">>", bin("^", x, y), cnst(vt(s), 1))), sc(s));
      }
      const s = halfS(a);
      const op = name.includes("add") ? "+" : name.includes("sub") ? "-" : name.includes("mul") ? "*" : "/";
      let r: Expr = bin(op, cast(vt("f32"), a.e), cast(vt("f32"), b.e), vt("f32"));
      if (name.endsWith("_sat")) r = callE("saturate", "f32", [r]);
      return rv(cast(vt(s), r), sc(s));
    }
    case "__hfma": case "__hfma_sat": {
      nargs(3);
      const a = val(0), b = val(1), c = val(2);
      const s = halfS(a);
      let r = callE("fma", "f32", [cast(vt("f32"), a.e), cast(vt("f32"), b.e), cast(vt("f32"), c.e)]);
      if (name.endsWith("_sat")) r = callE("saturate", "f32", [r]);
      return rv(cast(vt(s), r), sc(s));
    }
    case "__hneg": case "__habs": {
      nargs(1);
      const a = val(0); const s = halfS(a);
      const f = cast(vt("f32"), a.e);
      return rv(cast(vt(s), name === "__hneg" ? { op: "unop", fn: "-", t: vt("f32"), a: f } : callE("fabs", "f32", [f])), sc(s));
    }
    case "__hmax": case "__hmin": case "__hmax_nan": case "__hmin_nan": {
      nargs(2);
      const a = val(0), b = val(1); const s = halfS(a);
      const fn: MathFn = name.includes("max") ? "fmax" : "fmin";
      return rv(cast(vt(s), callE(fn, "f32", [cast(vt("f32"), a.e), cast(vt("f32"), b.e)])), sc(s));
    }
    case "__heq": case "__hne": case "__hlt": case "__hle": case "__hgt": case "__hge": {
      nargs(2);
      const a = val(0), b = val(1); halfS(a);
      const op = ({ __heq: "==", __hne: "!=", __hlt: "<", __hle: "<=", __hgt: ">", __hge: ">=" } as const)[name];
      return rv(bin(op, cast(vt("f32"), a.e), cast(vt("f32"), b.e)), sc("bool"));
    }
    case "__hisnan": case "__hisinf": { nargs(1); const a = val(0); halfS(a); return rv(cast(T_I32, { op: "call", fn: name === "__hisnan" ? "isnan" : "isinf", t: T_BOOL, args: [cast(vt("f32"), a.e)] }), sc("i32")); }
    case "hsqrt": case "hrsqrt": case "hexp": case "hexp2": case "hexp10": case "hlog": case "hlog2": case "hlog10": case "hsin": case "hcos":
    case "hceil": case "hfloor": case "htrunc": case "hrint": case "hrcp": {
      nargs(1);
      const a = val(0); const s = halfS(a);
      const f = cast(vt("f32"), a.e);
      const r = name === "hrcp" ? bin("/", cnst(vt("f32"), 1), f) : callE(name.slice(1) as MathFn, "f32", [f]);
      return rv(cast(vt(s), r), sc(s));
    }
    case "__hadd2": case "__hsub2": case "__hmul2": case "__h2div": case "__hfma2": case "__hneg2": case "__habs2": case "__hmax2": case "__hmin2":
    case "h2sqrt": case "h2exp": case "h2log": case "h2rcp": {
      const vals = argsE.map((_, i) => val(i));
      const s = halfS(vals[0]);
      const tv = vals.map((v) => L.tmp(v.e));
      const lanes = [0, 1].map((lane) => {
        const x = tv.map((v) => cast(vt("f32"), { op: "extract", t: vt(s), a: v, lane } as Expr));
        let r: Expr;
        switch (name) {
          case "__hadd2": r = bin("+", x[0], x[1]); break;
          case "__hsub2": r = bin("-", x[0], x[1]); break;
          case "__hmul2": r = bin("*", x[0], x[1]); break;
          case "__h2div": r = bin("/", x[0], x[1]); break;
          case "__hfma2": r = callE("fma", "f32", x); break;
          case "__hneg2": r = { op: "unop", fn: "-", t: vt("f32"), a: x[0] }; break;
          case "__habs2": r = callE("fabs", "f32", [x[0]]); break;
          case "__hmax2": r = callE("fmax", "f32", x); break;
          case "__hmin2": r = callE("fmin", "f32", x); break;
          case "h2rcp": r = bin("/", cnst(vt("f32"), 1), x[0]); break;
          default: r = callE(name.slice(2) as MathFn, "f32", [x[0]]);
        }
        return cast(vt(s), r);
      });
      return rv({ op: "vec", t: vt(s, 2), items: lanes }, vals[0].t);
    }

    // ------------------------------------------------ helper_math.h-style vector helpers
    case "dot": {
      nargs(2);
      const a = val(0), b = val(1);
      if (a.t.k !== "vec" || b.t.k !== "vec" || a.t.n !== b.t.n) fail("type", "dot expects two vectors of the same type", loc);
      const ta = L.tmp(a.e), tb = L.tmp(b.e);
      const s = a.t.s;
      let acc: Expr | null = null;
      for (let i = 0; i < a.t.n; i++) {
        const p = bin("*", { op: "extract", t: vt(s), a: ta, lane: i }, { op: "extract", t: vt(s), a: tb, lane: i });
        acc = acc ? bin("+", acc, p) : p;
      }
      return rv(acc!, sc(s));
    }
    case "length": {
      nargs(1);
      const a = val(0);
      if (a.t.k !== "vec") fail("type", "length expects a vector", loc);
      const ta = L.tmp(a.e);
      let acc: Expr | null = null;
      for (let i = 0; i < a.t.n; i++) { const x: Expr = { op: "extract", t: vt(a.t.s), a: ta, lane: i }; acc = acc ? bin("+", acc, bin("*", x, x)) : bin("*", x, x); }
      return rv(callE("sqrt", a.t.s, [acc!]), sc(a.t.s));
    }
    case "normalize": {
      nargs(1);
      const a = val(0);
      if (a.t.k !== "vec") fail("type", "normalize expects a vector", loc);
      const ta = L.tmp(a.e);
      let acc: Expr | null = null;
      for (let i = 0; i < a.t.n; i++) { const x: Expr = { op: "extract", t: vt(a.t.s), a: ta, lane: i }; acc = acc ? bin("+", acc, bin("*", x, x)) : bin("*", x, x); }
      const inv = L.tmp(callE("rsqrt", a.t.s, [acc!]));
      return rv({ op: "vec", t: ta.t, items: Array.from({ length: a.t.n }, (_, i) => bin("*", { op: "extract", t: vt((a.t as { s: Scalar }).s), a: ta, lane: i }, inv)) }, a.t);
    }
    case "cross": {
      nargs(2);
      const a = val(0), b = val(1);
      if (a.t.k !== "vec" || a.t.n !== 3) fail("type", "cross expects float3 arguments", loc);
      const ta = L.tmp(a.e), tb = L.tmp(b.e);
      const c = (v: Expr, i: number): Expr => ({ op: "extract", t: vt("f32"), a: v, lane: i });
      const items = [bin("-", bin("*", c(ta, 1), c(tb, 2)), bin("*", c(ta, 2), c(tb, 1))), bin("-", bin("*", c(ta, 2), c(tb, 0)), bin("*", c(ta, 0), c(tb, 2))), bin("-", bin("*", c(ta, 0), c(tb, 1)), bin("*", c(ta, 1), c(tb, 0)))];
      return rv({ op: "vec", t: ta.t, items }, a.t);
    }
    case "lerp": {
      nargs(3);
      const a = val(0), b = val(1), t = val(2);
      const tt = L.tmp(cast(vt("f32"), t.e));
      if (a.t.k === "vec") return vecZip(L, a, b, (x, y) => bin("+", x, bin("*", tt, bin("-", y, x))), loc);
      return rv(bin("+", cast(vt("f32"), a.e), bin("*", tt, bin("-", cast(vt("f32"), b.e), cast(vt("f32"), a.e)))), sc("f32"));
    }
    case "clamp": {
      nargs(3);
      const x = val(0), lo = val(1), hi = val(2);
      if (x.t.k === "vec") {
        const tl = L.tmp(lo.t.k === "vec" ? lo.e : cast(vt(x.t.s), lo.e)), th = L.tmp(hi.t.k === "vec" ? hi.e : cast(vt(x.t.s), hi.e));
        const es = x.t.s, n = x.t.n, tx = L.tmp(x.e);
        const pick = (v: Expr, i: number): Expr => (v.t.n > 1 ? { op: "extract", t: vt(es), a: v, lane: i } : v);
        return rv({ op: "vec", t: tx.t, items: Array.from({ length: n }, (_, i) => callE(isFloat(es) ? "fmin" : "min", es, [callE(isFloat(es) ? "fmax" : "max", es, [pick(tx, i), pick(tl, i)]), pick(th, i)])) }, x.t);
      }
      const s = commonScalar(scalarOf(x), commonScalar(scalarOf(lo), scalarOf(hi)) ?? "f32") ?? "f32";
      return rv(callE(isFloat(s) ? "fmin" : "min", s, [callE(isFloat(s) ? "fmax" : "max", s, [x.e, lo.e]), hi.e]), sc(s));
    }
    case "fracf": { nargs(1); const a = L.tmp(cast(vt("f32"), val(0).e)); return rv(bin("-", a, callE("floor", "f32", [a])), sc("f32")); }
  }
  fail("unsupported", `built-in function '${name}' is not supported`, loc);
}

function vecMap(L: Lowering, a: RV, f: (x: Expr, s: Scalar) => Expr, loc: SourceLoc): Val {
  void loc;
  const t = a.t as CType & { k: "vec" };
  const ta = L.tmp(a.e);
  return { k: "rv", e: { op: "vec", t: vt(t.s, t.n), items: Array.from({ length: t.n }, (_, i) => cast(vt(t.s), f({ op: "extract", t: vt(t.s), a: ta, lane: i }, t.s))) }, t };
}
function vecZip(L: Lowering, a: RV, b: RV, f: (x: Expr, y: Expr, s: Scalar) => Expr, loc: SourceLoc): Val {
  const t = (a.t.k === "vec" ? a.t : b.t) as CType & { k: "vec" };
  if (a.t.k === "vec" && b.t.k === "vec" && (a.t.n !== b.t.n || a.t.s !== b.t.s)) fail("type", "vector operands of different types", loc);
  const ta = L.tmp(a.e), tb = L.tmp(b.e);
  const pick = (v: Expr, i: number): Expr => (v.t.n > 1 ? { op: "extract", t: vt(t.s), a: v, lane: i } : cast(vt(t.s), v));
  return { k: "rv", e: { op: "vec", t: vt(t.s, t.n), items: Array.from({ length: t.n }, (_, i) => cast(vt(t.s), f(pick(ta, i), pick(tb, i), t.s))) }, t };
}

function warpOp(L: Lowering, fn: WarpFn, v: RV, extra: Expr[], width: number): Val {
  L.features.add("warp");
  // 64-bit, half2 and vector values are shuffled lane by lane / word by word by the backends: keep them typed here
  const t: VT = v.t.k === "vec" ? vt(v.t.s, v.t.n) : vt((v.t as { s: Scalar }).s);
  const r = L.newVar(fn, t);
  L.emit({ s: "warp", fn, args: [v.e, ...extra], width, result: r });
  return { k: "rv", e: readVar(r), t: v.t };
}

function syncReduce(L: Lowering, name: string, pred: Expr): Val {
  // block-wide reduction over a hidden shared counter: barrier; zero; barrier; count; barrier; read
  let root = L.roots.find((r) => (r as { syncCounter?: boolean }).syncCounter);
  if (!root) { root = L.newRoot("sync_count", "shared", 4, T_I32, { align: 4 }); (root as { syncCounter?: boolean }).syncCounter = true; L.sharedBytes += 4; }
  const tid0: Expr = bin("==", { op: "builtin", t: T_U32, name: "tid.linear" }, cnst(T_U32, 0));
  L.features.add("barrier");
  L.emit({ s: "barrier", scope: "block" });
  L.emit({ s: "if", c: tid0, then: [{ s: "store", root, off: i32(0), t: T_I32, e: i32(0) }], else: [] });
  L.emit({ s: "barrier", scope: "block" });
  L.emit({ s: "if", c: pred, then: [{ s: "atomic", fn: "add", root, off: i32(0), t: T_I32, args: [i32(1)], result: null }], else: [] });
  L.emit({ s: "barrier", scope: "block" });
  const cnt = L.newVar("count", T_I32);
  L.emit({ s: "let", v: cnt, e: { op: "load", t: T_I32, root, off: i32(0) } });
  L.emit({ s: "barrier", scope: "block" });
  const total = bin("*", bin("*", L.builtinComponent("blockDim", "x"), L.builtinComponent("blockDim", "y")), L.builtinComponent("blockDim", "z"));
  if (name === "__syncthreads_count") return { k: "rv", e: readVar(cnt), t: sc("i32") };
  if (name === "__syncthreads_or") return { k: "rv", e: cast(T_I32, bin("!=", readVar(cnt), i32(0))), t: sc("i32") };
  return { k: "rv", e: cast(T_I32, bin("==", cast(T_U32, readVar(cnt)), total)), t: sc("i32") };
}

function atomic(L: Lowering, name: string, fn: AtomicFn, argsE: AExpr[], loc: SourceLoc): Val {
  const n = fn === "cas" ? 3 : 2;
  if (argsE.length !== n) fail("type", `${name} expects ${n} arguments`, loc);
  const p = L.rvalue(L.expr(argsE[0]));
  if (p.k !== "ptr") fail("type", `${name}: first argument must be a pointer`, loc);
  const target = L.deref(p, loc) as Val & { k: "lv" };
  const lv = L.stableLV(target.lv);
  const t = lvType(lv);
  if (t.k !== "scalar") fail("type", `${name} on ${typeName(t)} is not supported`, loc);
  const s = t.s;
  const allowed: Record<AtomicFn, Scalar[]> = {
    add: ["i32", "u32", "u64", "f32", "f64", "i64"], sub: ["i32", "u32"], xchg: ["i32", "u32", "u64", "f32"], min: ["i32", "u32", "i64", "u64"], max: ["i32", "u32", "i64", "u64"],
    inc: ["u32"], dec: ["u32"], and: ["i32", "u32", "u64"], or: ["i32", "u32", "u64"], xor: ["i32", "u32", "u64"], cas: ["i32", "u32", "u64", "u16"],
  };
  if (!allowed[fn].includes(s)) fail("type", `no overload of ${name} for ${typeName(t)}*`, loc);
  const args: Expr[] = [];
  for (let i = 1; i < n; i++) {
    const v = L.rvalue(L.expr(argsE[i]));
    if (v.k !== "rv") fail("type", `${name}: argument ${i + 1} must be a number`, loc);
    args.push(cast(vt(s), (L.snapshot(v) as RV).e));
  }
  const result = L.newVar("old", vt(s));
  if (lv.k === "mem") {
    if (lv.root.space === "local") fail("type", `${name} on thread-local memory`, loc);
    L.emit({ s: "atomic", fn, root: lv.root, off: lv.off, t: vt(s), args, result });
    L.features.add(`atomic-${lv.root.space}`);
    if (s === "f32" || s === "f64") L.features.add("atomic-float");
    if (s === "u64" || s === "i64" || s === "f64") L.features.add("atomic-64");
  } else {
    // atomic on a register: a plain read-modify-write (only this thread can see it)
    const old = L.rvalue({ k: "lv", lv }) as RV;
    L.emit({ s: "let", v: result, e: old.e });
    const o = readVar(result);
    let nv: Expr;
    switch (fn) {
      case "add": nv = bin("+", o, args[0]); break;
      case "sub": nv = bin("-", o, args[0]); break;
      case "xchg": nv = args[0]; break;
      case "min": nv = { op: "call", fn: "min", t: vt(s), args: [o, args[0]] }; break;
      case "max": nv = { op: "call", fn: "max", t: vt(s), args: [o, args[0]] }; break;
      case "and": nv = bin("&", o, args[0]); break;
      case "or": nv = bin("|", o, args[0]); break;
      case "xor": nv = bin("^", o, args[0]); break;
      case "inc": nv = { op: "select", t: vt(s), c: bin(">=", o, args[0]), a: cnst(vt(s), 0), b: bin("+", o, cnst(vt(s), 1)) }; break;
      case "dec": nv = { op: "select", t: vt(s), c: bin("||", bin("==", o, cnst(vt(s), 0)), bin(">", o, args[0])), a: args[0], b: bin("-", o, cnst(vt(s), 1)) }; break;
      case "cas": nv = { op: "select", t: vt(s), c: bin("==", o, args[0]), a: args[1], b: o }; break;
    }
    L.assign(lv, { k: "rv", e: nv, t }, loc);
  }
  void bool;
  return { k: "rv", e: readVar(result), t };
}

function checkPrintf(fmt: string, args: Expr[], loc: SourceLoc): void {
  const re = /%([-+ #0]*)(\d+|\*)?(?:\.(\d+|\*))?(hh|h|ll|l|z|j|t)?([diuxXofeEgGcsaAp%])/g;
  let m: RegExpExecArray | null;
  let n = 0;
  while ((m = re.exec(fmt)) !== null) {
    if (m[5] === "%") continue;
    if (m[2] === "*" || m[3] === "*") fail("unsupported", "printf: '*' width/precision is not supported", loc);
    if (m[5] === "s" || m[5] === "p") fail("unsupported", `printf: %${m[5]} is not supported (string and pointer arguments)`, loc);
    const a = args[n++];
    if (!a) fail("type", `printf: format '${m[0]}' has no matching argument`, loc);
    const isF = "feEgGaA".includes(m[5]);
    if (isF !== isFloat(a.t.s)) fail("type", `printf: format '${m[0]}' expects ${isF ? "a floating point" : "an integer"} argument, got ${a.t.s}`, loc);
  }
  if (n !== args.length) fail("type", `printf: ${args.length} arguments for ${n} format specifiers`, loc);
}

function exprText(e: AExpr): string {
  switch (e.k) {
    case "ident": return e.name;
    case "num": return e.text;
    case "binary": return `${exprText(e.l)} ${e.op} ${exprText(e.r)}`;
    case "unary": return `${e.op}${exprText(e.e)}`;
    case "index": return `${exprText(e.base)}[${exprText(e.index)}]`;
    case "member": return `${exprText(e.base)}${e.arrow ? "->" : "."}${e.name}`;
    case "call": return `${exprText(e.callee)}(${e.args.map(exprText).join(", ")})`;
    default: return "...";
  }
}
