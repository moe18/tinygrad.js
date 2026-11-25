/**
 * Constant folding and algebraic simplification with exact C semantics for every IR type (wrapping 32/64-bit integers,
 * float rounding via Math.fround, IEEE half/bfloat16 rounding). Only operations whose result is bit-exact on every backend are
 * folded: transcendental functions are left to the backend, so optimized and direct translations compute the same values.
 */
import { bin, cnst, vtEq, type BinOp, type Expr, type VT } from "./ir.js";
import { isFloat, type Scalar } from "./types.js";

type CV = number | bigint | boolean;

const F32 = new Float32Array(1), U32A = new Uint32Array(F32.buffer);
const F64 = new Float64Array(1), U64A = new BigUint64Array(F64.buffer);

/** Round a number to the nearest IEEE binary16 value (round to nearest even), returned as a JS number. */
export function roundHalf(x: number): number {
  if (!Number.isFinite(x) || x === 0) return x;
  const a = Math.abs(x);
  if (a >= 65520) return x > 0 ? Infinity : -Infinity; // 65520 rounds up to inf
  let e = Math.floor(Math.log2(a));
  if (Math.pow(2, e) > a) e--;
  if (Math.pow(2, e + 1) <= a) e++;
  const ulpExp = Math.max(e, -14) - 10; // subnormals below 2^-14
  const q = a / Math.pow(2, ulpExp);
  let r = Math.floor(q);
  const frac = q - r;
  if (frac > 0.5 || (frac === 0.5 && r % 2 === 1)) r++;
  return Math.sign(x) * r * Math.pow(2, ulpExp);
}
export function roundBf16(x: number): number {
  if (!Number.isFinite(x)) return x;
  F32[0] = x; // f64 -> f32 first is a double rounding; exact when x is already an f32 value (it always is in kernels)
  let b = U32A[0];
  if ((b & 0x7fffffff) > 0x7f800000) return NaN;
  const lsb = (b >>> 16) & 1;
  b = (b + 0x7fff + lsb) >>> 0;
  U32A[0] = b & 0xffff0000;
  return F32[0];
}
export function halfBits(x: number): number {
  const h = roundHalf(x);
  if (Number.isNaN(h)) return 0x7e00;
  const sign = h < 0 || Object.is(h, -0) ? 0x8000 : 0;
  const a = Math.abs(h);
  if (a === Infinity) return sign | 0x7c00;
  if (a === 0) return sign;
  if (a < Math.pow(2, -14)) return sign | Math.round(a / Math.pow(2, -24));
  const e = Math.floor(Math.log2(a));
  const m = Math.round((a / Math.pow(2, e) - 1) * 1024);
  return sign | ((e + 15) << 10) | m;
}
export function halfFromBits(b: number): number {
  const s = b & 0x8000 ? -1 : 1, e = (b >> 10) & 0x1f, m = b & 0x3ff;
  if (e === 0) return s * m * Math.pow(2, -24);
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * Math.pow(2, e - 15);
}
export function bf16Bits(x: number): number { F32[0] = roundBf16(x); return U32A[0] >>> 16; }
export function bf16FromBits(b: number): number { U32A[0] = (b & 0xffff) << 16; return F32[0]; }

const INT_BITS: Partial<Record<Scalar, [number, boolean]>> = {
  i8: [8, true], u8: [8, false], i16: [16, true], u16: [16, false], i32: [32, true], u32: [32, false], i64: [64, true], u64: [64, false],
};

/** Normalize a value to the representation used for type s (number for <=32-bit ints and floats, bigint for 64-bit). */
export function normalize(s: Scalar, v: CV): CV {
  if (s === "bool") return typeof v === "boolean" ? v : typeof v === "bigint" ? v !== 0n : v !== 0;
  if (isFloat(s)) {
    const n = typeof v === "boolean" ? (v ? 1 : 0) : Number(v);
    return s === "f64" ? n : s === "f32" ? Math.fround(n) : s === "f16" ? roundHalf(Math.fround(n)) : roundBf16(n);
  }
  const [bits, signed] = INT_BITS[s]!;
  const b = typeof v === "boolean" ? (v ? 1n : 0n) : typeof v === "bigint" ? v : BigInt(Math.trunc(v));
  const r = signed ? BigInt.asIntN(bits, b) : BigInt.asUintN(bits, b);
  return bits === 64 ? r : Number(r);
}

/** C conversion of a constant; null if the result is undefined behaviour (float -> int out of range, NaN). */
export function convertConst(from: Scalar, to: Scalar, v: CV): CV | null {
  if (to === "bool") return normalize("bool", v);
  if (isFloat(to)) {
    if (typeof v === "bigint" && (v > 2n ** 53n || v < -(2n ** 53n)) && to !== "f64") return null; // avoid double rounding
    return normalize(to, typeof v === "bigint" ? Number(v) : v);
  }
  if (isFloat(from)) {
    const n = v as number;
    if (!Number.isFinite(n)) return null;
    const t = Math.trunc(n);
    const [bits, signed] = INT_BITS[to]!;
    const lo = signed ? -(2 ** (bits - 1)) : 0, hi = signed ? 2 ** (bits - 1) : 2 ** bits;
    if (t < lo || t >= hi) return null;
    return normalize(to, BigInt(t));
  }
  return normalize(to, v);
}

const asBig = (v: CV) => (typeof v === "bigint" ? v : typeof v === "boolean" ? (v ? 1n : 0n) : BigInt(v));

function evalBin(fn: BinOp, s: Scalar, a: CV, b: CV): CV | null {
  if (fn === "&&") return Boolean(a) && Boolean(b);
  if (fn === "||") return Boolean(a) || Boolean(b);
  if (isFloat(s)) {
    const x = a as number, y = b as number;
    switch (fn) {
      case "+": return normalize(s, x + y);
      case "-": return normalize(s, x - y);
      case "*": return normalize(s, x * y);
      // f64 x/y then fround: correctly rounded f32 division (double rounding is innocuous: 53 >= 2*24+2)
      case "/": return normalize(s, x / y);
      case "==": return x === y; case "!=": return x !== y;
      case "<": return x < y; case "<=": return x <= y; case ">": return x > y; case ">=": return x >= y;
      default: return null;
    }
  }
  if (s === "bool") {
    const x = Boolean(a), y = Boolean(b);
    switch (fn) { case "==": return x === y; case "!=": return x !== y; case "&": return x && y; case "|": return x || y; case "^": return x !== y; default: return null; }
  }
  const [bits, signed] = INT_BITS[s]!;
  const x = asBig(a), y = asBig(b);
  let r: bigint;
  switch (fn) {
    case "+": r = x + y; break;
    case "-": r = x - y; break;
    case "*": r = x * y; break;
    case "/": if (y === 0n) return null; r = x / y; break;   // BigInt division truncates toward zero, like C
    case "%": if (y === 0n) return null; r = x % y; break;   // sign of the dividend, like C
    case "&": r = x & y; break;
    case "|": r = x | y; break;
    case "^": r = x ^ y; break;
    case "<<": if (y < 0n || y >= BigInt(bits)) return null; r = x << y; break;
    case ">>": if (y < 0n || y >= BigInt(bits)) return null; r = x >> y; break; // x is already sign/zero-correct for its type
    case "==": return x === y; case "!=": return x !== y;
    case "<": return x < y; case "<=": return x <= y; case ">": return x > y; case ">=": return x >= y;
    default: return null;
  }
  r = signed ? BigInt.asIntN(bits, r) : BigInt.asUintN(bits, r);
  return bits === 64 ? r : Number(r);
}

const INVERT: Partial<Record<BinOp, BinOp>> = { "<": ">=", ">=": "<", ">": "<=", "<=": ">", "==": "!=", "!=": "==" };

export const isConst = (e: Expr): e is Expr & { op: "const" } => e.op === "const";
export const constValue = (e: Expr): CV | undefined => (e.op === "const" ? e.v : undefined);
export function isConstEq(e: Expr, n: number): boolean {
  if (e.op !== "const") return false;
  if (typeof e.v === "bigint") return e.v === BigInt(n);
  if (typeof e.v === "boolean") return (e.v ? 1 : 0) === n;
  return e.v === n && !(n === 0 && Object.is(e.v, -0) && isFloat(e.t.s));
}

function isPow2(v: CV): number {
  const b = asBig(v);
  if (b <= 0n || (b & (b - 1n)) !== 0n) return -1;
  return b.toString(2).length - 1;
}

/** Fold one node whose children are already folded. `simplify` enables algebraic identities (optimized mode). */
export function foldNode(e: Expr, simplify: boolean): Expr {
  switch (e.op) {
    case "unop": {
      if (e.a.op !== "const" || e.t.n !== 1) {
        if (e.a.op === "unop" && e.a.fn === e.fn && (e.fn === "-" || e.fn === "~" || e.fn === "!") && vtEq(e.a.t, e.t)) return e.a.a;
        // !(a < b) -> a >= b for integers (not for floats: NaN compares false both ways)
        if (e.fn === "!" && e.a.op === "binop" && INVERT[e.a.fn] && !isFloat(e.a.a.t.s)) return { ...e.a, fn: INVERT[e.a.fn]! };
        return e;
      }
      const v = e.a.v, s = e.t.s;
      if (e.fn === "!") return cnst(e.t, !v);
      if (e.fn === "-") return isFloat(s) ? cnst(e.t, -(v as number)) : cnst(e.t, normalize(s, -asBig(v)));
      if (e.fn === "~") return s === "bool" ? e : cnst(e.t, normalize(s, ~asBig(v)));
      return e;
    }
    case "binop": {
      const { a, b } = e;
      if (a.op === "const" && b.op === "const" && a.t.n === 1 && b.t.n === 1) {
        const r = evalBin(e.fn, a.t.s, a.v, b.v);
        if (r !== null) return cnst(e.t, r);
        return e;
      }
      // short-circuit with a constant left side is always exact
      if (e.fn === "&&" && a.op === "const") return a.v ? cast1(b) : cnst(e.t, false);
      if (e.fn === "||" && a.op === "const") return a.v ? cnst(e.t, true) : cast1(b);
      if (!simplify || e.t.n !== 1) return e;
      const s = a.t.s, int = !isFloat(s) && s !== "bool";
      switch (e.fn) {
        case "+":
          if (int && isConstEq(b, 0)) return a;
          if (int && isConstEq(a, 0)) return b;
          // (x + c1) + c2 -> x + (c1 + c2)
          if (int && b.op === "const" && a.op === "binop" && a.fn === "+" && a.b.op === "const") return foldNode(bin("+", a.a, foldNode(bin("+", a.b, b), true)), true);
          if (int && b.op === "const" && a.op === "binop" && a.fn === "-" && a.b.op === "const") return foldNode(bin("+", a.a, foldNode(bin("-", b, a.b), true)), true);
          break;
        case "-":
          if ((int || isFloat(s)) && isConstEq(b, 0)) return a;
          if (int && b.op === "const") return foldNode(bin("+", a, cnst(b.t, normalize(s, -asBig(b.v)))), true);
          break;
        case "*":
          if (isConstEq(b, 1)) return a;
          if (isConstEq(a, 1)) return b;
          if (int && (isConstEq(b, 0) || isConstEq(a, 0))) return cnst(e.t, normalize(s, 0));
          if (int && b.op === "const" && a.op === "binop" && a.fn === "*" && a.b.op === "const") return foldNode(bin("*", a.a, foldNode(bin("*", a.b, b), true)), true);
          if (int && a.op === "const" && b.op !== "const") return foldNode(bin("*", b, a), true);
          break;
        case "/":
          if (isConstEq(b, 1)) return a;
          // unsigned division by a power of two is a shift
          if (int && !s.startsWith("i") && b.op === "const" && isPow2(b.v) > 0) return bin(">>", a, cnst(b.t, normalize(s, isPow2(b.v))));
          break;
        case "%":
          if (int && !s.startsWith("i") && b.op === "const" && isPow2(b.v) >= 0) return bin("&", a, cnst(b.t, normalize(s, asBig(b.v) - 1n)));
          break;
        case "&":
          if (int && (isConstEq(b, 0) || isConstEq(a, 0))) return cnst(e.t, normalize(s, 0));
          if (int && b.op === "const" && asBig(b.v) === asBig(normalize(s, -1n))) return a;
          break;
        case "|": case "^":
          if (int && isConstEq(b, 0)) return a;
          if (int && isConstEq(a, 0)) return b;
          break;
        case "<<": case ">>":
          if (isConstEq(b, 0)) return a;
          break;
        case "&&":
          if (b.op === "const") return b.v ? cast1(a) : e; // x && false keeps evaluating x? x is pure: false
          break;
      }
      return e;
    }
    case "select": {
      if (e.c.op === "const") return e.c.v ? e.a : e.b;
      if (simplify && exprEq(e.a, e.b)) return e.a;
      return e;
    }
    case "cast": {
      const a = e.a;
      if (vtEq(a.t, e.t)) return a;
      if (a.op === "const" && a.t.n === 1 && e.t.n === 1) {
        const r = convertConst(a.t.s, e.t.s, a.v);
        if (r !== null) return cnst(e.t, r);
        return e;
      }
      if (a.op === "const" && e.t.n > 1 && a.t.n === 1) {
        const r = convertConst(a.t.s, e.t.s, a.v);
        if (r !== null) return { op: "vec", t: e.t, items: Array(e.t.n).fill(cnst({ s: e.t.s, n: 1 }, r)) };
      }
      // widening integer casts compose: i64(i32(x:i16)) -> i64(x)
      if (simplify && a.op === "cast" && isWideningInt(a.a.t, a.t) && isWideningInt(a.t, e.t) && sameSignedness(a.a.t.s, a.t.s)) return { ...e, a: a.a };
      return e;
    }
    case "extract": {
      if (e.a.op === "vec") return e.a.items[e.lane];
      return e;
    }
    case "call": {
      if (!e.args.every((x) => x.op === "const") || e.t.n !== 1) return e;
      const s = e.t.s;
      const v = e.args.map((x) => (x as { v: CV }).v);
      const f = (n: number) => cnst(e.t, normalize(s, n));
      if (isFloat(s)) {
        const x = v[0] as number, y = v[1] as number;
        switch (e.fn) {
          case "sqrt": return f(Math.sqrt(x)); // correctly rounded (double rounding is innocuous for sqrt)
          case "fabs": return f(Math.abs(x));
          case "floor": return f(Math.floor(x));
          case "ceil": return f(Math.ceil(x));
          case "trunc": return f(Math.trunc(x));
          case "round": return f(Math.sign(x) * Math.floor(Math.abs(x) + 0.5));
          case "fmin": return f(Number.isNaN(x) ? y : Number.isNaN(y) ? x : Math.min(x, y));
          case "fmax": return f(Number.isNaN(x) ? y : Number.isNaN(y) ? x : Math.max(x, y));
          case "copysign": return f(Math.abs(x) * (y < 0 || Object.is(y, -0) ? -1 : 1));
          default: return e;
        }
      }
      if (s === "bool") return e;
      const x = asBig(v[0]), y = v.length > 1 ? asBig(v[1]) : 0n;
      switch (e.fn) {
        case "min": return cnst(e.t, normalize(s, x < y ? x : y));
        case "max": return cnst(e.t, normalize(s, x > y ? x : y));
        case "abs": return cnst(e.t, normalize(s, x < 0n ? -x : x));
        default: return e;
      }
    }
    default: return e;
  }
}

const cast1 = (e: Expr): Expr => e; // && / || operands are already bool
function isWideningInt(a: VT, b: VT): boolean {
  const bits: Partial<Record<Scalar, number>> = { i8: 8, u8: 8, i16: 16, u16: 16, i32: 32, u32: 32, i64: 64, u64: 64 };
  return a.n === 1 && b.n === 1 && bits[a.s] !== undefined && bits[b.s] !== undefined && bits[a.s]! <= bits[b.s]!;
}
const sameSignedness = (a: Scalar, b: Scalar) => a.startsWith("i") === b.startsWith("i");

export function foldExpr(e: Expr, simplify = true): Expr {
  switch (e.op) {
    case "unop": case "cast": case "bitcast": case "extract": return foldNode({ ...e, a: foldExpr(e.a, simplify) } as Expr, simplify);
    case "binop": return foldNode({ ...e, a: foldExpr(e.a, simplify), b: foldExpr(e.b, simplify) }, simplify);
    case "select": return foldNode({ ...e, c: foldExpr(e.c, simplify), a: foldExpr(e.a, simplify), b: foldExpr(e.b, simplify) }, simplify);
    case "call": return foldNode({ ...e, args: e.args.map((x) => foldExpr(x, simplify)) }, simplify);
    case "load": return { ...e, off: foldExpr(e.off, simplify) };
    case "vec": {
      const items = e.items.map((x) => foldExpr(x, simplify));
      return { ...e, items };
    }
    default: return e;
  }
}

/** Structural equality of pure expressions. */
export function exprEq(a: Expr, b: Expr): boolean {
  if (a === b) return true;
  if (a.op !== b.op || !vtEq(a.t, b.t)) return false;
  switch (a.op) {
    case "const": { const bv = (b as typeof a).v; return a.v === bv || (typeof a.v === "number" && Number.isNaN(a.v) && Number.isNaN(bv as number)); }
    case "var": return a.v === (b as typeof a).v;
    case "param": return a.index === (b as typeof a).index;
    case "builtin": return a.name === (b as typeof a).name;
    case "unop": return a.fn === (b as typeof a).fn && exprEq(a.a, (b as typeof a).a);
    case "binop": return a.fn === (b as typeof a).fn && exprEq(a.a, (b as typeof a).a) && exprEq(a.b, (b as typeof a).b);
    case "select": { const o = b as typeof a; return exprEq(a.c, o.c) && exprEq(a.a, o.a) && exprEq(a.b, o.b); }
    case "cast": case "bitcast": return exprEq(a.a, (b as typeof a).a);
    case "extract": return a.lane === (b as typeof a).lane && exprEq(a.a, (b as typeof a).a);
    case "call": { const o = b as typeof a; return a.fn === o.fn && a.args.length === o.args.length && a.args.every((x, i) => exprEq(x, o.args[i])); }
    case "load": { const o = b as typeof a; return a.root === o.root && exprEq(a.off, o.off); }
    case "vec": { const o = b as typeof a; return a.items.every((x, i) => exprEq(x, o.items[i])); }
  }
}

/** Divide a byte-offset expression by an element size exactly (symbolically). Returns null if not provably divisible. */
export function divExact(e: Expr, k: number): Expr | null {
  if (k === 1) return e;
  if (e.op === "const") {
    const v = asBig(e.v);
    if (v % BigInt(k) !== 0n) return null;
    return cnst(e.t, normalize(e.t.s, v / BigInt(k)));
  }
  if (e.op === "binop") {
    if (e.fn === "*") {
      if (e.b.op === "const") {
        const m = asBig(e.b.v);
        if (m % BigInt(k) === 0n) { const q = m / BigInt(k); return q === 1n ? e.a : bin("*", e.a, cnst(e.b.t, normalize(e.b.t.s, q))); }
      }
      if (e.a.op === "const") return divExact(bin("*", e.b, e.a), k);
      const l = divExact(e.a, k); if (l) return bin("*", l, e.b);
      const r = divExact(e.b, k); if (r) return bin("*", e.a, r);
      return null;
    }
    if (e.fn === "<<" && e.b.op === "const") {
      const sh = Number(asBig(e.b.v)), p = isPow2(k);
      if (p >= 0 && sh >= p) return sh === p ? e.a : bin("<<", e.a, cnst(e.b.t, normalize(e.b.t.s, sh - p)));
      return null;
    }
    if (e.fn === "+" || e.fn === "-") {
      const l = divExact(e.a, k), r = l && divExact(e.b, k);
      return l && r ? bin(e.fn, l, r) : null;
    }
  }
  if (e.op === "cast" && e.a.t.n === 1 && !isFloat(e.a.t.s)) {
    // offsets are i32; a cast from another integer type of an exactly divisible expression stays divisible
    const inner = divExact(e.a, k);
    return inner ? { ...e, a: inner } : null;
  }
  return null;
}
