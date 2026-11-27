/** Encoding kernel scalar arguments, and formatting device printf output. */
import type { VT } from "../compiler/ir.js";
import { halfBits, bf16Bits } from "../compiler/fold.js";
import { SCALAR_SIZE, type Scalar } from "../compiler/types.js";
import { CudaError } from "./errors.js";
import type { ScalarValue } from "./backend.js";

const f32 = new Float32Array(1), u32v = new Uint32Array(f32.buffer);
const f64 = new Float64Array(1), u64v = new BigUint64Array(f64.buffer);

function scalarBits(s: Scalar, v: number | bigint | boolean): bigint {
  const n = typeof v === "boolean" ? (v ? 1 : 0) : v;
  switch (s) {
    case "f32": f32[0] = Number(n); return BigInt(u32v[0]);
    case "f64": f64[0] = Number(n); return u64v[0];
    case "f16": return BigInt(halfBits(Math.fround(Number(n))));
    case "bf16": return BigInt(bf16Bits(Number(n)));
    case "bool": return n ? 1n : 0n;
    default: {
      if (typeof n === "number" && !Number.isInteger(n)) throw new CudaError("cudaErrorInvalidValue", `expected an integer for a ${s} argument, got ${n}`);
      const bits = SCALAR_SIZE[s] * 8;
      return BigInt.asUintN(bits, BigInt(n));
    }
  }
}

/** Little-endian bytes of a scalar or vector value in C layout (used by the WASM backend and serialization). */
export function encodeScalarBytes(t: VT, value: ScalarValue): Uint8Array {
  const es = SCALAR_SIZE[t.s];
  const vals = Array.isArray(value) ? value : [value];
  if (vals.length !== t.n) throw new CudaError("cudaErrorInvalidValue", `expected ${t.n} component(s) for a ${t.s}x${t.n} argument, got ${vals.length}`);
  const out = new Uint8Array(es * t.n);
  vals.forEach((v, i) => {
    let b = scalarBits(t.s, v as number | bigint | boolean);
    for (let k = 0; k < es; k++) { out[i * es + k] = Number(b & 0xffn); b >>= 8n; }
  });
  return out;
}

/**
 * 32-bit words for the WGSL parameter buffer: 4-byte components one word each, 8-byte values as two words (lo, hi),
 * 1- and 2-byte components packed little-endian into words. f64 values (only after fp64 demotion) are sent as float.
 */
export function encodeScalarWords(t: VT, value: ScalarValue): Uint32Array {
  if (t.s === "f64") {
    const vals = Array.isArray(value) ? value : [value];
    return Uint32Array.from(vals.map((v) => Number(scalarBits("f32", v as number))));
  }
  const es = SCALAR_SIZE[t.s];
  const bytes = encodeScalarBytes(t, value);
  if (es >= 4) return new Uint32Array(bytes.buffer, 0, bytes.byteLength / 4).slice();
  const out = new Uint32Array(Math.ceil(bytes.length / 4));
  for (let i = 0; i < bytes.length; i++) out[i >> 2] |= bytes[i] << ((i & 3) * 8);
  return out;
}

/** C printf formatting of device printf records (argument words as written by the backends). */
export function formatPrintf(fmt: string, types: string[], words: Uint32Array | number[]): string {
  let wi = 0, ai = 0;
  const next = (): { v: number | bigint; s: string } => {
    const s = types[ai++] ?? "i32";
    if (s === "i64" || s === "u64") {
      const lo = BigInt(words[wi++] >>> 0), hi = BigInt(words[wi++] >>> 0);
      const u = lo | (hi << 32n);
      return { v: s === "i64" ? BigInt.asIntN(64, u) : u, s };
    }
    if (s === "f64") {
      // the WASM backend writes doubles as two words
      const lo = BigInt(words[wi++] >>> 0), hi = BigInt(words[wi++] >>> 0);
      u64v[0] = lo | (hi << 32n);
      return { v: f64[0], s };
    }
    const w = words[wi++] >>> 0;
    if (s === "f32" || s === "f16" || s === "bf16") { u32v[0] = w; return { v: f32[0], s }; }
    if (s === "i32" || s === "i8" || s === "i16") return { v: w | 0, s };
    if (s === "bool") return { v: w ? 1 : 0, s };
    return { v: w, s };
  };
  return fmt.replace(/%([-+ #0]*)(\d+)?(?:\.(\d+))?(hh|h|ll|l|z|j|t)?([diuxXofeEgGcaA%])/g, (_m, flags: string, width: string | undefined, prec: string | undefined, len: string | undefined, conv: string) => {
    if (conv === "%") return "%";
    const a = next();
    let body: string;
    const p = prec === undefined ? undefined : Number(prec);
    switch (conv) {
      case "d": case "i": body = BigInt(a.v).toString(); if (p !== undefined) body = padNum(body, p); break;
      case "u": {
        let v = BigInt(a.v);
        if (v < 0n) v = BigInt.asUintN(len === "ll" || len === "l" ? 64 : 32, v);
        body = v.toString(); if (p !== undefined) body = padNum(body, p); break;
      }
      case "x": case "X": case "o": {
        let v = BigInt(a.v);
        if (v < 0n) v = BigInt.asUintN(len === "ll" || len === "l" ? 64 : 32, v);
        body = v.toString(conv === "o" ? 8 : 16);
        if (conv === "X") body = body.toUpperCase();
        if (p !== undefined) body = padNum(body, p);
        if (flags.includes("#") && v !== 0n) body = (conv === "o" ? "0" : conv === "x" ? "0x" : "0X") + body;
        break;
      }
      case "c": body = String.fromCharCode(Number(a.v)); break;
      case "f": case "F": body = fmtFixed(Number(a.v), p ?? 6); break;
      case "e": case "E": body = fmtExp(Number(a.v), p ?? 6); if (conv === "E") body = body.toUpperCase(); break;
      case "g": case "G": body = fmtG(Number(a.v), p ?? 6, flags.includes("#")); if (conv === "G") body = body.toUpperCase(); break;
      case "a": case "A": body = Number(a.v).toString(16); break;
      default: body = String(a.v);
    }
    const num = "dieEfFgGaA".includes(conv);
    if (num && !body.startsWith("-")) { if (flags.includes("+")) body = "+" + body; else if (flags.includes(" ")) body = " " + body; }
    const w = width ? Number(width) : 0;
    if (body.length < w) {
      if (flags.includes("-")) body = body.padEnd(w);
      else if (flags.includes("0") && num && Number.isFinite(Number(a.v))) {
        const sign = /^[+\- ]/.test(body) ? body[0] : "";
        body = sign + body.slice(sign.length).padStart(w - sign.length, "0");
      } else body = body.padStart(w);
    }
    return body;
  });
}

function padNum(s: string, p: number): string { const neg = s.startsWith("-"); const d = neg ? s.slice(1) : s; return (neg ? "-" : "") + d.padStart(p, "0"); }
function special(v: number): string | null { if (Number.isNaN(v)) return "nan"; if (!Number.isFinite(v)) return v > 0 ? "inf" : "-inf"; return null; }
function fmtFixed(v: number, p: number): string { return special(v) ?? (Object.is(v, -0) ? "-" + (0).toFixed(p) : v.toFixed(p)); }
function fmtExp(v: number, p: number): string {
  const sp = special(v); if (sp) return sp;
  const s = v.toExponential(p); // 1.5e+0
  return s.replace(/e([+-])(\d)$/, "e$10$2");
}
function fmtG(v: number, p: number, alt: boolean): string {
  const sp = special(v); if (sp) return sp;
  const P = p === 0 ? 1 : p;
  if (v === 0) return Object.is(v, -0) ? "-0" : "0";
  const x = Math.floor(Math.log10(Math.abs(Number(v.toExponential(P - 1)))));
  let s = x < -4 || x >= P ? fmtExp(v, P - 1) : v.toFixed(Math.max(0, P - 1 - x));
  if (!alt) {
    if (s.includes("e")) s = s.replace(/\.?0+e/, "e");
    else if (s.includes(".")) s = s.replace(/\.?0+$/, "");
  }
  return s;
}
