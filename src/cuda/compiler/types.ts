/**
 * The C/CUDA type system used by semantic analysis: scalar, vector (float4, int2, half2, ...), pointer, array, struct types
 * with nvcc's (LP64) sizes and alignments, integer promotion and the usual arithmetic conversions.
 */
export type Scalar = "bool" | "i8" | "u8" | "i16" | "u16" | "i32" | "u32" | "i64" | "u64" | "f16" | "bf16" | "f32" | "f64";

export interface StructDef {
  name: string;
  fields: { name: string; type: CType; offset: number }[];
  size: number;
  align: number;
  isUnion: boolean;
  /** methods by name (resolved lazily by sema) */
  methods: Map<string, unknown[]>;
  /** static constexpr members: name -> constant value */
  statics: Map<string, { type: CType; value: number | bigint }>;
  /** template instantiation environment for methods */
  env?: unknown;
}

export type AddrSpace = "global" | "shared" | "local" | "constant";

export type CType =
  | { k: "void" }
  | { k: "scalar"; s: Scalar; name?: string }
  | { k: "vec"; s: Scalar; n: 1 | 2 | 3 | 4; name: string }
  | { k: "ptr"; to: CType; isConst?: boolean }
  | { k: "array"; of: CType; n: number }
  | { k: "struct"; def: StructDef }
  | { k: "nullptr" };

export const VOID: CType = { k: "void" };
export const sc = (s: Scalar, name?: string): CType & { k: "scalar" } => ({ k: "scalar", s, ...(name ? { name } : {}) });
export const BOOL = sc("bool"), I32 = sc("i32"), U32 = sc("u32"), I64 = sc("i64"), U64 = sc("u64"), F32 = sc("f32"), F64 = sc("f64");
export const F16 = sc("f16"), BF16 = sc("bf16"), I8 = sc("i8"), U8 = sc("u8"), I16 = sc("i16"), U16 = sc("u16");

export const SCALAR_SIZE: Record<Scalar, number> = { bool: 1, i8: 1, u8: 1, i16: 2, u16: 2, i32: 4, u32: 4, i64: 8, u64: 8, f16: 2, bf16: 2, f32: 4, f64: 8 };
export const isFloat = (s: Scalar) => s === "f16" || s === "bf16" || s === "f32" || s === "f64";
export const isInt = (s: Scalar) => !isFloat(s) && s !== "bool";
export const isSigned = (s: Scalar) => s === "i8" || s === "i16" || s === "i32" || s === "i64" || isFloat(s);
export const is64 = (s: Scalar) => s === "i64" || s === "u64" || s === "f64";

/** C spelling, for messages and IR dumps */
export const SCALAR_C_NAME: Record<Scalar, string> = {
  bool: "bool", i8: "signed char", u8: "unsigned char", i16: "short", u16: "unsigned short", i32: "int", u32: "unsigned int",
  i64: "long long", u64: "unsigned long long", f16: "__half", bf16: "__nv_bfloat16", f32: "float", f64: "double",
};

export const PRIM_SCALAR: Record<string, Scalar> = {
  bool: "bool", char: "i8", "signed char": "i8", "unsigned char": "u8", short: "i16", "unsigned short": "u16", int: "i32", "unsigned int": "u32",
  long: "i64", "unsigned long": "u64", "long long": "i64", "unsigned long long": "u64", float: "f32", double: "f64",
};

/** named builtin types: typedefs from stdint.h / cuda headers */
export const NAMED_SCALARS: Record<string, Scalar> = {
  size_t: "u64", ssize_t: "i64", ptrdiff_t: "i64", intptr_t: "i64", uintptr_t: "u64",
  int8_t: "i8", uint8_t: "u8", int16_t: "i16", uint16_t: "u16", int32_t: "i32", uint32_t: "u32", int64_t: "i64", uint64_t: "u64",
  uint: "u32", ushort: "u16", uchar: "u8", ulong: "u64", __half: "f16", half: "f16", __nv_bfloat16: "bf16", nv_bfloat16: "bf16",
  __bf16: "bf16", _Float16: "f16", __int128: "i64",
};

const VEC_BASES: [string, Scalar][] = [
  ["char", "i8"], ["uchar", "u8"], ["short", "i16"], ["ushort", "u16"], ["int", "i32"], ["uint", "u32"], ["long", "i64"], ["ulong", "u64"],
  ["longlong", "i64"], ["ulonglong", "u64"], ["float", "f32"], ["double", "f64"],
];
export const VECTOR_TYPES = new Map<string, { s: Scalar; n: 1 | 2 | 3 | 4 }>();
for (const [b, s] of VEC_BASES) for (const n of [1, 2, 3, 4] as const) VECTOR_TYPES.set(`${b}${n}`, { s, n });
VECTOR_TYPES.set("__half2", { s: "f16", n: 2 }); VECTOR_TYPES.set("half2", { s: "f16", n: 2 });
VECTOR_TYPES.set("__nv_bfloat162", { s: "bf16", n: 2 }); VECTOR_TYPES.set("nv_bfloat162", { s: "bf16", n: 2 });
VECTOR_TYPES.set("dim3", { s: "u32", n: 3 });

export const BUILTIN_TYPE_NAMES: string[] = [...Object.keys(NAMED_SCALARS), ...VECTOR_TYPES.keys()];

export const vecType = (s: Scalar, n: 1 | 2 | 3 | 4): CType => {
  if (n === 1) return sc(s);
  for (const [name, v] of VECTOR_TYPES) if (v.s === s && v.n === n && !name.startsWith("long") && name !== "dim3" && !name.startsWith("__")) return { k: "vec", s, n, name };
  return { k: "vec", s, n, name: `${SCALAR_C_NAME[s]}x${n}` };
};

/** CUDA's vector type alignment (vector_types.h): 2- and 4-component types are aligned to their size up to 16 bytes */
function vecAlign(s: Scalar, n: number): number {
  const es = SCALAR_SIZE[s];
  if (n === 3 || n === 1) return es;
  return Math.min(16, es * n);
}

export function sizeOf(t: CType): number {
  switch (t.k) {
    case "void": return 1;
    case "scalar": return SCALAR_SIZE[t.s];
    case "vec": return t.n === 3 ? SCALAR_SIZE[t.s] * 3 : Math.max(SCALAR_SIZE[t.s] * t.n, vecAlign(t.s, t.n));
    case "ptr": case "nullptr": return 8;
    case "array": return sizeOf(t.of) * Math.max(t.n, 0);
    case "struct": return t.def.size;
  }
}
export function alignOf(t: CType): number {
  switch (t.k) {
    case "void": return 1;
    case "scalar": return SCALAR_SIZE[t.s];
    case "vec": return vecAlign(t.s, t.n);
    case "ptr": case "nullptr": return 8;
    case "array": return alignOf(t.of);
    case "struct": return t.def.align;
  }
}

export function typeName(t: CType): string {
  switch (t.k) {
    case "void": return "void";
    case "scalar": return t.name ?? SCALAR_C_NAME[t.s];
    case "vec": return t.name;
    case "ptr": return typeName(t.to) + (t.isConst ? " const" : "") + "*";
    case "array": return `${typeName(t.of)}[${t.n < 0 ? "" : t.n}]`;
    case "struct": return t.def.name;
    case "nullptr": return "nullptr_t";
  }
}

export function sameType(a: CType, b: CType): boolean {
  if (a.k !== b.k) return false;
  switch (a.k) {
    case "scalar": return a.s === (b as typeof a).s;
    case "vec": return a.s === (b as typeof a).s && a.n === (b as typeof a).n;
    case "ptr": return sameType(a.to, (b as typeof a).to);
    case "array": return a.n === (b as typeof a).n && sameType(a.of, (b as typeof a).of);
    case "struct": return a.def === (b as typeof a).def;
    default: return true;
  }
}

export const isArith = (t: CType): t is CType & { k: "scalar" } => t.k === "scalar";
export const isIntegral = (t: CType) => t.k === "scalar" && (isInt(t.s) || t.s === "bool");
export const isFloating = (t: CType) => t.k === "scalar" && isFloat(t.s);

/** C integer promotion */
export function promote(s: Scalar): Scalar {
  if (s === "bool" || s === "i8" || s === "u8" || s === "i16" || s === "u16") return "i32";
  return s;
}

const RANK: Partial<Record<Scalar, number>> = { i32: 1, u32: 1, i64: 2, u64: 2 };

/** Usual arithmetic conversions. Returns null when the combination is ambiguous (CUDA's __half with integers). */
export function commonScalar(a: Scalar, b: Scalar): Scalar | null {
  if (a === "f64" || b === "f64") return "f64";
  if (a === "f32" || b === "f32") return "f32";
  if (a === "f16" && b === "f16") return "f16";
  if (a === "bf16" && b === "bf16") return "bf16";
  if (isFloat(a) || isFloat(b)) return null; // half/bf16 mixed with integers or each other: ambiguous in nvcc too
  const pa = promote(a), pb = promote(b);
  if (pa === pb) return pa;
  const sa = isSigned(pa), sb = isSigned(pb), ra = RANK[pa]!, rb = RANK[pb]!;
  if (sa === sb) return ra >= rb ? pa : pb;
  const [u, s] = sa ? [pb, pa] : [pa, pb];
  if (RANK[u]! >= RANK[s]!) return u;
  // signed type has higher rank: i64 can represent all u32 values
  return s;
}

export function layoutStruct(name: string, fields: { name: string; type: CType }[], isUnion: boolean, forcedAlign?: number): StructDef {
  let off = 0, align = 1, size = 0;
  const out: StructDef["fields"] = [];
  for (const f of fields) {
    const a = alignOf(f.type), s = sizeOf(f.type);
    align = Math.max(align, a);
    if (isUnion) { out.push({ ...f, offset: 0 }); size = Math.max(size, s); continue; }
    off = Math.ceil(off / a) * a;
    out.push({ ...f, offset: off });
    off += s;
    size = off;
  }
  if (forcedAlign && forcedAlign > align) align = forcedAlign;
  size = Math.max(Math.ceil(size / align) * align, fields.length ? 0 : 1);
  return { name, fields: out, size, align, isUnion, methods: new Map(), statics: new Map() };
}
