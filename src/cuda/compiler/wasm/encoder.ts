/** Minimal WebAssembly binary encoder: module sections, function bodies, MVP + atomics instructions. */

export const enum T { i32 = 0x7f, i64 = 0x7e, f32 = 0x7d, f64 = 0x7c, v128 = 0x7b }
export type ValType = T;

function uleb(n: number | bigint, out: number[]): void {
  let v = BigInt(n);
  do {
    let b = Number(v & 0x7fn);
    v >>= 7n;
    if (v !== 0n) b |= 0x80;
    out.push(b);
  } while (v !== 0n);
}
function sleb(n: number | bigint, out: number[]): void {
  let v = BigInt(n);
  for (;;) {
    const b = Number(v & 0x7fn);
    v >>= 7n;
    if ((v === 0n && (b & 0x40) === 0) || (v === -1n && (b & 0x40) !== 0)) { out.push(b); return; }
    out.push(b | 0x80);
  }
}
const f32buf = new Float32Array(1), f64buf = new Float64Array(1);

export class Func {
  readonly code: number[] = [];
  readonly locals: ValType[] = [];
  private depth = 0;
  constructor(readonly params: ValType[], readonly results: ValType[], readonly name: string) {}
  local(t: ValType): number { this.locals.push(t); return this.params.length + this.locals.length - 1; }
  op(...bytes: number[]): this { this.code.push(...bytes); return this; }
  get(i: number): this { this.code.push(0x20); uleb(i, this.code); return this; }
  set(i: number): this { this.code.push(0x21); uleb(i, this.code); return this; }
  tee(i: number): this { this.code.push(0x22); uleb(i, this.code); return this; }
  i32c(v: number): this { this.code.push(0x41); sleb(v | 0, this.code); return this; }
  i64c(v: bigint): this { this.code.push(0x42); sleb(BigInt.asIntN(64, v), this.code); return this; }
  f32c(v: number): this { f32buf[0] = v; this.code.push(0x43, ...new Uint8Array(f32buf.buffer)); return this; }
  f64c(v: number): this { f64buf[0] = v; this.code.push(0x44, ...new Uint8Array(f64buf.buffer)); return this; }
  /** structured control: returns the label depth for br targets */
  block(t: ValType | null = null): number { this.code.push(0x02, t ?? 0x40); return ++this.depth; }
  loop(t: ValType | null = null): number { this.code.push(0x03, t ?? 0x40); return ++this.depth; }
  if_(t: ValType | null = null): number { this.code.push(0x04, t ?? 0x40); return ++this.depth; }
  else_(): this { this.code.push(0x05); return this; }
  end(): this { this.code.push(0x0b); this.depth--; return this; }
  /** branch to the construct opened at label depth `label` (as returned by block/loop/if_) */
  br(label: number): this { this.code.push(0x0c); uleb(this.depth - label, this.code); return this; }
  brIf(label: number): this { this.code.push(0x0d); uleb(this.depth - label, this.code); return this; }
  ret(): this { this.code.push(0x0f); return this; }
  call(idx: number): this { this.code.push(0x10); uleb(idx, this.code); return this; }
  drop(): this { this.code.push(0x1a); return this; }
  select(): this { this.code.push(0x1b); return this; }
  unreachable(): this { this.code.push(0x00); return this; }
  /** memory access: opcode, alignment log2, static offset */
  mem(opcode: number, alignLog2: number, offset = 0): this { this.code.push(opcode); uleb(alignLog2, this.code); uleb(offset, this.code); return this; }
  /** 0xFE-prefixed atomic instruction */
  atomic(sub: number, alignLog2: number, offset = 0): this { this.code.push(0xfe); uleb(sub, this.code); uleb(alignLog2, this.code); uleb(offset, this.code); return this; }
  atomicFence(): this { this.code.push(0xfe, 0x03, 0x00); return this; }
  /** 0xFC-prefixed (saturating truncation, memory.fill/copy) */
  fc(sub: number, ...imm: number[]): this { this.code.push(0xfc); uleb(sub, this.code); this.code.push(...imm); return this; }
  get labelDepth(): number { return this.depth; }

  encode(): number[] {
    const body: number[] = [];
    // locals in runs of equal type
    const runs: [number, ValType][] = [];
    for (const t of this.locals) {
      if (runs.length && runs[runs.length - 1][1] === t) runs[runs.length - 1][0]++;
      else runs.push([1, t]);
    }
    uleb(runs.length, body);
    for (const [n, t] of runs) { uleb(n, body); body.push(t); }
    body.push(...this.code, 0x0b);
    const out: number[] = [];
    uleb(body.length, out);
    out.push(...body);
    return out;
  }
}

export interface Import { module: string; name: string; params: ValType[]; results: ValType[] }

export class ModuleBuilder {
  private types: string[] = [];
  private typeSigs: { params: ValType[]; results: ValType[] }[] = [];
  readonly imports: Import[] = [];
  readonly funcs: Func[] = [];
  private exports: { name: string; func: number }[] = [];
  constructor(readonly memory: { initial: number; maximum: number; shared: boolean }) {}
  private typeIndex(params: ValType[], results: ValType[]): number {
    const key = params.join(",") + ":" + results.join(",");
    let i = this.types.indexOf(key);
    if (i < 0) { i = this.types.length; this.types.push(key); this.typeSigs.push({ params, results }); }
    return i;
  }
  /** imports must be added before functions; returns the function index */
  importFunc(module: string, name: string, params: ValType[], results: ValType[]): number {
    if (this.funcs.length) throw new Error("imports must precede functions");
    const have = this.imports.findIndex((x) => x.module === module && x.name === name);
    if (have >= 0) return have;
    this.imports.push({ module, name, params, results });
    return this.imports.length - 1;
  }
  addFunc(f: Func): number { this.funcs.push(f); return this.imports.length + this.funcs.length - 1; }
  funcIndex(f: Func): number { return this.imports.length + this.funcs.indexOf(f); }
  export(name: string, func: number): void { this.exports.push({ name, func }); }

  encode(): Uint8Array {
    const out: number[] = [0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
    const section = (id: number, content: number[]) => { out.push(id); uleb(content.length, out); out.push(...content); };
    for (const f of this.funcs) this.typeIndex(f.params, f.results);
    for (const im of this.imports) this.typeIndex(im.params, im.results);
    // type
    const ts: number[] = [];
    uleb(this.typeSigs.length, ts);
    for (const s of this.typeSigs) { ts.push(0x60); uleb(s.params.length, ts); ts.push(...s.params); uleb(s.results.length, ts); ts.push(...s.results); }
    section(1, ts);
    // import: functions + memory
    const is: number[] = [];
    uleb(this.imports.length + 1, is);
    const str = (s: string) => { const b = new TextEncoder().encode(s); uleb(b.length, is); is.push(...b); };
    for (const im of this.imports) { str(im.module); str(im.name); is.push(0x00); uleb(this.typeIndex(im.params, im.results), is); }
    str("env"); str("memory"); is.push(0x02);
    is.push(this.memory.shared ? 0x03 : 0x01);
    uleb(this.memory.initial, is); uleb(this.memory.maximum, is);
    section(2, is);
    // function
    const fs: number[] = [];
    uleb(this.funcs.length, fs);
    for (const f of this.funcs) uleb(this.typeIndex(f.params, f.results), fs);
    section(3, fs);
    // export
    const es: number[] = [];
    uleb(this.exports.length, es);
    for (const e of this.exports) { const b = new TextEncoder().encode(e.name); uleb(b.length, es); es.push(...b); es.push(0x00); uleb(e.func, es); }
    section(7, es);
    // code
    const cs: number[] = [];
    uleb(this.funcs.length, cs);
    for (const f of this.funcs) cs.push(...f.encode());
    section(10, cs);
    return new Uint8Array(out);
  }
}

/** opcodes used by the code generator */
export const OP = {
  i32_load: 0x28, i64_load: 0x29, f32_load: 0x2a, f64_load: 0x2b, i32_load8_s: 0x2c, i32_load8_u: 0x2d, i32_load16_s: 0x2e, i32_load16_u: 0x2f,
  i32_store: 0x36, i64_store: 0x37, f32_store: 0x38, f64_store: 0x39, i32_store8: 0x3a, i32_store16: 0x3b,
  i32_eqz: 0x45, i32_eq: 0x46, i32_ne: 0x47, i32_lt_s: 0x48, i32_lt_u: 0x49, i32_gt_s: 0x4a, i32_gt_u: 0x4b, i32_le_s: 0x4c, i32_le_u: 0x4d, i32_ge_s: 0x4e, i32_ge_u: 0x4f,
  i64_eqz: 0x50, i64_eq: 0x51, i64_ne: 0x52, i64_lt_s: 0x53, i64_lt_u: 0x54, i64_gt_s: 0x55, i64_gt_u: 0x56, i64_le_s: 0x57, i64_le_u: 0x58, i64_ge_s: 0x59, i64_ge_u: 0x5a,
  f32_eq: 0x5b, f32_ne: 0x5c, f32_lt: 0x5d, f32_gt: 0x5e, f32_le: 0x5f, f32_ge: 0x60,
  f64_eq: 0x61, f64_ne: 0x62, f64_lt: 0x63, f64_gt: 0x64, f64_le: 0x65, f64_ge: 0x66,
  i32_clz: 0x67, i32_ctz: 0x68, i32_popcnt: 0x69, i32_add: 0x6a, i32_sub: 0x6b, i32_mul: 0x6c, i32_div_s: 0x6d, i32_div_u: 0x6e, i32_rem_s: 0x6f, i32_rem_u: 0x70,
  i32_and: 0x71, i32_or: 0x72, i32_xor: 0x73, i32_shl: 0x74, i32_shr_s: 0x75, i32_shr_u: 0x76, i32_rotl: 0x77,
  i64_clz: 0x79, i64_ctz: 0x7a, i64_popcnt: 0x7b, i64_add: 0x7c, i64_sub: 0x7d, i64_mul: 0x7e, i64_div_s: 0x7f, i64_div_u: 0x80, i64_rem_s: 0x81, i64_rem_u: 0x82,
  i64_and: 0x83, i64_or: 0x84, i64_xor: 0x85, i64_shl: 0x86, i64_shr_s: 0x87, i64_shr_u: 0x88,
  f32_abs: 0x8b, f32_neg: 0x8c, f32_ceil: 0x8d, f32_floor: 0x8e, f32_trunc: 0x8f, f32_nearest: 0x90, f32_sqrt: 0x91, f32_add: 0x92, f32_sub: 0x93, f32_mul: 0x94, f32_div: 0x95, f32_min: 0x96, f32_max: 0x97, f32_copysign: 0x98,
  f64_abs: 0x99, f64_neg: 0x9a, f64_ceil: 0x9b, f64_floor: 0x9c, f64_trunc: 0x9d, f64_nearest: 0x9e, f64_sqrt: 0x9f, f64_add: 0xa0, f64_sub: 0xa1, f64_mul: 0xa2, f64_div: 0xa3, f64_min: 0xa4, f64_max: 0xa5, f64_copysign: 0xa6,
  i32_wrap_i64: 0xa7, i64_extend_i32_s: 0xac, i64_extend_i32_u: 0xad,
  f32_convert_i32_s: 0xb2, f32_convert_i32_u: 0xb3, f32_convert_i64_s: 0xb4, f32_convert_i64_u: 0xb5, f32_demote_f64: 0xb6,
  f64_convert_i32_s: 0xb7, f64_convert_i32_u: 0xb8, f64_convert_i64_s: 0xb9, f64_convert_i64_u: 0xba, f64_promote_f32: 0xbb,
  i32_reinterpret_f32: 0xbc, i64_reinterpret_f64: 0xbd, f32_reinterpret_i32: 0xbe, f64_reinterpret_i64: 0xbf,
  i32_extend8_s: 0xc0, i32_extend16_s: 0xc1,
} as const;

/** 0xFC sub-opcodes: saturating float->int truncation (never traps) */
export const SAT = { i32_f32_s: 0, i32_f32_u: 1, i32_f64_s: 2, i32_f64_u: 3, i64_f32_s: 4, i64_f32_u: 5, i64_f64_s: 6, i64_f64_u: 7 } as const;

/** 0xFE atomic sub-opcodes */
export const AT = {
  i32_load: 0x10, i64_load: 0x11, i32_store: 0x17, i64_store: 0x18,
  i32_add: 0x1e, i64_add: 0x1f, i32_sub: 0x25, i64_sub: 0x26, i32_and: 0x2c, i64_and: 0x2d, i32_or: 0x33, i64_or: 0x34,
  i32_xor: 0x3a, i64_xor: 0x3b, i32_xchg: 0x41, i64_xchg: 0x42, i32_cmpxchg: 0x48, i64_cmpxchg: 0x49,
} as const;
