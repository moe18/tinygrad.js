/**
 * Serialization for distributed training coordinators: tensors (parameters, gradients, optimizer state), kernel bundles
 * (source + configuration + compiler identity) and device capability reports. The tensor format is shared with Python
 * (tinygrad_js.cuda.serialize_tensors / deserialize_tensors).
 *
 * Tensor container: "TGJT" magic, u32 version (1), u32 header length, UTF-8 JSON header, then 8-byte aligned raw
 * little-endian data. Header: { meta, tensors: [{ name, dtype, shape, offset, bytes }] }.
 */
import { DTYPE_SIZE, type DType } from "./runtime/memory.js";
import { COMPILER_VERSION } from "./compiler/module.js";
import { sha256Hex } from "./runtime/hash.js";
import type { CudaContext, HostArray } from "./runtime/context.js";

export interface TensorRecord { dtype: DType; shape: number[]; data: HostArray | Uint8Array }

const MAGIC = 0x544a4754; // "TGJT" little endian

export function serializeTensors(tensors: Record<string, TensorRecord>, meta: Record<string, unknown> = {}): Uint8Array {
  const entries: { name: string; dtype: DType; shape: number[]; offset: number; bytes: number }[] = [];
  let off = 0;
  for (const [name, t] of Object.entries(tensors)) {
    const n = t.shape.reduce((a, b) => a * b, 1);
    const bytes = n * DTYPE_SIZE[t.dtype];
    if (t.data.byteLength !== bytes) throw new Error(`tensor ${name}: ${t.data.byteLength} bytes of data for shape [${t.shape}] ${t.dtype} (${bytes} bytes)`);
    entries.push({ name, dtype: t.dtype, shape: t.shape, offset: off, bytes });
    off += Math.ceil(bytes / 8) * 8;
  }
  const header = new TextEncoder().encode(JSON.stringify({ meta, tensors: entries }));
  const hlen = Math.ceil(header.byteLength / 8) * 8;
  const out = new Uint8Array(16 + hlen + off);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, MAGIC, true); dv.setUint32(4, 1, true); dv.setUint32(8, header.byteLength, true);
  out.set(header, 16);
  for (const e of entries) {
    const d = tensors[e.name].data;
    out.set(new Uint8Array(d.buffer, d.byteOffset, d.byteLength), 16 + hlen + e.offset);
  }
  return out;
}

export function deserializeTensors(buf: Uint8Array): { meta: Record<string, unknown>; tensors: Record<string, { dtype: DType; shape: number[]; bytes: Uint8Array }> } {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  if (dv.getUint32(0, true) !== MAGIC) throw new Error("not a tinygrad.js tensor container (bad magic)");
  if (dv.getUint32(4, true) !== 1) throw new Error(`unsupported tensor container version ${dv.getUint32(4, true)}`);
  const hl = dv.getUint32(8, true);
  const header = JSON.parse(new TextDecoder().decode(buf.subarray(16, 16 + hl))) as { meta: Record<string, unknown>; tensors: { name: string; dtype: DType; shape: number[]; offset: number; bytes: number }[] };
  const base = 16 + Math.ceil(hl / 8) * 8;
  const tensors: Record<string, { dtype: DType; shape: number[]; bytes: Uint8Array }> = {};
  for (const e of header.tensors) {
    if (base + e.offset + e.bytes > buf.byteLength) throw new Error(`tensor ${e.name} extends past the end of the container`);
    tensors[e.name] = { dtype: e.dtype, shape: e.shape, bytes: buf.slice(base + e.offset, base + e.offset + e.bytes) };
  }
  return { meta: header.meta, tensors };
}

export interface KernelBundle {
  format: "tinygrad.js-cuda-kernel/1";
  compiler: string;
  source: string;
  sha256: string;
  kernel?: string;
  options?: Record<string, unknown>;
  launch?: { grid: number[]; block: number[]; sharedMem?: number };
  meta?: Record<string, unknown>;
}

export async function serializeKernel(source: string, opts: Omit<KernelBundle, "format" | "compiler" | "source" | "sha256"> = {}): Promise<string> {
  const b: KernelBundle = { format: "tinygrad.js-cuda-kernel/1", compiler: COMPILER_VERSION, source, sha256: await sha256Hex(source), ...opts };
  return JSON.stringify(b);
}
export async function deserializeKernel(json: string): Promise<KernelBundle> {
  const b = JSON.parse(json) as KernelBundle;
  if (b.format !== "tinygrad.js-cuda-kernel/1") throw new Error(`unknown kernel bundle format ${b.format}`);
  const h = await sha256Hex(b.source);
  if (h !== b.sha256) throw new Error("kernel bundle source does not match its sha256 (corrupted or tampered)");
  return b;
}

/** What a training coordinator needs to schedule work on this device. */
export function deviceReport(ctx: CudaContext, benchmarks?: Record<string, unknown>): Record<string, unknown> {
  const i = ctx.info();
  return {
    format: "tinygrad.js-device/1", compiler: COMPILER_VERSION, backend: i.backend, name: i.name, emulated: i.emulated,
    selection: ctx.selection, limits: i.limits, features: i.features, threads: i.threads ?? null, memory: ctx.backend.memoryStats(),
    notes: i.notes, benchmarks: benchmarks ?? null, timestamp: new Date().toISOString(),
  };
}
