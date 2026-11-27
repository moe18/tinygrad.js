// TEST-ONLY helper for wasm_engine.py: run one launch job (stdin JSON) on the WebAssembly backend, print the result JSON.
import { CudaContext, WasmBackend, CudaError } from "../../../dist/cuda.js";
let input = "";
for await (const c of process.stdin) input += c;
const job = JSON.parse(input);
const ctx = new CudaContext(new WasmBackend({ threads: 0 }), { requested: "cpu", selected: "wasm", reasons: [] });
let printf = "";
ctx.onPrintf = (s) => { printf += s; };
const b64 = (s) => Uint8Array.from(Buffer.from(s, "base64"));
const ptrs = {};
for (const [h, data] of Object.entries(job.mem)) { const bytes = b64(data); ptrs[h] = ctx.malloc(Math.max(4, bytes.length)); await ctx.memcpy(ptrs[h], bytes); }
const decode = (ctype, bytes) => {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (/double/.test(ctype)) return dv.getFloat64(0, true);
  if (/float/.test(ctype)) return dv.getFloat32(0, true);
  if (bytes.length === 8) return /unsigned/.test(ctype) ? dv.getBigUint64(0, true) : dv.getBigInt64(0, true);
  if (bytes.length === 1) return /unsigned|bool/.test(ctype) ? dv.getUint8(0) : dv.getInt8(0);
  if (bytes.length === 2) return /unsigned/.test(ctype) ? dv.getUint16(0, true) : dv.getInt16(0, true);
  return /unsigned/.test(ctype) ? dv.getUint32(0, true) : dv.getInt32(0, true);
};
try {
  const m = ctx.compile(job.source, { defines: job.defines ?? undefined });
  const args = job.args.map((a) => a.ptr ? ptrs[a.ptr.handle].add(a.ptr.offset) : decode(a.scalar.ctype ?? "int", b64(a.scalar.bytes)));
  const t0 = performance.now();
  await m.launch({ kernel: job.kernel, grid: job.grid, block: job.block, sharedMem: job.sharedMem, args });
  await ctx.synchronize();
  const ms = performance.now() - t0;
  const mem = {};
  for (const [h, data] of Object.entries(job.mem)) { const n = b64(data).length; const out = new Uint8Array(Math.max(4, n)); await ctx.memcpy(out, ptrs[h], Math.max(4, n)); mem[h] = Buffer.from(out.subarray(0, n)).toString("base64"); }
  process.stdout.write(JSON.stringify({ ms, mem, printf }));
} catch (e) {
  process.stdout.write(JSON.stringify({ error: { name: e instanceof CudaError ? e.errorName : "cudaErrorUnknown", message: e instanceof CudaError ? e.message.slice(e.errorName.length + 2) : e.message }, mem: {}, printf }));
}
ctx.destroy();
