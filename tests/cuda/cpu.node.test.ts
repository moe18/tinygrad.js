/** WebAssembly CPU backend + runtime API tests (Node.js; the same code runs in browsers). */
import { test, expect } from "@playwright/test";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { CASES } from "./cases.js";
import { randomKernels } from "./random-kernels.js";
import { WasmBackend } from "../../src/cuda/runtime/wasm.js";
import { CudaContext } from "../../src/cuda/runtime/context.js";
import { CudaError } from "../../src/cuda/runtime/errors.js";
import { createCudaContext } from "../../src/cuda/device.js";
import { serializeTensors, deserializeTensors, serializeKernel, deserializeKernel, deviceReport } from "../../src/cuda/serialize.js";
import { autotune } from "../../src/cuda/autotune.js";
import { sources } from "../../src/cuda/kernels/index.js";

const workerURL = pathToFileURL(path.resolve(import.meta.dirname, "../../dist/cuda-worker.js"));
const make = (threads: number) => new CudaContext(new WasmBackend({ threads, workerURL }), { requested: "cpu", selected: "wasm", reasons: [] });

for (const threads of [0, 4]) {
  test.describe(`kernel library on the CPU backend (${threads === 0 ? "calling thread" : `${threads} worker threads`})`, () => {
    test.describe.configure({ mode: "serial" });
    let ctx: CudaContext;
    test.beforeAll(() => { ctx = make(threads); ctx.onPrintf = () => {}; });
    test.afterAll(() => ctx.destroy());
    for (const c of CASES) {
      test(c.name, async () => {
        const r = await c.run(ctx);
        test.info().annotations.push({ type: "max-rel-error", description: `${r.maxErr.toExponential(2)} (tol ${r.tol})` });
        expect(r.maxErr, r.info).toBeLessThanOrEqual(r.tol);
      });
    }
  });
}

test.describe("runtime API (CPU backend)", () => {
  test("memory: malloc/memcpy/memset, pointer arithmetic, bounds, double free, use after free, limits", async () => {
    const ctx = new CudaContext(new WasmBackend({ threads: 0, memoryLimit: 1 << 20 }), { requested: "", selected: "", reasons: [] });
    const p = ctx.malloc(64);
    await ctx.memcpy(p, new Float32Array([1, 2, 3, 4]));
    ctx.memset(p.add(4), 0, 4);
    const out = new Float32Array(4);
    await ctx.memcpy(out, p, 16);
    expect(Array.from(out)).toEqual([1, 0, 3, 4]);
    await expect(ctx.memcpy(new Float32Array(32), p, 128)).rejects.toThrow(/exceed the allocation of 64 bytes/);
    expect(() => ctx.malloc(2 << 20)).toThrow(/cudaErrorMemoryAllocation/);
    ctx.free(p);
    expect(() => ctx.free(p)).toThrow(/double free/);
    await expect(ctx.memcpy(out, p, 16)).rejects.toThrow(/use of freed device memory/);
    expect(ctx.backend.memoryStats().allocations).toBe(0);
  });
  test("kernel argument validation and launch configuration errors", async () => {
    const ctx = make(0);
    const m = ctx.compile(sources.vectorAdd);
    const a = ctx.zeros([4]);
    await expect(m.launch({ kernel: "vector_add", grid: 1, block: 4, args: [a, a, a] })).rejects.toThrow(/expects 4 arguments/);
    await expect(m.launch({ kernel: "vector_add", grid: 1, block: 4, args: [a, a, 1, 4] })).rejects.toThrow(/must be a device pointer/);
    await expect(m.launch({ kernel: "vector_add", grid: 1, block: 4, args: [a, a, a, 1.5] })).rejects.toThrow(/expects an integer/);
    await expect(m.launch({ kernel: "vector_add", grid: 1, block: 2048, args: [a, a, a, 4] })).rejects.toThrow(/cudaErrorInvalidConfiguration/);
    await expect(m.launch({ kernel: "vector_add", grid: 0, block: 4, args: [a, a, a, 4] })).rejects.toThrow(/positive integers/);
    const other = make(0);
    await expect(m.launch({ kernel: "vector_add", grid: 1, block: 4, args: [other.zeros([4]), a, a, 4] })).rejects.toThrow(/different device/);
    expect(() => ctx.compile("__global__ void k() { int x = ; }")).toThrow(/CompileError|syntax/);
    expect(() => m.getFunction()).toThrow(/defines \d+ kernels/);
    ctx.destroy(); other.destroy();
  });
  test("out-of-bounds accesses are errors (cudaErrorIllegalAddress), and the error is sticky until reset", async () => {
    const ctx = make(0);
    const m = ctx.compile("__global__ void k(float* a, int i) { a[i] = 1.0f; }\n__global__ void r(const float* a, float* o, int i) { o[0] = a[i]; }");
    const a = ctx.zeros([8]);
    await m.launch({ kernel: "k", grid: 1, block: 1, args: [a, 7] });
    await m.launch({ kernel: "k", grid: 1, block: 1, args: [a.ptr.add(16), 4] });
    await expect(ctx.synchronize()).rejects.toThrow(/out-of-bounds store to 'a' at byte offset 16 by thread \(0,0,0\) in block \(0,0,0\)/);
    expect(ctx.getLastError()).toBe(700);
    (ctx.backend as WasmBackend).resetError();
    await m.launch({ kernel: "r", grid: 1, block: 1, args: [a, ctx.zeros([1]), -1] });
    await expect(ctx.synchronize()).rejects.toBeInstanceOf(CudaError);
    ctx.destroy();
  });
  test("launch statements: kernel<<<grid, block>>>(args)", async () => {
    const ctx = make(0);
    const m = ctx.compile(sources.vectorAdd);
    const a = ctx.tensor([1, 2, 3]), b = ctx.tensor([10, 20, 30]), c = ctx.zeros([3]);
    await m.exec("vector_add<<<dim3(1), 64>>>(a, b, c, n)", { a, b, c, n: 3 });
    expect(Array.from(await c.toArray() as Float32Array)).toEqual([11, 22, 33]);
    ctx.destroy();
  });
  test("__constant__ symbols: memcpyToSymbol / memcpyFromSymbol", async () => {
    const ctx = make(0);
    const m = ctx.compile("__constant__ float coef[4] = {1, 2, 3, 4};\n__global__ void k(float* o) { o[threadIdx.x] = coef[threadIdx.x] * 2.0f; }");
    const o = ctx.zeros([4]);
    await m.launch({ grid: 1, block: 4, args: [o] });
    expect(Array.from(await o.toArray() as Float32Array)).toEqual([2, 4, 6, 8]);
    await m.memcpyToSymbol("coef", new Float32Array([5, 6, 7, 8]));
    await m.launch({ grid: 1, block: 4, args: [o] });
    expect(Array.from(await o.toArray() as Float32Array)).toEqual([10, 12, 14, 16]);
    const back = new Float32Array(4);
    await m.memcpyFromSymbol(back, "coef");
    expect(Array.from(back)).toEqual([5, 6, 7, 8]);
    ctx.destroy();
  });
  test("double precision runs natively on the CPU backend", async () => {
    const ctx = make(0);
    const m = ctx.compile("__global__ void k(double* x) { x[0] = sqrt(2.0) * 1e300 * 1e8; x[1] = 0.1 + 0.2; }");
    const x = ctx.zeros([2], "float64");
    await m.launch({ grid: 1, block: 1, args: [x] });
    expect(Array.from(await x.toArray() as Float64Array)).toEqual([Math.SQRT2 * 1e300 * 1e8, 0.1 + 0.2]);
    ctx.destroy();
  });
  test("streams and events", async () => {
    const ctx = make(0);
    const s = ctx.streamCreate();
    const e0 = ctx.eventCreate(), e1 = ctx.eventCreate();
    ctx.eventRecord(e0, s);
    const m = ctx.compile(sources.matmul);
    const A = ctx.rand([128 * 128]), C = ctx.zeros([128 * 128]);
    await m.launch({ kernel: "matmul_naive", grid: [8, 8], block: [16, 16], args: [A, A, C, 128, 128, 128] });
    ctx.eventRecord(e1, s);
    await s.synchronize();
    expect(await ctx.eventElapsedTime(e0, e1)).toBeGreaterThanOrEqual(0);
    ctx.destroy();
  });
  test("device selection: explicit CPU, AUTO without WebGPU falls back with a reason, remote needs configuration", async () => {
    const cpu = await createCudaContext({ device: "cpu", threads: 0 });
    expect(cpu.selection.selected).toBe("wasm");
    expect(cpu.getDeviceProperties()).toMatchObject({ warpSize: 32, backend: "wasm", emulated: true });
    cpu.destroy();
    const auto = await createCudaContext({ device: "auto", threads: 0 });
    expect(auto.selection.selected).toBe("wasm");
    expect(auto.selection.reasons.join(" ")).toMatch(/WebGPU unavailable/);
    auto.destroy();
    await expect(createCudaContext({ device: "cuda_remote" })).rejects.toThrow(/needs a remote configuration/);
  });
  test("serialization: tensors, kernel bundles (integrity checked), device reports", async () => {
    const buf = serializeTensors({ w: { dtype: "float32", shape: [2, 2], data: new Float32Array([1, 2, 3, 4]) }, step: { dtype: "int64", shape: [1], data: new BigInt64Array([7n]) } }, { epoch: 3 });
    const r = deserializeTensors(buf);
    expect(r.meta).toEqual({ epoch: 3 });
    expect(Array.from(new Float32Array(r.tensors.w.bytes.buffer))).toEqual([1, 2, 3, 4]);
    expect(r.tensors.step).toMatchObject({ dtype: "int64", shape: [1] });
    const kb = await serializeKernel(sources.vectorAdd, { kernel: "vector_add", launch: { grid: [4], block: [256] } });
    expect((await deserializeKernel(kb)).kernel).toBe("vector_add");
    await expect(deserializeKernel(kb.replace("vector_add(", "vector_sub("))).rejects.toThrow(/does not match its sha256/);
    const ctx = make(0);
    expect(deviceReport(ctx)).toMatchObject({ format: "tinygrad.js-device/1", backend: "wasm" });
    ctx.destroy();
  });
  test("autotuning picks and caches the fastest configuration", async () => {
    const ctx = make(0);
    const m = ctx.compile(sources.vectorAdd);
    const n = 1 << 16, a = ctx.rand([n]), b = ctx.rand([n]), c = ctx.zeros([n]);
    const cands = [64, 256].map((bs) => ({ name: `block${bs}`, params: { kernel: "vector_add", grid: [Math.ceil(n / bs)], block: [bs], args: [a, b, c, n] } }));
    const r1 = await autotune(ctx, m, "vadd-test", cands as never);
    expect(r1.cached).toBe(false);
    expect(r1.results).toHaveLength(2);
    const r2 = await autotune(ctx, m, "vadd-test", cands as never);
    expect(r2).toMatchObject({ cached: true, best: r1.best });
    ctx.destroy();
  });
  test("randomized differential testing: runtime results equal exact constant folding (200 kernels)", async () => {
    const ctx = make(0);
    const ks = randomKernels(42, 200, { floatDivision: true });
    let mismatches = 0;
    for (const k of ks) {
      const run = async (src: string) => {
        const m = ctx.compile(src);
        const oi = ctx.zeros([4], "int32"), ou = ctx.zeros([4], "uint32"), of = ctx.zeros([4]);
        await m.launch({ grid: 1, block: 1, args: [ctx.tensor(k.inputs.i, [4], "int32"), ctx.tensor(k.inputs.u, [4], "uint32"), ctx.tensor(k.inputs.f), oi, ou, of] });
        return [...await oi.toArray() as Int32Array, ...await ou.toArray() as Uint32Array, ...await of.toArray() as Float32Array];
      };
      const [a, b] = [await run(k.source), await run(k.folded)];
      if (a.some((v, i) => !Object.is(v, b[i]) && !(Number.isNaN(v) && Number.isNaN(b[i])))) { mismatches++; if (mismatches < 3) console.log(k.source, a, b); }
    }
    expect(mismatches).toBe(0);
    ctx.destroy();
  });
});
