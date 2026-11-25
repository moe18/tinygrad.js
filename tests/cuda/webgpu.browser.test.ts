/** CUDA on WebGPU (real browser WebGPU; SwiftShader on GPU-less machines) and the CPU worker pool in a browser. */
import { test, expect, ISOLATED, NOT_ISOLATED } from "../helpers.js";
import type { Page } from "@playwright/test";
import { openCudaPage } from "./browser-helpers.js";
import { CASES } from "./cases.js";

type W = Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any

test.describe("kernel library on WebGPU", () => {
  test.describe.configure({ mode: "serial" });
  let page: Page;
  test.beforeAll(async ({ browser }) => {
    page = await openCudaPage(browser, ISOLATED);
    await page.evaluate(async () => {
      const w = window as unknown as W;
      w.ctx = await w.cuda.createCudaContext({ device: "webgpu" });
      w.ctx.onPrintf = () => {};
    });
  });
  test.afterAll(async () => { await page?.close(); });
  test("context uses the WebGPU adapter and reports capabilities", async () => {
    const info = await page.evaluate(() => (window as unknown as W).ctx.info());
    expect(info.backend).toBe("webgpu");
    expect(info.limits.warpSize).toBe(32);
    test.info().annotations.push({ type: "adapter", description: `${info.name} fallback=${info.isFallbackAdapter} subgroups=${JSON.stringify(info.features.subgroupSize)}` });
  });
  for (const c of CASES) {
    test(c.name, async () => {
      const r = await page.evaluate(async (name) => {
        const w = window as unknown as W;
        const cs = w.CASES.find((x: { name: string }) => x.name === name);
        try { return await cs.run(w.ctx); } catch (e) { w.ctx.backend.resetError?.(); return { maxErr: Infinity, tol: 0, info: (e as Error).message }; }
      }, c.name);
      test.info().annotations.push({ type: "max-rel-error", description: `${r.maxErr} (tol ${r.tol})` });
      expect(r.maxErr, r.info).toBeLessThanOrEqual(r.tol);
    });
  }
  test("randomized differential testing on WebGPU vs exact constant folding (150 kernels)", async () => {
    let r: any = await page.evaluate(async () => {
      const w = window as unknown as W;
      const ctx = w.ctx;
      // Integer results must be bit-exact. Float results: WGSL division is not correctly rounded (2.5 ulp), so float
      // division is excluded; WGSL lets implementations ignore the sign of zero; and shader compilers (SwiftShader/LLVM,
      // GPU drivers) may reassociate or contract float arithmetic, so floats must agree to within 4 ulp.
      const ks = w.randomKernels(7, 150, { floatDivision: false });
      const bits = new Float32Array(1), ib = new Int32Array(bits.buffer);
      const ord = (v: number) => { bits[0] = v; const x = ib[0]; return x < 0 ? -2147483648 - x : x; };
      const close = (x: number, y: number) => x === y || (Number.isNaN(x) && Number.isNaN(y)) || Math.abs(ord(x) - ord(y)) <= 4;
      const bad: string[] = [];
      let floatInexact = 0;
      for (const k of ks) {
        const run = async (src: string) => {
          const m = ctx.compile(src);
          const oi = ctx.zeros([4], "int32"), ou = ctx.zeros([4], "uint32"), of = ctx.zeros([4]);
          await m.launch({ grid: 1, block: 1, args: [ctx.tensor(k.inputs.i, [4], "int32"), ctx.tensor(k.inputs.u, [4], "uint32"), ctx.tensor(k.inputs.f), oi, ou, of] });
          return [...await oi.toArray(), ...await ou.toArray(), ...await of.toArray()];
        };
        const a = await run(k.source), b = await run(k.folded);
        const diff = a.map((v: number, i: number) => {
          if (Object.is(v, b[i])) return null;
          if (i >= 8 && close(v, b[i])) { if (v !== b[i]) floatInexact++; return null; }
          return `[${i}] ${Object.is(v, -0) ? "-0" : v} vs ${Object.is(b[i], -0) ? "-0" : b[i]}`;
        }).filter((x: string | null) => x);
        if (diff.length) bad.push(diff.join(", ") + "\n" + k.source);
      }
      return { bad, floatInexact };
    });
    test.info().annotations.push({ type: "float results differing by 1-4 ulp", description: String(r.floatInexact) });
    r = r.bad;
    expect(r.slice(0, 2).join("\n\n")).toBe("");
  });
  test("bounds checking reports out-of-bounds accesses that WebGPU would otherwise clamp", async () => {
    const r = await page.evaluate(async () => {
      const w = window as unknown as W;
      const ctx = await w.cuda.createCudaContext({ device: "webgpu" });
      const src = "__global__ void k(float* a, int n) { a[threadIdx.x + n] = 1.0f; }";
      const unchecked = ctx.compile(src);
      const checked = ctx.compile(src, { boundsCheck: true });
      const a = ctx.zeros([16]);
      await unchecked.launch({ grid: 1, block: 32, args: [a, 0] });
      await ctx.synchronize(); // clamped silently: no error
      await checked.launch({ grid: 1, block: 32, args: [a, 0] });
      try { await ctx.synchronize(); return "no error"; } catch (e) { return (e as Error).message; } finally { ctx.destroy(); }
    });
    expect(r).toMatch(/cudaErrorIllegalAddress: k: out-of-bounds store to 'a' at element\/byte 16 by thread \(16,0,0\) in block \(0,0,0\)/);
  });
  test("grids larger than maxComputeWorkgroupsPerDimension are split into several dispatches", async () => {
    const r = await page.evaluate(async () => {
      const w = window as unknown as W;
      const ctx = w.ctx;
      const m = ctx.compile("__global__ void k(unsigned* c) { if (threadIdx.x == 0) atomicAdd(&c[blockIdx.x & 1], 1u); }");
      const c = ctx.zeros([2], "uint32");
      await m.launch({ grid: [70000], block: [32], args: [c] });
      return Array.from(await c.toArray());
    });
    expect(r).toEqual([35000, 35000]);
  });
  test("aliased pointer arguments (the same buffer passed twice) and pointer offsets", async () => {
    const r = await page.evaluate(async () => {
      const w = window as unknown as W;
      const ctx = w.ctx;
      const m = ctx.compile("__global__ void k(const float* a, float* b, int n) { int i = threadIdx.x; if (i < n) b[i] = a[i] * 2.0f; }");
      const x = ctx.tensor([1, 2, 3, 4, 5, 6, 7, 8]);
      await m.launch({ grid: 1, block: 8, args: [x, x, 8] });                  // in place through aliased bindings
      await m.launch({ grid: 1, block: 8, args: [x.ptr.add(16), x.ptr, 4] });  // offsets into one buffer
      return Array.from(await x.toArray());
    });
    expect(r).toEqual([20, 24, 28, 32, 10, 12, 14, 16]);
  });
  test("GPU timestamps time kernels when the adapter supports timestamp-query", async () => {
    const r = await page.evaluate(async () => {
      const w = window as unknown as W;
      const ctx = w.ctx;
      const m = ctx.compile(w.cuda.kernels.sources.matmul);
      const n = 128, A = ctx.rand([n * n]), C = ctx.zeros([n * n]);
      await m.launch({ kernel: "matmul_naive", grid: [8, 8], block: [16, 16], args: [A, A, C, n, n, n], timing: true });
      const s = await ctx.synchronize();
      return { supported: ctx.info().features.timestamps, times: s.kernelTimes };
    });
    if (r.supported) expect(r.times[0].ms).toBeGreaterThan(0);
    else expect(r.times[0].ms).toBeNull();
  });
  test("a timed launch split into several dispatches reports one time covering every piece", async () => {
    const r = await page.evaluate(async () => {
      const w = window as unknown as W;
      const ctx = w.ctx;
      const m = ctx.compile(`__global__ void k(float* c) {
        float v = threadIdx.x;
        for (int i = 0; i < 64; ++i) v = v * 1.0001f + 1.0f;
        if (v == -1.0f) c[0] = v;
      }`);
      const c = ctx.zeros([1]);
      // 70000 blocks = a full piece of 65535 + one of 4465: the reported time must cover both, not the last piece alone
      await m.launch({ grid: [70000], block: [32], args: [c], timing: true });
      const split = (await ctx.synchronize()).kernelTimes;
      await m.launch({ grid: [70000 - 65535], block: [32], args: [c], timing: true });
      const last = (await ctx.synchronize()).kernelTimes;
      return { supported: ctx.info().features.timestamps, split, last };
    });
    expect(r.split).toHaveLength(1);
    if (!r.supported) { expect(r.split[0].ms).toBeNull(); return; }
    expect(r.split[0].ms).toBeGreaterThan(r.last[0].ms);
  });
  test("CUDA.compile + launch: the README example", async () => {
    const r = await page.evaluate(async () => {
      const w = window as unknown as W;
      const { CUDA } = w.cuda;
      CUDA.configure({ device: "webgpu" });
      const kernel = await CUDA.compile(`
    __global__ void vector_add(float* a, float* b, float* c, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) { c[i] = a[i] + b[i]; }
    }`);
      const ctx = kernel.ctx;
      const a = ctx.rand([1024]), b = ctx.rand([1024]), c = ctx.zeros([1024]);
      await kernel.launch({ grid: [4, 1, 1], block: [256, 1, 1], args: [a, b, c, 1024] });
      const [ah, bh, ch] = [await a.toArray(), await b.toArray(), await c.toArray()];
      return Math.max(...Array.from(ch as Float32Array, (v, i) => Math.abs(v - Math.fround(ah[i] + bh[i]))));
    });
    expect(r).toBe(0);
  });
  test("cuda* runtime API functions", async () => {
    const r = await page.evaluate(async () => {
      const w = window as unknown as W;
      const c = w.cuda;
      await c.cudaInitDevice({ device: "webgpu" });
      const p = await c.cudaMalloc(16);
      await c.cudaMemcpy(p, new Float32Array([1, 2, 3, 4]), 16, c.cudaMemcpyKind.cudaMemcpyHostToDevice);
      await c.cudaMemset(p, 0, 4);
      const out = new Float32Array(4);
      await c.cudaMemcpy(out, p, 16, c.cudaMemcpyKind.cudaMemcpyDeviceToHost);
      const props = await c.cudaGetDeviceProperties();
      c.cudaFree(p);
      return { out: Array.from(out), name: props.name, warp: props.warpSize, err: c.cudaGetLastError(), str: c.cudaGetErrorString(700), count: await c.cudaGetDeviceCount() };
    });
    expect(r.out).toEqual([0, 2, 3, 4]);
    expect(r.warp).toBe(32);
    expect(r.err).toBe(0);
    expect(r.str).toBe("an illegal memory access was encountered");
  });
  test("backend limits produce explicit errors: double precision, too many threads, 64-bit atomics", async () => {
    const r = await page.evaluate(async () => {
      const w = window as unknown as W;
      const ctx = w.ctx;
      const msgs: string[] = [];
      const t = async (src: string, block: number, args: unknown[]) => { try { await ctx.compile(src).launch({ grid: 1, block, args }); msgs.push("ok"); } catch (e) { msgs.push((e as Error).message); } };
      await t("__global__ void k(float* a) { a[0] = a[0] * 0.1; }", 1, [ctx.zeros([1])]);
      // twice the adapter's limit (256 on SwiftShader, 1024 on e.g. NVIDIA GPUs)
      await t("__global__ void k(float* a) { a[0] = 1.0f; }", 2 * ctx.info().limits.maxThreadsPerBlock, [ctx.zeros([1])]);
      await t("__global__ void k(unsigned long long* a) { atomicAdd(a, 1ull); }", 1, [ctx.zeros([1], "uint64")]);
      return msgs;
    });
    expect(r[0]).toMatch(/double precision is not supported by WebGPU.*0\.5f|double precision is not supported by WebGPU/);
    expect(r[1]).toMatch(/cudaErrorInvalidConfiguration: block of \d+ threads exceeds/);
    expect(r[2]).toMatch(/atomics are not supported by WebGPU/);
  });
  test("AUTO device selection: software WebGPU adapter -> CPU backend with a reason; allowSoftwareWebGPU keeps WebGPU", async () => {
    const r = await page.evaluate(async () => {
      const w = window as unknown as W;
      const a = await w.cuda.createCudaContext({ device: "auto" });
      const b = await w.cuda.createCudaContext({ device: "auto", allowSoftwareWebGPU: true });
      const c = await w.cuda.createCudaContext({ device: "auto", allowSoftwareWebGPU: true, requirements: [{ source: "__global__ void k(double* x) { x[0] = x[0] * 0.1; }", kernels: [{ name: "k", block: [64] }] }] });
      const out = { a: a.selection, b: b.selection, c: c.selection, fallback: b.info().isFallbackAdapter };
      a.destroy(); b.destroy(); c.destroy();
      return out;
    });
    if (r.fallback) {
      expect(r.a.selected).toBe("wasm");
      expect(r.a.reasons.join(" ")).toMatch(/software implementation/);
    } else expect(r.a.selected).toBe("webgpu");
    expect(r.b.selected).toBe("webgpu");
    expect(r.c.selected).toBe("wasm");
    expect(r.c.reasons.join(" ")).toMatch(/double precision/);
  });
});

for (const [label, origin, threads] of [["cross-origin isolated: worker pool", ISOLATED, 4], ["not isolated: calling thread", NOT_ISOLATED, 0]] as const) {
  test(`CPU backend in the browser (${label})`, async ({ browser }) => {
    const page = await openCudaPage(browser, origin);
    const r = await page.evaluate(async () => {
      const w = window as unknown as W;
      const ctx = await w.cuda.createCudaContext({ device: "cpu", workerURL: "/dist/cuda-worker.js" });
      ctx.onPrintf = () => {};
      const names = ["matmul naive", "reductions", "attention", "barrier semantics"];
      const res: string[] = [];
      for (const c of w.CASES) if (names.some((n) => c.name.startsWith(n))) { const x = await c.run(ctx); res.push(`${x.maxErr <= x.tol ? "ok" : "FAIL " + x.maxErr} ${c.name}`); }
      const info = ctx.info();
      ctx.destroy();
      return { res, threads: info.threads, name: info.name };
    });
    expect(r.res.filter((x) => !x.startsWith("ok"))).toEqual([]);
    if (!threads) expect(r.threads).toBe(1);
    if (threads) expect(r.threads).toBeGreaterThan(1);
    await page.close();
  });
}
