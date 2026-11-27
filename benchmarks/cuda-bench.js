// CUDA benchmark suite (runs in the browser page benchmarks/cuda-bench.html; driven by benchmarks/cuda.mjs).
//
// For each workload and each variant it reports, separately:
//   compileMs  first launch + synchronize minus the steady-state execution time (parse, lower, codegen, pipeline creation)
//   uploadMs   host -> device copies of the inputs
//   execMs     steady-state time per launch: R launches back to back, then one synchronize, divided by R (median of runs).
//              R is picked so one run lasts ~EXEC_TARGET_MS: with a fixed small R, the ~2-8 ms a browser needs to report
//              that the GPU finished dominated, and every fast kernel showed (sync latency) / R.
//   gpuMs      median GPU timestamp time per launch (WebGPU with timestamp-query only)
//   latencyMs  one launch + synchronize (median): what a single call costs end to end
//   downloadMs device -> host copy of the output
// Every result is checked against a host reference (maxRelErr) before its timing is reported.
import * as cuda from "../dist/cuda.js";

const median = (xs) => { const s = [...xs].sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };
const now = () => performance.now();
const rng = (seed) => () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 2 ** 32 - 0.5; };
const randArr = (n, seed = 1) => { const r = rng(seed); const a = new Float32Array(n); for (let i = 0; i < n; i++) a[i] = r(); return a; };
const ceil = (a, b) => Math.ceil(a / b);
const EXEC_TARGET_MS = 200;
/** launches per exec run, from an estimate of one launch's cost (GPU timestamp, else single-launch latency) */
const repsFor = (estMs, min, max = 4000) => Math.max(min, Math.min(max, Math.ceil(EXEC_TARGET_MS / Math.max(estMs, 0.005))));
const relErr = (got, ref) => {
  let m = 0, scale = 0;
  for (let i = 0; i < ref.length; i++) scale = Math.max(scale, Math.abs(ref[i]));
  for (let i = 0; i < ref.length; i++) m = Math.max(m, Math.abs(got[i] - ref[i]) / Math.max(scale, 1e-30));
  return m;
};

// ------------------------------------------------------------------ workloads
// each: { name, size(gpu), setup(ctx, big) -> { inputs: [{data, dtype}], outputs: [{n, dtype}], kernel, grid, block, args(ptrs), ref(), check(out) , flops, bytes } }
function workloads(big) {
  const W = [];
  {
    const n = big ? 1 << 22 : 1 << 20;
    const a = randArr(n, 1), b = randArr(n, 2);
    W.push({ name: `vector_add n=${n}`, src: cuda.kernels.sources.vectorAdd, kernel: "vector_add", grid: [ceil(n, 256)], block: [256],
      inputs: [a, b], outputs: [n], args: (I, O) => [I[0], I[1], O[0], n], bytes: 12 * n,
      ref: () => { const c = new Float32Array(n); for (let i = 0; i < n; i++) c[i] = Math.fround(a[i] + b[i]); return c; }, tol: 0,
      wgsl: "vector_add" });
  }
  for (const N of big ? [256, 512] : [256]) {
    const A = randArr(N * N, 3), B = randArr(N * N, 4);
    let refC = null;
    const ref = () => {
      if (refC) return refC;
      const C = new Float32Array(N * N);
      for (let i = 0; i < N; i++) for (let k = 0; k < N; k++) { const a = A[i * N + k]; for (let j = 0; j < N; j++) C[i * N + j] += a * B[k * N + j]; }
      return (refC = C);
    };
    const common = { src: cuda.kernels.sources.matmul, inputs: [A, B], outputs: [N * N], args: (I, O) => [I[0], I[1], O[0], N, N, N], flops: 2 * N ** 3, ref, tol: 1e-5 };
    W.push({ ...common, name: `matmul_naive ${N}`, kernel: "matmul_naive", grid: [ceil(N, 16), ceil(N, 16)], block: [16, 16] });
    W.push({ ...common, name: `matmul_tiled<16> ${N}`, kernel: "matmul_tiled<16>", grid: [ceil(N, 16), ceil(N, 16)], block: [16, 16], wgsl: "matmul_tiled" });
    W.push({ ...common, name: `matmul_regtile<64,64,8,4,4> ${N}`, kernel: "matmul_regtile<64, 64, 8, 4, 4>", grid: [ceil(N, 64), ceil(N, 64)], block: [16, 16] });
  }
  {
    const n = big ? 1 << 22 : 1 << 20;
    const x = randArr(n, 5);
    W.push({ name: `reduce_sum<256> n=${n}`, src: cuda.kernels.sources.reduction, kernel: "reduce_sum<256>", grid: [256], block: [256],
      inputs: [x], outputs: [1], zeroOutputs: true, args: (I, O) => [I[0], O[0], n], bytes: 4 * n,
      ref: () => { let s = 0; for (let i = 0; i < n; i++) s += x[i]; return new Float32Array([s]); }, tol: 1e-4, wgsl: "reduce_sum" });
  }
  {
    const rows = 1024, cols = 1024;
    const x = randArr(rows * cols, 6).map((v) => v * 8);
    W.push({ name: `softmax_rows<128> ${rows}x${cols}`, src: cuda.kernels.sources.softmax, kernel: "softmax_rows<128>", grid: [rows], block: [128],
      inputs: [x], outputs: [rows * cols], args: (I, O) => [I[0], O[0], cols], bytes: 8 * rows * cols, tol: 1e-5,
      ref: () => {
        const y = new Float32Array(rows * cols);
        for (let r = 0; r < rows; r++) {
          let m = -Infinity; for (let c = 0; c < cols; c++) m = Math.max(m, x[r * cols + c]);
          let s = 0; for (let c = 0; c < cols; c++) s += Math.exp(x[r * cols + c] - m);
          for (let c = 0; c < cols; c++) y[r * cols + c] = Math.exp(x[r * cols + c] - m) / s;
        }
        return y;
      } });
    const g = randArr(cols, 7), bb = randArr(cols, 8);
    W.push({ name: `layernorm_fwd<128> ${rows}x${cols}`, src: cuda.kernels.sources.norms, kernel: "layernorm_fwd<128>", grid: [rows], block: [128],
      inputs: [x, g, bb], outputs: [rows * cols, rows, rows], args: (I, O) => [I[0], I[1], I[2], O[0], O[1], O[2], cols, 1e-5], bytes: 8 * rows * cols, tol: 1e-4,
      ref: () => {
        const y = new Float32Array(rows * cols);
        for (let r = 0; r < rows; r++) {
          let m = 0; for (let c = 0; c < cols; c++) m += x[r * cols + c]; m /= cols;
          let v = 0; for (let c = 0; c < cols; c++) v += (x[r * cols + c] - m) ** 2; v /= cols;
          const rs = 1 / Math.sqrt(v + 1e-5);
          for (let c = 0; c < cols; c++) y[r * cols + c] = (x[r * cols + c] - m) * rs * g[c] + bb[c];
        }
        return y;
      } });
  }
  {
    const B = 1, H = 8, T = big ? 512 : 256, D = 16, scale = 1 / Math.sqrt(D);
    const Q = randArr(B * H * T * D, 9), K = randArr(B * H * T * D, 10), V = randArr(B * H * T * D, 11);
    let refO = null;
    const ref = () => {
      if (refO) return refO;
      const O = new Float32Array(B * H * T * D);
      for (let h = 0; h < H; h++) for (let t = 0; t < T; t++) {
        const q = (h * T + t) * D; const s = new Float64Array(t + 1); let m = -Infinity;
        for (let u = 0; u <= t; u++) { let d = 0; for (let e = 0; e < D; e++) d += Q[q + e] * K[(h * T + u) * D + e]; s[u] = d * scale; m = Math.max(m, s[u]); }
        let z = 0; for (let u = 0; u <= t; u++) { s[u] = Math.exp(s[u] - m); z += s[u]; }
        for (let e = 0; e < D; e++) { let o = 0; for (let u = 0; u <= t; u++) o += s[u] * V[(h * T + u) * D + e]; O[q + e] = o / z; }
      }
      return (refO = O);
    };
    const common = { src: cuda.kernels.sources.attention, inputs: [Q, K, V], flops: 2 * 2 * B * H * T * T * D / 2, ref, tol: 1e-4 };
    W.push({ ...common, name: `sdpa_naive causal H=${H} T=${T} D=${D}`, kernel: "sdpa_naive", grid: [ceil(B * H * T, 64)], block: [64],
      outputs: [B * H * T * D], args: (I, O) => [I[0], I[1], I[2], O[0], B, H, H, T, T, D, scale, 1] });
    W.push({ ...common, name: `flash_attention_fwd<32,16,16> causal H=${H} T=${T} D=${D}`, kernel: "flash_attention_fwd<32, 16, 16>", grid: [ceil(T, 32), B * H], block: [32],
      outputs: [B * H * T * D, B * H * T], args: (I, O) => [I[0], I[1], I[2], O[0], O[1], B, H, H, T, T, D, scale, 1] });
  }
  return W;
}

// ------------------------------------------------------------------ CUDA variants
async function benchCuda(ctx, w, { mode, runs, reps }) {
  const t0 = now();
  const m = ctx.compile(w.src, { mode });
  const parseMs = now() - t0;
  let tu = now();
  const I = w.inputs.map((a) => ctx.tensor(a));
  const O = w.outputs.map((n) => ctx.zeros([n]));
  await ctx.synchronize();
  const uploadMs = now() - tu;
  const args = w.args(I, O);
  const launch = (timing) => m.launch({ kernel: w.kernel, grid: w.grid, block: w.block, args, timing });
  const zero = () => { if (w.zeroOutputs) for (const o of O) ctx.memset(o.ptr, 0, o.nbytes); };
  // first launch: includes lowering, code generation and pipeline/module creation
  zero();
  let t = now(); await launch(false); await ctx.synchronize(); const firstMs = now() - t;
  zero(); await launch(false);
  const td = now(); const out = await O[0].toArray(); const downloadMs = now() - td;
  const maxRelErr = relErr(out, w.ref());
  const lat = [], exec = [], gpu = [];
  for (let r = 0; r < runs; r++) {
    zero(); t = now(); await launch(true); const rep1 = await ctx.synchronize(); lat.push(now() - t);
    const g = rep1.kernelTimes.at(-1)?.ms; if (g != null) gpu.push(g);
  }
  const gpuMs = gpu.length ? median(gpu) : null;
  let est = gpuMs;
  if (est == null) {  // no timestamps: time a short batch (single-launch latency would mostly be the sync wait on a GPU)
    t = now(); for (let k = 0; k < reps; k++) { zero(); await launch(false); } await ctx.synchronize(); est = (now() - t) / reps;
  }
  const R = repsFor(est, reps);
  for (let r = 0; r < runs; r++) {
    t = now();
    for (let k = 0; k < R; k++) { zero(); await launch(false); }
    await ctx.synchronize(); exec.push((now() - t) / R);
  }
  for (const x of [...I, ...O]) x.free?.();
  const execMs = median(exec);
  return { parseMs, compileMs: Math.max(0, firstMs - median(lat)), uploadMs, downloadMs, execMs, latencyMs: median(lat), gpuMs, reps: R, maxRelErr, ok: maxRelErr <= w.tol };
}

// ------------------------------------------------------------------ handwritten WGSL baselines (same algorithms, written directly in WGSL)
const WGSL = {
  vector_add: { code: `@group(0) @binding(0) var<storage, read> a: array<f32>; @group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>; @group(0) @binding(3) var<uniform> n: vec4<u32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x < n.x) { c[g.x] = a[g.x] + b[g.x]; } }`,
    dispatch: (w) => [[w.grid[0], 1, 1]], uniform: (w) => [w.outputs[0]] },
  matmul_tiled: { code: `@group(0) @binding(0) var<storage, read> A: array<f32>; @group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>; @group(0) @binding(3) var<uniform> dims: vec4<u32>;
var<workgroup> As: array<array<f32, 16>, 16>; var<workgroup> Bs: array<array<f32, 16>, 16>;
@compute @workgroup_size(16, 16) fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
  let N = dims.x; let row = wg.y * 16u + l.y; let col = wg.x * 16u + l.x; var acc = 0.0;
  for (var t = 0u; t < N; t += 16u) {
    As[l.y][l.x] = select(0.0, A[row * N + t + l.x], row < N && t + l.x < N);
    Bs[l.y][l.x] = select(0.0, B[(t + l.y) * N + col], col < N && t + l.y < N);
    workgroupBarrier();
    for (var k = 0u; k < 16u; k++) { acc += As[l.y][k] * Bs[k][l.x]; }
    workgroupBarrier();
  }
  if (row < N && col < N) { C[row * N + col] = acc; }
}`, dispatch: (w) => [[w.grid[0], w.grid[1], 1]], uniform: (w) => [Math.round(Math.sqrt(w.outputs[0]))] },
  // two passes (WGSL has no float atomics): per-workgroup tree sums, then one workgroup sums the partials
  reduce_sum: { code: `@group(0) @binding(0) var<storage, read> x: array<f32>; @group(0) @binding(1) var<storage, read_write> part: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>; @group(0) @binding(3) var<uniform> n: vec4<u32>;
var<workgroup> s: array<f32, 256>;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) l: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  var a = 0.0; for (var i = wg.x * 256u + l; i < n.x; i += 256u * 256u) { a += x[i]; }
  s[l] = a; workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) { if (l < st) { s[l] += s[l + st]; } workgroupBarrier(); }
  if (l == 0u) { part[wg.x] = s[0]; }
}
@compute @workgroup_size(256) fn finish(@builtin(local_invocation_index) l: u32) {
  s[l] = part[l]; workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) { if (l < st) { s[l] += s[l + st]; } workgroupBarrier(); }
  if (l == 0u) { out[0] = s[0]; }
}`, dispatch: () => [[256, 1, 1], [1, 1, 1]], entries: ["main", "finish"], partials: 256, uniform: (w) => [w.inputs[0].length] },
};

async function benchWGSL(device, w, { runs, reps }) {
  const spec = WGSL[w.wgsl];
  device.pushErrorScope("validation");
  const t0 = now();
  const mod = device.createShaderModule({ code: spec.code });
  const nb = w.inputs.length + (spec.partials ? 1 : 0) + 2;
  // one explicit layout for all entry points ("auto" layouts only contain the bindings each entry point uses)
  const bgl = device.createBindGroupLayout({ entries: Array.from({ length: nb }, (_, i) => ({ binding: i, visibility: GPUShaderStage.COMPUTE,
    buffer: { type: i === nb - 1 ? "uniform" : i < w.inputs.length ? "read-only-storage" : "storage" } })) });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
  const pipes = await Promise.all((spec.entries ?? ["main"]).map((e) => device.createComputePipelineAsync({ layout, compute: { module: mod, entryPoint: e } })));
  const compileMs = now() - t0;
  const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  let tu = now();
  const ins = w.inputs.map((a) => { const b = device.createBuffer({ size: a.byteLength, usage: S }); device.queue.writeBuffer(b, 0, a); return b; });
  const part = spec.partials ? device.createBuffer({ size: spec.partials * 4, usage: S }) : null;
  const out = device.createBuffer({ size: Math.max(16, w.outputs[0] * 4), usage: S });
  const uni = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(uni, 0, new Uint32Array([...spec.uniform(w), 0, 0, 0].slice(0, 4)));
  await device.queue.onSubmittedWorkDone();
  const uploadMs = now() - tu;
  const bufs = [...ins, ...(part ? [part] : []), out, uni];
  const group = device.createBindGroup({ layout: bgl, entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })) });
  const groups = pipes.map(() => group);
  const dims = spec.dispatch(w);
  const encodeRun = (enc) => {
    pipes.forEach((p, i) => { const pass = enc.beginComputePass(); pass.setPipeline(p); pass.setBindGroup(0, groups[i]); pass.dispatchWorkgroups(...dims[i]); pass.end(); });
  };
  const run = async (k) => { const enc = device.createCommandEncoder(); for (let i = 0; i < k; i++) encodeRun(enc); device.queue.submit([enc.finish()]); await device.queue.onSubmittedWorkDone(); };
  await run(1);
  const td = now();
  const rb = device.createBuffer({ size: w.outputs[0] * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  { const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(out, 0, rb, 0, w.outputs[0] * 4); device.queue.submit([enc.finish()]); }
  await rb.mapAsync(GPUMapMode.READ);
  const got = new Float32Array(rb.getMappedRange().slice(0));
  rb.unmap();
  const downloadMs = now() - td;
  const verr = await device.popErrorScope();
  if (verr) throw new Error(`WebGPU validation error: ${verr.message}`);
  const maxRelErr = relErr(got, w.ref());
  const lat = [], exec = [];
  for (let r = 0; r < runs; r++) { const t = now(); await run(1); lat.push(now() - t); }
  // no timestamps here: estimate one launch from a short batch (latency alone would mostly be the sync wait)
  let t = now(); await run(reps); const est = (now() - t) / reps;
  const R = repsFor(est, reps);
  for (let r = 0; r < runs; r++) { t = now(); await run(R); exec.push((now() - t) / R); }
  for (const b of [...bufs, rb]) b.destroy();
  return { compileMs, uploadMs, downloadMs, execMs: median(exec), latencyMs: median(lat), gpuMs: null, reps: R, maxRelErr, ok: maxRelErr <= w.tol };
}

// ------------------------------------------------------------------ driver
export async function runSuite({ quick = false, workerURL, log = () => {}, only = null } = {}) {
  const env = { userAgent: navigator.userAgent, isolated: globalThis.crossOriginIsolated, cores: navigator.hardwareConcurrency };
  const results = [];
  const gpuCtx = await cuda.createCudaContext({ device: "webgpu" }).catch((e) => { env.webgpuError = String(e); return null; });
  if (gpuCtx) { env.webgpu = gpuCtx.info(); gpuCtx.onPrintf = () => {}; }
  const variants = [];
  if (gpuCtx) {
    variants.push({ name: "CUDA→WGSL optimized", ctx: gpuCtx, mode: "optimized", gpu: true });
    variants.push({ name: "CUDA→WGSL direct", ctx: gpuCtx, mode: "direct", gpu: true });
  }
  const threads = Math.min(8, navigator.hardwareConcurrency || 4);
  const cpuPool = await cuda.createCudaContext({ device: "cpu", workerURL, threads });
  const cpu1 = await cuda.createCudaContext({ device: "cpu", threads: 0 });
  env.cpuPool = cpuPool.info(); env.cpu1 = cpu1.info();
  variants.push({ name: `CUDA→WASM ${cpuPool.info().threads} threads`, ctx: cpuPool, mode: "optimized" });
  variants.push({ name: "CUDA→WASM 1 thread", ctx: cpu1, mode: "optimized" });
  let rawDevice = null;
  if (gpuCtx) {
    const adapter = await navigator.gpu.requestAdapter();
    rawDevice = await adapter.requestDevice();
  }
  for (const w of workloads(!quick)) {
    if (only && !new RegExp(only).test(w.name)) continue;
    for (const v of variants) {
      const runs = quick ? 3 : v.gpu ? 7 : 3, reps = v.gpu ? 10 : 2;
      // the single-threaded CPU variant skips the largest O(n^3)/O(T^2) problems (minutes per run)
      if (!v.gpu && /512/.test(w.name) && v.ctx === cpu1) continue;
      log(`${w.name} — ${v.name}`);
      try { results.push({ workload: w.name, variant: v.name, flops: w.flops, bytes: w.bytes, ...(await benchCuda(v.ctx, w, { mode: v.mode, runs, reps })) }); }
      catch (e) { results.push({ workload: w.name, variant: v.name, error: String(e?.message ?? e) }); v.ctx.backend.resetError?.(); }
    }
    if (rawDevice && w.wgsl) {
      log(`${w.name} — handwritten WGSL`);
      try { results.push({ workload: w.name, variant: "handwritten WGSL", flops: w.flops, bytes: w.bytes, ...(await benchWGSL(rawDevice, w, { runs: quick ? 3 : 7, reps: 10 })) }); }
      catch (e) { results.push({ workload: w.name, variant: "handwritten WGSL", error: String(e?.message ?? e) }); }
    }
  }
  gpuCtx?.destroy(); cpuPool.destroy(); cpu1.destroy(); rawDevice?.destroy();
  return { env, results };
}
