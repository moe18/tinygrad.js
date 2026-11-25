/**
 * Numerical test cases for the CUDA kernel library and compiler features. Each case runs real kernels on a CudaContext and
 * compares against an independent float64 JavaScript reference. Shared by the Node (WebAssembly CPU backend) and browser
 * (WebGPU backend) test suites.
 */
import type { CudaContext } from "../../src/cuda/runtime/context.js";
import { sources } from "../../src/cuda/kernels/index.js";
import * as cublas from "../../src/cuda/libs/cublas.js";
import * as cudnn from "../../src/cuda/libs/cudnn.js";
import * as curand from "../../src/cuda/libs/curand.js";
import { halfFromBits } from "../../src/cuda/compiler/fold.js";

export interface CaseResult { maxErr: number; tol: number; info?: string }
export interface Case { name: string; run(ctx: CudaContext): Promise<CaseResult>; skip?: (ctx: CudaContext) => string | null }

/** deterministic pseudo-random floats in [lo, hi) */
export function rng(seed: number) {
  let s = seed >>> 0 || 1;
  return (n: number, lo = -1, hi = 1) => {
    const a = new Float32Array(n);
    for (let i = 0; i < n; i++) { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; a[i] = lo + (hi - lo) * ((s >>> 8) / 16777216); }
    return a;
  };
}
const maxRel = (got: ArrayLike<number>, want: ArrayLike<number>, floor = 1e-3) => {
  if (got.length !== want.length) throw new Error(`length ${got.length} != ${want.length}`);
  let m = 0;
  for (let i = 0; i < want.length; i++) {
    const g = Number(got[i]), w = Number(want[i]);
    if (g === w) continue;
    if (Number.isNaN(w) ? !Number.isNaN(g) : !(Math.abs(g - w) <= Math.abs(g - w))) return Infinity;
    const e = Math.abs(g - w) / Math.max(floor, Math.abs(w));
    if (e > m || Number.isNaN(e)) m = Number.isNaN(e) ? Infinity : e;
  }
  return m;
};
const f32 = async (t: { toArray(): Promise<unknown> }) => (await t.toArray()) as Float32Array;
const ceil = (a: number, b: number) => Math.ceil(a / b);

function matmulRef(A: Float32Array, B: Float32Array, M: number, N: number, K: number): Float64Array {
  const C = new Float64Array(M * N);
  for (let i = 0; i < M; i++) for (let k = 0; k < K; k++) { const a = A[i * K + k]; for (let j = 0; j < N; j++) C[i * N + j] += a * B[k * N + j]; }
  return C;
}
function erf(x: number): number {
  // Abramowitz-Stegun 7.1.26 is too coarse for 1e-6 tests: use a series / continued fraction in double
  const a = Math.abs(x);
  let r: number;
  if (a < 3) { let sum = a, term = a; for (let n = 1; n < 100; n++) { term *= -a * a / n; sum += term / (2 * n + 1); } r = (2 / Math.sqrt(Math.PI)) * sum; }
  else { let f = 0; for (let n = 60; n >= 1; n--) f = n / 2 / (a + f); r = 1 - Math.exp(-a * a) / Math.sqrt(Math.PI) / (a + f); }
  return x < 0 ? -r : r;
}

export const CASES: Case[] = [
  {
    name: "vector_add + float4 vector_add4 + saxpy",
    async run(ctx) {
      const n = 4099, r = rng(1);
      const ah = r(n), bh = r(n);
      const m = ctx.compile(sources.vectorAdd);
      const a = ctx.tensor(ah), b = ctx.tensor(bh), c = ctx.zeros([n]);
      await m.launch({ kernel: "vector_add", grid: [ceil(n, 256)], block: [256], args: [a, b, c, n] });
      const n4 = 1024; const a4 = ctx.tensor(ah.subarray(0, 4 * n4)), b4 = ctx.tensor(bh.subarray(0, 4 * n4)), c4 = ctx.zeros([4 * n4]);
      await m.launch({ kernel: "vector_add4", grid: [ceil(n4, 128)], block: [128], args: [a4, b4, c4, n4] });
      const y = ctx.tensor(bh);
      await m.launch({ kernel: "saxpy", grid: [8], block: [128], args: [n, 2.5, a, y] });
      const want = Array.from(ah, (v, i) => Math.fround(v + bh[i]));
      const wantS = Array.from(ah, (v, i) => Math.fround(Math.fround(2.5 * v) + bh[i]));
      return { maxErr: Math.max(maxRel(await f32(c), want), maxRel(await f32(c4), want.slice(0, 4 * n4)), maxRel(await f32(y), wantS)), tol: 1e-6 };
    },
  },
  {
    name: "elementwise switch (add/sub/mul/div/max/min)",
    async run(ctx) {
      const n = 1000, r = rng(2); const ah = r(n), bh = r(n, 0.5, 2);
      const m = ctx.compile(sources.vectorAdd);
      const a = ctx.tensor(ah), b = ctx.tensor(bh), o = ctx.zeros([n]);
      let err = 0;
      const ops = [(x: number, y: number) => x + y, (x: number, y: number) => x - y, (x: number, y: number) => x * y, (x: number, y: number) => x / y, Math.max, Math.min];
      for (let op = 0; op < 6; op++) {
        await m.launch({ kernel: "elementwise", grid: [ceil(n, 128)], block: [128], args: [a, b, o, n, op] });
        err = Math.max(err, maxRel(await f32(o), Array.from(ah, (v, i) => Math.fround(ops[op](v, bh[i])))));
      }
      return { maxErr: err, tol: 3e-7 };
    },
  },
  {
    name: "matmul naive / tiled<16> / register-tiled<64,64,8,4,4> (non-multiple sizes)",
    async run(ctx) {
      const M = 70, N = 53, K = 37, r = rng(3);
      const Ah = r(M * K), Bh = r(K * N);
      const want = matmulRef(Ah, Bh, M, N, K);
      const m = ctx.compile(sources.matmul);
      const A = ctx.tensor(Ah), B = ctx.tensor(Bh), C = ctx.zeros([M * N]);
      let err = 0;
      await m.launch({ kernel: "matmul_naive", grid: [ceil(N, 16), ceil(M, 16)], block: [16, 16], args: [A, B, C, M, N, K] });
      err = Math.max(err, maxRel(await f32(C), want, 1e-2));
      ctx.memset(C, 0);
      await m.launch({ kernel: "matmul_tiled<16>", grid: [ceil(N, 16), ceil(M, 16)], block: [16, 16], args: [A, B, C, M, N, K] });
      err = Math.max(err, maxRel(await f32(C), want, 1e-2));
      ctx.memset(C, 0);
      await m.launch({ kernel: "matmul_regtile<64, 64, 8, 4, 4>", grid: [ceil(N, 64), ceil(M, 64)], block: [16, 16], args: [A, B, C, M, N, K] });
      err = Math.max(err, maxRel(await f32(C), want, 1e-2));
      return { maxErr: err, tol: 1e-4 };
    },
  },
  {
    name: "batched matmul (blockIdx.z, size_t pointer offsets)",
    async run(ctx) {
      const Bn = 3, M = 9, N = 11, K = 7, r = rng(4);
      const Ah = r(Bn * M * K), Bh = r(Bn * K * N);
      const m = ctx.compile(sources.matmul);
      const C = ctx.zeros([Bn * M * N]);
      await m.launch({ kernel: "matmul_batched", grid: [1, 1, Bn], block: [16, 16], args: [ctx.tensor(Ah), ctx.tensor(Bh), C, M, N, K] });
      const want: number[] = [];
      for (let b = 0; b < Bn; b++) want.push(...matmulRef(Ah.subarray(b * M * K), Bh.subarray(b * K * N), M, N, K));
      return { maxErr: maxRel(await f32(C), want, 1e-2), tol: 1e-5 };
    },
  },
  {
    name: "transpose naive and tiled (padded shared tile)",
    async run(ctx) {
      const R = 45, Cc = 70, r = rng(5); const x = r(R * Cc);
      const m = ctx.compile(sources.transpose);
      const want = new Float32Array(R * Cc);
      for (let i = 0; i < R; i++) for (let j = 0; j < Cc; j++) want[j * R + i] = x[i * Cc + j];
      const o1 = ctx.zeros([R * Cc]), o2 = ctx.zeros([R * Cc]), X = ctx.tensor(x);
      await m.launch({ kernel: "transpose_naive", grid: [ceil(Cc, 16), ceil(R, 16)], block: [16, 16], args: [X, o1, R, Cc] });
      await m.launch({ kernel: "transpose_tiled", grid: [ceil(Cc, 16), ceil(R, 16)], block: [16, 16], args: [X, o2, R, Cc] });
      return { maxErr: Math.max(maxRel(await f32(o1), want), maxRel(await f32(o2), want)), tol: 0 };
    },
  },
  {
    name: "reductions: shared tree, warp shuffle, row max",
    async run(ctx) {
      const n = 100_003, r = rng(6); const x = r(n, 0, 1);
      const want = x.reduce((a, b) => a + b, 0);
      const m = ctx.compile(sources.reduction);
      const X = ctx.tensor(x);
      const s1 = ctx.zeros([1]), s2 = ctx.zeros([1]);
      await m.launch({ kernel: "reduce_sum<256>", grid: [64], block: [256], args: [X, s1, n] });
      await m.launch({ kernel: "reduce_sum_warp<256>", grid: [64], block: [256], args: [X, s2, n] });
      const rows = 17, cols = 300; const y = r(rows * cols);
      const mx = ctx.zeros([rows]);
      await m.launch({ kernel: "row_max<128>", grid: [rows], block: [128], args: [ctx.tensor(y), mx, cols] });
      const wantMax = Array.from({ length: rows }, (_, i) => Math.max(...y.subarray(i * cols, (i + 1) * cols)));
      const e = Math.max(Math.abs((await f32(s1))[0] - want) / want, Math.abs((await f32(s2))[0] - want) / want, maxRel(await f32(mx), wantMax));
      return { maxErr: e, tol: 1e-5 };
    },
  },
  {
    name: "prefix sums: multi-block Hillis-Steele scan and Blelloch scan",
    async run(ctx) {
      const n = 3000, B = 256, r = rng(7); const x = r(n, 0, 1);
      const m = ctx.compile(sources.scan);
      const nb = ceil(n, B);
      const out = ctx.zeros([n]), sums = ctx.zeros([nb]), sums2 = ctx.zeros([nb]), dummy = ctx.zeros([1]);
      await m.launch({ kernel: "scan_block<256>", grid: [nb], block: [B], args: [ctx.tensor(x), out, sums, n] });
      await m.launch({ kernel: "scan_block<256>", grid: [1], block: [B], args: [sums, sums2, dummy, nb] });
      await m.launch({ kernel: "scan_add<256>", grid: [nb], block: [B], args: [out, sums2, n] });
      const want = new Float64Array(n); let acc = 0; for (let i = 0; i < n; i++) { acc += x[i]; want[i] = acc; }
      const nb2 = 200; const ex = ctx.zeros([nb2]);
      await m.launch({ kernel: "scan_blelloch<128>", grid: [1], block: [128], args: [ctx.tensor(x.subarray(0, nb2)), ex, nb2] });
      const wantEx = new Float64Array(nb2); acc = 0; for (let i = 0; i < nb2; i++) { wantEx[i] = acc; acc += x[i]; }
      return { maxErr: Math.max(maxRel(await f32(out), want), maxRel((await f32(ex)).subarray(0, 128 * 2 > nb2 ? nb2 : 256), wantEx)), tol: 1e-5 };
    },
  },
  {
    name: "histograms with shared-memory atomics",
    async run(ctx) {
      const n = 50_000; const bytes = new Uint8Array(n); let s = 12345;
      for (let i = 0; i < n; i++) { s = (s * 1103515245 + 12345) >>> 0; bytes[i] = (s >>> 16) & 255; }
      const m = ctx.compile(sources.histogram);
      const hist = ctx.zeros([256], "uint32");
      const d = ctx.tensor(bytes, [n], "uint8");
      await m.launch({ kernel: "histogram256", grid: [16], block: [256], args: [d, hist, n] });
      const want = new Array(256).fill(0); for (const b of bytes) want[b]++;
      const r = rng(8); const x = r(10_000, 0, 1); const hf = ctx.zeros([10], "int32");
      await m.launch({ kernel: "histogram_float", grid: [ceil(10_000, 256)], block: [256], args: [ctx.tensor(x), hf, 10_000, 0, 1, 10] });
      const wf = new Array(10).fill(0); for (const v of x) wf[Math.min(9, Math.floor(Math.fround(Math.fround(v - 0) / 1) * 10))]++;
      return { maxErr: Math.max(maxRel(Array.from(await hist.toArray() as Uint32Array), want), maxRel(Array.from(await hf.toArray() as Int32Array), wf)), tol: 0 };
    },
  },
  {
    name: "activations forward/backward (relu, gelu erf/tanh, silu, fused bias+act)",
    async run(ctx) {
      const n = 2000, r = rng(9); const x = r(n, -4, 4), dy = r(n);
      const m = ctx.compile(sources.activations);
      const X = ctx.tensor(x), DY = ctx.tensor(dy), Y = ctx.zeros([n]), DX = ctx.zeros([n]);
      const g = [ceil(n, 256)], b = [256];
      let e = 0;
      const sig = (v: number) => 1 / (1 + Math.exp(-v));
      const geluT = (v: number) => 0.5 * v * (1 + Math.tanh(0.7978845608028654 * (v + 0.044715 * v ** 3)));
      const checks: [string, unknown[], (v: number, i: number) => number, boolean][] = [
        ["relu_fwd", [X, Y, n], (v) => Math.max(v, 0), false],
        ["relu_bwd", [X, DY, DX, n], (v, i) => (v > 0 ? dy[i] : 0), true],
        ["gelu_fwd", [X, Y, n, 0], (v) => 0.5 * v * (1 + erf(v / Math.SQRT2)), false],
        ["gelu_fwd", [X, Y, n, 1], geluT, false],
        ["gelu_bwd", [X, DY, DX, n, 0], (v, i) => dy[i] * (0.5 * (1 + erf(v / Math.SQRT2)) + v * Math.exp(-0.5 * v * v) / Math.sqrt(2 * Math.PI)), true],
        ["silu_fwd", [X, Y, n], (v) => v * sig(v), false],
        ["silu_bwd", [X, DY, DX, n], (v, i) => dy[i] * sig(v) * (1 + v * (1 - sig(v))), true],
      ];
      for (const [k, args, ref, isBwd] of checks) {
        await m.launch({ kernel: k, grid: g, block: b, args: args as never });
        e = Math.max(e, maxRel(await f32(isBwd ? DX : Y), Array.from(x, ref), 1e-2));
      }
      const rows = 20, cols = 100, xb = r(rows * cols), bias = r(cols);
      const out = ctx.zeros([rows * cols]);
      await m.launch({ kernel: "bias_act", grid: [ceil(rows * cols, 256)], block: [256], args: [ctx.tensor(xb), ctx.tensor(bias), out, rows, cols, 2] });
      e = Math.max(e, maxRel(await f32(out), Array.from(xb, (v, i) => geluT(v + bias[i % cols])), 1e-2));
      return { maxErr: e, tol: 2e-4 };
    },
  },
  {
    name: "softmax rows forward/backward (warp shuffles)",
    async run(ctx) {
      const rows = 13, cols = 777, r = rng(10); const x = r(rows * cols, -5, 5), dy = r(rows * cols);
      const m = ctx.compile(sources.softmax);
      const Y = ctx.zeros([rows * cols]), DX = ctx.zeros([rows * cols]);
      await m.launch({ kernel: "softmax_rows<128>", grid: [rows], block: [128], args: [ctx.tensor(x), Y, cols] });
      const want = new Float64Array(rows * cols);
      for (let i = 0; i < rows; i++) { const row = x.subarray(i * cols, (i + 1) * cols); const mx = Math.max(...row); let s = 0; for (const v of row) s += Math.exp(v - mx); row.forEach((v, j) => (want[i * cols + j] = Math.exp(v - mx) / s)); }
      const y = await f32(Y);
      await m.launch({ kernel: "softmax_rows_bwd<128>", grid: [rows], block: [128], args: [Y, ctx.tensor(dy), DX, cols] });
      const wantDx = new Float64Array(rows * cols);
      for (let i = 0; i < rows; i++) { let d = 0; for (let j = 0; j < cols; j++) d += dy[i * cols + j] * want[i * cols + j]; for (let j = 0; j < cols; j++) wantDx[i * cols + j] = want[i * cols + j] * (dy[i * cols + j] - d); }
      return { maxErr: Math.max(maxRel(y, want, 1e-4), maxRel(await f32(DX), wantDx, 1e-4)), tol: 3e-4 };
    },
  },
  {
    name: "layernorm and rmsnorm forward/backward",
    async run(ctx) {
      const rows = 9, cols = 200, r = rng(11); const x = r(rows * cols, -2, 2), gm = r(cols, 0.5, 1.5), bt = r(cols), dy = r(rows * cols);
      const m = ctx.compile(sources.norms);
      const Y = ctx.zeros([rows * cols]), mean = ctx.zeros([rows]), rstd = ctx.zeros([rows]);
      const X = ctx.tensor(x), G = ctx.tensor(gm), B = ctx.tensor(bt), DY = ctx.tensor(dy);
      await m.launch({ kernel: "layernorm_fwd<128>", grid: [rows], block: [128], args: [X, G, B, Y, mean, rstd, cols, 1e-5] });
      const DX = ctx.zeros([rows * cols]), DG = ctx.zeros([cols]), DB = ctx.zeros([cols]);
      await m.launch({ kernel: "layernorm_bwd<128>", grid: [rows], block: [128], args: [X, G, DY, mean, rstd, DX, DG, DB, cols] });
      const wy = new Float64Array(rows * cols), wdx = new Float64Array(rows * cols), wdg = new Float64Array(cols), wdb = new Float64Array(cols);
      for (let i = 0; i < rows; i++) {
        const row = Array.from(x.subarray(i * cols, (i + 1) * cols));
        const mu = row.reduce((a, b) => a + b) / cols, v = row.reduce((a, b) => a + (b - mu) ** 2, 0) / cols, rs = 1 / Math.sqrt(v + 1e-5);
        const xh = row.map((t) => (t - mu) * rs);
        let mg = 0, mgx = 0;
        for (let j = 0; j < cols; j++) { wy[i * cols + j] = xh[j] * gm[j] + bt[j]; const g = dy[i * cols + j] * gm[j]; mg += g; mgx += g * xh[j]; wdg[j] += dy[i * cols + j] * xh[j]; wdb[j] += dy[i * cols + j]; }
        mg /= cols; mgx /= cols;
        for (let j = 0; j < cols; j++) wdx[i * cols + j] = rs * (dy[i * cols + j] * gm[j] - mg - xh[j] * mgx);
      }
      let e = Math.max(maxRel(await f32(Y), wy, 1e-2), maxRel(await f32(DX), wdx, 1e-2), maxRel(await f32(DG), wdg, 1e-2), maxRel(await f32(DB), wdb, 1e-2));
      // rmsnorm
      const RY = ctx.zeros([rows * cols]), rr = ctx.zeros([rows]), RDX = ctx.zeros([rows * cols]), RDW = ctx.zeros([cols]);
      await m.launch({ kernel: "rmsnorm_fwd<128>", grid: [rows], block: [128], args: [X, G, RY, rr, cols, 1e-6] });
      await m.launch({ kernel: "rmsnorm_bwd<128>", grid: [rows], block: [128], args: [X, G, DY, rr, RDX, RDW, cols] });
      const wry = new Float64Array(rows * cols), wrdx = new Float64Array(rows * cols), wrdw = new Float64Array(cols);
      for (let i = 0; i < rows; i++) {
        let ss = 0; for (let j = 0; j < cols; j++) ss += x[i * cols + j] ** 2;
        const rinv = 1 / Math.sqrt(ss / cols + 1e-6);
        let dot = 0; for (let j = 0; j < cols; j++) { wry[i * cols + j] = x[i * cols + j] * rinv * gm[j]; dot += dy[i * cols + j] * gm[j] * x[i * cols + j]; wrdw[j] += dy[i * cols + j] * x[i * cols + j] * rinv; }
        for (let j = 0; j < cols; j++) wrdx[i * cols + j] = dy[i * cols + j] * gm[j] * rinv - x[i * cols + j] * dot * rinv ** 3 / cols;
      }
      e = Math.max(e, maxRel(await f32(RY), wry, 1e-2), maxRel(await f32(RDX), wrdx, 1e-2), maxRel(await f32(RDW), wrdw, 1e-2));
      return { maxErr: e, tol: 1e-4 };
    },
  },
  {
    name: "conv2d (stride 2, padding 1), max/avg pooling, maxpool backward",
    async run(ctx) {
      const N = 2, C = 3, H = 9, W = 8, O = 4, KH = 3, KW = 3, st = 2, pad = 1;
      const OH = Math.floor((H + 2 * pad - KH) / st) + 1, OW = Math.floor((W + 2 * pad - KW) / st) + 1;
      const r = rng(12); const x = r(N * C * H * W), w = r(O * C * KH * KW), bias = r(O);
      const m = ctx.compile(sources.conv);
      const Y = ctx.zeros([N * O * OH * OW]);
      await m.launch({ kernel: "conv2d_fwd", grid: [ceil(N * O * OH * OW, 128)], block: [128], args: [ctx.tensor(x), ctx.tensor(w), ctx.tensor(bias), Y, N, C, H, W, O, KH, KW, st, pad, OH, OW] });
      const want: number[] = [];
      for (let n = 0; n < N; n++) for (let o = 0; o < O; o++) for (let oh = 0; oh < OH; oh++) for (let ow = 0; ow < OW; ow++) {
        let a = bias[o];
        for (let c = 0; c < C; c++) for (let kh = 0; kh < KH; kh++) for (let kw = 0; kw < KW; kw++) {
          const ih = oh * st - pad + kh, iw = ow * st - pad + kw;
          if (ih >= 0 && ih < H && iw >= 0 && iw < W) a += x[((n * C + c) * H + ih) * W + iw] * w[((o * C + c) * KH + kh) * KW + kw];
        }
        want.push(a);
      }
      let e = maxRel(await f32(Y), want, 1e-2);
      const PH = 4, PW = 4; const P = ctx.zeros([N * C * PH * PW]), AM = ctx.zeros([N * C * PH * PW], "int32"), AV = ctx.zeros([N * C * PH * PW]);
      const X = ctx.tensor(x);
      await m.launch({ kernel: "maxpool2d_fwd", grid: [1], block: [128], args: [X, P, AM, N, C, H, W, 2, 2, PH, PW] });
      await m.launch({ kernel: "avgpool2d_fwd", grid: [1], block: [128], args: [X, AV, N, C, H, W, 2, 2, PH, PW] });
      const wmax: number[] = [], wavg: number[] = [];
      for (let nc = 0; nc < N * C; nc++) for (let oh = 0; oh < PH; oh++) for (let ow = 0; ow < PW; ow++) {
        let mx = -Infinity, s = 0;
        for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) { const v = x[(nc * H + oh * 2 + i) * W + ow * 2 + j]; mx = Math.max(mx, v); s += v; }
        wmax.push(mx); wavg.push(s / 4);
      }
      e = Math.max(e, maxRel(await f32(P), wmax), maxRel(await f32(AV), wavg));
      const DX = ctx.zeros([N * C * H * W]);
      await m.launch({ kernel: "maxpool2d_bwd", grid: [1], block: [256], args: [ctx.ones([N * C * PH * PW]), AM, DX, N * C * PH * PW] });
      const dxs = (await f32(DX)).reduce((a, b) => a + b, 0);
      e = Math.max(e, Math.abs(dxs - N * C * PH * PW) / 96);
      return { maxErr: e, tol: 1e-5 };
    },
  },
  {
    name: "embedding forward and atomic backward (duplicate ids)",
    async run(ctx) {
      const vocab = 10, dim = 6, ids = [3, 1, 3, 9, 0, 3], n = ids.length, r = rng(13);
      const table = r(vocab * dim), dout = r(n * dim);
      const m = ctx.compile(sources.embedding);
      const I = ctx.tensor(ids, [n], "int32"), out = ctx.zeros([n * dim]), dt = ctx.zeros([vocab * dim]);
      await m.launch({ kernel: "embedding_fwd", grid: [1], block: [64], args: [I, ctx.tensor(table), out, n, dim, vocab] });
      await m.launch({ kernel: "embedding_bwd", grid: [1], block: [64], args: [I, ctx.tensor(dout), dt, n, dim, vocab] });
      const wo: number[] = [], wd = new Float64Array(vocab * dim);
      ids.forEach((id, t) => { for (let d = 0; d < dim; d++) { wo.push(table[id * dim + d]); wd[id * dim + d] += dout[t * dim + d]; } });
      return { maxErr: Math.max(maxRel(await f32(out), wo), maxRel(await f32(dt), wd)), tol: 1e-6 };
    },
  },
  {
    name: "fused cross-entropy forward/backward",
    async run(ctx) {
      const rows = 7, cols = 300, r = rng(14); const x = r(rows * cols, -3, 3); const t = Array.from({ length: rows }, (_, i) => (i * 37) % cols);
      const m = ctx.compile(sources.crossEntropy);
      const L = ctx.zeros([rows]), D = ctx.zeros([rows * cols]);
      await m.launch({ kernel: "cross_entropy_fwd_bwd<128>", grid: [rows], block: [128], args: [ctx.tensor(x), ctx.tensor(t, [rows], "int32"), L, D, cols, rows] });
      const wl: number[] = [], wd = new Float64Array(rows * cols);
      for (let i = 0; i < rows; i++) {
        const row = x.subarray(i * cols, (i + 1) * cols); const mx = Math.max(...row); let s = 0; for (const v of row) s += Math.exp(v - mx);
        const lse = mx + Math.log(s); wl.push(lse - row[t[i]]);
        for (let j = 0; j < cols; j++) wd[i * cols + j] = (Math.exp(row[j] - lse) - (j === t[i] ? 1 : 0)) / rows;
      }
      return { maxErr: Math.max(maxRel(await f32(L), wl, 1e-2), maxRel(await f32(D), wd, 1e-4)), tol: 2e-4 };
    },
  },
  {
    name: "attention: naive SDPA (causal + GQA) vs flash attention",
    async run(ctx) {
      const B = 2, H = 4, HKV = 2, T = 37, S = 37, D = 16, r = rng(15);
      const q = r(B * H * T * D), k = r(B * HKV * S * D), v = r(B * HKV * S * D);
      const scale = 1 / Math.sqrt(D);
      const want = (causal: boolean) => {
        const o = new Float64Array(B * H * T * D);
        for (let b = 0; b < B; b++) for (let h = 0; h < H; h++) for (let i = 0; i < T; i++) {
          const hk = Math.floor(h / (H / HKV));
          const last = causal ? Math.min(S - 1, i + (S - T)) : S - 1;
          const sc: number[] = [];
          for (let j = 0; j <= last; j++) { let s = 0; for (let d = 0; d < D; d++) s += q[((b * H + h) * T + i) * D + d] * k[((b * HKV + hk) * S + j) * D + d]; sc.push(s * scale); }
          const mx = Math.max(...sc); const p = sc.map((s) => Math.exp(s - mx)); const l = p.reduce((a, c) => a + c);
          for (let d = 0; d < D; d++) { let a = 0; for (let j = 0; j <= last; j++) a += p[j] * v[((b * HKV + hk) * S + j) * D + d]; o[((b * H + h) * T + i) * D + d] = a / l; }
        }
        return o;
      };
      const m = ctx.compile(sources.attention);
      const Q = ctx.tensor(q), K = ctx.tensor(k), V = ctx.tensor(v);
      let e = 0;
      for (const causal of [0, 1]) {
        const O1 = ctx.zeros([B * H * T * D]), O2 = ctx.zeros([B * H * T * D]), LSE = ctx.zeros([B * H * T]);
        await m.launch({ kernel: "sdpa_naive", grid: [ceil(B * H * T, 64)], block: [64], args: [Q, K, V, O1, B, H, HKV, T, S, D, scale, causal] });
        await m.launch({ kernel: "flash_attention_fwd<32, 16, 16>", grid: [ceil(T, 32), B * H], block: [32], args: [Q, K, V, O2, LSE, B, H, HKV, T, S, D, scale, causal] });
        const w = want(Boolean(causal));
        e = Math.max(e, maxRel(await f32(O1), w, 1e-2), maxRel(await f32(O2), w, 1e-2));
      }
      return { maxErr: e, tol: 2e-4 };
    },
  },
  {
    name: "rotary embeddings forward/backward round trip, KV cache append",
    async run(ctx) {
      const B = 1, H = 2, T = 5, D = 8, r = rng(16); const x = r(B * H * T * D);
      const m = ctx.compile(sources.rope);
      const X = ctx.tensor(x);
      const n = B * H * T * D / 2;
      await m.launch({ kernel: "rope_apply", grid: [1], block: [64], args: [X, B, H, T, D, 3, 10000] });
      const want = new Float64Array(x.length);
      for (let row = 0; row < B * H * T; row++) for (let i = 0; i < D / 2; i++) {
        const t = row % T; const ang = (3 + t) * Math.pow(10000, -2 * i / D); const a = x[row * D + 2 * i], b = x[row * D + 2 * i + 1];
        want[row * D + 2 * i] = a * Math.cos(ang) - b * Math.sin(ang); want[row * D + 2 * i + 1] = a * Math.sin(ang) + b * Math.cos(ang);
      }
      let e = maxRel(await f32(X), want, 1e-2);
      await m.launch({ kernel: "rope_bwd", grid: [1], block: [64], args: [X, B, H, T, D, 3, 10000] });
      e = Math.max(e, maxRel(await f32(X), x, 1e-2));
      void n;
      const kc = ctx.compile(sources.kvcache);
      const MAXT = 8, HK = 2; const kk = r(B * HK * 2 * D), vv = r(B * HK * 2 * D);
      const KC = ctx.zeros([B * HK * MAXT * D]), VC = ctx.zeros([B * HK * MAXT * D]);
      await kc.launch({ kernel: "kv_cache_append", grid: [1], block: [64], args: [ctx.tensor(kk), ctx.tensor(vv), KC, VC, B, HK, 2, D, MAXT, 5] });
      const kch = await f32(KC);
      const wk = new Float64Array(B * HK * MAXT * D);
      for (let bh = 0; bh < B * HK; bh++) for (let t = 0; t < 2; t++) for (let d = 0; d < D; d++) wk[(bh * MAXT + 5 + t) * D + d] = kk[(bh * 2 + t) * D + d];
      e = Math.max(e, maxRel(kch, wk));
      return { maxErr: e, tol: 5e-4 };
    },
  },
  {
    name: "quantized matmul: INT8 (scalar and __dp4a), INT4 grouped, fp16 storage",
    async run(ctx) {
      const M = 5, N = 6, K = 16, r = rng(17);
      const xq = Int8Array.from(r(M * K, -127, 127), Math.round), wq = Int8Array.from(r(N * K, -127, 127), Math.round);
      const sx = r(M, 0.01, 0.1), sw = r(N, 0.01, 0.1);
      const want: number[] = [];
      for (let m2 = 0; m2 < M; m2++) for (let n = 0; n < N; n++) { let a = 0; for (let k = 0; k < K; k++) a += xq[m2 * K + k] * wq[n * K + k]; want.push(Math.fround(Math.fround(a * sx[m2]) * sw[n])); }
      const m = ctx.compile(sources.quantized);
      const Y = ctx.zeros([M * N]), Y4 = ctx.zeros([M * N]);
      await m.launch({ kernel: "int8_matmul", grid: [1, 1], block: [16, 16], args: [ctx.tensor(xq, [M * K], "int8"), ctx.tensor(sx), ctx.tensor(wq, [N * K], "int8"), ctx.tensor(sw), Y, M, N, K] });
      await m.launch({ kernel: "int8_matmul_dp4a", grid: [1, 1], block: [16, 16], args: [ctx.tensor(new Int32Array(xq.buffer)), ctx.tensor(sx), ctx.tensor(new Int32Array(wq.buffer)), ctx.tensor(sw), Y4, M, N, K / 4] });
      let e = Math.max(maxRel(await f32(Y), want), maxRel(await f32(Y4), want));
      // int4: nibble values 0..15 (zero point 8), group size 8
      const G = 8, groups = K / G;
      const nib = Array.from({ length: N * K }, (_, i) => (i * 7 + 3) % 16);
      const packed = new Uint8Array(N * K / 2); for (let i = 0; i < N * K; i += 2) packed[i / 2] = nib[i] | (nib[i + 1] << 4);
      const sc = r(N * groups, 0.05, 0.2), x = r(M * K);
      const Y5 = ctx.zeros([M * N]);
      await m.launch({ kernel: "int4_matmul", grid: [1, 1], block: [16, 16], args: [ctx.tensor(x), ctx.tensor(packed, [N * K / 2], "uint8"), ctx.tensor(sc), Y5, M, N, K, G] });
      const w4: number[] = [];
      for (let m2 = 0; m2 < M; m2++) for (let n = 0; n < N; n++) { let a = 0; for (let g = 0; g < groups; g++) { let p = 0; for (let k = g * G; k < (g + 1) * G; k++) p += x[m2 * K + k] * (nib[n * K + k] - 8); a += p * sc[n * groups + g]; } w4.push(a); }
      e = Math.max(e, maxRel(await f32(Y5), w4, 1e-2));
      // fp16 matmul
      const ah = r(M * K), bh = r(K * N);
      const A16 = ctx.tensor(Array.from(ah), [M * K], "float16"), B16 = ctx.tensor(Array.from(bh), [K * N], "float16");
      const a16 = Array.from(await A16.toArray() as Uint16Array, halfFromBits), b16 = Array.from(await B16.toArray() as Uint16Array, halfFromBits);
      const C16 = ctx.zeros([M * N]);
      await m.launch({ kernel: "matmul_f16", grid: [1, 1], block: [16, 16], args: [A16, B16, C16, M, N, K] });
      e = Math.max(e, maxRel(await f32(C16), matmulRef(Float32Array.from(a16), Float32Array.from(b16), M, N, K), 1e-2));
      return { maxErr: e, tol: 1e-5 };
    },
  },
  {
    name: "sparse CSR SpMV / SpMM",
    async run(ctx) {
      const rows = 20, colsN = 15; const rp = [0], ci: number[] = [], vals: number[] = []; const r = rng(18);
      const dense = new Float64Array(rows * colsN);
      for (let i = 0; i < rows; i++) { for (let j = 0; j < colsN; j++) if ((i * 7 + j * 3) % 5 === 0) { const v = r(1)[0]; ci.push(j); vals.push(v); dense[i * colsN + j] = Math.fround(v); } rp.push(ci.length); }
      const x = r(colsN); const Nn = 4; const X = r(colsN * Nn);
      const m = ctx.compile(sources.sparse);
      const RP = ctx.tensor(rp, [rp.length], "int32"), CI = ctx.tensor(ci, [ci.length], "int32"), VA = ctx.tensor(vals);
      const y = ctx.zeros([rows]), Y = ctx.zeros([rows * Nn]);
      await m.launch({ kernel: "csr_spmv", grid: [1], block: [64], args: [RP, CI, VA, ctx.tensor(x), y, rows] });
      await m.launch({ kernel: "csr_spmm", grid: [1, rows], block: [32], args: [RP, CI, VA, ctx.tensor(X), Y, rows, Nn] });
      const wy = Array.from({ length: rows }, (_, i) => { let s = 0; for (let j = 0; j < colsN; j++) s += dense[i * colsN + j] * x[j]; return s; });
      const wY: number[] = []; for (let i = 0; i < rows; i++) for (let c = 0; c < Nn; c++) { let s = 0; for (let j = 0; j < colsN; j++) s += dense[i * colsN + j] * X[j * Nn + c]; wY.push(s); }
      return { maxErr: Math.max(maxRel(await f32(y), wy, 1e-2), maxRel(await f32(Y), wY, 1e-2)), tol: 1e-5 };
    },
  },
  {
    name: "mixture of experts: top-k routing, capacity dispatch, weighted combine",
    async run(ctx) {
      const tokens = 9, E = 4, k = 2, dim = 5, cap = 8, r = rng(19);
      const logits = r(tokens * E, -2, 2), x = r(tokens * dim);
      const m = ctx.compile(sources.moe);
      const EX = ctx.zeros([tokens * k], "int32"), W = ctx.zeros([tokens * k]);
      await m.launch({ kernel: "moe_topk", grid: [1], block: [32], args: [ctx.tensor(logits), EX, W, tokens, E, k] });
      const ex = Array.from(await EX.toArray() as Int32Array), w = await f32(W);
      const wantEx: number[] = [], wantW: number[] = [];
      for (let t = 0; t < tokens; t++) {
        const l = Array.from(logits.subarray(t * E, (t + 1) * E)); const order = l.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, k);
        const mx = Math.max(...l), z = l.reduce((a, v) => a + Math.exp(v - mx), 0);
        const ws = order.map(([v]) => Math.exp(v - mx) / z); const s = ws.reduce((a, b) => a + b);
        order.forEach(([, i], j) => { wantEx.push(i); wantW.push(ws[j] / s); });
      }
      let e = Math.max(maxRel(ex, wantEx), maxRel(w, wantW));
      const BUF = ctx.zeros([E * cap * dim]), CNT = ctx.zeros([E], "int32"), SLOT = ctx.zeros([tokens * k], "int32");
      await m.launch({ kernel: "moe_dispatch", grid: [tokens * k], block: [32], args: [ctx.tensor(x), EX, BUF, CNT, SLOT, tokens, k, dim, cap] });
      // identity experts: combine must give sum_j w_j * x (all slots fit the capacity)
      const Y = ctx.zeros([tokens * dim]);
      await m.launch({ kernel: "moe_combine", grid: [1], block: [64], args: [BUF, SLOT, W, Y, tokens, k, dim] });
      const wy: number[] = []; for (let t = 0; t < tokens; t++) for (let d = 0; d < dim; d++) wy.push((w[t * k] + w[t * k + 1]) * x[t * dim + d]);
      e = Math.max(e, maxRel(await f32(Y), wy, 1e-3));
      return { maxErr: e, tol: 1e-5 };
    },
  },
  {
    name: "optimizers: SGD momentum/nesterov, Adam, AdamW, global-norm clipping, accumulation",
    async run(ctx) {
      const n = 500, r = rng(20); const p0 = r(n), g = r(n);
      const m = ctx.compile(sources.optimizers);
      let e = 0;
      // SGD (nesterov, wd), two steps
      { const P = ctx.tensor(p0), G = ctx.tensor(g), BUF = ctx.zeros([n]);
        for (let s = 0; s < 2; s++) await m.launch({ kernel: "sgd_step", grid: [2], block: [256], args: [P, G, BUF, n, 0.1, 0.9, 0.01, 1, s === 0 ? 1 : 0] });
        const want = Array.from(p0, (p, i) => { let b = 0, w = p; for (let s = 0; s < 2; s++) { const d = g[i] + 0.01 * w; b = s === 0 ? d : 0.9 * b + d; w -= 0.1 * (d + 0.9 * b); } return w; });
        e = Math.max(e, maxRel(await f32(P), want, 1e-2)); }
      for (const decoupled of [0, 1]) {
        const P = ctx.tensor(p0), G = ctx.tensor(g), M1 = ctx.zeros([n]), V1 = ctx.zeros([n]);
        for (let s = 1; s <= 3; s++) await m.launch({ kernel: "adam_step", grid: [2], block: [256], args: [P, G, M1, V1, n, 1e-2, 0.9, 0.999, 1e-8, 0.1, s, decoupled] });
        const want = Array.from(p0, (p, i) => { let mm = 0, vv = 0, w = p; for (let s = 1; s <= 3; s++) { let gr = g[i]; if (decoupled) w *= 1 - 1e-2 * 0.1; else gr += 0.1 * w; mm = 0.9 * mm + 0.1 * gr; vv = 0.999 * vv + 0.001 * gr * gr; w -= 1e-2 * (mm / (1 - 0.9 ** s)) / (Math.sqrt(vv / (1 - 0.999 ** s)) + 1e-8); } return w; });
        e = Math.max(e, maxRel(await f32(P), want, 1e-2));
      }
      const G = ctx.tensor(g), SS = ctx.zeros([1]);
      await m.launch({ kernel: "sumsq<256>", grid: [2], block: [256], args: [G, SS, n] });
      await m.launch({ kernel: "clip_by_norm", grid: [2], block: [256], args: [G, SS, n, 1.0] });
      const norm = Math.sqrt(g.reduce((a, b) => a + b * b, 0)); const c = Math.min(1, 1 / (norm + 1e-6));
      e = Math.max(e, maxRel(await f32(G), Array.from(g, (v) => v * c), 1e-2));
      const ACC = ctx.tensor(p0);
      await m.launch({ kernel: "accumulate", grid: [2], block: [256], args: [ACC, ctx.tensor(g), n, 0.25] });
      e = Math.max(e, maxRel(await f32(ACC), Array.from(p0, (v, i) => v + 0.25 * g[i]), 1e-2));
      // powf/sqrtf: WGSL allows a few ulp more error than CUDA's library
      return { maxErr: e, tol: 2e-4 };
    },
  },
  {
    name: "mixed precision: f16/bf16 casts (exact RNE), loss-scale overflow detection, master weights",
    async run(ctx) {
      const vals = [0, 1, -2.5, 65504, 65520, 1e-8, 6.1e-5, 3.14159, 1 / 3, -0.0001, 1e5];
      const n = vals.length;
      const m = ctx.compile(sources.mixedPrecision);
      const X = ctx.tensor(vals), H16 = ctx.empty([n], "float16"), BF = ctx.empty([n], "bfloat16"), BACK = ctx.zeros([n]);
      await m.launch({ kernel: "cast_f32_to_f16", grid: [1], block: [32], args: [X, H16, n] });
      await m.launch({ kernel: "cast_f32_to_bf16", grid: [1], block: [32], args: [X, BF, n] });
      await m.launch({ kernel: "cast_f16_to_f32", grid: [1], block: [32], args: [H16, BACK, n] });
      const { halfBits, bf16Bits } = await import("../../src/cuda/compiler/fold.js");
      const h = Array.from(await H16.toArray() as Uint16Array), b = Array.from(await BF.toArray() as Uint16Array);
      let e = Math.max(maxRel(h, vals.map((v) => halfBits(Math.fround(v)))), maxRel(b, vals.map((v) => bf16Bits(Math.fround(v)))));
      e = Math.max(e, maxRel(await f32(BACK), vals.map((v) => halfFromBits(halfBits(Math.fround(v))))));
      const g16 = ctx.tensor([1, 2, Infinity, 4], [4], "float16"), g32 = ctx.zeros([4]), inf = ctx.zeros([1], "int32");
      await m.launch({ kernel: "unscale_check", grid: [1], block: [32], args: [g16, g32, inf, 4, 2.0] });
      e = Math.max(e, maxRel(Array.from(await inf.toArray() as Int32Array), [1]));
      const inf2 = ctx.zeros([1], "int32");
      await m.launch({ kernel: "unscale_check", grid: [1], block: [32], args: [ctx.tensor([1, 2, 3, 4], [4], "float16"), g32, inf2, 4, 0.5] });
      e = Math.max(e, maxRel(Array.from(await inf2.toArray() as Int32Array), [0]), maxRel(await f32(g32), [0.5, 1, 1.5, 2]));
      const master = ctx.tensor([1, 2, 3, 4]), p16 = ctx.empty([4], "float16");
      await m.launch({ kernel: "master_step", grid: [1], block: [32], args: [master, p16, g32, inf2, 4, 0.1] });
      e = Math.max(e, maxRel(await f32(master), [0.95, 1.9, 2.85, 3.8].map(Math.fround), 1e-3));
      return { maxErr: e, tol: 1e-6 };
    },
  },
  {
    name: "cuBLAS: sgemm (all transposes, alpha/beta, column-major), strided batched, gemv, axpy, dot, nrm2",
    async run(ctx) {
      const h = cublas.cublasCreate(ctx);
      const m = 7, n = 5, k = 6, r = rng(21);
      let e = 0;
      const col = (A: Float32Array, rows: number, i: number, j: number) => A[i + j * rows];
      for (const [ta, tb] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
        const A = r(m * k), B = r(k * n), C0 = r(m * n);
        const lda = ta ? k : m, ldb = tb ? n : k;
        const dC = ctx.tensor(C0);
        await cublas.cublasSgemm(h, ta, tb, m, n, k, 1.5, ctx.tensor(A).ptr, lda, ctx.tensor(B).ptr, ldb, 0.5, dC.ptr, m);
        const want: number[] = [];
        for (let j = 0; j < n; j++) for (let i = 0; i < m; i++) {
          let s = 0;
          for (let l = 0; l < k; l++) s += (ta ? col(A, lda, l, i) : col(A, lda, i, l)) * (tb ? col(B, ldb, j, l) : col(B, ldb, l, j));
          want[i + j * m] = 1.5 * s + 0.5 * C0[i + j * m];
        }
        e = Math.max(e, maxRel(await f32(dC), want, 1e-2));
      }
      const x = r(k), y = r(m), A = r(m * k);
      const dy = ctx.tensor(y);
      await cublas.cublasSgemv(h, 0, m, k, 2, ctx.tensor(A).ptr, m, ctx.tensor(x).ptr, 1, 1, dy.ptr, 1);
      const wy = Array.from(y, (v, i) => { let s = 0; for (let l = 0; l < k; l++) s += A[i + l * m] * x[l]; return 2 * s + v; });
      e = Math.max(e, maxRel(await f32(dy), wy, 1e-2));
      const v1 = r(1000), v2 = r(1000);
      const d1 = ctx.tensor(v1), d2 = ctx.tensor(v2);
      const dot = await cublas.cublasSdot(h, 1000, d1.ptr, 1, d2.ptr, 1);
      const nrm = await cublas.cublasSnrm2(h, 1000, d1.ptr, 1);
      e = Math.max(e, Math.abs(dot - v1.reduce((a, b, i) => a + b * v2[i], 0)) / 10, Math.abs(nrm - Math.sqrt(v1.reduce((a, b) => a + b * b, 0))) / 10);
      await cublas.cublasSaxpy(h, 1000, -2, d1.ptr, 1, d2.ptr, 1);
      e = Math.max(e, maxRel(await f32(d2), Array.from(v2, (v, i) => v - 2 * v1[i]), 1e-2));
      return { maxErr: e, tol: 1e-5 };
    },
  },
  {
    name: "cuDNN: activation, channel softmax, convolution, pooling",
    async run(ctx) {
      const h = cudnn.cudnnCreate(ctx);
      const r = rng(22);
      const xd = { n: 2, c: 3, h: 5, w: 5 }; const x = r(150);
      const X = ctx.tensor(x), Y = ctx.zeros([150]);
      await cudnn.cudnnActivationForward(h, cudnn.CUDNN_ACTIVATION_TANH, 0, 1, xd, X.ptr, 0, Y.ptr);
      let e = maxRel(await f32(Y), Array.from(x, Math.tanh), 1e-2);
      await cudnn.cudnnSoftmaxForward(h, cudnn.CUDNN_SOFTMAX_ACCURATE, cudnn.CUDNN_SOFTMAX_MODE_CHANNEL, 1, xd, X.ptr, 0, Y.ptr);
      const ws: number[] = new Array(150);
      for (let n = 0; n < 2; n++) for (let p = 0; p < 25; p++) { const vs = [0, 1, 2].map((c) => x[(n * 3 + c) * 25 + p]); const mx = Math.max(...vs); const z = vs.reduce((a, v) => a + Math.exp(v - mx), 0); vs.forEach((v, c) => (ws[(n * 3 + c) * 25 + p] = Math.exp(v - mx) / z)); }
      e = Math.max(e, maxRel(await f32(Y), ws, 1e-3));
      const wd = { k: 2, c: 3, h: 3, w: 3 }, conv = { padH: 1, padW: 1, strideH: 1, strideW: 1 };
      const yd = cudnn.cudnnGetConvolution2dForwardOutputDim(conv, xd, wd);
      const w = r(54); const CY = ctx.zeros([yd.n * yd.c * yd.h * yd.w]);
      await cudnn.cudnnConvolutionForward(h, 1, xd, X.ptr, wd, ctx.tensor(w).ptr, conv, 0, yd, CY.ptr);
      const wc: number[] = [];
      for (let n = 0; n < 2; n++) for (let o = 0; o < 2; o++) for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) {
        let a = 0; for (let c = 0; c < 3; c++) for (let p = 0; p < 3; p++) for (let q = 0; q < 3; q++) { const ii = i - 1 + p, jj = j - 1 + q; if (ii >= 0 && ii < 5 && jj >= 0 && jj < 5) a += x[((n * 3 + c) * 5 + ii) * 5 + jj] * w[((o * 3 + c) * 3 + p) * 3 + q]; }
        wc.push(a);
      }
      e = Math.max(e, maxRel(await f32(CY), wc, 1e-2));
      const pd = { mode: cudnn.CUDNN_POOLING_MAX, windowH: 2, windowW: 2, padH: 0, padW: 0, strideH: 2, strideW: 2 };
      const PY = ctx.zeros([2 * 3 * 2 * 2]);
      await cudnn.cudnnPoolingForward(h, pd, 1, xd, X.ptr, 0, { n: 2, c: 3, h: 2, w: 2 }, PY.ptr);
      const wp: number[] = []; for (let nc = 0; nc < 6; nc++) for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) wp.push(Math.max(x[(nc * 5 + 2 * i) * 5 + 2 * j], x[(nc * 5 + 2 * i) * 5 + 2 * j + 1], x[(nc * 5 + 2 * i + 1) * 5 + 2 * j], x[(nc * 5 + 2 * i + 1) * 5 + 2 * j + 1]));
      e = Math.max(e, maxRel(await f32(PY), wp));
      return { maxErr: e, tol: 2e-4 };
    },
  },
  {
    name: "cuRAND: Philox uniform/normal statistics and reproducibility",
    async run(ctx) {
      const g = curand.curandCreateGenerator(ctx);
      curand.curandSetPseudoRandomGeneratorSeed(g, 1234);
      const n = 40_000;
      const U = ctx.zeros([n]), Nn = ctx.zeros([n]);
      await curand.curandGenerateUniform(g, U.ptr, n);
      await curand.curandGenerateNormal(g, Nn.ptr, n, 1, 2);
      const u = await f32(U), z = await f32(Nn);
      const mean = (a: Float32Array) => a.reduce((s, v) => s + v, 0) / a.length;
      const varr = (a: Float32Array) => { const m = mean(a); return a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length; };
      const inRange = u.every((v) => v > 0 && v <= 1);
      // reproducible: same seed and offset regenerate the same values
      const g2 = curand.curandCreateGenerator(ctx); curand.curandSetPseudoRandomGeneratorSeed(g2, 1234);
      const U2 = ctx.zeros([n]); await curand.curandGenerateUniform(g2, U2.ptr, n);
      const same = maxRel(await f32(U2), u);
      const err = Math.max(Math.abs(mean(u) - 0.5) * 10, Math.abs(varr(u) - 1 / 12) * 50, Math.abs(mean(z) - 1) / 2, Math.abs(Math.sqrt(varr(z)) - 2) / 2, inRange ? 0 : 1, same);
      return { maxErr: err, tol: 0.05, info: `uniform mean ${mean(u).toFixed(4)} var ${varr(u).toFixed(4)}, normal mean ${mean(z).toFixed(3)} std ${Math.sqrt(varr(z)).toFixed(3)}` };
    },
  },
];

// ------------------------------------------------------------------ compiler feature cases
const FEATURES = `
#include <cuda_fp16.h>
struct Pair { float a; int b; };
struct Acc { float s; __device__ Acc() : s(0.0f) {} __device__ void add(float v) { s += v; } };
__device__ float2 operator*(float2 x, float s) { return make_float2(x.x * s, x.y * s); }
template <typename T> __device__ T clampv(T x, T lo, T hi) { return x < lo ? lo : (x > hi ? hi : x); }
__constant__ float c_table[4] = {1.0f, 2.0f, 3.0f, 4.0f};
__device__ int g_counter;
__global__ void features(float* out, int* iout, Pair p, int n) {
  int i = threadIdx.x;
  // structs by value, methods, constructors, operator overloading, templates
  Acc acc;
  for (int k = 0; k < 4; ++k) acc.add(c_table[k]);
  float2 v = make_float2(1.0f, 2.0f) * 3.0f;
  out[0] = acc.s;                   // 10
  out[1] = v.y;                     // 6
  out[2] = clampv(7.5f, 0.0f, 5.0f);    // 5
  out[3] = (float)clampv<int>(-3, 0, 9); // 0
  out[4] = p.a * p.b;               // struct parameter
  // integer semantics
  iout[0] = -7 / 2;                 // -3 (truncation)
  iout[1] = -7 % 3;                 // -1
  iout[2] = (int)(0xffffffffu >> 28); // 15
  iout[3] = __popc(0xf0f0u) + __clz(1) + __ffs(8);   // 8 + 31 + 4
  unsigned int u = 3000000000u;
  iout[4] = (int)(u / 7u);          // unsigned division
  long long big = 1LL << 40;
  iout[5] = (int)(big >> 38);       // 4
  iout[6] = (int)__brev(1u) == (int)0x80000000 ? 1 : 0;
  iout[7] = __umulhi(0x80000000u, 6u); // 3
  // math functions
  out[5] = sqrtf(2.0f);
  out[6] = expf(1.0f);
  out[7] = powf(-2.0f, 3.0f);       // -8
  out[8] = fmodf(7.5f, 2.0f);       // 1.5
  out[9] = roundf(-2.5f);           // -3 (away from zero)
  out[10] = rintf(2.5f);            // 2 (half to even)
  out[11] = fminf(NAN, 3.0f);       // 3
  out[12] = __half2float(__hmul(__float2half(1.5f), __float2half(2.0f))); // 3
  out[13] = erff(0.5f);
  out[14] = tanhf(20.0f);           // 1
  out[15] = copysignf(2.0f, -0.0f); // -2
  out[16] = (float)isinf(INFINITY) + (float)isnan(NAN);  // 2
  out[17] = log1pf(1e-7f);
  // device variable through atomics, switch with fallthrough, do-while, goto-free control flow
  if (i == 0) atomicAdd(&g_counter, 5);
  int s = 0;
  switch (n) { case 1: s += 1; case 2: s += 2; break; default: s = 100; }
  iout[8] = s;                      // n = 1: 3
  int t = 0;
  do { t += 2; } while (t < 7);
  iout[9] = t;                      // 8
  int lsum = 0;
  for (int k = 0; k < 10; ++k) { if (k == 3) continue; if (k == 8) break; lsum += k; }
  iout[10] = lsum;                  // 0+1+2+4+5+6+7 = 25
}
__global__ void read_counter(int* o) { o[0] = g_counter; }
`;

CASES.push({
  name: "compiler features: structs, methods, operators, templates, __constant__/__device__, integer & math semantics, control flow",
  async run(ctx) {
    const m = ctx.compile(FEATURES);
    const out = ctx.zeros([18]), io = ctx.zeros([11], "int32");
    await m.launch({ kernel: "features", grid: 1, block: 1, args: [out, io, { a: 2.5, b: 4 }, 1] });
    const c = ctx.zeros([1], "int32");
    await m.launch({ kernel: "read_counter", grid: 1, block: 1, args: [c] });
    const o = await f32(out), ii = Array.from(await io.toArray() as Int32Array);
    const wantF = [10, 6, 5, 0, 10, Math.SQRT2, Math.E, -8, 1.5, -3, 2, 3, 3, erf(0.5), 1, -2, 2, Math.log1p(1e-7)];
    const wantI = [-3, -1, 15, 43, Math.floor(3000000000 / 7), 4, 1, 3, 3, 8, 25];
    const counter = Array.from(await c.toArray() as Int32Array)[0];
    return { maxErr: Math.max(maxRel(o, wantF, 1e-3), maxRel(ii, wantI), Math.abs(counter - 5)), tol: 1e-5, info: `floats ${Array.from(o).map((v) => v.toPrecision(5)).join(",")} ints ${ii}` };
  },
});

CASES.push({
  name: "barrier semantics: early return before barrier, divergent loops, __syncthreads_count/or/and, warp ballot/any/all/shuffles (width 8)",
  async run(ctx) {
    const m = ctx.compile(`
__global__ void bar(int* o, int n) {
  __shared__ int s[128];
  int t = threadIdx.x;
  s[t] = t;
  __syncthreads();
  int c = __syncthreads_count(t % 3 == 0);
  int a = __syncthreads_and(t < 1000);
  int r = __syncthreads_or(t == 77);
  unsigned b = __ballot_sync(0xffffffff, (t & 1) == 0);
  int any = __any_sync(0xffffffff, t == 33), all = __all_sync(0xffffffff, t < 64);
  int x = __shfl_xor_sync(0xffffffff, t, 3, 8);
  int up = __shfl_up_sync(0xffffffff, t, 2, 8);
  int bc = __shfl_sync(0xffffffff, t * 10, 5);
  if (t == 0) { o[0] = c; o[1] = a; o[2] = r; o[3] = (int)b; }
  if (t == 33) { o[4] = any; o[5] = all; o[6] = x; o[7] = up; o[8] = bc; }
  if (t >= n) return;
  // loop with a barrier and a trip count that depends on uniform data
  int acc = 0;
  for (int k = 0; k < 4; ++k) { acc += s[(t + k) % 128]; __syncthreads(); }
  o[9 + t] = acc;
}`);
    const o = ctx.zeros([9 + 128], "int32");
    await m.launch({ grid: 1, block: 128, args: [o, 100] });
    const r = Array.from(await o.toArray() as Int32Array);
    const want = [43, 1, 1, 0x55555555 | 0, 1, 1, 0, 0, 0];
    // lane 33: warp 1 lanes 32..63; shfl_xor width 8: 33^3 = 34? (lane 1 ^ 3 = 2 -> thread 34)
    want[6] = 34; want[7] = 33; /* lane 1 of its 8-segment: (1 & 7) < 2 keeps its own value */ want[8] = 37 * 10;
    for (let t = 0; t < 128; t++) want[9 + t] = t < 100 ? [0, 1, 2, 3].reduce((a, k) => a + ((t + k) % 128), 0) : 0;
    return { maxErr: maxRel(r, want), tol: 0, info: r.join(",") };
  },
});

CASES.push({
  name: "device printf and assert",
  async run(ctx) {
    const m = ctx.compile(`
#include <assert.h>
__global__ void p(int n) {
  if (threadIdx.x == 1 && blockIdx.x == 2) printf("block %d thread %d: %5.2f %x %c %lld\\n", blockIdx.x, threadIdx.x, 3.14159f, 255u, 'Z', -12345678901LL);
  assert(threadIdx.x < n);
}`);
    await m.launch({ grid: 3, block: 4, args: [100] });
    const r = await ctx.synchronize();
    const text = r.printf.join("");
    let err = text === "block 2 thread 1:  3.14 ff Z -12345678901\n" ? 0 : 1;
    // a failing assert is reported at synchronization as cudaErrorAssert with the source location
    await m.launch({ grid: 1, block: 8, args: [4] });
    try { await ctx.synchronize(); err += 1; } catch (e) { if (!/cudaErrorAssert: .*Assertion `threadIdx.x < n` failed/.test((e as Error).message)) err += 1; }
    (ctx.backend as { resetError?: () => void }).resetError?.();
    return { maxErr: err, tol: 0, info: text };
  },
});
