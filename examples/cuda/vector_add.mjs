// CUDA kernels from Node.js on the WebAssembly CPU backend (the same API runs on WebGPU in browsers).
//   npm run build && node examples/cuda/vector_add.mjs
import { CUDA, kernels, cublas } from "../../dist/cuda.js";

CUDA.configure({ device: "cpu", workerURL: new URL("../../dist/cuda-worker.js", import.meta.url) });

const kernel = await CUDA.compile(`
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { c[i] = a[i] + b[i]; }
}`);
const ctx = kernel.ctx;
console.log(`device: ${ctx.info().name} (${ctx.selection.reasons.join("; ") || ctx.selection.selected})`);

const a = ctx.rand([1024]), b = ctx.rand([1024]), c = ctx.zeros([1024]);
await kernel.launch({ grid: [4, 1, 1], block: [256, 1, 1], args: [a, b, c, 1024] });
const [ah, bh, ch] = await Promise.all([a.toArray(), b.toArray(), c.toArray()]);
console.log("c[:4] =", Array.from(ch.subarray(0, 4)), "max |c - (a+b)| =", Math.max(...ch.map((v, i) => Math.abs(v - Math.fround(ah[i] + bh[i])))));

// a library kernel with templates and shared memory, and the launch-statement syntax
const mm = ctx.compile(kernels.sources.matmul);
const n = 128, A = ctx.rand([n * n], 1), B = ctx.rand([n * n], 2), C = ctx.zeros([n * n]);
await mm.exec("matmul_tiled<16><<<dim3(n / 16, n / 16), dim3(16, 16)>>>(A, B, C, n, n, n)", { A, B, C, n });
// the same product through the cuBLAS-style API (column-major: C^T = B^T A^T)
const C2 = ctx.zeros([n * n]);
await cublas.cublasSgemm(cublas.cublasCreate(ctx), cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_N, n, n, n, 1, B.ptr, n, A.ptr, n, 0, C2.ptr, n);
const [c1, c2] = [await C.toArray(), await C2.toArray()];
console.log("matmul_tiled vs cublasSgemm max diff:", Math.max(...c1.map((v, i) => Math.abs(v - c2[i]))));

// device printf and error reporting
const dbg = ctx.compile(`__global__ void k(float* x, int n) {
  if (threadIdx.x == 0) printf("hello from block %d\\n", blockIdx.x);
  x[threadIdx.x + n] = 1.0f;   // out of bounds for n > 0
}`);
ctx.onPrintf = (s) => process.stdout.write(s);
await dbg.launch({ grid: 2, block: 32, args: [ctx.zeros([32]), 0] });
await ctx.synchronize();
await dbg.launch({ grid: 1, block: 32, args: [ctx.zeros([32]), 8] });
try { await ctx.synchronize(); } catch (e) { console.log("caught:", e.message); }
ctx.destroy();
