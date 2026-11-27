/**
 * cuBLAS compatibility subset (single precision), implemented with CUDA kernels compiled by tinygrad.js. Semantics follow
 * cuBLAS: column-major storage, leading dimensions, op(A) transposition, alpha/beta scaling. This is not NVIDIA's library.
 * Results are written asynchronously like cuBLAS; read them back with ctx.memcpy / tensor.toArray().
 */
import type { CudaContext, CudaModule } from "../runtime/context.js";
import type { DevicePtr } from "../runtime/memory.js";
import { CudaError } from "../runtime/errors.js";

export const CUBLAS_OP_N = 0, CUBLAS_OP_T = 1;

const SRC = `
#define TS 16
// column-major: A(i,l) with lda; op(A) is m x k
__global__ void sgemm_cm(int transa, int transb, int m, int n, int k, float alpha, const float* A, int lda, const float* B, int ldb,
                         float beta, float* C, int ldc, long long strideA, long long strideB, long long strideC) {
  __shared__ float As[TS][TS + 1];
  __shared__ float Bs[TS][TS + 1];
  int bz = blockIdx.z;
  const float* a = A + bz * strideA;
  const float* b = B + bz * strideB;
  float* c = C + bz * strideC;
  int i = blockIdx.y * TS + threadIdx.y;   // row of C
  int j = blockIdx.x * TS + threadIdx.x;   // column of C
  float acc = 0.0f;
  for (int l0 = 0; l0 < k; l0 += TS) {
    int la = l0 + threadIdx.x, lb = l0 + threadIdx.y;
    As[threadIdx.y][threadIdx.x] = (i < m && la < k) ? (transa ? a[la + (long long)i * lda] : a[i + (long long)la * lda]) : 0.0f;
    Bs[threadIdx.y][threadIdx.x] = (lb < k && j < n) ? (transb ? b[j + (long long)lb * ldb] : b[lb + (long long)j * ldb]) : 0.0f;
    __syncthreads();
    #pragma unroll
    for (int l = 0; l < TS; ++l) acc += As[threadIdx.y][l] * Bs[l][threadIdx.x];
    __syncthreads();
  }
  if (i < m && j < n) {
    long long ci = i + (long long)j * ldc;
    c[ci] = beta == 0.0f ? alpha * acc : alpha * acc + beta * c[ci];
  }
}
__global__ void sgemv_cm(int trans, int m, int n, float alpha, const float* A, int lda, const float* x, int incx, float beta, float* y, int incy) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  int rows = trans ? n : m, len = trans ? m : n;
  if (r >= rows) return;
  float s = 0.0f;
  for (int l = 0; l < len; ++l) s += (trans ? A[l + (long long)r * lda] : A[r + (long long)l * lda]) * x[l * incx];
  y[r * incy] = beta == 0.0f ? alpha * s : alpha * s + beta * y[r * incy];
}
__global__ void saxpy(int n, float alpha, const float* x, int incx, float* y, int incy) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i * incy] += alpha * x[i * incx];
}
__global__ void sscal(int n, float alpha, float* x, int incx) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i * incx] *= alpha;
}
__global__ void scopy(int n, const float* x, int incx, float* y, int incy) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i * incy] = x[i * incx];
}
// mode: 0 dot, 1 sum of squares, 2 sum of |x|
__global__ void sreduce(int n, const float* x, int incx, const float* y, int incy, float* out, int mode) {
  __shared__ float s[256];
  float a = 0.0f;
  for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += 256 * gridDim.x) {
    float v = x[i * incx];
    a += mode == 0 ? v * y[i * incy] : mode == 1 ? v * v : fabsf(v);
  }
  s[threadIdx.x] = a;
  __syncthreads();
  for (int st = 128; st > 0; st >>= 1) { if (threadIdx.x < st) s[threadIdx.x] += s[threadIdx.x + st]; __syncthreads(); }
  if (threadIdx.x == 0) atomicAdd(out, s[0]);
}
`;

export class CublasHandle {
  readonly module: CudaModule;
  constructor(readonly ctx: CudaContext) { this.module = ctx.compile(SRC, { name: "cublas" }); }
  destroy(): void { /* kernels are cached by the context */ }
}

export function cublasCreate(ctx: CudaContext): CublasHandle { return new CublasHandle(ctx); }

function check(cond: boolean, msg: string): void { if (!cond) throw new CudaError("cudaErrorInvalidValue", `cuBLAS: ${msg}`); }

/** C = alpha * op(A) @ op(B) + beta * C (column-major) */
export async function cublasSgemm(h: CublasHandle, transa: number, transb: number, m: number, n: number, k: number, alpha: number,
  A: DevicePtr, lda: number, B: DevicePtr, ldb: number, beta: number, C: DevicePtr, ldc: number): Promise<void> {
  return cublasSgemmStridedBatched(h, transa, transb, m, n, k, alpha, A, lda, 0, B, ldb, 0, beta, C, ldc, 0, 1);
}

export async function cublasSgemmStridedBatched(h: CublasHandle, transa: number, transb: number, m: number, n: number, k: number, alpha: number,
  A: DevicePtr, lda: number, strideA: number, B: DevicePtr, ldb: number, strideB: number, beta: number, C: DevicePtr, ldc: number, strideC: number, batch: number): Promise<void> {
  check(lda >= Math.max(1, transa ? k : m) && ldb >= Math.max(1, transb ? n : k) && ldc >= Math.max(1, m), "invalid leading dimension");
  if (m === 0 || n === 0 || batch === 0) return;
  await h.module.launch({
    kernel: "sgemm_cm", grid: [Math.ceil(n / 16), Math.ceil(m / 16), batch], block: [16, 16],
    args: [transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, BigInt(strideA), BigInt(strideB), BigInt(strideC)],
  });
}

export async function cublasSgemv(h: CublasHandle, trans: number, m: number, n: number, alpha: number, A: DevicePtr, lda: number,
  x: DevicePtr, incx: number, beta: number, y: DevicePtr, incy: number): Promise<void> {
  const rows = trans ? n : m;
  await h.module.launch({ kernel: "sgemv_cm", grid: [Math.ceil(rows / 128)], block: [128], args: [trans, m, n, alpha, A, lda, x, incx, beta, y, incy] });
}
export async function cublasSaxpy(h: CublasHandle, n: number, alpha: number, x: DevicePtr, incx: number, y: DevicePtr, incy: number): Promise<void> {
  await h.module.launch({ kernel: "saxpy", grid: [Math.ceil(n / 256)], block: [256], args: [n, alpha, x, incx, y, incy] });
}
export async function cublasSscal(h: CublasHandle, n: number, alpha: number, x: DevicePtr, incx: number): Promise<void> {
  await h.module.launch({ kernel: "sscal", grid: [Math.ceil(n / 256)], block: [256], args: [n, alpha, x, incx] });
}
export async function cublasScopy(h: CublasHandle, n: number, x: DevicePtr, incx: number, y: DevicePtr, incy: number): Promise<void> {
  await h.module.launch({ kernel: "scopy", grid: [Math.ceil(n / 256)], block: [256], args: [n, x, incx, y, incy] });
}
async function reduce(h: CublasHandle, n: number, x: DevicePtr, incx: number, y: DevicePtr | null, incy: number, mode: number): Promise<number> {
  const out = h.ctx.malloc(4);
  h.ctx.memset(out, 0, 4);
  await h.module.launch({ kernel: "sreduce", grid: [Math.min(64, Math.max(1, Math.ceil(n / 256)))], block: [256], args: [n, x, incx, y ?? x, incy, out, mode] });
  const r = new Float32Array(1);
  await h.ctx.memcpy(r, out, 4);
  h.ctx.free(out);
  return r[0];
}
/** result returned to the host (like CUBLAS_POINTER_MODE_HOST) */
export function cublasSdot(h: CublasHandle, n: number, x: DevicePtr, incx: number, y: DevicePtr, incy: number): Promise<number> { return reduce(h, n, x, incx, y, incy, 0); }
export async function cublasSnrm2(h: CublasHandle, n: number, x: DevicePtr, incx: number): Promise<number> { return Math.sqrt(await reduce(h, n, x, incx, null, 1, 1)); }
export function cublasSasum(h: CublasHandle, n: number, x: DevicePtr, incx: number): Promise<number> { return reduce(h, n, x, incx, null, 1, 2); }
