/** Fundamental parallel kernels: vector ops, matrix multiplication (naive, tiled, register-blocked), transpose, reductions, scans, histograms. */

export const vectorAdd = `
__global__ void vector_add(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
// float4-vectorized: each thread adds four consecutive elements (n4 = n / 4)
__global__ void vector_add4(const float4* __restrict__ a, const float4* __restrict__ b, float4* __restrict__ c, int n4) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) c[i] = a[i] + b[i];
}
__global__ void saxpy(int n, float alpha, const float* __restrict__ x, float* __restrict__ y) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) y[i] = alpha * x[i] + y[i];
}
// elementwise ops selected by an opcode: 0 add, 1 sub, 2 mul, 3 div, 4 max, 5 min
__global__ void elementwise(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, int n, int op) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = a[i], y = b[i], r;
  switch (op) {
    case 0: r = x + y; break;
    case 1: r = x - y; break;
    case 2: r = x * y; break;
    case 3: r = x / y; break;
    case 4: r = fmaxf(x, y); break;
    default: r = fminf(x, y); break;
  }
  out[i] = r;
}
__global__ void fill(float* __restrict__ out, float value, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = value;
}
__global__ void scale(float* __restrict__ x, float s, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] *= s;
}
`;

export const matmul = `
// C[M,N] = A[M,K] @ B[K,N], row-major
__global__ void matmul_naive(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
  }
}

// shared-memory tiling: block TILE x TILE, each thread one output
template <int TILE>
__global__ void matmul_tiled(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  int ty = threadIdx.y, tx = threadIdx.x;
  int row = blockIdx.y * TILE + ty, col = blockIdx.x * TILE + tx;
  float acc = 0.0f;
  for (int t = 0; t < K; t += TILE) {
    As[ty][tx] = (row < M && t + tx < K) ? A[row * K + t + tx] : 0.0f;
    Bs[ty][tx] = (t + ty < K && col < N) ? B[(t + ty) * N + col] : 0.0f;
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = acc;
}

// register blocking: a BMxBN tile per block of (BN/TN)x(BM/TM) threads, each thread a TMxTN micro-tile kept in registers
template <int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_regtile(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
  __shared__ float As[BK][BM];
  __shared__ float Bs[BK][BN];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int nthreads = (BM / TM) * (BN / TN);
  const int tid = ty * (BN / TN) + tx;
  const int rowBase = blockIdx.y * BM, colBase = blockIdx.x * BN;
  float acc[TM][TN];
  #pragma unroll
  for (int i = 0; i < TM; ++i)
    #pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
  for (int k0 = 0; k0 < K; k0 += BK) {
    // cooperative loads (A tile stored transposed for conflict-free reads)
    for (int i = tid; i < BM * BK; i += nthreads) {
      int r = i / BK, c = i % BK;
      int gr = rowBase + r, gc = k0 + c;
      As[c][r] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
    }
    for (int i = tid; i < BK * BN; i += nthreads) {
      int r = i / BN, c = i % BN;
      int gr = k0 + r, gc = colBase + c;
      Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < BK; ++k) {
      float a[TM], b[TN];
      #pragma unroll
      for (int i = 0; i < TM; ++i) a[i] = As[k][ty * TM + i];
      #pragma unroll
      for (int j = 0; j < TN; ++j) b[j] = Bs[k][tx * TN + j];
      #pragma unroll
      for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];
    }
    __syncthreads();
  }
  #pragma unroll
  for (int i = 0; i < TM; ++i)
    #pragma unroll
    for (int j = 0; j < TN; ++j) {
      int r = rowBase + ty * TM + i, c = colBase + tx * TN + j;
      if (r < M && c < N) C[r * N + c] = acc[i][j];
    }
}

// batched: C[b] = A[b] @ B[b]
__global__ void matmul_batched(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
  int b = blockIdx.z;
  int row = blockIdx.y * blockDim.y + threadIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;
  const float* a = A + (size_t)b * M * K;
  const float* bb = B + (size_t)b * K * N;
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) acc += a[row * K + k] * bb[k * N + col];
  C[(size_t)b * M * N + row * N + col] = acc;
}
`;

export const transpose = `
__global__ void transpose_naive(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
  int c = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y * blockDim.y + threadIdx.y;
  if (r < rows && c < cols) out[c * rows + r] = in[r * cols + c];
}
// tiled with +1 padding against shared-memory bank conflicts; coalesced reads and writes
#define TT 16
__global__ void transpose_tiled(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
  __shared__ float tile[TT][TT + 1];
  int x = blockIdx.x * TT + threadIdx.x, y = blockIdx.y * TT + threadIdx.y;
  if (x < cols && y < rows) tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
  __syncthreads();
  x = blockIdx.y * TT + threadIdx.x;
  y = blockIdx.x * TT + threadIdx.y;
  if (x < rows && y < cols) out[y * rows + x] = tile[threadIdx.x][threadIdx.y];
}
`;

export const reduction = `
// warp-level sum with shuffles
__device__ __forceinline__ float warp_reduce_sum(float v) {
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
  return v;
}
// grid-stride partial sums per block (shared memory tree), atomically accumulated into *out
template <int BLOCK>
__global__ void reduce_sum(const float* __restrict__ in, float* __restrict__ out, int n) {
  __shared__ float s[BLOCK];
  float v = 0.0f;
  for (int i = blockIdx.x * BLOCK + threadIdx.x; i < n; i += BLOCK * gridDim.x) v += in[i];
  s[threadIdx.x] = v;
  __syncthreads();
  #pragma unroll
  for (int st = BLOCK / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st) s[threadIdx.x] += s[threadIdx.x + st];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(out, s[0]);
}
// same with a warp-shuffle final stage
template <int BLOCK>
__global__ void reduce_sum_warp(const float* __restrict__ in, float* __restrict__ out, int n) {
  __shared__ float warps[BLOCK / 32];
  float v = 0.0f;
  for (int i = blockIdx.x * BLOCK + threadIdx.x; i < n; i += BLOCK * gridDim.x) v += in[i];
  v = warp_reduce_sum(v);
  int lane = threadIdx.x % 32, w = threadIdx.x / 32;
  if (lane == 0) warps[w] = v;
  __syncthreads();
  v = threadIdx.x < BLOCK / 32 ? warps[threadIdx.x] : 0.0f;
  v = warp_reduce_sum(v);
  if (threadIdx.x == 0) atomicAdd(out, v);
}
// per-row max and sum (rows x cols), one block per row
template <int BLOCK>
__global__ void row_max(const float* __restrict__ x, float* __restrict__ out, int cols) {
  __shared__ float s[BLOCK];
  const float* r = x + (size_t)blockIdx.x * cols;
  float m = -INFINITY;
  for (int i = threadIdx.x; i < cols; i += BLOCK) m = fmaxf(m, r[i]);
  s[threadIdx.x] = m;
  __syncthreads();
  for (int st = BLOCK / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st) s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + st]);
    __syncthreads();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = s[0];
}
`;

export const scan = `
// inclusive prefix sum within blocks (Hillis-Steele, double buffered), plus block totals
template <int BLOCK>
__global__ void scan_block(const float* __restrict__ in, float* __restrict__ out, float* __restrict__ block_sums, int n) {
  __shared__ float buf[2][BLOCK];
  int i = blockIdx.x * BLOCK + threadIdx.x;
  int p = 0;
  buf[0][threadIdx.x] = i < n ? in[i] : 0.0f;
  __syncthreads();
  for (int off = 1; off < BLOCK; off <<= 1) {
    float v = buf[p][threadIdx.x];
    if (threadIdx.x >= off) v += buf[p][threadIdx.x - off];
    buf[1 - p][threadIdx.x] = v;
    p = 1 - p;
    __syncthreads();
  }
  if (i < n) out[i] = buf[p][threadIdx.x];
  if (threadIdx.x == BLOCK - 1) block_sums[blockIdx.x] = buf[p][threadIdx.x];
}
// add the exclusive scan of block totals (computed by a second scan_block pass) to each block
template <int BLOCK>
__global__ void scan_add(float* __restrict__ out, const float* __restrict__ scanned_sums, int n) {
  int i = blockIdx.x * BLOCK + threadIdx.x;
  if (blockIdx.x > 0 && i < n) out[i] += scanned_sums[blockIdx.x - 1];
}
// work-efficient exclusive scan of one block of 2*BLOCK elements (Blelloch)
template <int BLOCK>
__global__ void scan_blelloch(const float* __restrict__ in, float* __restrict__ out, int n) {
  __shared__ float t[2 * BLOCK];
  int tid = threadIdx.x;
  int base = blockIdx.x * 2 * BLOCK;
  t[2 * tid] = base + 2 * tid < n ? in[base + 2 * tid] : 0.0f;
  t[2 * tid + 1] = base + 2 * tid + 1 < n ? in[base + 2 * tid + 1] : 0.0f;
  int off = 1;
  for (int d = BLOCK; d > 0; d >>= 1) {
    __syncthreads();
    if (tid < d) { int ai = off * (2 * tid + 1) - 1, bi = off * (2 * tid + 2) - 1; t[bi] += t[ai]; }
    off <<= 1;
  }
  if (tid == 0) t[2 * BLOCK - 1] = 0.0f;
  for (int d = 1; d <= BLOCK; d <<= 1) {
    off >>= 1;
    __syncthreads();
    if (tid < d) { int ai = off * (2 * tid + 1) - 1, bi = off * (2 * tid + 2) - 1; float v = t[ai]; t[ai] = t[bi]; t[bi] += v; }
  }
  __syncthreads();
  if (base + 2 * tid < n) out[base + 2 * tid] = t[2 * tid];
  if (base + 2 * tid + 1 < n) out[base + 2 * tid + 1] = t[2 * tid + 1];
}
`;

export const histogram = `
// histogram of byte values with per-block shared-memory bins, merged with global atomics
__global__ void histogram256(const unsigned char* __restrict__ data, unsigned int* __restrict__ hist, int n) {
  __shared__ unsigned int local[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) local[i] = 0;
  __syncthreads();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) atomicAdd(&local[data[i]], 1u);
  __syncthreads();
  for (int i = threadIdx.x; i < 256; i += blockDim.x) if (local[i]) atomicAdd(&hist[i], local[i]);
}
// float histogram into nbins equal bins over [lo, hi)
__global__ void histogram_float(const float* __restrict__ x, int* __restrict__ hist, int n, float lo, float hi, int nbins) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = x[i];
  if (v < lo || v >= hi) return;
  int b = (int)((v - lo) / (hi - lo) * nbins);
  atomicAdd(&hist[min(b, nbins - 1)], 1);
}
`;
