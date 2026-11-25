/** Quantized, sparse and mixture-of-experts kernels. */

export const quantized = `
// INT8 weight-only quantized matmul: y[M,N] = x[M,K] @ (w_q[N,K] * scale[N])^T, int32 accumulation of int8 activations
// x_q[M,K] int8 with per-row activation scales sx[M] (W8A8): y = (sum_k x_q*w_q) * sx[m] * sw[n]
__global__ void int8_matmul(const signed char* __restrict__ xq, const float* __restrict__ sx, const signed char* __restrict__ wq,
                            const float* __restrict__ sw, float* __restrict__ y, int M, int N, int K) {
  int n = blockIdx.x * blockDim.x + threadIdx.x, m = blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || n >= N) return;
  int acc = 0;
  for (int k = 0; k < K; ++k) acc += (int)xq[m * K + k] * (int)wq[n * K + k];
  y[m * N + n] = (float)acc * sx[m] * sw[n];
}
// same with packed 4-byte dot products (__dp4a); K must be a multiple of 4
__global__ void int8_matmul_dp4a(const int* __restrict__ xq4, const float* __restrict__ sx, const int* __restrict__ wq4,
                                 const float* __restrict__ sw, float* __restrict__ y, int M, int N, int K4) {
  int n = blockIdx.x * blockDim.x + threadIdx.x, m = blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || n >= N) return;
  int acc = 0;
  for (int k = 0; k < K4; ++k) acc = __dp4a(xq4[m * K4 + k], wq4[n * K4 + k], acc);
  y[m * N + n] = (float)acc * sx[m] * sw[n];
}
// per-row symmetric INT8 quantization: q = round(x / s), s = max|x| / 127
template <int BLOCK>
__global__ void quantize_rows_int8(const float* __restrict__ x, signed char* __restrict__ q, float* __restrict__ scales, int cols) {
  __shared__ float s[BLOCK];
  size_t base = (size_t)blockIdx.x * cols;
  float m = 0.0f;
  for (int i = threadIdx.x; i < cols; i += BLOCK) m = fmaxf(m, fabsf(x[base + i]));
  s[threadIdx.x] = m;
  __syncthreads();
  for (int st = BLOCK / 2; st > 0; st >>= 1) { if (threadIdx.x < st) s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + st]); __syncthreads(); }
  float sc = s[0] > 0.0f ? s[0] / 127.0f : 1.0f;
  if (threadIdx.x == 0) scales[blockIdx.x] = sc;
  for (int i = threadIdx.x; i < cols; i += BLOCK) q[base + i] = (signed char)max(-127, min(127, __float2int_rn(x[base + i] / sc)));
}
// INT4 weight-only matmul: w packed two per byte (low nibble = even k), signed 4-bit with zero point 8, per-group scales
// (group size G along K). y[M,N] = x[M,K] @ dequant(w)[N,K]^T
__global__ void int4_matmul(const float* __restrict__ x, const unsigned char* __restrict__ wp, const float* __restrict__ scales,
                            float* __restrict__ y, int M, int N, int K, int G) {
  int n = blockIdx.x * blockDim.x + threadIdx.x, m = blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || n >= N) return;
  float acc = 0.0f;
  int groups = K / G;
  for (int g = 0; g < groups; ++g) {
    float part = 0.0f;
    for (int k = g * G; k < (g + 1) * G; k += 2) {
      unsigned char byte = wp[(n * K + k) / 2];
      float w0 = (float)((int)(byte & 15) - 8), w1 = (float)((int)(byte >> 4) - 8);
      part += x[m * K + k] * w0 + x[m * K + k + 1] * w1;
    }
    acc += part * scales[n * groups + g];
  }
  y[m * N + n] = acc;
}
// fp16 storage, fp32 compute matmul (mixed precision inference)
__global__ void matmul_f16(const __half* __restrict__ A, const __half* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
  int col = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= M || col >= N) return;
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) acc += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
  C[row * N + col] = acc;
}
`;

export const sparse = `
// CSR sparse matrix x dense vector: y = A x, one thread per row
__global__ void csr_spmv(const int* __restrict__ rowptr, const int* __restrict__ cols, const float* __restrict__ vals,
                         const float* __restrict__ x, float* __restrict__ y, int rows) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  float s = 0.0f;
  for (int j = rowptr[r]; j < rowptr[r + 1]; ++j) s += vals[j] * x[cols[j]];
  y[r] = s;
}
// CSR x dense matrix: Y[rows, N] = A @ X[cols, N]
__global__ void csr_spmm(const int* __restrict__ rowptr, const int* __restrict__ cols, const float* __restrict__ vals,
                         const float* __restrict__ X, float* __restrict__ Y, int rows, int N) {
  int r = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows || c >= N) return;
  float s = 0.0f;
  for (int j = rowptr[r]; j < rowptr[r + 1]; ++j) s += vals[j] * X[cols[j] * N + c];
  Y[r * N + c] = s;
}
`;

export const moe = `
// top-k routing: per token softmax over E expert logits, keep the k best (renormalized); k <= 8, E <= 64
__global__ void moe_topk(const float* __restrict__ logits, int* __restrict__ experts, float* __restrict__ weights, int tokens, int E, int k) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= tokens) return;
  const float* l = logits + (size_t)t * E;
  float m = -INFINITY;
  for (int e = 0; e < E; ++e) m = fmaxf(m, l[e]);
  float z = 0.0f;
  for (int e = 0; e < E; ++e) z += expf(l[e] - m);
  int chosen[8];
  float wsum = 0.0f;
  for (int i = 0; i < k; ++i) {
    int best = -1; float bv = -INFINITY;
    for (int e = 0; e < E; ++e) {
      bool taken = false;
      for (int p = 0; p < i; ++p) if (chosen[p] == e) taken = true;
      if (!taken && l[e] > bv) { bv = l[e]; best = e; }
    }
    chosen[i] = best;
    float w = expf(bv - m) / z;
    experts[t * k + i] = best;
    weights[t * k + i] = w;
    wsum += w;
  }
  for (int i = 0; i < k; ++i) weights[t * k + i] /= wsum;
}
// dispatch: copy each (token, slot) row into its expert's contiguous buffer; counts[e] gives the fill position
__global__ void moe_dispatch(const float* __restrict__ x, const int* __restrict__ experts, float* __restrict__ buf, int* __restrict__ counts,
                             int* __restrict__ slot_of, int tokens, int k, int dim, int capacity) {
  int a = blockIdx.x;            // one block per (token, choice)
  if (a >= tokens * k) return;
  __shared__ int pos;
  int e = experts[a];
  if (threadIdx.x == 0) {
    pos = atomicAdd(&counts[e], 1);
    slot_of[a] = pos < capacity ? e * capacity + pos : -1;   // dropped when the expert is full
  }
  __syncthreads();
  if (pos >= capacity) return;
  const float* src = x + (size_t)(a / k) * dim;
  float* dst = buf + ((size_t)e * capacity + pos) * dim;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) dst[d] = src[d];
}
// combine: y[token] = sum over its k choices of weight * expert_out[slot]
__global__ void moe_combine(const float* __restrict__ expert_out, const int* __restrict__ slot_of, const float* __restrict__ weights,
                            float* __restrict__ y, int tokens, int k, int dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= tokens * dim) return;
  int t = i / dim, d = i % dim;
  float acc = 0.0f;
  for (int j = 0; j < k; ++j) {
    int s = slot_of[t * k + j];
    if (s >= 0) acc += weights[t * k + j] * expert_out[(size_t)s * dim + d];
  }
  y[i] = acc;
}
`;
