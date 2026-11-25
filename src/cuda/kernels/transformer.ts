/** Transformer kernels: attention (naive, flash, causal, grouped-query), rotary embeddings, KV cache. Layout [B, H, T, D]. */

export const attention = `
// naive scaled dot-product attention: one thread per (b, h, q) row; scores are recomputed for the two softmax passes
__global__ void sdpa_naive(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V, float* __restrict__ O,
                           int B, int H, int HKV, int T, int S, int D, float scale, int causal) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= B * H * T) return;
  int q = idx % T, h = (idx / T) % H, b = idx / (T * H);
  int hk = h / (H / HKV);                       // grouped-query attention: query heads share kv heads
  const float* qp = Q + ((size_t)(b * H + h) * T + q) * D;
  const float* kb = K + (size_t)(b * HKV + hk) * S * D;
  const float* vb = V + (size_t)(b * HKV + hk) * S * D;
  int last = causal ? min(S - 1, q + (S - T)) : S - 1;
  float m = -INFINITY;
  for (int j = 0; j <= last; ++j) {
    float s = 0.0f;
    for (int d = 0; d < D; ++d) s += qp[d] * kb[j * D + d];
    m = fmaxf(m, s * scale);
  }
  float l = 0.0f;
  float* op = O + ((size_t)(b * H + h) * T + q) * D;
  for (int d = 0; d < D; ++d) op[d] = 0.0f;
  for (int j = 0; j <= last; ++j) {
    float s = 0.0f;
    for (int d = 0; d < D; ++d) s += qp[d] * kb[j * D + d];
    float p = expf(s * scale - m);
    l += p;
    for (int d = 0; d < D; ++d) op[d] += p * vb[j * D + d];
  }
  for (int d = 0; d < D; ++d) op[d] /= l;
}

// FlashAttention-style forward: BR query rows per block (one thread per row), K/V tiles of BC rows staged in shared
// memory, online softmax (running max m and normalizer l), O accumulated in registers. D <= DMAX.
template <int BR, int BC, int DMAX>
__global__ void flash_attention_fwd(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V, float* __restrict__ O,
                                    float* __restrict__ LSE, int B, int H, int HKV, int T, int S, int D, float scale, int causal) {
  __shared__ float Ks[BC][DMAX];
  __shared__ float Vs[BC][DMAX];
  int bh = blockIdx.y;
  int b = bh / H, h = bh % H, hk = h / (H / HKV);
  int q = blockIdx.x * BR + threadIdx.x;
  bool valid = q < T;
  const float* qp = Q + ((size_t)bh * T + (valid ? q : 0)) * D;
  float qr[DMAX], acc[DMAX];
  #pragma unroll
  for (int d = 0; d < DMAX; ++d) { qr[d] = d < D ? qp[d] * scale : 0.0f; acc[d] = 0.0f; }
  float m = -INFINITY, l = 0.0f;
  int qlast = causal ? q + (S - T) : S - 1;
  int blockLast = causal ? min(S - 1, blockIdx.x * BR + BR - 1 + (S - T)) : S - 1;
  for (int j0 = 0; j0 <= blockLast; j0 += BC) {
    // cooperative tile load
    for (int i = threadIdx.x; i < BC * DMAX; i += BR) {
      int r = i / DMAX, d = i % DMAX;
      int j = j0 + r;
      bool in = j < S && d < D;
      size_t off = ((size_t)(b * HKV + hk) * S + (in ? j : 0)) * D + (in ? d : 0);
      Ks[r][d] = in ? K[off] : 0.0f;
      Vs[r][d] = in ? V[off] : 0.0f;
    }
    __syncthreads();
    if (valid) {
      for (int r = 0; r < BC; ++r) {
        int j = j0 + r;
        if (j >= S || j > qlast) break;
        float s = 0.0f;
        #pragma unroll
        for (int d = 0; d < DMAX; ++d) s += qr[d] * Ks[r][d];
        float mn = fmaxf(m, s);
        float corr = expf(m - mn), p = expf(s - mn);
        l = l * corr + p;
        #pragma unroll
        for (int d = 0; d < DMAX; ++d) acc[d] = acc[d] * corr + p * Vs[r][d];
        m = mn;
      }
    }
    __syncthreads();
  }
  if (valid) {
    float* op = O + ((size_t)bh * T + q) * D;
    float inv = 1.0f / l;
    for (int d = 0; d < D; ++d) op[d] = acc[d] * inv;
    LSE[(size_t)bh * T + q] = m + logf(l);
  }
}
`;

export const rope = `
// rotary position embeddings on [B, H, T, D] (pairs (2i, 2i+1) rotated by pos * base^(-2i/D)), in place
__global__ void rope_apply(float* __restrict__ x, int B, int H, int T, int D, int pos0, float base) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int half = D / 2;
  if (idx >= B * H * T * half) return;
  int i = idx % half, t = (idx / half) % T;
  size_t row = (size_t)(idx / half) * D;
  float inv = powf(base, -2.0f * i / D);
  float ang = (pos0 + t) * inv;
  float c = cosf(ang), s = sinf(ang);
  float a = x[row + 2 * i], b = x[row + 2 * i + 1];
  x[row + 2 * i] = a * c - b * s;
  x[row + 2 * i + 1] = a * s + b * c;
}
// backward of the rotation: rotate the gradient by -angle
__global__ void rope_bwd(float* __restrict__ dx, int B, int H, int T, int D, int pos0, float base) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int half = D / 2;
  if (idx >= B * H * T * half) return;
  int i = idx % half, t = (idx / half) % T;
  size_t row = (size_t)(idx / half) * D;
  float ang = (pos0 + t) * powf(base, -2.0f * i / D);
  float c = cosf(ang), s = sinf(ang);
  float a = dx[row + 2 * i], b = dx[row + 2 * i + 1];
  dx[row + 2 * i] = a * c + b * s;
  dx[row + 2 * i + 1] = -a * s + b * c;
}
`;

export const kvcache = `
// append T new positions of k/v [B, HKV, T, D] into caches [B, HKV, MAXT, D] at position pos
__global__ void kv_cache_append(const float* __restrict__ k, const float* __restrict__ v, float* __restrict__ kc, float* __restrict__ vc,
                                int B, int HKV, int T, int D, int MAXT, int pos) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= B * HKV * T * D) return;
  int d = idx % D, t = (idx / D) % T, bh = idx / (D * T);
  if (pos + t >= MAXT) return;
  size_t dst = ((size_t)bh * MAXT + pos + t) * D + d;
  kc[dst] = k[idx];
  vc[dst] = v[idx];
}
// fused residual add + LayerNorm (no affine): y = LN(x + r); one block per row
template <int BLOCK>
__global__ void add_layernorm(const float* __restrict__ x, const float* __restrict__ r, float* __restrict__ sum_out, float* __restrict__ y, int cols, float eps) {
  __shared__ float s1[BLOCK], s2[BLOCK];
  size_t base = (size_t)blockIdx.x * cols;
  float a = 0.0f, b = 0.0f;
  for (int i = threadIdx.x; i < cols; i += BLOCK) { float v = x[base + i] + r[base + i]; sum_out[base + i] = v; a += v; b += v * v; }
  s1[threadIdx.x] = a; s2[threadIdx.x] = b;
  __syncthreads();
  for (int st = BLOCK / 2; st > 0; st >>= 1) { if (threadIdx.x < st) { s1[threadIdx.x] += s1[threadIdx.x + st]; s2[threadIdx.x] += s2[threadIdx.x + st]; } __syncthreads(); }
  float mu = s1[0] / cols, rs = rsqrtf(fmaxf(s2[0] / cols - mu * mu, 0.0f) + eps);
  for (int i = threadIdx.x; i < cols; i += BLOCK) y[base + i] = (sum_out[base + i] - mu) * rs;
}
`;
