/** Neural network kernels with forward and backward passes. */

export const activations = `
__device__ __forceinline__ float gelu_erf(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)); }
__device__ __forceinline__ float gelu_tanh(float x) { return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x))); }
__device__ __forceinline__ float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

__global__ void relu_fwd(const float* __restrict__ x, float* __restrict__ y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = fmaxf(x[i], 0.0f);
}
__global__ void relu_bwd(const float* __restrict__ x, const float* __restrict__ dy, float* __restrict__ dx, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dx[i] = x[i] > 0.0f ? dy[i] : 0.0f;
}
// approximate: 0 = exact (erf), 1 = tanh approximation
__global__ void gelu_fwd(const float* __restrict__ x, float* __restrict__ y, int n, int approximate) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = approximate ? gelu_tanh(x[i]) : gelu_erf(x[i]);
}
__global__ void gelu_bwd(const float* __restrict__ x, const float* __restrict__ dy, float* __restrict__ dx, int n, int approximate) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = x[i], g;
  if (approximate) {
    const float k = 0.7978845608028654f;
    float u = k * (v + 0.044715f * v * v * v);
    float t = tanhf(u);
    g = 0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * k * (1.0f + 3.0f * 0.044715f * v * v);
  } else {
    float cdf = 0.5f * (1.0f + erff(v * 0.70710678118654752f));
    float pdf = expf(-0.5f * v * v) * 0.3989422804014327f;
    g = cdf + v * pdf;
  }
  dx[i] = dy[i] * g;
}
__global__ void silu_fwd(const float* __restrict__ x, float* __restrict__ y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { float v = x[i]; y[i] = v * sigmoidf(v); }
}
__global__ void silu_bwd(const float* __restrict__ x, const float* __restrict__ dy, float* __restrict__ dx, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { float v = x[i], s = sigmoidf(v); dx[i] = dy[i] * s * (1.0f + v * (1.0f - s)); }
}
// fused bias + activation: y = act(x + bias[col]); act: 0 none, 1 relu, 2 gelu(tanh), 3 silu
__global__ void bias_act(const float* __restrict__ x, const float* __restrict__ bias, float* __restrict__ y, int rows, int cols, int act) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * cols) return;
  float v = x[i] + bias[i % cols];
  if (act == 1) v = fmaxf(v, 0.0f);
  else if (act == 2) v = gelu_tanh(v);
  else if (act == 3) v = v * sigmoidf(v);
  y[i] = v;
}
`;

export const softmax = `
__device__ __forceinline__ float warp_max(float v) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
  return v;
}
__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}
// numerically stable row softmax: one block per row; BLOCK threads, BLOCK multiple of 32
template <int BLOCK>
__global__ void softmax_rows(const float* __restrict__ x, float* __restrict__ y, int cols) {
  __shared__ float red[BLOCK / 32];
  __shared__ float stat;
  const float* r = x + (size_t)blockIdx.x * cols;
  float* o = y + (size_t)blockIdx.x * cols;
  int lane = threadIdx.x % 32, w = threadIdx.x / 32;
  float m = -INFINITY;
  for (int i = threadIdx.x; i < cols; i += BLOCK) m = fmaxf(m, r[i]);
  m = warp_max(m);
  if (lane == 0) red[w] = m;
  __syncthreads();
  if (threadIdx.x == 0) { float t = red[0]; for (int i = 1; i < BLOCK / 32; i++) t = fmaxf(t, red[i]); stat = t; }
  __syncthreads();
  m = stat;
  float s = 0.0f;
  for (int i = threadIdx.x; i < cols; i += BLOCK) s += expf(r[i] - m);
  s = warp_sum(s);
  __syncthreads();
  if (lane == 0) red[w] = s;
  __syncthreads();
  if (threadIdx.x == 0) { float t = 0.0f; for (int i = 0; i < BLOCK / 32; i++) t += red[i]; stat = t; }
  __syncthreads();
  float inv = 1.0f / stat;
  for (int i = threadIdx.x; i < cols; i += BLOCK) o[i] = expf(r[i] - m) * inv;
}
// dx = y * (dy - sum(dy * y)) per row
template <int BLOCK>
__global__ void softmax_rows_bwd(const float* __restrict__ y, const float* __restrict__ dy, float* __restrict__ dx, int cols) {
  __shared__ float s[BLOCK];
  size_t base = (size_t)blockIdx.x * cols;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < cols; i += BLOCK) acc += dy[base + i] * y[base + i];
  s[threadIdx.x] = acc;
  __syncthreads();
  for (int st = BLOCK / 2; st > 0; st >>= 1) { if (threadIdx.x < st) s[threadIdx.x] += s[threadIdx.x + st]; __syncthreads(); }
  float dot = s[0];
  for (int i = threadIdx.x; i < cols; i += BLOCK) dx[base + i] = y[base + i] * (dy[base + i] - dot);
}
`;

export const norms = `
// LayerNorm over the last dimension: one block per row. Saves mean and rstd for the backward pass.
template <int BLOCK>
__global__ void layernorm_fwd(const float* __restrict__ x, const float* __restrict__ gamma, const float* __restrict__ beta,
                              float* __restrict__ y, float* __restrict__ mean, float* __restrict__ rstd, int cols, float eps) {
  __shared__ float s1[BLOCK], s2[BLOCK];
  size_t base = (size_t)blockIdx.x * cols;
  float a = 0.0f, b = 0.0f;
  for (int i = threadIdx.x; i < cols; i += BLOCK) { float v = x[base + i]; a += v; b += v * v; }
  s1[threadIdx.x] = a; s2[threadIdx.x] = b;
  __syncthreads();
  for (int st = BLOCK / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st) { s1[threadIdx.x] += s1[threadIdx.x + st]; s2[threadIdx.x] += s2[threadIdx.x + st]; }
    __syncthreads();
  }
  float mu = s1[0] / cols;
  float var = fmaxf(s2[0] / cols - mu * mu, 0.0f);
  float rs = rsqrtf(var + eps);
  if (threadIdx.x == 0) { mean[blockIdx.x] = mu; rstd[blockIdx.x] = rs; }
  for (int i = threadIdx.x; i < cols; i += BLOCK) y[base + i] = (x[base + i] - mu) * rs * gamma[i] + beta[i];
}
// dx per row; dgamma/dbeta accumulated with atomics
template <int BLOCK>
__global__ void layernorm_bwd(const float* __restrict__ x, const float* __restrict__ gamma, const float* __restrict__ dy,
                              const float* __restrict__ mean, const float* __restrict__ rstd,
                              float* __restrict__ dx, float* __restrict__ dgamma, float* __restrict__ dbeta, int cols) {
  __shared__ float s1[BLOCK], s2[BLOCK];
  size_t base = (size_t)blockIdx.x * cols;
  float mu = mean[blockIdx.x], rs = rstd[blockIdx.x];
  float a = 0.0f, b = 0.0f;
  for (int i = threadIdx.x; i < cols; i += BLOCK) {
    float xh = (x[base + i] - mu) * rs;
    float g = dy[base + i] * gamma[i];
    a += g; b += g * xh;
    atomicAdd(&dgamma[i], dy[base + i] * xh);
    atomicAdd(&dbeta[i], dy[base + i]);
  }
  s1[threadIdx.x] = a; s2[threadIdx.x] = b;
  __syncthreads();
  for (int st = BLOCK / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st) { s1[threadIdx.x] += s1[threadIdx.x + st]; s2[threadIdx.x] += s2[threadIdx.x + st]; }
    __syncthreads();
  }
  float mg = s1[0] / cols, mgx = s2[0] / cols;
  for (int i = threadIdx.x; i < cols; i += BLOCK) {
    float xh = (x[base + i] - mu) * rs;
    dx[base + i] = rs * (dy[base + i] * gamma[i] - mg - xh * mgx);
  }
}
// RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight
template <int BLOCK>
__global__ void rmsnorm_fwd(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y, float* __restrict__ rrms, int cols, float eps) {
  __shared__ float s[BLOCK];
  size_t base = (size_t)blockIdx.x * cols;
  float a = 0.0f;
  for (int i = threadIdx.x; i < cols; i += BLOCK) { float v = x[base + i]; a += v * v; }
  s[threadIdx.x] = a;
  __syncthreads();
  for (int st = BLOCK / 2; st > 0; st >>= 1) { if (threadIdx.x < st) s[threadIdx.x] += s[threadIdx.x + st]; __syncthreads(); }
  float r = rsqrtf(s[0] / cols + eps);
  if (threadIdx.x == 0) rrms[blockIdx.x] = r;
  for (int i = threadIdx.x; i < cols; i += BLOCK) y[base + i] = x[base + i] * r * w[i];
}
template <int BLOCK>
__global__ void rmsnorm_bwd(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ dy, const float* __restrict__ rrms,
                            float* __restrict__ dx, float* __restrict__ dw, int cols) {
  __shared__ float s[BLOCK];
  size_t base = (size_t)blockIdx.x * cols;
  float r = rrms[blockIdx.x];
  float a = 0.0f;
  for (int i = threadIdx.x; i < cols; i += BLOCK) { a += dy[base + i] * w[i] * x[base + i]; atomicAdd(&dw[i], dy[base + i] * x[base + i] * r); }
  s[threadIdx.x] = a;
  __syncthreads();
  for (int st = BLOCK / 2; st > 0; st >>= 1) { if (threadIdx.x < st) s[threadIdx.x] += s[threadIdx.x + st]; __syncthreads(); }
  float c = s[0] * r * r * r / cols;
  for (int i = threadIdx.x; i < cols; i += BLOCK) dx[base + i] = dy[base + i] * w[i] * r - x[base + i] * c;
}
`;

export const conv = `
// direct 2D convolution, NCHW input, OIHW weights, stride/padding, one thread per output element
__global__ void conv2d_fwd(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ bias, float* __restrict__ y,
                           int N, int C, int H, int W, int O, int KH, int KW, int stride, int pad, int OH, int OW) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * O * OH * OW;
  if (idx >= total) return;
  int ow = idx % OW, oh = (idx / OW) % OH, o = (idx / (OW * OH)) % O, n = idx / (OW * OH * O);
  float acc = bias ? bias[o] : 0.0f;
  for (int c = 0; c < C; ++c)
    for (int kh = 0; kh < KH; ++kh) {
      int ih = oh * stride - pad + kh;
      if (ih < 0 || ih >= H) continue;
      for (int kw = 0; kw < KW; ++kw) {
        int iw = ow * stride - pad + kw;
        if (iw < 0 || iw >= W) continue;
        acc += x[((n * C + c) * H + ih) * W + iw] * w[((o * C + c) * KH + kh) * KW + kw];
      }
    }
  y[idx] = acc;
}
__global__ void maxpool2d_fwd(const float* __restrict__ x, float* __restrict__ y, int* __restrict__ argmax, int N, int C, int H, int W, int K, int stride, int OH, int OW) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N * C * OH * OW) return;
  int ow = idx % OW, oh = (idx / OW) % OH, nc = idx / (OW * OH);
  float m = -INFINITY; int am = -1;
  for (int kh = 0; kh < K; ++kh) for (int kw = 0; kw < K; ++kw) {
    int ih = oh * stride + kh, iw = ow * stride + kw;
    if (ih < H && iw < W) { int j = (nc * H + ih) * W + iw; float v = x[j]; if (v > m) { m = v; am = j; } }
  }
  y[idx] = m;
  argmax[idx] = am;
}
__global__ void maxpool2d_bwd(const float* __restrict__ dy, const int* __restrict__ argmax, float* __restrict__ dx, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && argmax[i] >= 0) atomicAdd(&dx[argmax[i]], dy[i]);
}
__global__ void avgpool2d_fwd(const float* __restrict__ x, float* __restrict__ y, int N, int C, int H, int W, int K, int stride, int OH, int OW) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N * C * OH * OW) return;
  int ow = idx % OW, oh = (idx / OW) % OH, nc = idx / (OW * OH);
  float s = 0.0f;
  for (int kh = 0; kh < K; ++kh) for (int kw = 0; kw < K; ++kw) s += x[(nc * H + oh * stride + kh) * W + ow * stride + kw];
  y[idx] = s / (K * K);
}
`;

export const embedding = `
__global__ void embedding_fwd(const int* __restrict__ ids, const float* __restrict__ table, float* __restrict__ out, int n, int dim, int vocab) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * dim) return;
  int t = i / dim, d = i % dim;
  int id = ids[t];
  out[i] = (id >= 0 && id < vocab) ? table[id * dim + d] : 0.0f;
}
// scatter-add of output gradients into the table gradient (duplicate ids accumulate atomically)
__global__ void embedding_bwd(const int* __restrict__ ids, const float* __restrict__ dout, float* __restrict__ dtable, int n, int dim, int vocab) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * dim) return;
  int t = i / dim, d = i % dim;
  int id = ids[t];
  if (id >= 0 && id < vocab) atomicAdd(&dtable[id * dim + d], dout[i]);
}
`;

export const crossEntropy = `
// fused softmax + negative log likelihood per row; loss[row] and dlogits = (softmax - onehot) / rows
template <int BLOCK>
__global__ void cross_entropy_fwd_bwd(const float* __restrict__ logits, const int* __restrict__ targets, float* __restrict__ loss,
                                      float* __restrict__ dlogits, int cols, int rows) {
  __shared__ float s[BLOCK];
  size_t base = (size_t)blockIdx.x * cols;
  float m = -INFINITY;
  for (int i = threadIdx.x; i < cols; i += BLOCK) m = fmaxf(m, logits[base + i]);
  s[threadIdx.x] = m;
  __syncthreads();
  for (int st = BLOCK / 2; st > 0; st >>= 1) { if (threadIdx.x < st) s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + st]); __syncthreads(); }
  m = s[0];
  __syncthreads();
  float sum = 0.0f;
  for (int i = threadIdx.x; i < cols; i += BLOCK) sum += expf(logits[base + i] - m);
  s[threadIdx.x] = sum;
  __syncthreads();
  for (int st = BLOCK / 2; st > 0; st >>= 1) { if (threadIdx.x < st) s[threadIdx.x] += s[threadIdx.x + st]; __syncthreads(); }
  float lse = m + logf(s[0]);
  int t = targets[blockIdx.x];
  if (threadIdx.x == 0) loss[blockIdx.x] = lse - logits[base + t];
  for (int i = threadIdx.x; i < cols; i += BLOCK)
    dlogits[base + i] = (expf(logits[base + i] - lse) - (i == t ? 1.0f : 0.0f)) / rows;
}
`;
