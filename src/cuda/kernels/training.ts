/** Optimizer and training kernels: SGD, Adam, AdamW, gradient clipping, accumulation, mixed precision. */

export const optimizers = `
// SGD with momentum, Nesterov and weight decay (PyTorch semantics)
__global__ void sgd_step(float* __restrict__ p, const float* __restrict__ g, float* __restrict__ buf, int n,
                         float lr, float momentum, float weight_decay, int nesterov, int first) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float d = g[i] + weight_decay * p[i];
  if (momentum != 0.0f) {
    float b = first ? d : momentum * buf[i] + d;
    buf[i] = b;
    d = nesterov ? d + momentum * b : b;
  }
  p[i] -= lr * d;
}
// Adam / AdamW: decoupled = 1 applies weight decay to the parameter (AdamW), 0 adds it to the gradient (Adam)
__global__ void adam_step(float* __restrict__ p, const float* __restrict__ g, float* __restrict__ m, float* __restrict__ v, int n,
                          float lr, float beta1, float beta2, float eps, float weight_decay, int step, int decoupled) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float grad = g[i];
  if (decoupled) p[i] *= 1.0f - lr * weight_decay;
  else grad += weight_decay * p[i];
  float mi = beta1 * m[i] + (1.0f - beta1) * grad;
  float vi = beta2 * v[i] + (1.0f - beta2) * grad * grad;
  m[i] = mi; v[i] = vi;
  float bc1 = 1.0f - powf(beta1, (float)step), bc2 = 1.0f - powf(beta2, (float)step);
  p[i] -= lr * (mi / bc1) / (sqrtf(vi / bc2) + eps);
}
// sum of squares (for the global gradient norm), accumulated atomically
template <int BLOCK>
__global__ void sumsq(const float* __restrict__ x, float* __restrict__ out, int n) {
  __shared__ float s[BLOCK];
  float a = 0.0f;
  for (int i = blockIdx.x * BLOCK + threadIdx.x; i < n; i += BLOCK * gridDim.x) a += x[i] * x[i];
  s[threadIdx.x] = a;
  __syncthreads();
  for (int st = BLOCK / 2; st > 0; st >>= 1) { if (threadIdx.x < st) s[threadIdx.x] += s[threadIdx.x + st]; __syncthreads(); }
  if (threadIdx.x == 0) atomicAdd(out, s[0]);
}
// clip by global norm: g *= min(1, max_norm / (sqrt(sumsq) + 1e-6))
__global__ void clip_by_norm(float* __restrict__ g, const float* __restrict__ sumsq_total, int n, float max_norm) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float c = fminf(1.0f, max_norm / (sqrtf(sumsq_total[0]) + 1e-6f));
  g[i] *= c;
}
// gradient accumulation: acc += g * scale
__global__ void accumulate(float* __restrict__ acc, const float* __restrict__ g, int n, float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) acc[i] += g[i] * scale;
}
`;

export const mixedPrecision = `
__global__ void cast_f32_to_f16(const float* __restrict__ x, __half* __restrict__ y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = __float2half(x[i]);
}
__global__ void cast_f16_to_f32(const __half* __restrict__ x, float* __restrict__ y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = __half2float(x[i]);
}
__global__ void cast_f32_to_bf16(const float* __restrict__ x, __nv_bfloat16* __restrict__ y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = __float2bfloat16(x[i]);
}
// unscale fp16 gradients into fp32, flag non-finite values (dynamic loss scaling)
__global__ void unscale_check(const __half* __restrict__ g16, float* __restrict__ g32, int* __restrict__ found_inf, int n, float inv_scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = __half2float(g16[i]) * inv_scale;
  if (!isfinite(v)) atomicExch(found_inf, 1);
  g32[i] = v;
}
// fp32 master-weight SGD step that also refreshes the fp16 copy; skipped when an overflow was found
__global__ void master_step(float* __restrict__ master, __half* __restrict__ p16, const float* __restrict__ g, const int* __restrict__ found_inf, int n, float lr) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || found_inf[0]) return;
  float w = master[i] - lr * g[i];
  master[i] = w;
  p16[i] = __float2half(w);
}
`;
