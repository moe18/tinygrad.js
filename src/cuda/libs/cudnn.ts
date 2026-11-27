/**
 * cuDNN compatibility subset: activation, softmax, 2D convolution and pooling forward passes on NCHW float tensors,
 * implemented with tinygrad.js-compiled CUDA kernels. Descriptors are plain objects; only the listed modes exist.
 */
import type { CudaContext, CudaModule } from "../runtime/context.js";
import type { DevicePtr } from "../runtime/memory.js";
import { CudaError } from "../runtime/errors.js";

export const CUDNN_ACTIVATION_SIGMOID = 0, CUDNN_ACTIVATION_RELU = 1, CUDNN_ACTIVATION_TANH = 2, CUDNN_ACTIVATION_CLIPPED_RELU = 3, CUDNN_ACTIVATION_ELU = 4;
export const CUDNN_SOFTMAX_ACCURATE = 1, CUDNN_SOFTMAX_LOG = 2;
export const CUDNN_SOFTMAX_MODE_INSTANCE = 0, CUDNN_SOFTMAX_MODE_CHANNEL = 1;
export const CUDNN_POOLING_MAX = 0, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1;

export interface TensorDesc { n: number; c: number; h: number; w: number }
export interface FilterDesc { k: number; c: number; h: number; w: number }
export interface ConvDesc { padH: number; padW: number; strideH: number; strideW: number; dilationH?: number; dilationW?: number }
export interface PoolDesc { mode: number; windowH: number; windowW: number; padH: number; padW: number; strideH: number; strideW: number }

const SRC = `
__global__ void activation(int mode, int n, float alpha, float coef, const float* x, float beta, float* y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = x[i], r;
  if (mode == 0) r = 1.0f / (1.0f + expf(-v));
  else if (mode == 1) r = fmaxf(v, 0.0f);
  else if (mode == 2) r = tanhf(v);
  else if (mode == 3) r = fminf(fmaxf(v, 0.0f), coef);
  else r = v > 0.0f ? v : coef * (expf(v) - 1.0f);
  y[i] = beta == 0.0f ? alpha * r : alpha * r + beta * y[i];
}
// softmax over C for every (n, h, w) (channel mode) or over C*H*W per n (instance mode)
__global__ void softmax_nchw(int algo, int N, int C, int HW, int instance, float alpha, const float* x, float beta, float* y) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int groups = instance ? N : N * HW;
  if (idx >= groups) return;
  int len = instance ? C * HW : C;
  int n = instance ? idx : idx / HW, p = instance ? 0 : idx % HW;
  int stride = instance ? 1 : HW;
  long long base = (long long)n * C * HW + p;
  float m = -INFINITY;
  for (int c = 0; c < len; ++c) m = fmaxf(m, x[base + (long long)c * stride]);
  float s = 0.0f;
  for (int c = 0; c < len; ++c) s += expf(x[base + (long long)c * stride] - m);
  for (int c = 0; c < len; ++c) {
    long long i = base + (long long)c * stride;
    float r = algo == 2 ? x[i] - m - logf(s) : expf(x[i] - m) / s;
    y[i] = beta == 0.0f ? alpha * r : alpha * r + beta * y[i];
  }
}
__global__ void conv_fwd(const float* x, const float* w, float* y, int N, int C, int H, int W, int K, int R, int S,
                         int padH, int padW, int sh, int sw, int dh, int dw, int OH, int OW, float alpha, float beta) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N * K * OH * OW) return;
  int ow = idx % OW, oh = (idx / OW) % OH, k = (idx / (OW * OH)) % K, n = idx / (OW * OH * K);
  float acc = 0.0f;
  for (int c = 0; c < C; ++c)
    for (int r = 0; r < R; ++r) {
      int ih = oh * sh - padH + r * dh;
      if (ih < 0 || ih >= H) continue;
      for (int s = 0; s < S; ++s) {
        int iw = ow * sw - padW + s * dw;
        if (iw >= 0 && iw < W) acc += x[((n * C + c) * H + ih) * W + iw] * w[((k * C + c) * R + r) * S + s];
      }
    }
  y[idx] = beta == 0.0f ? alpha * acc : alpha * acc + beta * y[idx];
}
__global__ void pool_fwd(int mode, const float* x, float* y, int NC, int H, int W, int wh, int ww, int ph, int pw, int sh, int sw, int OH, int OW, float alpha, float beta) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= NC * OH * OW) return;
  int ow = idx % OW, oh = (idx / OW) % OH, nc = idx / (OW * OH);
  float m = -INFINITY, s = 0.0f;
  for (int i = 0; i < wh; ++i) for (int j = 0; j < ww; ++j) {
    int ih = oh * sh - ph + i, iw = ow * sw - pw + j;
    bool in = ih >= 0 && ih < H && iw >= 0 && iw < W;
    float v = in ? x[(nc * H + ih) * W + iw] : 0.0f;
    if (in) m = fmaxf(m, v);
    s += v;
  }
  float r = mode == 0 ? m : s / (wh * ww);
  y[idx] = beta == 0.0f ? alpha * r : alpha * r + beta * y[idx];
}
`;

export class CudnnHandle {
  readonly module: CudaModule;
  constructor(readonly ctx: CudaContext) { this.module = ctx.compile(SRC, { name: "cudnn" }); }
}
export function cudnnCreate(ctx: CudaContext): CudnnHandle { return new CudnnHandle(ctx); }
const size = (d: TensorDesc) => d.n * d.c * d.h * d.w;

export async function cudnnActivationForward(h: CudnnHandle, mode: number, coef: number, alpha: number, xDesc: TensorDesc, x: DevicePtr, beta: number, y: DevicePtr): Promise<void> {
  const n = size(xDesc);
  await h.module.launch({ kernel: "activation", grid: [Math.ceil(n / 256)], block: [256], args: [mode, n, alpha, coef, x, beta, y] });
}
export async function cudnnSoftmaxForward(h: CudnnHandle, algo: number, mode: number, alpha: number, d: TensorDesc, x: DevicePtr, beta: number, y: DevicePtr): Promise<void> {
  const inst = mode === CUDNN_SOFTMAX_MODE_INSTANCE ? 1 : 0;
  const groups = inst ? d.n : d.n * d.h * d.w;
  await h.module.launch({ kernel: "softmax_nchw", grid: [Math.ceil(groups / 128)], block: [128], args: [algo, d.n, d.c, d.h * d.w, inst, alpha, x, beta, y] });
}
export function cudnnGetConvolution2dForwardOutputDim(conv: ConvDesc, x: TensorDesc, w: FilterDesc): TensorDesc {
  const dh = conv.dilationH ?? 1, dw = conv.dilationW ?? 1;
  const oh = Math.floor((x.h + 2 * conv.padH - dh * (w.h - 1) - 1) / conv.strideH) + 1;
  const ow = Math.floor((x.w + 2 * conv.padW - dw * (w.w - 1) - 1) / conv.strideW) + 1;
  return { n: x.n, c: w.k, h: oh, w: ow };
}
export async function cudnnConvolutionForward(h: CudnnHandle, alpha: number, xDesc: TensorDesc, x: DevicePtr, wDesc: FilterDesc, w: DevicePtr,
  conv: ConvDesc, beta: number, yDesc: TensorDesc, y: DevicePtr): Promise<void> {
  if (wDesc.c !== xDesc.c) throw new CudaError("cudaErrorInvalidValue", `cuDNN: filter has ${wDesc.c} input channels, input has ${xDesc.c}`);
  const o = cudnnGetConvolution2dForwardOutputDim(conv, xDesc, wDesc);
  if (o.h !== yDesc.h || o.w !== yDesc.w || o.c !== yDesc.c) throw new CudaError("cudaErrorInvalidValue", "cuDNN: output descriptor does not match the convolution");
  const n = size(o);
  await h.module.launch({
    kernel: "conv_fwd", grid: [Math.ceil(n / 128)], block: [128],
    args: [x, w, y, xDesc.n, xDesc.c, xDesc.h, xDesc.w, wDesc.k, wDesc.h, wDesc.w, conv.padH, conv.padW, conv.strideH, conv.strideW, conv.dilationH ?? 1, conv.dilationW ?? 1, o.h, o.w, alpha, beta],
  });
}
export async function cudnnPoolingForward(h: CudnnHandle, p: PoolDesc, alpha: number, xDesc: TensorDesc, x: DevicePtr, beta: number, yDesc: TensorDesc, y: DevicePtr): Promise<void> {
  const n = size(yDesc);
  await h.module.launch({
    kernel: "pool_fwd", grid: [Math.ceil(n / 128)], block: [128],
    args: [p.mode, x, y, xDesc.n * xDesc.c, xDesc.h, xDesc.w, p.windowH, p.windowW, p.padH, p.padW, p.strideH, p.strideW, yDesc.h, yDesc.w, alpha, beta],
  });
}
