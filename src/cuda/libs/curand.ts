/**
 * cuRAND host API subset backed by the Philox4x32-10 device generator in <curand_kernel.h>. Element i of a generate call
 * is drawn from subsequence i of the generator with the given seed and a running offset, so results are reproducible
 * across backends. The ordering of values differs from NVIDIA's host API (which interleaves Philox outputs differently).
 */
import type { CudaContext, CudaModule } from "../runtime/context.js";
import type { DevicePtr } from "../runtime/memory.js";
import { CudaError } from "../runtime/errors.js";

export const CURAND_RNG_PSEUDO_DEFAULT = 100, CURAND_RNG_PSEUDO_PHILOX4_32_10 = 161;

const SRC = `
#include <curand_kernel.h>
__global__ void gen_uniform(float* out, int n, unsigned long long seed, unsigned long long offset) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  curandStatePhilox4_32_10_t st;
  curand_init(seed, i, offset, &st);
  out[i] = curand_uniform(&st);
}
__global__ void gen_normal(float* out, int n, unsigned long long seed, unsigned long long offset, float mean, float stddev) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  curandStatePhilox4_32_10_t st;
  curand_init(seed, i, offset, &st);
  out[i] = mean + stddev * curand_normal(&st);
}
__global__ void gen_bits(unsigned int* out, int n, unsigned long long seed, unsigned long long offset) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  curandStatePhilox4_32_10_t st;
  curand_init(seed, i, offset, &st);
  out[i] = curand(&st);
}
`;

export class CurandGenerator {
  seed = 0n;
  offset = 0n;
  readonly module: CudaModule;
  constructor(readonly ctx: CudaContext, readonly type: number) {
    if (type !== CURAND_RNG_PSEUDO_DEFAULT && type !== CURAND_RNG_PSEUDO_PHILOX4_32_10)
      throw new CudaError("cudaErrorNotSupported", `cuRAND generator type ${type} is not supported (only Philox4_32_10)`);
    this.module = ctx.compile(SRC, { name: "curand" });
  }
}
export function curandCreateGenerator(ctx: CudaContext, type = CURAND_RNG_PSEUDO_PHILOX4_32_10): CurandGenerator { return new CurandGenerator(ctx, type); }
export function curandSetPseudoRandomGeneratorSeed(g: CurandGenerator, seed: number | bigint): void { g.seed = BigInt(seed); }
export function curandSetGeneratorOffset(g: CurandGenerator, offset: number | bigint): void { g.offset = BigInt(offset); }
export async function curandGenerateUniform(g: CurandGenerator, out: DevicePtr, n: number): Promise<void> {
  await g.module.launch({ kernel: "gen_uniform", grid: [Math.ceil(n / 256)], block: [256], args: [out, n, g.seed, g.offset] });
  g.offset += 1n;
}
export async function curandGenerateNormal(g: CurandGenerator, out: DevicePtr, n: number, mean: number, stddev: number): Promise<void> {
  await g.module.launch({ kernel: "gen_normal", grid: [Math.ceil(n / 256)], block: [256], args: [out, n, g.seed, g.offset, mean, stddev] });
  g.offset += 2n;
}
export async function curandGenerate(g: CurandGenerator, out: DevicePtr, n: number): Promise<void> {
  await g.module.launch({ kernel: "gen_bits", grid: [Math.ceil(n / 256)], block: [256], args: [out, n, g.seed, g.offset] });
  g.offset += 1n;
}
