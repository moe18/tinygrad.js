/**
 * Built-in headers. There is no file system: `#include <cuda_runtime.h>` and friends resolve here. Most CUDA declarations
 * (threadIdx, atomics, math functions, half types) are compiler built-ins, so these headers mostly hold macros. The cuRAND
 * device API is implemented here in CUDA C and compiled like user code.
 */

const MATH = `
#ifndef __TGJS_MATH_H
#define __TGJS_MATH_H
#define M_E 2.71828182845904523536
#define M_LOG2E 1.44269504088896340736
#define M_LOG10E 0.434294481903251827651
#define M_LN2 0.693147180559945309417
#define M_LN10 2.30258509299404568402
#define M_PI 3.14159265358979323846
#define M_PI_2 1.57079632679489661923
#define M_PI_4 0.785398163397448309616
#define M_1_PI 0.318309886183790671538
#define M_2_PI 0.636619772367581343076
#define M_2_SQRTPI 1.12837916709551257390
#define M_SQRT2 1.41421356237309504880
#define M_SQRT1_2 0.707106781186547524401
#define INFINITY __builtin_inff()
#define HUGE_VALF __builtin_inff()
#define HUGE_VAL __builtin_inf()
#define NAN __builtin_nanf("")
#define CUDART_INF_F __builtin_inff()
#define CUDART_NAN_F __builtin_nanf("")
#define CUDART_PI_F 3.141592654f
#define CUDART_PI 3.1415926535897931e+0
#define CUDART_SQRT_2PI_F 2.506628275f
#define CUDART_INF __builtin_inf()
#endif
`;

const FLOAT = `
#ifndef __TGJS_FLOAT_H
#define __TGJS_FLOAT_H
#define FLT_MAX 3.402823466e+38f
#define FLT_MIN 1.175494351e-38f
#define FLT_EPSILON 1.192092896e-07f
#define FLT_TRUE_MIN 1.401298464e-45f
#define FLT_MANT_DIG 24
#define FLT_DIG 6
#define DBL_MAX 1.7976931348623158e+308
#define DBL_MIN 2.2250738585072014e-308
#define DBL_EPSILON 2.2204460492503131e-016
#define DBL_MANT_DIG 53
#endif
`;

const LIMITS = `
#ifndef __TGJS_LIMITS_H
#define __TGJS_LIMITS_H
#define CHAR_BIT 8
#define SCHAR_MIN (-128)
#define SCHAR_MAX 127
#define UCHAR_MAX 255
#define CHAR_MIN (-128)
#define CHAR_MAX 127
#define SHRT_MIN (-32768)
#define SHRT_MAX 32767
#define USHRT_MAX 65535
#define INT_MIN (-2147483647 - 1)
#define INT_MAX 2147483647
#define UINT_MAX 4294967295u
#define LONG_MIN (-9223372036854775807L - 1)
#define LONG_MAX 9223372036854775807L
#define ULONG_MAX 18446744073709551615UL
#define LLONG_MIN (-9223372036854775807LL - 1)
#define LLONG_MAX 9223372036854775807LL
#define ULLONG_MAX 18446744073709551615ULL
#define INT8_MIN (-128)
#define INT8_MAX 127
#define UINT8_MAX 255
#define INT16_MIN (-32768)
#define INT16_MAX 32767
#define UINT16_MAX 65535
#define INT32_MIN (-2147483647 - 1)
#define INT32_MAX 2147483647
#define UINT32_MAX 4294967295u
#define INT64_MAX 9223372036854775807LL
#define INT64_MIN (-9223372036854775807LL - 1)
#define UINT64_MAX 18446744073709551615ULL
#define SIZE_MAX 18446744073709551615ULL
#endif
`;

const RUNTIME = `
#ifndef __TGJS_CUDA_RUNTIME_H
#define __TGJS_CUDA_RUNTIME_H
#include <math.h>
#include <float.h>
#include <limits.h>
#define __CUDA_RUNTIME_H__ 1
#endif
`;

const FP16 = `
#ifndef __TGJS_CUDA_FP16_H
#define __TGJS_CUDA_FP16_H
#define __CUDA_FP16_H__ 1
#define CUDART_INF_FP16 __float2half(__builtin_inff())
#define CUDART_ZERO_FP16 __float2half(0.0f)
#define CUDART_ONE_FP16 __float2half(1.0f)
#endif
`;

const BF16 = `
#ifndef __TGJS_CUDA_BF16_H
#define __TGJS_CUDA_BF16_H
#define __CUDA_BF16_H__ 1
#endif
`;

/**
 * cuRAND device API: Philox4x32-10 (Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3", SC'11), with the state
 * layout and the curand_init / curand / curand_uniform / curand_normal semantics of NVIDIA's curand_philox4x32_x.h.
 * curandState (NVIDIA's default XORWOW generator) is mapped to Philox with a compile-time warning: XORWOW's subsequence
 * skip-ahead needs NVIDIA's precomputed matrix tables, which are not reproduced here.
 */
const CURAND = `
#ifndef __TGJS_CURAND_KERNEL_H
#define __TGJS_CURAND_KERNEL_H
#define CURAND_2POW32_INV (2.3283064e-10f)
#define CURAND_2POW32_INV_2PI (2.3283064e-10f * 6.2831855f)
#define CURAND_PHILOX_M0 0xD2511F53u
#define CURAND_PHILOX_M1 0xCD9E8D57u
#define CURAND_PHILOX_W0 0x9E3779B9u
#define CURAND_PHILOX_W1 0xBB67AE85u

struct curandStatePhilox4_32_10 {
  uint4 ctr;
  uint4 output;
  uint2 key;
  unsigned int STATE;
  int boxmuller_flag;
  int boxmuller_flag_double;
  float boxmuller_extra;
  double boxmuller_extra_double;
};
typedef struct curandStatePhilox4_32_10 curandStatePhilox4_32_10_t;
typedef struct curandStatePhilox4_32_10 curandState;
typedef struct curandStatePhilox4_32_10 curandState_t;
typedef struct curandStatePhilox4_32_10 curandStateXORWOW;
typedef struct curandStatePhilox4_32_10 curandStateXORWOW_t;

__device__ __forceinline__ uint4 __tgjs_philox_round(uint4 ctr, uint2 key) {
  unsigned int hi0 = __umulhi(CURAND_PHILOX_M0, ctr.x);
  unsigned int hi1 = __umulhi(CURAND_PHILOX_M1, ctr.z);
  unsigned int lo0 = CURAND_PHILOX_M0 * ctr.x;
  unsigned int lo1 = CURAND_PHILOX_M1 * ctr.z;
  return make_uint4(hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0);
}
__device__ __forceinline__ uint4 curand_Philox4x32_10(uint4 c, uint2 k) {
  #pragma unroll
  for (int r = 0; r < 9; r++) {
    c = __tgjs_philox_round(c, k);
    k.x += CURAND_PHILOX_W0; k.y += CURAND_PHILOX_W1;
  }
  return __tgjs_philox_round(c, k);
}
__device__ __forceinline__ void __tgjs_philox_incr(curandStatePhilox4_32_10_t* s, unsigned long long n) {
  unsigned int nlo = (unsigned int)n, nhi = (unsigned int)(n >> 32);
  s->ctr.x += nlo;
  if (s->ctr.x < nlo) nhi++;
  s->ctr.y += nhi;
  if (nhi <= s->ctr.y) return;
  if (++s->ctr.z) return;
  ++s->ctr.w;
}
__device__ __forceinline__ void __tgjs_philox_incr_hi(curandStatePhilox4_32_10_t* s, unsigned long long n) {
  unsigned int nlo = (unsigned int)n, nhi = (unsigned int)(n >> 32);
  s->ctr.z += nlo;
  if (s->ctr.z < nlo) nhi++;
  s->ctr.w += nhi;
}
__device__ __forceinline__ void __tgjs_philox_incr1(curandStatePhilox4_32_10_t* s) {
  if (++s->ctr.x) return;
  if (++s->ctr.y) return;
  if (++s->ctr.z) return;
  ++s->ctr.w;
}
__device__ __forceinline__ void skipahead(unsigned long long n, curandStatePhilox4_32_10_t* state) {
  state->STATE += (unsigned int)(n & 3);
  n /= 4;
  if (state->STATE > 3) { n += 1; state->STATE -= 4; }
  __tgjs_philox_incr(state, n);
  state->output = curand_Philox4x32_10(state->ctr, state->key);
}
__device__ __forceinline__ void skipahead_sequence(unsigned long long n, curandStatePhilox4_32_10_t* state) {
  __tgjs_philox_incr_hi(state, n);
  state->output = curand_Philox4x32_10(state->ctr, state->key);
}
__device__ __forceinline__ void curand_init(unsigned long long seed, unsigned long long subsequence, unsigned long long offset,
                                            curandStatePhilox4_32_10_t* state) {
  state->ctr = make_uint4(0u, 0u, 0u, 0u);
  state->key.x = (unsigned int)seed;
  state->key.y = (unsigned int)(seed >> 32);
  state->STATE = 0;
  state->boxmuller_flag = 0;
  state->boxmuller_flag_double = 0;
  state->boxmuller_extra = 0.0f;
  state->boxmuller_extra_double = 0.0;
  skipahead_sequence(subsequence, state);
  skipahead(offset, state);
}
__device__ __forceinline__ unsigned int curand(curandStatePhilox4_32_10_t* state) {
  unsigned int ret;
  unsigned int st = state->STATE;
  if (st == 0) ret = state->output.x;
  else if (st == 1) ret = state->output.y;
  else if (st == 2) ret = state->output.z;
  else ret = state->output.w;
  state->STATE = st + 1;
  if (state->STATE == 4) {
    __tgjs_philox_incr1(state);
    state->output = curand_Philox4x32_10(state->ctr, state->key);
    state->STATE = 0;
  }
  return ret;
}
__device__ __forceinline__ uint4 curand4(curandStatePhilox4_32_10_t* state) {
  uint4 tmp = state->output;
  uint4 r;
  __tgjs_philox_incr1(state);
  state->output = curand_Philox4x32_10(state->ctr, state->key);
  unsigned int st = state->STATE;
  if (st == 0) return tmp;
  if (st == 1) r = make_uint4(tmp.y, tmp.z, tmp.w, state->output.x);
  else if (st == 2) r = make_uint4(tmp.z, tmp.w, state->output.x, state->output.y);
  else r = make_uint4(tmp.w, state->output.x, state->output.y, state->output.z);
  return r;
}
__device__ __forceinline__ float _curand_uniform(unsigned int x) { return x * CURAND_2POW32_INV + (CURAND_2POW32_INV / 2.0f); }
__device__ __forceinline__ float curand_uniform(curandStatePhilox4_32_10_t* state) { return _curand_uniform(curand(state)); }
__device__ __forceinline__ float4 curand_uniform4(curandStatePhilox4_32_10_t* state) {
  uint4 x = curand4(state);
  return make_float4(_curand_uniform(x.x), _curand_uniform(x.y), _curand_uniform(x.z), _curand_uniform(x.w));
}
__device__ __forceinline__ float2 _curand_box_muller(unsigned int x, unsigned int y) {
  float u = x * CURAND_2POW32_INV + (CURAND_2POW32_INV / 2.0f);
  float v = y * CURAND_2POW32_INV_2PI + (CURAND_2POW32_INV_2PI / 2.0f);
  float s = sqrtf(-2.0f * logf(u));
  return make_float2(s * sinf(v), s * cosf(v));
}
__device__ __forceinline__ float curand_normal(curandStatePhilox4_32_10_t* state) {
  if (state->boxmuller_flag != 1) {
    unsigned int x = curand(state);
    unsigned int y = curand(state);
    float2 v = _curand_box_muller(x, y);
    state->boxmuller_extra = v.y;
    state->boxmuller_flag = 1;
    return v.x;
  }
  state->boxmuller_flag = 0;
  return state->boxmuller_extra;
}
__device__ __forceinline__ float2 curand_normal2(curandStatePhilox4_32_10_t* state) {
  unsigned int x = curand(state);
  unsigned int y = curand(state);
  return _curand_box_muller(x, y);
}
__device__ __forceinline__ float curand_log_normal(curandStatePhilox4_32_10_t* state, float mean, float stddev) {
  return expf(mean + stddev * curand_normal(state));
}
#endif
`;

export const BUILTIN_HEADERS: Record<string, string> = {
  "cuda_runtime.h": RUNTIME, "cuda_runtime_api.h": RUNTIME, "cuda.h": RUNTIME, "device_launch_parameters.h": "", "device_functions.h": "",
  "cuda_device_runtime_api.h": "", "sm_20_atomic_functions.h": "", "sm_60_atomic_functions.h": "", "crt/host_defines.h": "",
  "math.h": MATH, cmath: MATH, "math_constants.h": MATH, "float.h": FLOAT, cfloat: FLOAT, "limits.h": LIMITS, climits: LIMITS,
  "stdint.h": "", cstdint: "", "stddef.h": "", cstddef: "", "assert.h": "", cassert: "", "stdio.h": "", cstdio: "",
  "cuda_fp16.h": FP16, "cuda_fp16.hpp": FP16, "cuda_bf16.h": BF16, "cuda_bf16.hpp": BF16, "vector_types.h": "", "vector_functions.h": "",
  "curand_kernel.h": CURAND, "curand.h": "", "curand_philox4x32_x.h": CURAND,
};

/** headers that only declare host APIs: including them is harmless for device code */
export const HOST_ONLY_HEADERS = new Set([
  "iostream", "vector", "string", "algorithm", "stdlib.h", "cstdlib", "string.h", "cstring", "time.h", "ctime", "chrono", "random",
  "cublas_v2.h", "cublas.h", "cudnn.h", "thrust/device_vector.h", "thrust/host_vector.h", "helper_cuda.h", "helper_functions.h",
  "numeric", "memory", "functional", "utility", "array", "map", "unordered_map", "iomanip", "fstream", "sstream", "cstdarg", "cassert",
]);
