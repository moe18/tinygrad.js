/**
 * Math functions imported by CPU kernels. They run in double precision; float kernels round the result once, so they are at
 * least as accurate as CUDA's single-precision library (which allows 1-4 ulp).
 */

/** erf with |error| < 1.2e-16 relative (W. J. Cody's rational approximations, as in many libm's). */
export function erf(x: number): number {
  if (Number.isNaN(x)) return NaN;
  const a = Math.abs(x);
  if (a < 0.5) {
    const z = x * x;
    const p = [3.16112374387056560e00, 1.13864154151050156e02, 3.77485237685302021e02, 3.20937758913846947e03, 1.85777706184603153e-1];
    const q = [2.36012909523441209e01, 2.44024637934444173e02, 1.28261652607737228e03, 2.84423683343917062e03];
    let num = p[4] * z, den = z;
    for (let i = 0; i < 3; i++) { num = (num + p[i]) * z; den = (den + q[i]) * z; }
    return (x * (num + p[3])) / (den + q[3]);
  }
  return x < 0 ? -(1 - erfcPos(a)) : 1 - erfcPos(a);
}
function erfcPos(a: number): number {
  if (a >= 27) return 0;
  if (a < 4) {
    const p = [5.64188496988670089e-1, 8.88314979438837594e00, 6.61191906371416295e01, 2.98635138197400131e02, 8.81952221241769090e02, 1.71204761263407058e03, 2.05107837782607147e03, 1.23033935479799725e03, 2.15311535474403846e-8];
    const q = [1.57449261107098347e01, 1.17693950891312499e02, 5.37181101862009858e02, 1.62138957456669019e03, 3.29079923573345963e03, 4.36261909014324716e03, 3.43936767414372164e03, 1.23033935480374942e03];
    let num = p[8] * a, den = a;
    for (let i = 0; i < 7; i++) { num = (num + p[i]) * a; den = (den + q[i]) * a; }
    const r = (num + p[7]) / (den + q[7]);
    const z = Math.trunc(a * 16) / 16;
    return Math.exp(-z * z) * Math.exp(-(a - z) * (a + z)) * r;
  }
  const z = 1 / (a * a);
  const p = [3.05326634961232344e-1, 3.60344899949804439e-1, 1.25781726111229246e-1, 1.60837851487422766e-2, 6.58749161529837803e-4, 1.63153871373020978e-2];
  const q = [2.56852019228982242e00, 1.87295284992346725e00, 5.27905102951428412e-1, 6.05183413124413191e-2, 2.33520497626869185e-3];
  let num = p[5] * z, den = z;
  for (let i = 0; i < 4; i++) { num = (num + p[i]) * z; den = (den + q[i]) * z; }
  let r = (z * (num + p[4])) / (den + q[4]);
  r = (0.564189583547756287 - r) / a;
  const zz = Math.trunc(a * 16) / 16;
  return Math.exp(-zz * zz) * Math.exp(-(a - zz) * (a + zz)) * r;
}
export function erfc(x: number): number {
  if (Number.isNaN(x)) return NaN;
  if (x < 0.5) return 1 - erf(x);
  return erfcPos(x);
}

const LANCZOS = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
  12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
export function lgamma(x: number): number {
  if (Number.isNaN(x)) return NaN;
  if (!Number.isFinite(x)) return Infinity;
  if (x <= 0 && Number.isInteger(x)) return Infinity;
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
  // Stirling series for large x (more accurate there than Lanczos)
  if (x > 15) {
    const z = 1 / (x * x);
    return (x - 0.5) * Math.log(x) - x + 0.91893853320467274 + (1 / x) * (1 / 12 - z * (1 / 360 - z * (1 / 1260 - z / 1680)));
  }
  const y = x - 1;
  let a = LANCZOS[0];
  const t = y + 7.5;
  for (let i = 1; i < 9; i++) a += LANCZOS[i] / (y + i);
  return 0.91893853320467274 + (y + 0.5) * Math.log(t) - t + Math.log(a);
}
export function tgamma(x: number): number {
  if (Number.isNaN(x) || x === -Infinity) return NaN;
  if (x === Infinity) return Infinity;
  if (x === 0) return 1 / x;
  if (x < 0 && Number.isInteger(x)) return NaN;
  if (x > 171.7) return Infinity;
  if (Number.isInteger(x) && x <= 23) { let r = 1; for (let i = 2; i < x; i++) r *= i; return r; }
  if (x < 0.5) return Math.PI / (Math.sin(Math.PI * x) * tgamma(1 - x));
  const r = Math.exp(lgamma(x));
  return r;
}

/** IEEE remainder: x - n*y, n = x/y rounded half to even (exact) */
export function remainder(x: number, y: number): number {
  if (Number.isNaN(x) || Number.isNaN(y) || !Number.isFinite(x) || y === 0) return NaN;
  if (!Number.isFinite(y)) return x;
  const ay = Math.abs(y);
  let t = Math.abs(x % (2 * ay)); // JS % is the exact C fmod
  if (!Number.isFinite(2 * ay)) t = Math.abs(x);
  let r = t;
  if (t > 0.5 * ay) { const d = t - ay; r = d >= 0.5 * ay ? d - ay : d; }
  return x < 0 || Object.is(x, -0) ? -r : r;
}

export const MATH_IMPORTS: Record<string, (...a: number[]) => number> = {
  exp: Math.exp, exp2: (x) => Math.pow(2, x), exp10: (x) => Math.pow(10, x), log: Math.log, log2: Math.log2, log10: Math.log10,
  expm1: Math.expm1, log1p: Math.log1p, sin: Math.sin, cos: Math.cos, tan: Math.tan, asin: Math.asin, acos: Math.acos, atan: Math.atan,
  sinh: Math.sinh, cosh: Math.cosh, tanh: Math.tanh, asinh: Math.asinh, acosh: Math.acosh, atanh: Math.atanh, cbrt: Math.cbrt,
  erf, erfc, lgamma, tgamma, normcdf: (x) => 0.5 * erfc(-x / Math.SQRT2),
  pow: (x, y) => cpow(x, y), atan2: Math.atan2, hypot: Math.hypot, fmod: (x, y) => x % y, remainder,
};

/** C pow semantics (JS Math.pow differs for pow(1, NaN) and pow(-1, +-Infinity)) */
function cpow(x: number, y: number): number {
  if (x === 1 || y === 0) return 1;
  if (x === -1 && !Number.isFinite(y) && !Number.isNaN(y)) return 1;
  return Math.pow(x, y);
}
