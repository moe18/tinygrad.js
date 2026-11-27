/**
 * Randomized differential testing: random integer/float expression kernels are run with runtime inputs on a backend and
 * compared against the same expressions with the inputs as compile-time constants, which the compiler's constant folder
 * evaluates with exact C semantics (fold.ts). Undefined behaviour is avoided by construction (divisors are forced
 * non-zero, shift amounts masked). Float division is excluded on backends without correctly rounded division (WGSL).
 */
export interface RandomKernel { source: string; folded: string; inputs: { i: number[]; u: number[]; f: number[] }; outputs: { kind: "i" | "u" | "f"; n: number } }

export function randomKernels(seed: number, count: number, opts: { floatDivision: boolean }): RandomKernel[] {
  let s = seed >>> 0 || 1;
  const rnd = (n: number) => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s % n; };
  const out: RandomKernel[] = [];
  for (let k = 0; k < count; k++) {
    const ints = Array.from({ length: 4 }, () => (rnd(2) ? rnd(2000) - 1000 : (rnd(0x7fffffff) | 0) - 0x3fffffff));
    const uns = Array.from({ length: 4 }, () => rnd(0xffffffff) >>> 0);
    const flts = Array.from({ length: 4 }, () => Math.fround((rnd(20000) - 10000) / 37));
    const intExpr = (d: number): string => {
      if (d <= 0 || rnd(5) === 0) return rnd(3) ? `i${rnd(4)}` : String(rnd(200) - 100);
      const a = intExpr(d - 1), b = intExpr(d - 1);
      switch (rnd(12)) {
        case 0: return `(${a} + ${b})`;
        case 1: return `(${a} - ${b})`;
        case 2: return `(${a} * ${b})`;
        case 3: return `(${a} / ((${b}) | 1))`;
        case 4: return `(${a} % ((${b}) | 1))`;
        case 5: return `(${a} & ${b})`;
        case 6: return `(${a} ^ ${b})`;
        case 7: return `(${a} << ((${b}) & 15))`;
        case 8: return `(${a} >> ((${b}) & 31))`;
        case 9: return `((${a}) < (${b}) ? ${a} : ${b})`;
        case 10: return `(int)((unsigned)(${a}) / (((unsigned)(${b})) | 1u))`;
        default: return `(-(${a}))`;
      }
    };
    const unsExpr = (d: number): string => {
      if (d <= 0 || rnd(5) === 0) return rnd(3) ? `u${rnd(4)}` : `${rnd(100000)}u`;
      const a = unsExpr(d - 1), b = unsExpr(d - 1);
      switch (rnd(9)) {
        case 0: return `(${a} + ${b})`;
        case 1: return `(${a} - ${b})`;
        case 2: return `(${a} * ${b})`;
        case 3: return `(${a} / (${b} | 1u))`;
        case 4: return `(${a} % (${b} | 1u))`;
        case 5: return `(${a} >> (${b} & 31u))`;
        case 6: return `(${a} | ${b})`;
        case 7: return `(unsigned)((long long)(${a}) * 3 - (long long)(${b}))`;
        default: return `(${a} > ${b} ? ${a} - ${b} : ${b} - ${a})`;
      }
    };
    const fltExpr = (d: number): string => {
      if (d <= 0 || rnd(5) === 0) return rnd(3) ? `f${rnd(4)}` : `${((rnd(2000) - 1000) / 8).toFixed(3)}f`;
      const a = fltExpr(d - 1), b = fltExpr(d - 1);
      switch (rnd(opts.floatDivision ? 8 : 7)) {
        case 0: return `(${a} + ${b})`;
        case 1: return `(${a} - ${b})`;
        case 2: return `(${a} * ${b})`;
        case 3: return `fabsf(${a})`;
        case 4: return `floorf(${a})`;
        case 5: return `(float)(int)(${a} * 0.25f)`;
        case 6: return `fmaxf(${a}, ${b})`;
        default: return `(${a} / (fabsf(${b}) + 1.0f))`;
      }
    };
    const n = 4;
    const iexprs = Array.from({ length: n }, () => intExpr(4));
    const uexprs = Array.from({ length: n }, () => unsExpr(4));
    const fexprs = Array.from({ length: n }, () => fltExpr(3));
    const body = (prefix: string) => [
      ...iexprs.map((e, j) => `  ${prefix}oi[${j}] = ${e};`),
      ...uexprs.map((e, j) => `  ${prefix}ou[${j}] = ${e};`),
      ...fexprs.map((e, j) => `  ${prefix}of[${j}] = ${e};`),
    ].join("\n");
    const decl = (vals: number[], kind: string, t: string) => vals.map((v, j) => `  const ${t} ${kind}${j} = ${kind === "f" ? (Object.is(v, -0) ? "-0.0f" : `${v}f`) : kind === "u" ? `${v}u` : v === -2147483648 ? "(-2147483647 - 1)" : String(v)};`).join("\n");
    const source = `__global__ void rk(const int* I, const unsigned* U, const float* F, int* oi, unsigned* ou, float* of) {
  int i0 = I[0], i1 = I[1], i2 = I[2], i3 = I[3];
  unsigned u0 = U[0], u1 = U[1], u2 = U[2], u3 = U[3];
  float f0 = F[0], f1 = F[1], f2 = F[2], f3 = F[3];
${body("")}
}`;
    const folded = `__global__ void rk(const int* I, const unsigned* U, const float* F, int* oi, unsigned* ou, float* of) {
${decl(ints, "i", "int")}
${decl(uns, "u", "unsigned")}
${decl(flts.map((f) => Number(f.toPrecision(9))), "f", "float")}
${body("")}
}`;
    out.push({ source, folded: folded.replace(/const float f(\d) = ([^;]+);/g, (_m, j) => `const float f${j} = ${floatLit(flts[Number(j)])};`), inputs: { i: ints, u: uns, f: flts }, outputs: { kind: "i", n } });
  }
  return out;
}

function floatLit(v: number): string {
  // exact float literal via hexadecimal floating constants
  if (v === 0) return Object.is(v, -0) ? "-0.0f" : "0.0f";
  const neg = v < 0; const a = Math.abs(v);
  let e = Math.floor(Math.log2(a)); let m = a / Math.pow(2, e);
  if (m >= 2) { m /= 2; e++; } if (m < 1) { m *= 2; e--; }
  const frac = Math.round((m - 1) * 2 ** 24);
  return `${neg ? "-" : ""}0x1.${frac.toString(16).padStart(6, "0")}p${e}f`;
}
