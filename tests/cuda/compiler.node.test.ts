/** CUDA compiler unit tests (Node.js): preprocessor, lexer, parser, semantic analysis, IR, passes, WGSL and WASM generation. */
import { test, expect } from "@playwright/test";
import { tokenize, parseNumber } from "../../src/cuda/compiler/lexer.js";
import { preprocess } from "../../src/cuda/compiler/preprocess.js";
import { DiagnosticSink, CompileError } from "../../src/cuda/compiler/diagnostics.js";
import { parseModule } from "../../src/cuda/compiler/module.js";
import { printKernel, walkStmts, type Kernel } from "../../src/cuda/compiler/ir.js";
import { generateWGSL, DEFAULT_WGSL_CAPS, type WGSLCaps } from "../../src/cuda/compiler/wgsl.js";
import { generateWasm } from "../../src/cuda/compiler/wasm/codegen.js";
import { roundHalf, halfBits, foldExpr } from "../../src/cuda/compiler/fold.js";
import { sources } from "../../src/cuda/kernels/index.js";
import { formatPrintf } from "../../src/cuda/runtime/encode.js";

const caps: WGSLCaps = { ...DEFAULT_WGSL_CAPS, maxComputeWorkgroupStorageSize: 32768, maxStorageBuffersPerShaderStage: 10 };
const lower = (src: string, ref: string, block: [number, number, number] = [64, 1, 1], mode: "optimized" | "direct" = "optimized") =>
  parseModule(src).instantiate(ref, { block, dynamicShared: 0 }, { mode });
const err = (f: () => unknown): CompileError => {
  try { f(); } catch (e) { if (e instanceof CompileError) return e; throw e; }
  throw new Error("expected a CompileError");
};
const count = (k: Kernel, pred: (s: import("../../src/cuda/compiler/ir.js").Stmt) => boolean) => { let n = 0; walkStmts(k.body, (s) => { if (pred(s)) n++; }); return n; };

test.describe("lexer", () => {
  test("tokens, literals, CUDA launch punctuation", () => {
    const t = tokenize("a<<<g, b>>>(x); x >>= 2; 1.5e-3f 0x1fu 07 'c' \"s\\n\"").map((x) => `${x.k}:${x.v}`);
    expect(t).toContain("punct:<<<");
    expect(t).toContain("punct:>>>");
    expect(t).toContain("punct:>>=");
    expect(t).toContain("num:1.5e-3f");
    expect(t).toContain("char:c");
    expect(t).toContain("str:s\n");
  });
  test("integer literal types follow C++ rules (LP64)", () => {
    expect(parseNumber("2147483647").type).toBe("int");
    expect(parseNumber("2147483648").type).toBe("long");
    expect(parseNumber("0xffffffff").type).toBe("unsigned int");
    expect(parseNumber("1ull").type).toBe("unsigned long long");
    expect(parseNumber("0.5f")).toMatchObject({ isFloat: true, type: "float" });
    expect(parseNumber("0x1.8p1").value).toBe(3);
    expect(() => parseNumber("1f")).toThrow(/1\.0f/);
  });
});

test.describe("preprocessor", () => {
  const pp = (src: string, defines?: Record<string, string>) => preprocess(src, "t.cu", { defines }, new DiagnosticSink()).filter((t) => t.k !== "eof" && t.loc.file === "t.cu").map((t) => t.v).join(" ");
  test("object and function-like macros, # and ##, variadic", () => {
    expect(pp("#define N 4\n#define SQ(x) ((x)*(x))\nint a = SQ(N + 1);")).toBe("int a = ( ( 4 + 1 ) * ( 4 + 1 ) ) ;");
    expect(pp("#define CAT(a,b) a##b\nint CAT(foo, 2);")).toBe("int foo2 ;");
    expect(pp("#define S(x) #x\nS(a + b)")).toBe("a + b");
    expect(pp("#define F(fmt, ...) printf(fmt, __VA_ARGS__)\nF(\"%d\", 1, 2)")).toBe("printf ( %d , 1 , 2 )");
  });
  test("conditionals, defined(), __CUDA_ARCH__, user defines", () => {
    expect(pp("#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700\nyes\n#else\nno\n#endif")).toBe("yes");
    expect(pp("#ifdef FOO\na\n#elif BAR > 2\nb\n#else\nc\n#endif", { BAR: "3" })).toBe("b");
    expect(pp("#ifndef X\n#define X 1\n#endif\nX")).toBe("1");
  });
  test("includes: built-in headers, user headers, host-only headers", () => {
    const sink = new DiagnosticSink();
    const toks = preprocess('#include <iostream>\n#include "util.h"\nVALUE', "t.cu", { headers: { "util.h": "#define VALUE 42" } }, sink);
    expect(toks.filter((t) => t.loc.file === "t.cu" && t.k !== "eof").map((t) => t.v)).toEqual(["42"]);
    expect(sink.warnings.some((w) => /host-only/.test(w.message))).toBe(true);
    expect(() => preprocess('#include "missing.h"', "t.cu", {}, new DiagnosticSink())).toThrow(/cannot find header 'missing.h'/);
    expect(() => preprocess("#if 1\nx", "t.cu", {}, new DiagnosticSink())).toThrow(/unterminated #if/);
    expect(() => preprocess("#error stop here", "t.cu", {}, new DiagnosticSink())).toThrow(/#error stop here/);
  });
});

test.describe("parser and semantic analysis", () => {
  test("host code is skipped: main(), std::vector, <<<>>> launches", () => {
    const m = parseModule(`#include <vector>
__global__ void k(float* x) { x[threadIdx.x] = 1.0f; }
int main() { std::vector<float> v(10); float* d; cudaMalloc(&d, 40); k<<<1, 10>>>(d); return 0; }`);
    expect(m.kernels.map((k) => k.name)).toEqual(["k"]);
  });
  test("templates, deduction, explicit arguments, struct templates, static members", () => {
    const k = lower(`
template <typename T, int N> struct Vec { T v[N]; static constexpr int size = N; __device__ T sum() const { T s = 0; for (int i = 0; i < N; i++) s += v[i]; return s; } };
template <typename T> __device__ T twice(T x) { return x + x; }
template <int B> __global__ void k(float* o) { Vec<float, 4> a = {{1, 2, 3, 4}}; o[0] = twice(a.sum()) + Vec<int, B>::size + twice<int>(3); }`, "k<8>", [1, 1, 1]);
    expect(printKernel(k)).toContain("store.f32 o[0] = 34.0");
  });
  test("errors carry locations and codes", () => {
    const e1 = err(() => lower("__global__ void k(float* x) {\n  x[0] = y;\n}", "k"));
    expect(e1.code).toBe("name");
    expect(e1.loc?.line).toBe(2);
    const e2 = err(() => lower("__global__ void k(float* x) { x[0] = 1.0f % 2.0f; }", "k"));
    expect(e2.message).toMatch(/fmodf/);
    const e3 = err(() => lower("__global__ void k(float* x) { goto end; end: ; }", "k"));
    expect(e3.code).toBe("unsupported");
    const e4 = err(() => lower("__device__ int f(int n) { return n ? f(n - 1) : 0; }\n__global__ void k(int* x) { x[0] = f(3); }", "k"));
    expect(e4.message).toMatch(/recursi/);
    const e5 = err(() => lower("__global__ void k(float* a, float* b, int c) { float* p = c ? a : b; p[0] = 1; }", "k"));
    expect(e5.message).toMatch(/different buffers/);
    const e6 = err(() => lower("__global__ void k(float** p) { }", "k"));
    expect(e6.message).toMatch(/pointers to pointers/);
    expect(err(() => lower("__global__ void k(int* x) { x[0] = clock(); }", "k")).message).toMatch(/clock\(\)/);
    expect(err(() => lower("__global__ void k(float* x) { asm(\"nop\"); }", "k")).code).toBe("unsupported");
  });
  test("integer promotion and usual arithmetic conversions", () => {
    const k = lower("__global__ void k(unsigned* o, int a, unsigned b, char c, long long d) { o[0] = a + b; o[1] = c + c; o[2] = (unsigned)(d + a); }", "k", [1, 1, 1], "direct");
    const ir = printKernel(k);
    expect(ir).toMatch(/\(u32\(a_\d+\) \+ b_\d+\)/);        // int + unsigned -> unsigned
    expect(ir).toMatch(/\(i32\(c_\d+\) \+ i32\(c_\d+\)\)/); // char + char -> int
    expect(ir).toMatch(/\(d_\d+ \+ i64\(a_\d+\)\)/);        // long long + int -> long long
  });
  test("struct scalar replacement and register promotion of local arrays", () => {
    const k = lower("struct P { float x, y; };\n__global__ void k(float* o) { P p = {1.0f, 2.0f}; float a[4]; for (int i = 0; i < 4; i++) a[i] = i * p.y; o[0] = a[3] + p.x; }", "k", [1, 1, 1]);
    expect(k.roots.filter((r) => r.space === "local")).toHaveLength(0);
    expect(printKernel(k)).toContain("store.f32 o[0] = 7.0");
  });
});

test.describe("optimizer", () => {
  const src = `#define T 16
__global__ void mm(const float* A, const float* B, float* C, int n) {
  __shared__ float As[T][T];
  int r = threadIdx.y, c = threadIdx.x;
  float acc = 0.0f;
  As[r][c] = A[r * n + c];
  __syncthreads();
  __syncthreads();
  #pragma unroll
  for (int k = 0; k < T; ++k) acc += As[r][k] * B[k * n + c];
  C[r * n + c] = acc;
  __syncthreads();
}`;
  test("direct mode keeps the loop; optimized mode unrolls, folds, hoists and removes redundant barriers", () => {
    const d = lower(src, "mm", [16, 16, 1], "direct");
    const o = lower(src, "mm", [16, 16, 1], "optimized");
    expect(count(d, (s) => s.s === "loop")).toBe(1);
    expect(count(o, (s) => s.s === "loop")).toBe(0);
    expect(count(d, (s) => s.s === "barrier")).toBe(3);
    expect(count(o, (s) => s.s === "barrier")).toBe(1);   // duplicate and trailing barriers removed
    const passes = (o as Kernel & { passes: { name: string; changes: number }[] }).passes;
    for (const p of ["unroll", "copyprop", "dce", "barriers"]) expect(passes.find((x) => x.name === p && x.changes > 0), p).toBeTruthy();
  });
  test("constant folding is exact (wrapping ints, float rounding, half precision)", () => {
    const k = lower(`__global__ void k(int* i, float* f) {
  i[0] = 2147483647 + 1; i[1] = -7 / 2; i[2] = (int)(3000000000u >> 1); i[3] = (int)(unsigned char)300;
  f[0] = 0.1f + 0.2f; f[1] = __half2float(__float2half(0.1f)); f[2] = sqrtf(2.0f); }`, "k", [1, 1, 1]);
    const ir = printKernel(k);
    expect(ir).toContain("i[0] = -2147483648");
    expect(ir).toContain("i[4] = -3");
    expect(ir).toContain("i[8] = 1500000000");
    expect(ir).toContain("i[12] = 44");
    expect(ir).toContain(`f[0] = ${Math.fround(Math.fround(0.1) + Math.fround(0.2))}`);
    expect(ir).toContain(`f[4] = ${roundHalf(Math.fround(0.1))}`);
    expect(ir).toContain(`f[8] = ${Math.fround(Math.SQRT2)}`);
  });
  test("transcendental functions are not folded (folding would change results vs the backend)", () => {
    expect(printKernel(lower("__global__ void k(float* f) { f[0] = expf(1.0f); }", "k", [1, 1, 1]))).toContain("exp.f32(1.0)");
  });
});

test.describe("code generation", () => {
  test("every library kernel compiles for WGSL and WebAssembly, in both modes", async () => {
    const targs: Record<string, string> = {
      matmul_tiled: "<16>", matmul_regtile: "<64, 64, 8, 4, 4>", reduce_sum: "<256>", reduce_sum_warp: "<256>", row_max: "<128>", scan_block: "<256>", scan_add: "<256>",
      scan_blelloch: "<128>", softmax_rows: "<128>", softmax_rows_bwd: "<128>", layernorm_fwd: "<128>", layernorm_bwd: "<128>", rmsnorm_fwd: "<128>", rmsnorm_bwd: "<128>",
      cross_entropy_fwd_bwd: "<128>", flash_attention_fwd: "<32, 32, 64>", quantize_rows_int8: "<128>", sumsq: "<256>", add_layernorm: "<128>",
    };
    const two = new Set(["matmul_tiled", "matmul_regtile", "matmul_naive", "transpose_tiled", "transpose_naive"]);
    let n = 0;
    for (const src of Object.values(sources)) {
      const m = parseModule(src);
      for (const k of m.kernels) for (const mode of ["optimized", "direct"] as const) {
        const block: [number, number, number] = two.has(k.name) ? [16, 16, 1] : k.name === "flash_attention_fwd" ? [32, 1, 1] : [128, 1, 1];
        const ref = k.name + (targs[k.name] ?? "");
        const w = generateWGSL(m.instantiate(ref, { block, dynamicShared: 0 }, { mode }), caps, { boundsCheck: true }, new DiagnosticSink());
        expect(w.code).toContain("@compute");
        const wasm = generateWasm(m.instantiate(ref, { block, dynamicShared: 0 }, { mode }), { shared: true, maxMemoryPages: 100 }, new DiagnosticSink());
        await WebAssembly.compile(wasm.bytes as BufferSource);
        n++;
      }
    }
    expect(n).toBeGreaterThan(120);
  });
  test("WGSL: binding layout, workgroup size, typed vs word-addressed buffers, f64 policy", () => {
    const k = lower("__global__ void k(const float* a, unsigned char* b, int n) { int i = threadIdx.x; if (i < n) b[i] = a[i] > 0.0f; }", "k", [64, 1, 1]);
    const w = generateWGSL(k, caps, {}, new DiagnosticSink());
    expect(w.workgroupSize).toEqual([64, 1, 1]);
    expect(w.code).toContain("var<storage, read> buf0_a: array<f32>");
    expect(w.code).toContain("array<atomic<u32>>"); // byte stores into a shared word need atomic read-modify-write
    expect(w.notes.join(" ")).toMatch(/8\/16-bit stores/);
    const d = lower("__global__ void k(float* a) { a[0] = a[0] * 0.5; a[1] = a[1] * 0.1; }", "k", [1, 1, 1]);
    expect(() => generateWGSL(d, caps, {}, new DiagnosticSink())).toThrow(/double precision is not supported by WebGPU/);
    const d2 = lower("__global__ void k(float* a) { a[0] = a[0] * 0.5; }", "k", [1, 1, 1]);
    expect(generateWGSL(d2, caps, {}, new DiagnosticSink()).code).not.toMatch(/f64/); // exact demotion
    const s = new DiagnosticSink();
    generateWGSL(lower("__global__ void k(float* a) { a[1] = a[1] * 0.1; }", "k", [1, 1, 1]), caps, { fp64: "demote" }, s);
    expect(s.warnings.some((w2) => /demoted/.test(w2.message))).toBe(true);
    expect(() => generateWGSL(lower("__global__ void k(float* a) { }", "k", [512, 1, 1]), caps, {}, new DiagnosticSink())).toThrow(/exceeds this GPU's maxComputeInvocationsPerWorkgroup/);
  });
  test("WGSL: parameters and pointer offsets are read from the uniform buffer once, at kernel entry, in both modes", () => {
    // optimized mode used to propagate `bitcast<i32>(P[2].z)` into the k loop (a 1.7-2.3x slowdown on D3D12/NVIDIA)
    const gridStride = "__global__ void k(const float* x, float* y, int n) { for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) y[i] = 2.0f * x[i]; }";
    for (const mode of ["optimized", "direct"] as const) {
      for (const [src, ref, block] of [[sources.matmul, "matmul_naive", [16, 16, 1]], [gridStride, "k", [256, 1, 1]]] as const) {
        const code = generateWGSL(lower(src, ref, [...block], mode), caps, {}, new DiagnosticSink()).code;
        const body = code.slice(code.indexOf("@compute"));
        expect(body.slice(body.indexOf("loop {"))).not.toMatch(/P\[\d+\]/);  // pointer offsets, sizes, gridDim
        const reads = body.match(/let uw\d+ = P\[\d+\]\.[xyzw];/g) ?? [];
        expect(new Set(reads).size).toBe(reads.length);                    // each word once
      }
      const mm = generateWGSL(lower(sources.matmul, "matmul_naive", [16, 16, 1], mode), caps, {}, new DiagnosticSink()).code;
      expect(mm.match(/let prm\d+_i32 = bitcast<i32>\(uw\d+\);/g)).toHaveLength(3);  // M, N, K
    }
  });
  test("WebAssembly: phase splitting at barriers and spilled per-thread slots", () => {
    const k = lower(sources.reduction, "reduce_sum<256>", [256, 1, 1]);
    const w = generateWasm(k, { shared: false, maxMemoryPages: 100 }, new DiagnosticSink());
    expect(w.phases).toBeGreaterThan(2);
    expect(w.text).toMatch(/thread-phase/);
    const plain = generateWasm(lower(sources.vectorAdd, "vector_add", [256, 1, 1]), { shared: false, maxMemoryPages: 100 }, new DiagnosticSink());
    expect(plain.phases).toBe(1);
  });
  test("printf formatting", () => {
    const f32 = new Uint32Array(new Float32Array([3.14159]).buffer)[0];
    expect(formatPrintf("%d|%5.2f|%x|%-4d|%04d|%c|%u|%e|%g", ["i32", "f32", "u32", "i32", "i32", "i8", "i32", "f32", "f32"], [-3, f32, 255, 7, 42, 65, -1, f32, f32]))
      .toBe("-3| 3.14|ff|7   |0042|A|4294967295|3.141590e+00|3.14159");
    expect(halfBits(65504)).toBe(0x7bff);
    expect(foldExpr).toBeTruthy();
  });
});

test.describe("WGSL __syncthreads memory ordering", () => {
  const sb = (src: string, ref: string, aliasGroups?: number[]) =>
    (generateWGSL(lower(src, ref, [256, 1, 1]), caps, { aliasGroups }, new DiagnosticSink()).code.match(/storageBarrier\(\)/g) ?? []).length;
  test("storageBarrier is elided when no thread can observe another thread's global writes across the barrier", () => {
    // reads only `in`, writes `out` once with a commutative atomic whose result is unused
    expect(sb(sources.reduction, "reduce_sum<256>", [0, 1, 2])).toBe(0);
    // a single store outside loops, never read back
    expect(sb("__global__ void k(const float* a, float* b) { __shared__ float s[256]; s[threadIdx.x] = a[threadIdx.x]; __syncthreads(); b[threadIdx.x] = s[255 - threadIdx.x]; }", "k", [0, 1])).toBe(0);
  });
  test("storageBarrier is kept whenever global memory may carry data between threads", () => {
    const comm = "__global__ void k(float* a, float* b) { a[threadIdx.x] = 1.0f; __syncthreads(); b[threadIdx.x] = a[255 - threadIdx.x]; }";
    expect(sb(comm, "k", [0, 1])).toBeGreaterThan(0);                                   // written buffer is read back
    expect(sb("__global__ void k(const float* a, float* b) { b[threadIdx.x] = 1.0f; __syncthreads(); b[threadIdx.x] = a[255 - threadIdx.x]; }", "k", [0, 1])).toBeGreaterThan(0); // two store sites
    expect(sb("__global__ void k(const float* a, float* b) { for (int i = 0; i < 4; i++) { b[(threadIdx.x + i) & 255] = a[i]; __syncthreads(); } }", "k", [0, 1])).toBeGreaterThan(0); // store in a loop
    expect(sb("__global__ void k(const float* a, int* c, float* b) { int o = atomicAdd(c, 1); __syncthreads(); b[o & 255] = a[0]; }", "k", [0, 1, 2])).toBeGreaterThan(0); // atomic result used
    // aliasing: `a` and `b` bound to one buffer, or alias information unavailable
    expect(sb("__global__ void k(const float* a, float* b) { b[threadIdx.x] = 1.0f; __syncthreads(); b[0] = a[threadIdx.x]; }", "k", [0, 0])).toBeGreaterThan(0);
    expect(sb(sources.reduction, "reduce_sum<256>")).toBeGreaterThan(0);
  });
});
