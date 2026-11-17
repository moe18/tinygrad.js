import { test, expect, openSession, reference, compareResults, expectGpuExecution, type Session } from "./helpers.js";

test.describe.configure({ mode: "serial" });
let s: Session;
test.beforeAll(async ({ browser, tinygradVersion }) => { s = await openSession(browser, { tinygradVersion }); });
test.afterAll(async () => { await s?.close(); });

test("spec example: basic tensor addition", async () => {
  const r = await s.run("from tinygrad import Tensor\na = Tensor([1, 2, 3])\nb = Tensor([4, 5, 6])\nprint((a + b).numpy())");
  expect(r.ok, JSON.stringify(r.error)).toBe(true);
  expect(r.stdout.trim()).toBe("[5 7 9]");
  expect(r.gpu!.kernelsDispatched).toBeGreaterThan(0);
});

test("spec example: matrix multiplication", async () => {
  const r = await s.run("from tinygrad import Tensor\na = Tensor.randn(128, 128)\nb = Tensor.randn(128, 128)\nc = a @ b\nprint(c.shape)");
  expect(r.ok, JSON.stringify(r.error)).toBe(true);
  expect(r.stdout.trim()).toBe("(128, 128)");
});

// Strict tolerance for results computed from exact inputs. Values derived from Tensor.randn go through Box-Muller
// (log, cos); WGSL leaves transcendental precision implementation-defined (cos/sin: absolute error <= 2^-11 on
// [-pi, pi]), so those are compared with a looser, documented tolerance.
const STRICT = { rtol: 1e-4, atol: 1e-5 };
const TRANSCENDENTAL_RNG = { rtol: 2e-3, atol: 2e-3 };
for (const name of ["basic", "broadcast", "matmul"]) {
  test(`${name} ops match native tinygrad CPU`, async ({ tinygradVersion }) => {
    const r = await s.program(name);
    expectGpuExecution(r);
    const ref = reference(tinygradVersion)[name];
    for (const key of Object.keys(ref)) {
      if (key === "device") continue;
      const tol = key.startsWith("randn") ? TRANSCENDENTAL_RNG : STRICT;
      const maxErr = compareResults(r.result[key], ref[key], `${name}.${key}`, tol);
      test.info().annotations.push({ type: `max-abs-error ${key}`, description: maxErr.toExponential(2) });
    }
  });
}

test("GPU buffer transfers (sizes, dtypes, padding, assign)", async ({ tinygradVersion }) => {
  const r = await s.program("transfers");
  expectGpuExecution(r);
  for (const [k, v] of Object.entries(r.result)) if (k.startsWith("f32_") || k.startsWith("rt_")) expect(v, k).toBe(true);
  compareResults(r.result, reference(tinygradVersion).transfers, "transfers");
  expect(r.run.gpu!.bytesUploaded).toBeGreaterThan(1 << 20);
  expect(r.run.gpu!.bytesDownloaded).toBeGreaterThan(1 << 20);
});

test("results agree with tinygrad's PYTHON (CPU) device in the same browser", async () => {
  const r = await s.run(`
import numpy as np
from tinygrad import Tensor
a = np.linspace(-2, 2, 48, dtype=np.float32).reshape(6, 8); b = np.cos(np.arange(40, dtype=np.float32)).reshape(8, 5)
def f(dev):
  x, y = Tensor(a, device=dev), Tensor(b, device=dev)
  return ((x @ y).relu() + x.sum(axis=1, keepdim=True)).softmax(axis=1).numpy()
gpu, cpu = f("WEBGPU"), f("PYTHON")
print(float(np.abs(gpu - cpu).max()))`);
  expect(r.ok, JSON.stringify(r.error)).toBe(true);
  expect(Number(r.stdout.trim())).toBeLessThan(1e-5);
});
