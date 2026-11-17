import { test, expect, openSession, reference, compareResults, expectGpuExecution, type Session } from "./helpers.js";

test.describe.configure({ mode: "serial" });
let s: Session;
test.beforeAll(async ({ browser, tinygradVersion }) => { s = await openSession(browser, { tinygradVersion }); });
test.afterAll(async () => { await s?.close(); });

test("spec example: automatic differentiation (verbatim, requires_grad=True)", async ({ tinygradVersion }) => {
  const code = "from tinygrad import Tensor\nx = Tensor([2.0], requires_grad=True)\ny = (x * x).sum()\ny.backward()\nprint(x.grad.numpy())";
  const r = await s.run(code);
  if (tinygradVersion === "0.12.0") {
    expect(r.ok, JSON.stringify(r.error)).toBe(true);
    expect(r.stdout.trim()).toBe("[4.]");
  } else {
    // tinygrad 0.13+ removed the requires_grad argument upstream; this is upstream behaviour, not a tinygrad.js bug.
    expect(r.ok).toBe(false);
    expect(r.error!.type).toBe("TypeError");
    expect(r.error!.message).toContain("requires_grad");
  }
});

test("spec example: automatic differentiation (tinygrad >= 0.13 form)", async ({ tinygradVersion }) => {
  test.skip(tinygradVersion === "0.12.0", "tinygrad 0.12 only computes gradients for requires_grad=True tensors (covered by the verbatim test)");
  const r = await s.run("from tinygrad import Tensor\nx = Tensor([2.0])\ny = (x * x).sum()\ny.backward()\nprint(x.grad.numpy())");
  expect(r.ok, JSON.stringify(r.error)).toBe(true);
  expect(r.stdout.trim()).toBe("[4.]");
  expect(r.gpu!.kernelsDispatched).toBeGreaterThan(0);
});

test("gradients match native tinygrad CPU", async ({ tinygradVersion }) => {
  const r = await s.program("autograd");
  expectGpuExecution(r);
  compareResults(r.result, reference(tinygradVersion).autograd, "autograd", { rtol: 1e-4, atol: 1e-6 });
});
