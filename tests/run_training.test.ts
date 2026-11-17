import { test, expect, openSession, compareResults, expectGpuExecution, type Session } from "./helpers.js";

test.describe.configure({ mode: "serial" });
let s: Session;
test.beforeAll(async ({ browser, tinygradVersion }) => { s = await openSession(browser, { tinygradVersion }); });
test.afterAll(async () => { await s?.close(); });

test("run_training: TinyJit matches plain execution, is faster, and guards against stale replays", async ({ tinygradVersion }) => {
  const r = await s.program("run_training");
  expectGpuExecution(r);
  const res = r.result as Record<string, any>;

  // 1. identical training with and without TinyJit
  expect(res.plain_jit_used).toBe(false);
  expect(res.jit_used, res.summary).toBe(true);
  compareResults(res.jit_losses, res.plain_losses, "jit vs plain losses", { rtol: 1e-5, atol: 1e-6 });
  expect(res.jit_losses[59]).toBeLessThan(res.jit_losses[0] * 0.5);
  expect(res.warmup_losses).toEqual(res.jit_losses.slice(0, 2));
  expect(res.ms_jit, `jit ${res.ms_jit} ms/step vs plain ${res.ms_plain} ms/step`).toBeLessThan(res.ms_plain);
  test.info().annotations.push({ type: "summary", description: res.summary });

  // 2. reading tensor data inside the step
  const [itemJit, itemReason, itemStep, itemDecreased] = res.item_fallback;
  expect(itemDecreased).toBe(true);
  if (tinygradVersion === "0.12.0") {
    expect(itemJit).toBe(true); // tinygrad 0.12 allows it (replay verified identical natively)
  } else {
    expect(itemJit).toBe(false);
    expect(itemReason).toMatch(/^JitError: cannot access tensor data during JIT capture/);
    expect(itemStep).toBe(3); // warmup 2 -> TinyJit call 1 at step 2 runs normally, call 2 (recording) raises
  }

  // 3./4. stale replays are detected before they happen
  expect(res.scalar_fallback).toEqual([false, "TinyJit would replay stale inputs: argument 2 changed from 1.0 to 0.5", 6]);
  expect(res.shape_fallback[0]).toBe(false);
  expect(res.shape_fallback[1]).toMatch(/argument 0 changed from shape \(128, 2\) .* to shape \(256, 2\)/);
  expect(res.shape_fallback[2]).toBe(8);

  // 5. per-step minibatches keep TinyJit
  expect(res.minibatch).toEqual([true, null, true]);
  expect(r.run.stdout).toMatch(/step    0  loss [\d.]+\nstep   20  loss [\d.]+\nstep   39  loss [\d.]+/);
});

test("run_training can be interrupted", async () => {
  const r = await s.page.evaluate(async () => {
    const rt = (window as any).rt;
    const p = rt.run(`
from tinygrad import Tensor, nn
from tinygrad.nn.optim import SGD
from tinygrad_js import run_training
w = Tensor([[0.5]]); opt = SGD([w], lr=0.01)
def step(x):
  opt.zero_grad(); loss = (x @ w).square().mean(); loss.backward(); opt.step(); return loss
run_training(step, 10_000_000, args=(Tensor.ones(4, 1).contiguous(),))`);
    await new Promise((res) => setTimeout(res, 1500));
    const mode = await rt.interrupt();
    return { mode, result: await p };
  });
  if (r.mode === "signal") expect(r.result.status).toBe("interrupted");
  else expect(r.result.status).toBe("terminated");
  expect((await s.run("1 + 1")).result).toBe("2");
});
