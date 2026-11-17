import fs from "node:fs";
import path from "node:path";
import { test, expect, openSession, reference, compareResults, expectGpuExecution, type Session } from "./helpers.js";

test.describe.configure({ mode: "serial" });
let s: Session;
test.beforeAll(async ({ browser, tinygradVersion }) => { s = await openSession(browser, { tinygradVersion }); });
test.afterAll(async () => { await s?.close(); });

test("MLP training with SGD and Adam: loss decreases on WebGPU and matches native CPU", async ({ tinygradVersion }) => {
  const r = await s.program("training");
  expectGpuExecution(r);
  const ref = reference(tinygradVersion).training as Record<string, number[] | number>;
  for (const opt of ["sgd", "adam"]) {
    const losses = r.result[`${opt}_losses`] as number[];
    expect(losses.length).toBe(60);
    expect(losses[59], `${opt}: final loss < 50% of initial`).toBeLessThan(losses[0] * 0.5);
    expect(r.result[`${opt}_acc`] as number, `${opt} accuracy`).toBeGreaterThan(0.85);
    // same seed -> same initial weights and data; float32 rounding differences may grow over 60 steps
    compareResults(losses[0], (ref[`${opt}_losses`] as number[])[0], `${opt} initial loss`, { rtol: 1e-5, atol: 1e-6 });
    compareResults(losses, ref[`${opt}_losses`], `${opt} losses`, { rtol: 1e-3, atol: 1e-4 });
  }
  test.info().annotations.push({ type: "kernels", description: String(r.run.gpu!.kernelsDispatched) });
});

test("examples/mlp_training.py (TinyJit) trains on WebGPU", async () => {
  const code = fs.readFileSync(path.resolve(import.meta.dirname, "../examples/python/mlp_training.py"), "utf8");
  const r = await s.run(code);
  expect(r.ok, r.error?.traceback).toBe(true);
  const m = r.stdout.match(/final loss ([\d.]+) \(initial ([\d.]+)\), accuracy ([\d.]+)/);
  expect(m, r.stdout).not.toBeNull();
  expect(Number(m![1])).toBeLessThan(Number(m![2]) * 0.5);
  expect(Number(m![3])).toBeGreaterThan(0.85);
  expect(r.gpu!.kernelsDispatched).toBeGreaterThan(100);
});
