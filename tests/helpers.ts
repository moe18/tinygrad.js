import fs from "node:fs";
import path from "node:path";
import { test as base, expect, type Browser, type Page } from "@playwright/test";
import type { CreateTinygradOptions, RunResult, RuntimeInfo } from "../src/types.js";

export interface TestOptions { tinygradVersion: "0.14.0" | "0.12.0" }
export const ISOLATED = "http://127.0.0.1:8780";      // COOP/COEP -> SharedArrayBuffer -> signal-based interrupt
export const NOT_ISOLATED = "http://127.0.0.1:8781";

export const test = base.extend<TestOptions>({ tinygradVersion: ["0.14.0", { option: true }] });
export { expect };

const root = path.resolve(import.meta.dirname, "..");
const prelude = fs.readFileSync(path.join(root, "tests/programs/_prelude.py"), "utf8");

/** Counts tinygrad kernels compiled for the PYTHON (CPU) device: GPU tests assert it stays 0. */
const INSTRUMENT = `
import tinygrad.runtime.ops_python as _tgw_op
_tgw_python_programs = [0]
_tgw_orig_init = _tgw_op.PythonProgram.__init__
def _tgw_counting_init(self, *a, **k):
  _tgw_python_programs[0] += 1
  return _tgw_orig_init(self, *a, **k)
_tgw_op.PythonProgram.__init__ = _tgw_counting_init
`;

export interface Session {
  page: Page;
  info: RuntimeInfo;
  run(code: string, opts?: { timeoutMs?: number }): Promise<RunResult>;
  /** Run tests/programs/<name>.py and return its RESULT_JSON plus GPU evidence. */
  program(name: string): Promise<{ result: Record<string, unknown>; run: RunResult; pythonDeviceKernels: number }>;
  close(): Promise<void>;
}

export async function openSession(browser: Browser, opts: CreateTinygradOptions & { tinygradVersion?: string } = {},
                                  origin = ISOLATED): Promise<Session> {
  const page = await browser.newPage();
  page.on("pageerror", (e) => console.log("[pageerror]", e.message));
  await page.goto(`${origin}/tests/fixtures/index.html`);
  await page.waitForFunction(() => (window as unknown as { tgReady?: boolean }).tgReady === true);
  const info = (await page.evaluate(async (o) => {
    const w = window as unknown as Record<string, any>;
    w.rt = await w.tg.createTinygrad({ pyodideIndexURL: w.PYODIDE, ...o });
    return w.rt.getDeviceInfo();
  }, opts)) as RuntimeInfo;
  const run = (code: string, o: { timeoutMs?: number } = {}) =>
    page.evaluate(([c, ro]) => (window as unknown as Record<string, any>).rt.run(c, ro), [code, o] as const) as Promise<RunResult>;
  const instrumented = await run(INSTRUMENT);
  if (!instrumented.ok) throw new Error(`instrumentation failed: ${JSON.stringify(instrumented.error)}`);
  return {
    page, info, run,
    async program(name) {
      const src = fs.readFileSync(path.join(root, "tests/programs", `${name}.py`), "utf8");
      const r = await run(prelude + "\n" + src);
      if (!r.ok) throw new Error(`${name} failed: ${r.error?.traceback || r.error?.message}\n${r.stdout}\n${r.stderr}`);
      const line = r.stdout.split("\n").find((l) => l.startsWith("RESULT_JSON:"));
      if (!line) throw new Error(`${name}: no RESULT_JSON in output:\n${r.stdout}`);
      const counter = await run("_tgw_python_programs[0]");
      return { result: JSON.parse(line.slice("RESULT_JSON:".length)), run: r, pythonDeviceKernels: Number(counter.result) };
    },
    async close() {
      await page.evaluate(() => (window as unknown as Record<string, any>).rt?.destroy()).catch(() => {});
      await page.close();
    },
  };
}

export function reference(version: string): Record<string, Record<string, unknown>> {
  const file = path.join(root, "tests/reference", `cpu-${version}.json`);
  return JSON.parse(fs.readFileSync(file, "utf8")).results;
}

interface ArrayValue { shape: number[]; dtype: string; data: number[] }
const isArray = (v: unknown): v is ArrayValue => typeof v === "object" && v !== null && "data" in v && "shape" in v;

/** Recursively compare a browser result with the native CPU reference. Returns the max abs error seen for floats. */
export function compareResults(got: unknown, want: unknown, where: string, tol = { rtol: 1e-4, atol: 1e-5 }): number {
  let maxErr = 0;
  if (isArray(want)) {
    expect(isArray(got), `${where}: expected an array`).toBe(true);
    const g = got as ArrayValue;
    expect(g.shape, `${where}: shape`).toEqual(want.shape);
    expect(g.dtype, `${where}: dtype`).toEqual(want.dtype);
    for (let i = 0; i < want.data.length; i++) {
      const a = g.data[i], b = want.data[i];
      if (Number.isNaN(b)) { expect(Number.isNaN(a), `${where}[${i}] NaN`).toBe(true); continue; }
      const err = Math.abs(a - b);
      maxErr = Math.max(maxErr, err);
      if (err > tol.atol + tol.rtol * Math.abs(b)) expect(a, `${where}[${i}] (rtol=${tol.rtol}, atol=${tol.atol})`).toBeCloseTo(b, 6);
    }
  } else if (Array.isArray(want)) {
    expect(Array.isArray(got), `${where}: expected list`).toBe(true);
    expect((got as unknown[]).length, `${where}: length`).toBe(want.length);
    want.forEach((w, i) => { maxErr = Math.max(maxErr, compareResults((got as unknown[])[i], w, `${where}[${i}]`, tol)); });
  } else if (typeof want === "number") {
    expect(typeof got, `${where}: type`).toBe("number");
    const err = Math.abs((got as number) - want);
    maxErr = err;
    if (err > tol.atol + tol.rtol * Math.abs(want)) expect(got, `${where} (rtol=${tol.rtol}, atol=${tol.atol})`).toBeCloseTo(want, 6);
  } else if (typeof want === "object" && want !== null) {
    for (const k of Object.keys(want)) {
      if (k === "device") continue;
      maxErr = Math.max(maxErr, compareResults((got as Record<string, unknown>)[k], (want as Record<string, unknown>)[k], `${where}.${k}`, tol));
    }
  } else {
    expect(got, where).toEqual(want);
  }
  return maxErr;
}

/** Assert GPU evidence for a run: kernels were dispatched through WebGPU and none were compiled for the CPU device. */
export function expectGpuExecution(r: { run: RunResult; pythonDeviceKernels: number; result: Record<string, unknown> }) {
  expect(r.result.device).toBe("WEBGPU");
  expect(r.run.gpu, "GPU counters").not.toBeNull();
  expect(r.run.gpu!.kernelsDispatched, "kernels dispatched through WebGPU").toBeGreaterThan(0);
  expect(r.pythonDeviceKernels, "kernels compiled for tinygrad's PYTHON (CPU) device").toBe(0);
}
