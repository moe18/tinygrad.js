import { chromium } from "@playwright/test";
import { test, expect, openSession, NOT_ISOLATED, ISOLATED, type Session } from "./helpers.js";

test.describe.configure({ mode: "serial" });

test.describe("lifecycle (cross-origin isolated page)", () => {
  let s: Session;
  test.beforeAll(async ({ browser }) => { s = await openSession(browser); });
  test.afterAll(async () => { await s?.close(); });

  test("Pyodide initializes and tinygrad imports", async () => {
    expect(s.info.pyodideVersion).toBe("314.0.8");
    expect(s.info.pythonVersion).toMatch(/^3\.14/);
    expect(s.info.tinygradVersion).toBe("0.14.0");
    expect(s.info.crossOriginIsolated).toBe(true);
    expect(s.info.interruptMode).toBe("signal");
    for (const k of ["pyodideMs", "packagesMs", "webgpuMs", "tinygradImportMs", "totalMs"] as const)
      expect(s.info.timings[k], k).toBeGreaterThan(0);
    const r = await s.run("import tinygrad, sys\nprint(tinygrad.__file__.startswith(sys.prefix) or tinygrad.__file__)");
    expect(r.ok).toBe(true);
  });

  test("repeated executions share one Python environment", async () => {
    expect((await s.run("from tinygrad import Tensor\ncounter = Tensor([0.0])")).ok).toBe(true);
    for (let i = 1; i <= 5; i++) {
      const r = await s.run("counter = counter + 1\nprint(counter.item())");
      expect(r.stdout.trim()).toBe(`${i}.0`);
    }
    const r = await s.run("counter.item()");
    expect(r.result).toBe("5.0");
  });

  test("stdout and stderr are captured and streamed", async () => {
    const r = await s.page.evaluate(async () => {
      const chunks: string[] = [];
      const res = await (window as any).rt.run("import sys\nprint('out1')\nprint('err1', file=sys.stderr)\nprint('no newline', end='')",
        { onStdout: (t: string) => chunks.push(t) });
      return { res, chunks };
    });
    expect(r.res.stdout).toBe("out1\nno newline");
    expect(r.res.stderr).toBe("err1\n");
    expect(r.chunks.join("")).toBe("out1\nno newline");
  });

  test("syntax errors are reported", async () => {
    const r = await s.run("def broken(:\n  pass");
    expect(r.ok).toBe(false);
    expect(r.error!.type).toBe("SyntaxError");
  });

  test("interrupt() stops an infinite loop with KeyboardInterrupt and keeps state", async () => {
    await s.run("keep = 123");
    const r = await s.page.evaluate(async () => {
      const rt = (window as any).rt;
      const p = rt.run("while True:\n  pass");
      await new Promise((res) => setTimeout(res, 500));
      const mode = await rt.interrupt();
      return { mode, result: await p };
    });
    expect(r.mode).toBe("signal");
    expect(r.result.status).toBe("interrupted");
    expect(r.result.error.type).toBe("KeyboardInterrupt");
    expect((await s.run("keep")).result).toBe("123");
  });

  test("interrupt() stops a long-running GPU workload", async () => {
    const r = await s.page.evaluate(async () => {
      const rt = (window as any).rt;
      const p = rt.run("from tinygrad import Tensor\nx = Tensor.ones(64, 64).contiguous()\nwhile True:\n  x = (x @ x * 0.01).realize()");
      await new Promise((res) => setTimeout(res, 1500));
      const mode = await rt.interrupt();
      return { mode, result: await p };
    });
    expect(r.mode).toBe("signal");
    expect(r.result.status).toBe("interrupted");
    expect((await s.run("from tinygrad import Tensor\nprint((Tensor([1, 2]) * 2).tolist())")).stdout.trim()).toBe("[2, 4]");
  });

  test("timeoutMs interrupts runaway code", async () => {
    const r = await s.run("while True: pass", { timeoutMs: 500 });
    expect(r.ok).toBe(false);
    expect(r.error!.type).toBe("TimeoutError");
    expect((await s.run("1 + 1")).result).toBe("2");
  });

  test("network access is disabled by default", async () => {
    const r = await s.run("import js\njs.fetch");
    expect(r.ok).toBe(false);
    expect(r.error!.message).toMatch(/network access \(fetch\) is disabled/);
    const r2 = await s.run("import pyodide.http\nawait pyodide.http.pyfetch('/package.json')");
    expect(r2.ok).toBe(false);
  });

  test("reset() clears user globals", async () => {
    await s.run("zzz = 1");
    await s.page.evaluate(() => (window as any).rt.reset());
    expect((await s.run("zzz")).error!.type).toBe("NameError");
  });

  test("destroy() shuts the worker down; later runs reject", async () => {
    const r = await s.page.evaluate(async () => {
      const rt = (window as any).rt;
      await rt.destroy();
      try { await rt.run("1"); return "ran"; } catch (e) { return String(e); }
    });
    expect(r).toMatch(/destroyed/);
  });
});

test("network: 'allow' keeps fetch available", async ({ browser }) => {
  const s = await openSession(browser, { network: "allow" });
  try {
    const r = await s.run("import pyodide.http, json\nr = await pyodide.http.pyfetch('/package.json')\njson.loads(await r.string())['name']");
    expect(r.ok, JSON.stringify(r.error)).toBe(true);
    expect(r.result).toBe("'tinygrad-web'");
  } finally { await s.close(); }
});

test("without cross-origin isolation, interrupt() terminates and restarts the worker", async ({ browser }) => {
  const s = await openSession(browser, {}, NOT_ISOLATED);
  try {
    expect(s.info.interruptMode).toBe("terminate");
    await s.run("keep = 1");
    const r = await s.page.evaluate(async () => {
      const rt = (window as any).rt;
      const p = rt.run("while True: pass");
      await new Promise((res) => setTimeout(res, 300));
      const mode = await rt.interrupt();
      return { mode, result: await p };
    });
    expect(r.mode).toBe("terminate");
    expect(r.result.status).toBe("terminated");
    // fresh environment: old state is gone, GPU works again
    expect((await s.run("keep")).error!.type).toBe("NameError");
    const ok = await s.run("from tinygrad import Tensor\nprint((Tensor([1.0, 2.0]) * 2).numpy())");
    expect(ok.stdout.trim()).toBe("[2. 4.]");
    const t = await s.run("while True: pass", { timeoutMs: 400 });
    expect(t.status).toBe("terminated");
    expect(t.error!.type).toBe("TimeoutError");
  } finally { await s.close(); }
});

test.describe("unsupported WebGPU environment (browser without a WebGPU adapter)", () => {
  test("webgpu fails clearly, auto falls back to CPU, cpu works", async () => {
    const browser = await chromium.launch({ args: [] }); // no --enable-unsafe-webgpu: headless Chromium has no adapter
    try {
      const page = await browser.newPage();
      await page.goto(`${ISOLATED}/tests/fixtures/index.html`);
      await page.waitForFunction(() => (window as any).tgReady === true);
      const support = await page.evaluate(() => (window as any).tg.checkSupport());
      expect(support.adapter).toBe(false);
      expect(support.reason).toMatch(/adapter/);
      const err = await page.evaluate(() => (window as any).tg.createTinygrad({ pyodideIndexURL: (window as any).PYODIDE })
        .then(() => "created", (e: Error) => e.message));
      expect(err).toMatch(/WebGPU initialization failed: .*no GPU adapter/);
      await page.close();

      for (const device of ["auto", "cpu"] as const) {
        const s = await openSession(browser, { device });
        try {
          expect(s.info.tinygradDevice).toBe("PYTHON");
          expect(s.info.gpu).toBeNull();
          if (device === "auto") expect(s.info.diagnostics.join(" ")).toMatch(/WebGPU unavailable/);
          const r = await s.run("from tinygrad import Tensor, Device\nx = Tensor([2.0])\n(x * x).sum().backward()\nprint(Device.DEFAULT, (Tensor([1, 2, 3]) + Tensor([4, 5, 6])).numpy(), x.grad.numpy())");
          expect(r.ok, JSON.stringify(r.error)).toBe(true);
          expect(r.stdout.trim()).toBe("PYTHON [5 7 9] [4.]");
          expect(r.gpu).toBeNull();
        } finally { await s.close(); }
      }
    } finally { await browser.close(); }
  });
});
