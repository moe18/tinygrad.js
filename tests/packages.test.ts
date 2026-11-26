// loadPackagesOnImport: Pyodide packages are loaded the first time code imports them, with user code still offline.
// Needs one extra package in the self-hosted Pyodide subset:  npm run fetch:assets -- --pyodide six
import fs from "node:fs";
import path from "node:path";
import { test, expect, openSession } from "./helpers.js";

const vendor = path.resolve(import.meta.dirname, "..", "vendor", "pyodide");
const hasSix = fs.existsSync(vendor) && fs.readdirSync(vendor).some((f) => f.startsWith("six-"));

test.describe("loadPackagesOnImport", () => {
  test.skip(!hasSix, "run `npm run fetch:assets -- --pyodide six` to include the test package");

  test("loads an imported package on first use, and user code stays offline", async ({ browser }) => {
    const s = await openSession(browser, { device: "cpu", loadPackagesOnImport: true });
    try {
      const statuses = await s.page.evaluate(async () => {
        const w = window as unknown as Record<string, any>;
        const seen: string[] = [];
        const rt = await w.tg.createTinygrad({ pyodideIndexURL: w.PYODIDE, device: "cpu", loadPackagesOnImport: true, onStatus: (m: string) => seen.push(m) });
        const r = await rt.run("import six\nsix.__name__");
        await rt.destroy();
        return { seen, ok: r.ok, result: r.result };
      });
      expect(statuses.ok).toBe(true);
      expect(statuses.result).toBe("'six'");
      expect(statuses.seen.some((m) => /Loading six/.test(m))).toBe(true);

      const r = await s.run("import six\nsix.moves.range(3)[-1]");
      expect(r.ok, JSON.stringify(r.error)).toBe(true);
      expect(r.result).toBe("2");
      // a second run does not load anything again and still works
      expect((await s.run("six.PY3")).result).toBe("True");

      // the package loader may fetch; user code may not
      const net = await s.run("import js\njs.fetch");
      expect(net.ok).toBe(false);
      expect(net.error!.message).toMatch(/network access \(fetch\) is disabled/);

      // a module that is not a Pyodide package fails in Python as usual
      const missing = await s.run("import not_a_real_module_xyz");
      expect(missing.error!.type).toBe("ModuleNotFoundError");
    } finally { await s.close(); }
  });

  test("is off by default", async ({ browser }) => {
    const s = await openSession(browser, { device: "cpu" });
    try {
      const r = await s.run("import six");
      expect(r.ok).toBe(false);
      expect(r.error!.type).toBe("ModuleNotFoundError");
    } finally { await s.close(); }
  });
});
