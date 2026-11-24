import fs from "node:fs";
import path from "node:path";
import { test, expect, ISOLATED } from "./helpers.js";

const dist = path.resolve(import.meta.dirname, "../examples/vite-demo/dist/index.html");

test.describe("Vite demo (examples/vite-demo, production build)", () => {
  test.skip(!fs.existsSync(dist), "build the demo first: cd examples/vite-demo && npm install && npm run build");

  test("loads, runs every example on WebGPU, and Stop interrupts", async ({ page }) => {
    test.setTimeout(480_000);
    await page.goto(`${ISOLATED}/examples/vite-demo/dist/`);
    await expect(page.locator("#state")).toHaveText("ready", { timeout: 120_000 });
    await expect(page.locator("#device-info")).toContainText("WEBGPU");
    const names = await page.locator("#examples option").allTextContents();
    expect(names.length).toBeGreaterThanOrEqual(5);
    for (const name of names) {
      await page.locator("#examples").selectOption(name);
      await page.locator("#run").click();
      await expect(page.locator("#state")).toHaveText("ready", { timeout: 240_000 });
      await expect(page.locator("#run-info"), name).toContainText("completed");
    }
    const text = await page.locator("#console").textContent();
    expect(text).toContain("a + b = [5 7 9]");
    expect(text).toContain("dy/dx at x=2: [4.]");
    expect(text).toMatch(/final loss [\d.]+ \(initial [\d.]+\), accuracy/);
    expect(text).toContain("@compute"); // generated WGSL was printed
    expect(text).toMatch(/200 steps in [\d.]+s, TinyJit, loss [\d.]+ -> [\d.]+/); // run_training example
    expect(text).toMatch(/restored model agrees: True\nfinal accuracy 0\.[6-9]\d* on webgpu/); // PyTorch example (import torch)
    await page.screenshot({ path: "test-results/demo.png", fullPage: true });

    await page.locator("#code").fill("while True:\n  pass");
    await page.locator("#run").click();
    await expect(page.locator("#state")).toHaveText("running…");
    await page.waitForTimeout(500);
    await page.locator("#stop").click();
    await expect(page.locator("#state")).toHaveText("ready", { timeout: 30_000 });
    await expect(page.locator("#run-info")).toContainText("interrupted (KeyboardInterrupt)");
  });
});
