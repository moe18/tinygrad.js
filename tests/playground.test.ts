import { test, expect, ISOLATED, NOT_ISOLATED } from "./helpers.js";

// examples/playground is a plain HTML page (no build step) served from the repository root.
test.describe("playground (examples/playground)", () => {
  test("trains every dataset on WebGPU, paints points mid-training, runs Python, and Stop interrupts", async ({ page }) => {
    test.setTimeout(300_000);
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(`${ISOLATED}/examples/playground/`);
    await expect(page.locator("#train")).toBeEnabled({ timeout: 120_000 });
    await expect(page.locator("#device")).toContainText("WEBGPU");

    const step = async () => Number(await page.locator("#s-step").textContent());
    const train = async (minSteps: number) => {
      await page.locator("#train").click();
      await expect.poll(step, { timeout: 30_000 }).toBeLessThan(100);         // a dataset switch starts fresh weights
      await expect.poll(step, { timeout: 180_000 }).toBeGreaterThanOrEqual(minSteps);
      await page.locator("#train").click();                                     // pause
    };

    for (const ds of ["circle", "xor", "moons"]) {
      await page.locator(`[data-ds=${ds}]`).click();
      await train(500);
      const acc = parseFloat((await page.locator("#s-acc").textContent()) ?? "0");
      expect(acc, `${ds} accuracy`).toBeGreaterThan(95);
      await expect(page.locator("#status")).toContainText("training on WEBGPU");
    }
    await page.locator("[data-ds=spiral]").click();
    await page.locator("#width").selectOption("32");
    await page.locator("#depth").selectOption("3");
    await train(1500);
    expect(parseFloat((await page.locator("#s-acc").textContent()) ?? "0"), "spiral accuracy").toBeGreaterThan(85);
    await page.screenshot({ path: "test-results/playground.png", fullPage: true });

    // painting points while training rebuilds the data/JIT and keeps training
    await page.locator("#train").click();
    const box = (await page.locator("#board").boundingBox())!;
    for (let i = 0; i < 6; i++) await page.mouse.click(box.x + box.width * (0.1 + i * 0.03), box.y + box.height * 0.1);
    const before = await step();
    await expect.poll(step, { timeout: 60_000 }).toBeGreaterThan(before + 50);
    await page.locator("#train").click();

    // the Python box shares the playground's session
    const run = async (snippet: string) => {
      await page.locator("#snippets").selectOption(snippet);
      await page.locator("#run").click();
      await expect(page.locator("#run")).toBeEnabled({ timeout: 60_000 });
    };
    await run("Inspect the model");
    await run("Autograd");
    await page.locator("#wgsl").click();
    await expect(page.locator("#out")).toContainText("@compute", { timeout: 60_000 });
    const out = page.locator("#out");
    await expect(out).toContainText("layer 3: weight (1, 32)");
    await expect(out).toContainText("[12. 27.]");

    // Stop interrupts an infinite loop and the session survives
    await page.locator("#snippets").selectOption("Infinite loop (try Stop)");
    await page.locator("#run").click();
    await expect(page.locator("#stop")).toBeEnabled();
    await page.waitForTimeout(500);
    await page.locator("#stop").click();
    await expect(page.locator("#run")).toBeEnabled({ timeout: 30_000 });
    await expect(out).toContainText("KeyboardInterrupt");
    await run("Predict a point");
    await expect(out).toContainText(/\n[\d.]+(e-\d+)?\n\[completed/);
    expect(errors).toEqual([]);
  });

  // GitHub Pages can't send COOP/COEP, so there Stop terminates the worker; the page rebuilds its Python state.
  test("without cross-origin isolation, Stop restarts the worker and the playground recovers", async ({ page }) => {
    test.setTimeout(180_000);
    await page.goto(`${NOT_ISOLATED}/examples/playground/`);
    await expect(page.locator("#train")).toBeEnabled({ timeout: 120_000 });
    await expect(page.locator("#device")).toContainText("terminate");
    await page.locator("#snippets").selectOption("Infinite loop (try Stop)");
    await page.locator("#run").click();
    await expect(page.locator("#stop")).toBeEnabled();
    await page.waitForTimeout(500);
    await page.locator("#stop").click();
    await expect(page.locator("#run")).toBeEnabled({ timeout: 60_000 });
    await expect(page.locator("#out")).toContainText("[stopped via terminate]");
    await page.locator("#train").click();
    await expect.poll(async () => Number(await page.locator("#s-step").textContent()), { timeout: 60_000 }).toBeGreaterThan(50);
    await page.locator("#train").click();
    await expect(page.locator("#status")).toContainText("training on WEBGPU");
  });
});
