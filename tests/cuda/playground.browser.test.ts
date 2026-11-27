/** examples/cuda-playground: plain HTML page (no build step) served from the repository root. */
import { test, expect, ISOLATED, NOT_ISOLATED } from "../helpers.js";

test.describe("CUDA playground (examples/cuda-playground)", () => {
  test("runs examples, reports compile and runtime errors, shows IR/WGSL, compares backends, visualizations", async ({ page }) => {
    test.setTimeout(240_000);
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(`${ISOLATED}/examples/cuda-playground/?backend=webgpu`);
    await expect(page.locator("#run")).toBeEnabled({ timeout: 60_000 });
    await expect(page.locator("#status")).toContainText("WebGPU");
    for (const ex of ["vector add", "shared-memory reduction", "warp shuffle reduction", "tiled matmul", "histogram (atomics)", "softmax (one block per row)", "printf & debugging"]) {
      await page.locator("#example").selectOption(ex);
      await page.locator("#run").click();
      await expect(page.locator("#summary"), ex).toContainText("✓ ran", { timeout: 60_000 });
    }
    await expect(page.locator("#output")).toContainText("block 1 thread 3 -> global 7");
    await expect(page.locator("#output")).toContainText("0, 1, 4, 9, 16, 25, 36, 49");
    // shared-memory reduction of 65536 random values in [0, 1): the single output is ~32768
    await page.locator("#example").selectOption("shared-memory reduction");
    await page.locator("#run").click();
    await expect(page.locator("#summary")).toContainText("✓ ran");
    const sum = Number((await page.locator("#output").textContent())!.match(/out \[float32 × 1\]\n\s+([\d.]+)/)![1]);
    expect(Math.abs(sum - 32768)).toBeLessThan(300);
    // generated code
    await page.locator("[data-tab=code]").click();
    await expect(page.locator("#code")).toContainText("@compute @workgroup_size(256, 1, 1)");
    await page.locator("[data-tab=ir]").click();
    await expect(page.locator("#ir")).toContainText("barrier");
    // runtime error with bounds checks
    await page.locator("[data-tab=out]").click();
    await page.locator("#example").selectOption("out-of-bounds bug");
    await page.locator("#run").click();
    await expect(page.locator("#output")).toContainText(/cudaErrorIllegalAddress: buggy: out-of-bounds store to 'a' .* by thread \(32,0,0\)/);
    // compile error with a clickable line diagnostic
    await page.locator("#src").fill("__global__ void k(float* a) {\n  a[0] = undefined_name;\n}");
    await page.locator("#run").click();
    await expect(page.locator("#summary")).toContainText("compile error");
    await expect(page.locator(".diag.error")).toContainText(/line 2:\d+: .*undefined_name/);
    // benchmark & compare across WebGPU and both CPU variants
    await page.locator("#example").selectOption("vector add");
    await page.locator("#bench").click();
    await expect(page.locator("#summary table")).toBeVisible({ timeout: 120_000 });
    await expect(page.locator("#summary")).toContainText("CPU, worker pool");
    expect((await page.locator("#summary").textContent())!.match(/max rel\. diff 0\.0e\+0/g)?.length).toBe(3);
    // visualizations
    await page.locator("[data-tab=learn]").click();
    for (const v of ["index", "coalesce", "reduce", "tile"]) {
      await page.locator("#viz").selectOption(v);
      if (v === "reduce" || v === "tile") for (let i = 0; i < 5; i++) await page.locator("#viz-step").click();
      await expect(page.locator("#viz-info")).not.toBeEmpty();
    }
    await expect(page.locator("#viz-info")).toContainText("is complete");
    await page.locator("#viz").selectOption("coalesce");
    await page.locator("#viz-opt").selectOption("2");
    await expect(page.locator("#viz-info")).toContainText("32 memory transaction(s)");
    expect(errors).toEqual([]);
  });

  test("embedding: postMessage API sets code, runs it and returns results; CPU fallback without isolation", async ({ page }) => {
    test.setTimeout(120_000);
    await page.goto(`${NOT_ISOLATED}/tests/cuda/page.html`);
    const r = await page.evaluate(async () => {
      const f = document.createElement("iframe");
      f.src = "/examples/cuda-playground/?embed=1&backend=cpu";
      document.body.appendChild(f);
      await new Promise<void>((res) => addEventListener("message", function h(e) { if (e.data?.type === "tgjs-cuda:loaded") { removeEventListener("message", h); res(); } }));
      let id = 0;
      const call = (msg: Record<string, unknown>) => new Promise<Record<string, unknown>>((res) => {
        const my = ++id;
        addEventListener("message", function h(e) { if (e.data?.id === my) { removeEventListener("message", h); res(e.data); } });
        f.contentWindow!.postMessage({ ...msg, id: my }, "*");
      });
      const ready = await call({ type: "tgjs-cuda:set", source: "__global__ void axpb(const float* x, float* y, float a, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = a * x[i] + 1.0f; }",
        grid: [1], block: [64], args: { x: { n: 8, init: "arange" }, y: { n: 8, init: "zeros" }, a: 3, n: 8 } });
      const res = await call({ type: "tgjs-cuda:run" });
      const state = await call({ type: "tgjs-cuda:get" });
      return { ready, res, state };
    });
    expect(r.ready.kernels).toEqual(["axpb"]);
    expect(r.res).toMatchObject({ type: "tgjs-cuda:result", ok: true, kernel: "axpb" });
    expect((r.res.output as Record<string, number[]>).y).toEqual([1, 4, 7, 10, 13, 16, 19, 22]);
    expect((r.state.backend as { selected: string }).selected).toBe("wasm");
  });
});
