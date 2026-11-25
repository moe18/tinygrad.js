import path from "node:path";
import * as esbuild from "esbuild";
import type { Browser, Page } from "@playwright/test";

const root = path.resolve(import.meta.dirname, "../..");
let built: Promise<unknown> | null = null;
/** bundle tests/cuda/browser-entry.ts (once per test process) */
export function buildBundle(): Promise<unknown> {
  built ??= esbuild.build({ entryPoints: [path.join(root, "tests/cuda/browser-entry.ts")], bundle: true, format: "esm", outfile: path.join(root, ".cache/cuda-tests.js"), logLevel: "warning", sourcemap: true });
  return built;
}
export async function openCudaPage(browser: Browser, origin: string): Promise<Page> {
  await buildBundle();
  const page = await browser.newPage();
  page.on("pageerror", (e) => console.log("[pageerror]", e.message));
  await page.goto(`${origin}/tests/cuda/page.html`);
  await page.waitForFunction(() => (window as unknown as { cudaReady?: boolean }).cudaReady === true);
  return page;
}
