import { defineConfig } from "@playwright/test";
import type { TestOptions } from "./tests/helpers.js";

// Real WebGPU is required. Headless Chromium exposes an adapter only with --enable-unsafe-webgpu; on machines without a
// GPU it is SwiftShader (Dawn's software Vulkan): real WebGPU API + real WGSL compute shaders, executed on the CPU.
// Set TGW_CHROMIUM_ARGS to pass extra flags (e.g. "--use-angle=vulkan" on a GPU machine), or TGW_HEADED=1.
const extraArgs = (process.env.TGW_CHROMIUM_ARGS ?? "").split(" ").filter(Boolean);

export default defineConfig<TestOptions>({
  testDir: "tests",
  testMatch: /.*\.test\.ts$/,
  timeout: 180_000,
  expect: { timeout: 30_000 },
  workers: 1,
  fullyParallel: false,
  reporter: [["list"], ["json", { outputFile: "test-results/results.json" }]],
  use: {
    headless: !process.env.TGW_HEADED,
    launchOptions: { args: ["--enable-unsafe-webgpu", ...extraArgs] },
    trace: "off",
  },
  projects: [
    { name: "tinygrad-0.14.0", use: { tinygradVersion: "0.14.0" } },
    { name: "tinygrad-0.12.0", use: { tinygradVersion: "0.12.0" }, testIgnore: /init\.test\.ts|demo\.test\.ts/ },
  ],
  webServer: [
    { command: "node scripts/serve.mjs", env: { PORT: "8780", COI: "1" }, url: "http://127.0.0.1:8780/package.json", reuseExistingServer: true },
    { command: "node scripts/serve.mjs", env: { PORT: "8781", COI: "0" }, url: "http://127.0.0.1:8781/package.json", reuseExistingServer: true },
  ],
});
