import { defineConfig } from "@playwright/test";
import type { TestOptions } from "./tests/helpers.js";

// Real WebGPU is required. Headless Chromium exposes an adapter only with --enable-unsafe-webgpu; on machines without a
// GPU it is SwiftShader (Dawn's software Vulkan): real WebGPU API + real WGSL compute shaders, executed on the CPU.
// Set TGW_CHROMIUM_ARGS to pass extra flags (e.g. "--use-angle=vulkan" on a GPU machine), or TGW_HEADED=1.
// TGW_CHANNEL=chromium uses full Chromium instead of the headless shell (on Windows only that one returns a hardware adapter;
// add TGW_CHROMIUM_ARGS=--force-high-performance-gpu on laptops with two GPUs).
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
    ...(process.env.TGW_CHANNEL ? { channel: process.env.TGW_CHANNEL } : {}),
    launchOptions: { args: ["--enable-unsafe-webgpu", ...extraArgs] },
    trace: "off",
  },
  projects: [
    // testIgnore is matched against the absolute path: anchored to tests/cuda/, because a bare /cuda\// also matched a
    // checkout folder like ".../tinygrad.js-cuda/" and silently ignored every test
    { name: "tinygrad-0.14.0", use: { tinygradVersion: "0.14.0" }, testIgnore: /[\\/]tests[\\/]cuda[\\/]/ },
    { name: "tinygrad-0.12.0", use: { tinygradVersion: "0.12.0" }, testIgnore: /init\.test\.ts|demo\.test\.ts|playground\.test\.ts|[\\/]tests[\\/]cuda[\\/]/ },
    // CUDA compiler + CPU backend in Node.js (no browser), and the browser suites (WebGPU, tinygrad.cuda in Pyodide, playground)
    { name: "cuda-node", testMatch: /cuda\/.*\.node\.test\.ts$/ },
    { name: "cuda-browser", testMatch: /cuda\/.*\.browser\.test\.ts$/, use: { tinygradVersion: "0.14.0" } },
  ],
  webServer: [
    { command: "node scripts/serve.mjs", env: { PORT: "8780", COI: "1" }, url: "http://127.0.0.1:8780/package.json", reuseExistingServer: true },
    { command: "node scripts/serve.mjs", env: { PORT: "8781", COI: "0" }, url: "http://127.0.0.1:8781/package.json", reuseExistingServer: true },
  ],
});
