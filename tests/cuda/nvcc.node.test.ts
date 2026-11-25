/**
 * The kernel library is plain CUDA: every source must also compile with NVIDIA's nvcc, not only with tinygrad.js's
 * compiler (which accepts some things nvcc doesn't, e.g. helper_math.h-style float4 operators without the header).
 * Opt-in: set TGJS_NVCC to an nvcc command, e.g. TGJS_NVCC=/usr/local/cuda/bin/nvcc, or on Windows with CUDA in WSL
 * TGJS_NVCC="wsl /usr/local/cuda/bin/nvcc". Skipped otherwise.
 */
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { test, expect } from "@playwright/test";
import { sources } from "../../src/cuda/kernels/index.js";

const nvcc = (process.env.TGJS_NVCC ?? "").split(" ").filter(Boolean);
// __dp4a, warp shuffles with masks etc. need a real target (nvcc's default, sm_52, lacks some); override with your own -arch
const arch = nvcc.some((a) => a.startsWith("-arch") || a.startsWith("--gpu-architecture")) ? [] : ["-arch=sm_70"];

test.describe("kernel library compiles with nvcc", () => {
  test.skip(!nvcc.length, "set TGJS_NVCC to an nvcc command to run");
  const dir = path.resolve("test-results/nvcc");
  test.beforeAll(() => fs.mkdirSync(dir, { recursive: true }));
  for (const [name, src] of Object.entries(sources)) {
    test(name, () => {
      fs.writeFileSync(path.join(dir, `${name}.cu`), src);
      let out = "";
      try {
        // relative paths, so a `wsl ...` command resolves them from the inherited working directory
        execFileSync(nvcc[0], [...nvcc.slice(1), ...arch, "-std=c++17", "-c", `${name}.cu`, "-o", `${name}.o`], { cwd: dir, stdio: "pipe" });
      } catch (e) {
        out = String((e as { stderr?: Buffer }).stderr ?? e);
      }
      expect(out).toBe("");
    });
  }
});
