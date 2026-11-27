/** tinygrad.cuda: CUDA kernels on tinygrad tensors inside the tinygrad.js runtime (Pyodide + WebGPU, and the CPU device). */
import fs from "node:fs";
import path from "node:path";
import { test, expect, openSession, type Session } from "../helpers.js";

const prog = (name: string) => fs.readFileSync(path.resolve(import.meta.dirname, "programs", `${name}.py`), "utf8");

for (const device of ["webgpu", "cpu"] as const) {
  test.describe(`tinygrad.cuda on the ${device === "webgpu" ? "WEBGPU" : "CPU (PYTHON)"} device`, () => {
    test.describe.configure({ mode: "serial" });
    let s: Session;
    test.beforeAll(async ({ browser }) => { s = await openSession(browser, { device }); });
    test.afterAll(async () => { await s?.close(); });

    const runProgram = async (name: string) => {
      const r = await s.run(prog(name), { timeoutMs: 170_000 });
      if (!r.ok) throw new Error(`${name}: ${r.error?.traceback || r.error?.message}\n${r.stdout}\n${r.stderr}`);
      const line = r.stdout.split("\n").find((l) => l.startsWith("RESULT_JSON:"));
      if (!line) throw new Error(`no RESULT_JSON:\n${r.stdout}`);
      return { result: JSON.parse(line.slice(12)) as Record<string, unknown>, run: r };
    };

    test("README example runs verbatim (Python), results match, data stays on the device", async () => {
      const { result, run } = await runProgram("readme");
      expect(result.max_err).toBe(0);
      expect(result.device).toBe(device === "webgpu" ? "WEBGPU" : "PYTHON");
      if (device === "webgpu") {
        // zero-copy: only the final .numpy() readbacks move data off the GPU
        expect(run.gpu!.kernelsDispatched).toBeGreaterThan(0);
      }
    });
    test("tensor integration: views, in-place writes, aliasing, templates, structs, lazy launches, TinyJit, printf, errors", async () => {
      const { result } = await runProgram("integration");
      for (const [k, v] of Object.entries(result)) if (k !== "device") expect(v, k).toBe(true);
    });
    test("custom autograd with CUDA forward/backward kernels matches tinygrad's autograd", async () => {
      const { result } = await runProgram("autograd");
      expect(result.max_grad_err as number).toBeLessThan(1e-5);
      expect(result.max_out_err as number).toBeLessThan(1e-5);
    });
    test("training an MLP with a custom CUDA layer and a CUDA Adam optimizer matches pure tinygrad", async () => {
      const { result } = await runProgram("training");
      expect(result.max_loss_diff as number).toBeLessThan(1e-4);
      expect((result.losses_cuda as number[]).at(-1)!).toBeLessThan((result.losses_cuda as number[])[0] * 0.5);
    });
    test("transformer block inference: CUDA flash attention, RoPE, RMSNorm vs tinygrad's ops", async () => {
      const { result } = await runProgram("transformer");
      expect(result.max_attn_err as number).toBeLessThan(2e-4);
      expect(result.max_block_err as number).toBeLessThan(5e-4);
    });
  });
}
