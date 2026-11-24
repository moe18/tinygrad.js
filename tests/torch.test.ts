import fs from "node:fs";
import path from "node:path";
import { test, expect, openSession, torchReference, compareResults, expectGpuExecution, type Session } from "./helpers.js";

// The PyTorch-compatible shim (`import torch`, docs/TORCH.md). tests/torch_programs/*.py are plain PyTorch scripts; their results under
// real PyTorch are in tests/reference/torch.json (tests/reference/generate_torch.py). Here the same scripts run unmodified in the
// browser on WebGPU and must give the same numbers.
let s: Session;
test.beforeAll(async ({ browser, tinygradVersion }) => { s = await openSession(browser, { tinygradVersion }); });
test.afterAll(async () => { await s?.close(); });

// float32 results; training programs accumulate rounding differences over their steps (same tolerances as the native check)
const TOL = { rtol: 2e-3, atol: 2e-4 };
const PROGRAMS: [name: string, what: string][] = [
  ["tensor_basics", "tensor creation, arithmetic, indexing, reductions, in-place operations, printing"],
  ["autograd", "backward, grad accumulation, no_grad, detach, retain_graph, autograd.grad, higher-order gradients"],
  ["polynomial", "the 'Learning PyTorch with Examples' polynomial fit, five ways"],
  ["nn_layers", "nn layers, losses, attention, transformer and recurrent layers"],
  ["optimizers", "SGD/Adam/AdamW/RMSprop/Adagrad/Adadelta update rules, parameter groups, LR schedulers, gradient clipping"],
  ["classifier", "Quickstart-style classifier with Dataset, DataLoader and train/test loops"],
  ["cnn", "the CIFAR-10 tutorial CNN and a small ResNet with BatchNorm"],
  ["gpt", "a nanoGPT-style transformer: training with AdamW parameter groups, LR schedule, clipping, generation"],
  ["patterns", "GAN, DQN target network / EMA, policy gradient, fine-tuning, hooks and module introspection"],
  ["save_load", "torch.save / torch.load round trips and loading a checkpoint written by real PyTorch"],
  ["snippets", "learner-style snippets: regression from scratch, positional encoding, VAE loss, char LSTM, k-means, 40 one-liners"],
];

for (const [name, what] of PROGRAMS) {
  test(`torch: ${name} matches real PyTorch (${what})`, async ({ tinygradVersion }) => {
    test.skip(tinygradVersion === "0.12.0", "the torch shim needs tinygrad >= 0.14");
    test.setTimeout(900_000);
    expect(s.info.torchShim).toBe(true);
    const ref = torchReference();
    const r = await s.program(name, "torch_programs");
    expectGpuExecution(r);
    const maxErr = compareResults(r.result, ref.results[name], name, TOL);
    test.info().annotations.push({ type: "result", description:
      `PyTorch ${ref.torch}; max abs error ${maxErr.toExponential(2)}; ${r.run.gpu!.kernelsDispatched} kernels; ${(r.run.durationMs / 1000).toFixed(1)} s` });
  });
}

test("torch: run_training runs a PyTorch-style step under TinyJit with the same losses as plain execution", async ({ tinygradVersion }) => {
  test.skip(tinygradVersion === "0.12.0", "the torch shim needs tinygrad >= 0.14");
  test.setTimeout(600_000);
  const code = fs.readFileSync(path.resolve(import.meta.dirname, "torch_programs/_run_training.py"), "utf8");
  const r = await s.run(code);
  expect(r.ok, r.error?.traceback).toBe(true);
  const res = JSON.parse(r.stdout.split("\n").find((l) => l.startsWith("RESULT_JSON:"))!.slice("RESULT_JSON:".length)) as Record<string, any>;
  for (const k of ["sgd", "adam", "sched"]) {
    expect(res[k].plain_jit, k).toBe(false);
    expect(res[k].jit_used, `${k}: ${res[k].reason}`).toBe(true);
    compareResults(res[k].jit_losses, res[k].plain_losses, `${k}: jit vs plain losses`, { rtol: 1e-4, atol: 1e-6 });
    compareResults(res[k].jit_params, res[k].plain_params, `${k}: jit vs plain parameters`, { rtol: 1e-4, atol: 1e-6 });
    expect(res[k].jit_losses.at(-1), `${k} trains`).toBeLessThan(res[k].jit_losses[0] * 0.9);
  }
  expect(res.sgd.ms_jit, `jit ${res.sgd.ms_jit} ms/step vs plain ${res.sgd.ms_plain} ms/step`).toBeLessThan(res.sgd.ms_plain);
  test.info().annotations.push({ type: "summary", description: res.sgd.summary });
});

test("torch: the shim identifies itself, and unsupported APIs fail with a clear error", async ({ tinygradVersion }) => {
  if (tinygradVersion === "0.12.0") {
    expect(s.info.torchShim).toBe(false);
    expect(s.info.diagnostics.join(" ")).toMatch(/`import torch` is unavailable: the tinygrad.js torch shim needs tinygrad >= 0.14/);
    const r = await s.run("import torch");
    expect(r.ok).toBe(false);
    expect(r.error!.type).toBe("ModuleNotFoundError");
    return;
  }
  const r = await s.run(`
import torch
print(torch.__version__, torch.__tinygrad_js_shim__, torch.cuda.is_available(), torch.tensor([1.0]).device)
for what in ("class F(torch.autograd.Function): pass", "torch.linalg.eigh(torch.eye(2))", "torch.eye(2).register_hook(print)", "torch.nn.Transformer()"):
  try: exec(what); print("no error:", what)
  except (NotImplementedError, AttributeError) as e: print(type(e).__name__, "|", e)
try: import torchvision
except ImportError as e: print("ImportError |", e)
`);
  expect(r.ok, r.error?.traceback).toBe(true);
  const lines = r.stdout.trim().split("\n");
  expect(lines[0]).toBe("0.1.0+tinygrad.js True False webgpu");
  expect(lines).toHaveLength(6);
  for (const l of lines.slice(1, 5)) expect(l).toMatch(/^(NotImplementedError|AttributeError) \| .*tinygrad\.js torch shim/);
  expect(lines[5]).toMatch(/^ImportError \| No module named 'torchvision'/);
});

test("torch: can be turned off", async ({ browser, tinygradVersion }) => {
  test.skip(tinygradVersion === "0.12.0", "the torch shim needs tinygrad >= 0.14");
  const off = await openSession(browser, { tinygradVersion, torch: false });
  try {
    expect(off.info.torchShim).toBe(false);
    const r = await off.run("import torch");
    expect(r.ok).toBe(false);
    expect(r.error!.type).toBe("ModuleNotFoundError");
  } finally { await off.close(); }
});
