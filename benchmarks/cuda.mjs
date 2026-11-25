#!/usr/bin/env node
// CUDA benchmark runner.
//
//   npm run build && npm run bench:cuda -- [--quick] [--label NAME] [--headed] [--skip-tinygrad] [--only REGEX]
//
// Phase 1 (benchmarks/cuda-bench.js, plain page): each library workload through CUDA→WGSL (optimized and direct
// compilation), CUDA→WASM (worker pool and one thread) and, where one exists, a handwritten WGSL kernel of the same
// algorithm. Phase 2 (benchmarks/cuda_tinygrad.py, tinygrad.js runtime): CUDA kernels on tinygrad tensors vs the
// equivalent tinygrad operations on WEBGPU and on the CPU (PYTHON) device. Every timed result is verified first.
// Results: benchmarks/results/cuda-<label>.{json,md}; open benchmarks/dashboard.html to browse them.
// Extra Chromium flags: TGW_CHROMIUM_ARGS="--use-angle=vulkan ..." (e.g. to select a hardware adapter).
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const arg = (name, def) => { const i = process.argv.indexOf(name); return i === -1 ? def : process.argv[i + 1]; };
const quick = process.argv.includes("--quick");
const label = arg("--label", `${new Date().toISOString().slice(0, 10)}-${os.hostname().slice(0, 12)}`);
const port = 8791;

const server = spawn(process.execPath, [path.join(root, "scripts/serve.mjs")], { env: { ...process.env, PORT: String(port), COI: "1" }, stdio: "ignore" });
await new Promise((r) => setTimeout(r, 500));
const browser = await chromium.launch({
  headless: !process.argv.includes("--headed"),
  args: ["--enable-unsafe-webgpu", "--enable-features=WebGPUDeveloperFeatures", ...(process.env.TGW_CHROMIUM_ARGS ?? "").split(" ").filter(Boolean)],
});
const out = { label, date: new Date().toISOString(), quick, host: { cpu: os.cpus()[0]?.model, cores: os.cpus().length, platform: `${os.platform()} ${os.release()}` }, browser: browser.version() };
try {
  const page = await browser.newPage();
  page.on("console", (m) => { if (m.type() === "log") console.log(m.text()); });
  await page.goto(`http://127.0.0.1:${port}/benchmarks/cuda-bench.html`);
  await page.waitForFunction(() => window.benchReady === true);
  page.setDefaultTimeout(0);
  const suite = await page.evaluate((q) => window.runSuite({ quick: q[0], only: q[1], workerURL: "/dist/cuda-worker.js", log: (s) => console.log(s) }), [quick, arg("--only", null)]);
  out.env = suite.env;
  out.kernels = suite.results;
  await page.close();

  if (!process.argv.includes("--skip-tinygrad")) {
    const p2 = await browser.newPage();
    p2.on("console", (m) => { if (m.type() === "log") console.log(m.text()); });
    await p2.goto(`http://127.0.0.1:${port}/tests/fixtures/index.html`);
    await p2.waitForFunction(() => window.tgReady === true);
    p2.setDefaultTimeout(0);
    const src = fs.readFileSync(path.join(root, "benchmarks/cuda_tinygrad.py"), "utf8");
    const r = await p2.evaluate(async ([c]) => {
      const rt = await window.tg.createTinygrad({ pyodideIndexURL: window.PYODIDE });
      const res = await rt.run(c);
      return { ...res, info: rt.getDeviceInfo() };
    }, [`BENCH_QUICK = ${quick ? 1 : 0}\nBENCH_DEVICES = "WEBGPU,PYTHON"\n` + src]);
    if (!r.ok) throw new Error(r.error.traceback || r.error.message);
    out.tinygrad = JSON.parse(r.stdout.split("\n").find((l) => l.startsWith("BENCH_JSON:")).slice(11));
    out.tinygrad.runtime = { tinygradVersion: r.info.tinygradVersion, pyodideVersion: r.info.pyodideVersion };
    await p2.close();
  }
} finally {
  await browser.close();
  server.kill();
}

// ---------------------------------------------------------------- report
const f = (v, d = 2) => (v == null ? "–" : Number(v).toFixed(d));
const g = out.env.webgpu;
let md = `# CUDA benchmarks: ${label}\n\n- date: ${out.date}${quick ? " (quick mode)" : ""}\n- browser: Chromium ${out.browser}, cross-origin isolated: ${out.env.isolated}\n`;
md += `- WebGPU: ${g ? `${g.name} (software fallback adapter: ${g.isFallbackAdapter}; timestamps: ${g.features.timestamps}; subgroups: ${JSON.stringify(g.features.subgroupSize)})` : `unavailable (${out.env.webgpuError})`}\n`;
md += `- CPU: ${out.host.cpu} × ${out.host.cores} (${out.host.platform}); WASM pool: ${out.env.cpuPool.threads} threads (scalar WebAssembly, no SIMD)\n\n`;
md += "Times in ms per launch. **exec**: steady state, launches queued back to back (median). **latency**: one launch + synchronize. **compile**: extra time of the first launch (lowering, codegen, pipeline creation; ~0 when an earlier workload already compiled the same kernel for the same block size). **gpu**: GPU timestamp (when supported). Results are verified against host references (max rel. error) before being reported.\n\n";
const workloadsSeen = [...new Set(out.kernels.map((r) => r.workload))];
md += "| workload | variant | exec | latency | gpu | compile | upload | download | GFLOP/s | GB/s | max rel err |\n|---|---|---|---|---|---|---|---|---|---|---|\n";
for (const w of workloadsSeen) for (const r of out.kernels.filter((x) => x.workload === w)) {
  if (r.error) { md += `| ${w} | ${r.variant} | error: ${r.error.replace(/\|/g, "/").slice(0, 120)} |||||||||\n`; continue; }
  md += `| ${w} | ${r.variant} | ${f(r.execMs, 3)} | ${f(r.latencyMs)} | ${f(r.gpuMs, 3)} | ${f(r.compileMs, 1)} | ${f(r.uploadMs)} | ${f(r.downloadMs)} | ${r.flops ? f(r.flops / r.execMs / 1e6, 1) : "–"} | ${r.bytes ? f(r.bytes / r.execMs / 1e6, 1) : "–"} | ${r.ok ? "" : "**FAIL** "}${r.maxRelErr.toExponential(1)} |\n`;
}
if (out.tinygrad) {
  md += `\n## CUDA kernels on tinygrad tensors vs tinygrad operations (tinygrad ${out.tinygrad.runtime.tinygradVersion})\n\nPer call: realize + synchronize, median; both produce the same result (checked).\n\n| device | workload | tinygrad.cuda kernel | tinygrad op | ratio (op / kernel) | max rel err |\n|---|---|---|---|---|---|\n`;
  for (const r of out.tinygrad.runs) md += `| ${r.device} | ${r.workload} | ${f(r.cuda_ms, 3)} | ${f(r.tinygrad_ms, 3)} | ${r.ok ? f(r.tinygrad_ms / r.cuda_ms) + "×" : "**results differ**"} | ${r.maxRelErr.toExponential(1)} |\n`;
}
md += "\nA ratio above 1× means the CUDA kernel was faster on this machine; it says nothing about other hardware. On a software WebGPU adapter (SwiftShader) GPU numbers measure a CPU emulation of a GPU.\n";
fs.mkdirSync(path.join(root, "benchmarks/results"), { recursive: true });
fs.writeFileSync(path.join(root, `benchmarks/results/cuda-${label}.json`), JSON.stringify(out, null, 1));
fs.writeFileSync(path.join(root, `benchmarks/results/cuda-${label}.md`), md);
const idx = path.join(root, "benchmarks/results/index.json");
const list = fs.existsSync(idx) ? JSON.parse(fs.readFileSync(idx, "utf8")) : [];
if (!list.includes(`cuda-${label}.json`)) list.push(`cuda-${label}.json`);
fs.writeFileSync(idx, JSON.stringify(list, null, 1));
console.log(md);
