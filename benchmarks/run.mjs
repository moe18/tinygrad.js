#!/usr/bin/env node
// Reproducible benchmark runner.
//
//   npm run build && npm run bench -- [--quick] [--label NAME] [--native /path/to/python] [--headed]
//
// Measures (median of runs) Pyodide/package/WebGPU/import init time in fresh workers, then runs benchmarks/bench.py in
// the browser on WEBGPU and on tinygrad.js's interpreted CPU device. With --native, also runs bench.py with native
// upstream tinygrad (DEV=CPU, clang) as a reference. Results: benchmarks/results/<label>.{json,md}.
// Extra Chromium flags: TGW_CHROMIUM_ARGS="--use-angle=vulkan ..." (e.g. to select a hardware adapter).
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn, execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const arg = (name, def) => { const i = process.argv.indexOf(name); return i === -1 ? def : process.argv[i + 1]; };
const quick = process.argv.includes("--quick");
const label = arg("--label", `${new Date().toISOString().slice(0, 10)}-${os.hostname().slice(0, 12)}`);
const nativePython = arg("--native", null);
const port = 8790;
const median = (xs) => { const s = [...xs].sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };

const server = spawn(process.execPath, [path.join(root, "scripts/serve.mjs")], { env: { ...process.env, PORT: String(port), COI: "1" }, stdio: "ignore" });
await new Promise((r) => setTimeout(r, 500));
const browser = await chromium.launch({
  headless: !process.argv.includes("--headed"),
  args: ["--enable-unsafe-webgpu", ...(process.env.TGW_CHROMIUM_ARGS ?? "").split(" ").filter(Boolean)],
});
const benchSrc = fs.readFileSync(path.join(root, "benchmarks/bench.py"), "utf8");
const out = { label, date: new Date().toISOString(), quick, host: { cpu: os.cpus()[0]?.model, cores: os.cpus().length, platform: `${os.platform()} ${os.release()}` } };

async function newRuntime(page) {
  await page.goto(`http://127.0.0.1:${port}/tests/fixtures/index.html`);
  await page.waitForFunction(() => window.tgReady === true);
  return page.evaluate(async () => { window.rt = await window.tg.createTinygrad({ pyodideIndexURL: window.PYODIDE }); return window.rt.getDeviceInfo(); });
}

try {
  // init timings: fresh page + worker each time (browser HTTP cache is disabled by the server's no-store header)
  const inits = [];
  for (let i = 0; i < (quick ? 2 : 5); i++) {
    const page = await browser.newPage();
    inits.push(await newRuntime(page));
    await page.close();
  }
  const keys = ["pyodideMs", "packagesMs", "webgpuMs", "tinygradImportMs", "totalMs"];
  out.init = Object.fromEntries(keys.map((k) => [k, median(inits.map((x) => x.timings[k]))]));
  out.runtime = { ...inits[0], timings: undefined };
  out.userAgent = await browser.version();
  console.log("init (median ms):", out.init);

  const page = await browser.newPage();
  await newRuntime(page);
  const code = `BENCH_QUICK = ${quick ? 1 : 0}\nBENCH_DEVICES = "WEBGPU,CPU"\n` + benchSrc;
  const r = await page.evaluate((c) => window.rt.run(c), code);
  if (!r.ok) throw new Error(r.error.traceback || r.error.message);
  out.browser = JSON.parse(r.stdout.split("\n").find((l) => l.startsWith("BENCH_JSON:")).slice(11));
  await page.close();

  if (nativePython) {
    const s = execFileSync(nativePython, [path.join(root, "benchmarks/bench.py")], { env: { ...process.env, DEV: "CPU", BENCH_QUICK: quick ? "1" : "0" }, cwd: os.tmpdir(), maxBuffer: 1 << 26 }).toString();
    out.native = JSON.parse(s.split("\n").find((l) => l.startsWith("BENCH_JSON:")).slice(11));
    out.native.version = execFileSync(nativePython, ["-c", "import importlib.metadata as m; print(m.version('tinygrad'))"]).toString().trim();
  }
} finally {
  await browser.close();
  server.kill();
}

// ---------------------------------------------------------------- report
const rows = [...out.browser.runs.map((x) => ({ ...x, where: x.device === "WEBGPU" ? "browser WebGPU" : "browser CPU (interpreted)" })),
  ...(out.native?.runs ?? []).map((x) => ({ ...x, where: `native CPU (tinygrad ${out.native.version})` }))];
const f = (v, d = 2) => (v === undefined ? "" : Number(v).toFixed(d));
const g = out.runtime.gpu ?? {};
let md = `# Benchmark: ${label}\n\n`;
md += `- date: ${out.date}${quick ? " (quick mode)" : ""}\n- browser: Chromium ${out.userAgent}\n- WebGPU adapter: ${g.vendor} / ${g.architecture} ${g.description ?? ""} (fallback adapter: ${g.isFallbackAdapter})\n`;
md += `- host CPU: ${out.host.cpu} × ${out.host.cores}, ${out.host.platform}\n- tinygrad ${out.runtime.tinygradVersion}, Pyodide ${out.runtime.pyodideVersion}, Python ${out.runtime.pythonVersion}\n\n`;
md += `## Initialization (median of ${quick ? 2 : 5} fresh workers)\n\n| Pyodide load | numpy + tinygrad wheel | WebGPU adapter+device | import tinygrad + backend | total |\n|---|---|---|---|---|\n`;
md += `| ${f(out.init.pyodideMs, 0)} ms | ${f(out.init.packagesMs, 0)} ms | ${f(out.init.webgpuMs, 0)} ms | ${f(out.init.tinygradImportMs, 0)} ms | ${f(out.init.totalMs, 0)} ms |\n\n`;
md += `WebGPU init runs concurrently with Pyodide loading, so it is not on the critical path.\n\n`;
const table = (title, header, filter, cols) => {
  const rs = rows.filter(filter);
  if (!rs.length) return;
  md += `## ${title}\n\n| where | ${header.join(" | ")} |\n|${"---|".repeat(header.length + 1)}\n`;
  for (const x of rs) md += `| ${x.where} | ${cols(x).join(" | ")} |\n`;
  md += "\n";
};
table("First kernel (compile + run) vs warm run", ["first", "warm"], (x) => x.bench === "first_kernel", (x) => [`${f(x.first_ms)} ms`, `${f(x.warm_ms, 3)} ms`]);
table("Elementwise `x*y+x` (realize + synchronize)", ["elements", "time", "GB/s"], (x) => x.bench === "elementwise", (x) => [x.n, `${f(x.ms, 3)} ms`, f(x.gbps, 3)]);
table("Matmul n×n (realize + synchronize, no readback)", ["n", "time", "GFLOP/s"], (x) => x.bench === "matmul", (x) => [x.n, `${f(x.ms, 3)} ms`, f(x.gflops, 3)]);
table("Readback (`.numpy()`)", ["bytes", "time", "MB/s"], (x) => x.bench === "readback", (x) => [x.bytes, `${f(x.ms, 3)} ms`, f(x.mbps, 1)]);
table("MLP training (2→64→1, batch 256, Adam)", ["TinyJit", "ms/step", "loss first → last"], (x) => x.bench === "mlp_train",
  (x) => [x.jit ? "yes" : "no", f(x.ms_per_step, 2), `${f(x.first_loss, 4)} → ${f(x.last_loss, 4)}`]);

const dir = path.join(root, "benchmarks/results");
fs.mkdirSync(dir, { recursive: true });
fs.writeFileSync(path.join(dir, `${label}.json`), JSON.stringify(out, null, 2));
fs.writeFileSync(path.join(dir, `${label}.md`), md);
console.log(md);
console.log(`wrote benchmarks/results/${label}.{json,md}`);
