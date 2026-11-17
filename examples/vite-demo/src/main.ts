import { createTinygrad, checkSupport, type TinygradRuntime, type RuntimeInfo, type RunResult } from "tinygrad.js";
import { EXAMPLES } from "./examples";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const code = $<HTMLTextAreaElement>("code"), out = $<HTMLPreElement>("console");
const runBtn = $<HTMLButtonElement>("run"), stopBtn = $<HTMLButtonElement>("stop"), state = $("state");
const examples = $<HTMLSelectElement>("examples");

for (const name of Object.keys(EXAMPLES)) examples.add(new Option(name, name));
const show = (name: string) => { code.value = EXAMPLES[name]; examples.value = name; };
show(Object.keys(EXAMPLES)[0]);
examples.onchange = () => show(examples.value);
$("clear").onclick = () => { out.textContent = ""; };

function print(text: string, cls?: "err" | "sys") {
  const atBottom = out.scrollTop + out.clientHeight >= out.scrollHeight - 4;
  if (cls) { const s = document.createElement("span"); s.className = cls; s.textContent = text; out.append(s); }
  else out.append(text);
  if (atBottom) out.scrollTop = out.scrollHeight;
}

function dl(el: HTMLElement, rows: [string, string][]) {
  el.replaceChildren(...rows.flatMap(([k, v]) => {
    const dt = document.createElement("dt"); dt.textContent = k;
    const dd = document.createElement("dd"); dd.textContent = v;
    return [dt, dd];
  }));
}

function showDevice(info: RuntimeInfo) {
  const g = info.gpu, t = info.timings;
  dl($("device-info"), [
    ["tinygrad device", info.tinygradDevice],
    ["adapter", g ? `${g.vendor || "?"} ${g.architecture || ""} ${g.description || ""}`.trim() : "none (CPU)"],
    ["fallback adapter", g ? String(g.isFallbackAdapter) : "–"],
    ["features", g ? g.features.join(", ") : "–"],
    ["max buffer", g ? `${(g.limits.maxBufferSize / 2 ** 20).toFixed(0)} MiB` : "–"],
    ["versions", `tinygrad ${info.tinygradVersion} · Pyodide ${info.pyodideVersion} · Python ${info.pythonVersion}`],
    ["sync / stop", `${info.syncMode ?? "–"} / ${info.interruptMode}`],
    ["init", `${(t.totalMs / 1000).toFixed(2)} s (Pyodide ${t.pyodideMs.toFixed(0)} ms, packages ${t.packagesMs.toFixed(0)} ms, WebGPU ${t.webgpuMs.toFixed(0)} ms, import ${t.tinygradImportMs.toFixed(0)} ms)`],
    ...info.diagnostics.map((d, i): [string, string] => [i ? "" : "notes", d]),
  ]);
}

function showRun(r: RunResult, wallMs: number) {
  dl($("run-info"), [
    ["status", r.status + (r.error ? ` (${r.error.type})` : "")],
    ["time", `${r.durationMs.toFixed(1)} ms in Python, ${wallMs.toFixed(1)} ms wall`],
    ...(r.gpu ? [
      ["GPU kernels", `${r.gpu.kernelsDispatched} dispatched, ${r.gpu.kernelsCompiled} compiled`],
      ["GPU transfers", `${(r.gpu.bytesUploaded / 1024).toFixed(1)} KiB up, ${(r.gpu.bytesDownloaded / 1024).toFixed(1)} KiB down`],
    ] as [string, string][] : []),
  ]);
}

let rt: TinygradRuntime | null = null;
let running = false;
const setRunning = (v: boolean) => { running = v; runBtn.disabled = v || !rt; stopBtn.disabled = !v; state.textContent = v ? "running…" : "ready"; };

async function run() {
  if (!rt || running) return;
  setRunning(true);
  print(`>>> run ${examples.value && EXAMPLES[examples.value] === code.value ? `"${examples.value}"` : "code"}\n`, "sys");
  const t0 = performance.now();
  try {
    const r = await rt.run(code.value, { onStdout: (t) => print(t), onStderr: (t) => print(t, "err") });
    if (r.error) print((r.error.traceback || `${r.error.type}: ${r.error.message}`) + "\n", "err");
    else if (r.result !== undefined) print(r.result + "\n");
    showRun(r, performance.now() - t0);
  } catch (e) {
    print(String(e) + "\n", "err");
  } finally {
    setRunning(false);
  }
}

runBtn.onclick = run;
stopBtn.onclick = async () => {
  if (!rt) return;
  state.textContent = "stopping…";
  const mode = await rt.interrupt();
  if (mode === "terminate") print("worker terminated and restarted (Python state was reset)\n", "sys");
};
code.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); void run(); }
  else if (e.key === "Tab") {
    e.preventDefault();
    const { selectionStart: s, selectionEnd: en } = code;
    code.setRangeText("  ", s, en, "end");
  }
});

(async () => {
  const support = await checkSupport();
  if (support.reason) print(`note: ${support.reason}\n`, "sys");
  const device = support.webgpu && support.adapter && support.jspi ? "webgpu" : "auto";
  try {
    rt = await createTinygrad({
      device,
      pyodideIndexURL: import.meta.env.VITE_PYODIDE_URL ?? new URL("pyodide/", document.baseURI).href,
      onStatus: (m) => { state.textContent = m + "…"; },
    });
    const info = rt.getDeviceInfo();
    showDevice(info);
    print(`ready: tinygrad ${info.tinygradVersion} on ${info.tinygradDevice} in ${(info.timings.totalMs / 1000).toFixed(1)} s\n`, "sys");
    (window as unknown as { tinygradRuntime: TinygradRuntime }).tinygradRuntime = rt;
    setRunning(false);
  } catch (e) {
    state.textContent = "failed";
    dl($("device-info"), [["error", String(e)]]);
    print(`initialization failed: ${e}\n`, "err");
  }
})();
