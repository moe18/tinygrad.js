/// <reference lib="webworker" />
/**
 * tinygrad.js worker: owns Pyodide, upstream tinygrad and the WebGPU device. One worker = one Python environment.
 * Started by Runtime (runtime.ts); all communication goes through the messages in protocol.ts.
 */
import { WebGPUBridge } from "./bridge.js";
import { PYTHON_FILES } from "./generated/python.js";
import type { FromWorker, InitConfig, RunReply, ToWorker } from "./protocol.js";
import type { GpuStats, GpuStatsDelta, InitTimings, RuntimeInfo } from "./types.js";

declare const self: DedicatedWorkerGlobalScope;

// Minimal structural types for the parts of the Pyodide API we use (avoids a hard dependency on the pyodide package).
interface PyProxyLike { toJs(opts?: { dict_converter?: (e: Iterable<[string, unknown]>) => unknown }): unknown; destroy(): void }
interface PyodideLike {
  version: string;
  FS: { mkdirTree(p: string): void; writeFile(p: string, d: string): void };
  loadPackage(names: string | string[], opts?: { messageCallback?: (m: string) => void; errorCallback?: (m: string) => void }): Promise<unknown>;
  runPython(code: string): unknown;
  runPythonAsync(code: string): Promise<unknown>;
  registerJsModule(name: string, mod: object): void;
  setStdout(o: { write: (b: Uint8Array) => number }): void;
  setStderr(o: { write: (b: Uint8Array) => number }): void;
  setInterruptBuffer(b: Int32Array): void;
  globals: { get(name: string): unknown };
}

const post = (m: FromWorker) => self.postMessage(m);
const status = (message: string) => post({ type: "status", message });
const LIB_DIR = "/home/pyodide/.tinygrad_js";

let py: PyodideLike | null = null;
let bridge: WebGPUBridge | null = null;
let info: RuntimeInfo | null = null;
let currentRun = 0;
let interruptBuffer: Int32Array | undefined;

function makeWriter(stream: "stdout" | "stderr") {
  const dec = new TextDecoder();
  return {
    write(buf: Uint8Array) {
      const text = dec.decode(buf, { stream: true });
      if (text) post({ type: "stream", id: currentRun, stream, text });
      return buf.length;
    },
  };
}

function hasJSPI(): boolean {
  return typeof (WebAssembly as unknown as { Suspending?: unknown }).Suspending === "function";
}

function lockDownNetwork(): void {
  // Best effort: remove network-capable APIs from the worker scope. Not a security boundary (see README).
  const names = ["fetch", "XMLHttpRequest", "WebSocket", "EventSource", "importScripts", "Worker", "SharedWorker", "WebTransport", "caches"];
  for (const name of names) {
    try {
      Object.defineProperty(self, name, {
        configurable: false, enumerable: false,
        get() { throw new Error(`network access (${name}) is disabled in this tinygrad.js runtime`); },
      });
    } catch { /* not present in this browser */ }
  }
}

async function init(config: InitConfig): Promise<RuntimeInfo> {
  const t0 = performance.now();
  const timings: InitTimings = { pyodideMs: 0, packagesMs: 0, webgpuMs: 0, tinygradImportMs: 0, totalMs: 0 };
  const diagnostics: string[] = [];
  let device: "WEBGPU" | "PYTHON" = config.device === "cpu" ? "PYTHON" : "WEBGPU";

  // Start WebGPU init concurrently with Pyodide's download/compile.
  const wantGpu = config.device !== "cpu";
  let gpuError: string | null = null;
  const gpuPromise = wantGpu
    ? (async () => {
        const s = performance.now();
        if (!hasJSPI()) throw new Error("this browser lacks WebAssembly JSPI (WebAssembly.Suspending), which tinygrad.js needs to read GPU results synchronously from Python");
        try { return await WebGPUBridge.create({ maxBufferMemory: config.maxGpuMemoryBytes }); }
        finally { timings.webgpuMs = performance.now() - s; }
      })().catch((e: unknown) => { gpuError = e instanceof Error ? e.message : String(e); return null; })
    : Promise.resolve(null);

  status("loading Pyodide");
  let s = performance.now();
  const indexURL = config.pyodideIndexURL.endsWith("/") ? config.pyodideIndexURL : config.pyodideIndexURL + "/";
  const mod = (await import(/* @vite-ignore */ /* webpackIgnore: true */ indexURL + "pyodide.mjs")) as { loadPyodide: (o: object) => Promise<PyodideLike> };
  py = await mod.loadPyodide({ indexURL, stdout: () => {}, stderr: () => {} });
  timings.pyodideMs = performance.now() - s;

  status("loading numpy and tinygrad");
  s = performance.now();
  const loadErrors: string[] = [];
  await py.loadPackage(["numpy", config.tinygradWheelURL, ...config.packages], { messageCallback: () => {}, errorCallback: (m) => loadErrors.push(m) });
  if (loadErrors.length) throw new Error("failed to load packages:\n" + loadErrors.join("\n"));
  timings.packagesMs = performance.now() - s;

  for (const [rel, src] of Object.entries(PYTHON_FILES)) {
    const full = `${LIB_DIR}/${rel}`;
    py.FS.mkdirTree(full.slice(0, full.lastIndexOf("/")));
    py.FS.writeFile(full, src);
  }

  status("initializing WebGPU");
  bridge = await gpuPromise;
  if (wantGpu && !bridge) {
    if (config.device === "auto") {
      device = "PYTHON";
      diagnostics.push(`WebGPU unavailable, using tinygrad's PYTHON (CPU) device: ${gpuError}`);
    } else throw new Error(`WebGPU initialization failed: ${gpuError}`);
  }
  if (bridge?.info().isFallbackAdapter)
    diagnostics.push("the WebGPU adapter is a software fallback adapter (e.g. SwiftShader): kernels run through WebGPU but on the CPU");
  if (device === "PYTHON") diagnostics.push("the PYTHON device is tinygrad's pure-Python interpreter: correct but very slow; keep tensors small");

  status("importing tinygrad");
  s = performance.now();
  if (bridge) py.registerJsModule("_tinygrad_js_bridge", bridge);
  py.runPython(`
import sys
sys.path.insert(0, ${JSON.stringify(LIB_DIR)})
import tinygrad_js, tinygrad_js._runner
tinygrad_js.install(${bridge ? "__import__('_tinygrad_js_bridge')" : "None"}, default_device=${JSON.stringify(device)})
import tinygrad
from tinygrad import Tensor, Device
assert Device.DEFAULT == ${JSON.stringify(device)}, Device.DEFAULT
`);
  if (bridge) {
    // open the device now so that errors surface during init rather than in the first user run
    await py.runPythonAsync("from tinygrad import Device; Device['WEBGPU']");
  }
  let torchShim = false;
  if (config.torch) {
    // `import torch` -> the PyTorch-compatible shim in tinygrad_js/_shims (only a sys.path entry; nothing is imported until used)
    const reason = py.runPython("tinygrad_js.torch_shim_unavailable_reason()") as string | undefined | null;
    if (reason) diagnostics.push(`\`import torch\` is unavailable: ${reason}`);
    else torchShim = Boolean(py.runPython("tinygrad_js.install_torch()"));
  }
  const patches = py.runPython("tinygrad_js.applied_patches()") as PyProxyLike;
  for (const p of patches.toJs() as string[]) diagnostics.push(`tinygrad.js patch: ${p}`);
  patches.destroy();
  timings.tinygradImportMs = performance.now() - s;

  if (config.interruptBuffer) {
    interruptBuffer = config.interruptBuffer;
    py.setInterruptBuffer(interruptBuffer);
    bridge?.setInterruptBuffer(interruptBuffer);
  }
  py.setStdout(makeWriter("stdout"));
  py.setStderr(makeWriter("stderr"));
  if (config.network === "none") lockDownNetwork();

  const versions = py.runPython(
    "import sys, importlib.metadata as _md; (_md.version('tinygrad'), sys.version.split()[0])",
  ) as PyProxyLike;
  const [tinygradVersion, pythonVersion] = versions.toJs() as [string, string];
  versions.destroy();
  timings.totalMs = performance.now() - t0;
  info = {
    tinygradDevice: device,
    gpu: bridge ? bridge.info() : null,
    tinygradVersion,
    pyodideVersion: py.version,
    pythonVersion,
    syncMode: bridge ? "jspi" : null,
    interruptMode: interruptBuffer ? "signal" : "terminate",
    crossOriginIsolated: Boolean(self.crossOriginIsolated),
    diagnostics,
    torchShim,
    timings,
  };
  return info;
}

const DELTA_KEYS: (keyof GpuStatsDelta)[] = ["kernelsCompiled", "kernelsDispatched", "bytesUploaded", "bytesDownloaded", "readbacks"];

async function run(id: number, code: string): Promise<RunReply> {
  if (!py) throw new Error("runtime is not initialized");
  currentRun = id;
  if (interruptBuffer) Atomics.store(interruptBuffer, 0, 0);
  const before: GpuStats | null = bridge ? { ...bridge.counters } : null;
  const runFn = (py.globals.get("tinygrad_js") as { _runner: { run: (c: string) => Promise<PyProxyLike> } })._runner.run;
  const proxy = await runFn(code);
  const r = proxy.toJs({ dict_converter: Object.fromEntries }) as {
    ok: boolean; result?: string | null; type?: string; message?: string; traceback?: string; duration_ms: number;
  };
  proxy.destroy();
  let gpu: GpuStatsDelta | null = null;
  if (bridge && before) {
    gpu = {} as GpuStatsDelta;
    for (const k of DELTA_KEYS) gpu[k] = bridge.counters[k] - before[k];
  }
  return {
    ok: r.ok,
    result: r.result ?? undefined,
    error: r.ok ? undefined : { type: r.type ?? "Error", message: r.message ?? "", traceback: r.traceback ?? "" },
    durationMs: r.duration_ms,
    gpu,
  };
}

// Requests are processed one at a time, in order.
let queue: Promise<void> = Promise.resolve();
self.onmessage = (ev: MessageEvent<ToWorker>) => {
  const msg = ev.data;
  queue = queue.then(async () => {
    try {
      let value: RuntimeInfo | RunReply | null = null;
      switch (msg.type) {
        case "init": value = await init(msg.config); break;
        case "run": value = await run(msg.id, msg.code); break;
        case "info":
          if (info && bridge) info = { ...info, gpu: bridge.info() };
          value = info;
          break;
        case "reset": py?.runPython("import tinygrad_js._runner as r; r.reset_globals()"); break;
        case "dispose": bridge?.destroy(); bridge = null; break;
      }
      post({ type: "reply", id: msg.id, ok: true, value });
    } catch (e) {
      post({ type: "reply", id: msg.id, ok: false, error: e instanceof Error ? e.message : String(e) });
    }
  });
};
