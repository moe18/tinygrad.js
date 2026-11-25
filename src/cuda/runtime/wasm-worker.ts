/**
 * CPU kernel worker (browser Web Worker or Node worker_thread). Each worker shares the device's WebAssembly.Memory and
 * executes thread blocks fetched from an atomic counter, so independent blocks run in parallel on all cores.
 */
import { Executor, KernelFaultError, type ExecModule } from "./wasm-exec.js";

type Msg =
  | { type: "init"; memory: WebAssembly.Memory; index: number }
  | { type: "run"; job: number; id: number; module?: WebAssembly.Module; printf?: ExecModule["printf"]; args: number; scratch: number };

let exec: Executor | null = null;
let index = 0;
const modules = new Map<number, ExecModule>();

function handle(msg: Msg, post: (m: unknown) => void): void {
  if (msg.type === "init") { exec = new Executor(msg.memory); index = msg.index; post({ type: "ready", index }); return; }
  if (msg.type === "run") {
    if (msg.module) modules.set(msg.id, { module: msg.module, printf: msg.printf ?? [] });
    const m = modules.get(msg.id);
    if (!exec || !m) { post({ type: "done", job: msg.job, error: { message: "worker not initialized" } }); return; }
    const t0 = performance.now();
    try {
      const printf = exec.run(msg.id, m, msg.args, msg.scratch, index);
      post({ type: "done", job: msg.job, printf, ms: performance.now() - t0 });
    } catch (e) {
      if (e instanceof KernelFaultError) post({ type: "done", job: msg.job, fault: e.fault });
      else post({ type: "done", job: msg.job, error: { message: e instanceof Error ? e.message : String(e) } });
    }
  }
}

declare const WorkerGlobalScope: unknown;
const isBrowserWorker = typeof self !== "undefined" && typeof WorkerGlobalScope !== "undefined";
if (isBrowserWorker) {
  (self as unknown as { onmessage: (ev: MessageEvent<Msg>) => void }).onmessage = (ev) => handle(ev.data, (m) => (self as unknown as { postMessage(m: unknown): void }).postMessage(m));
} else {
  // Node.js worker_threads
  const NODE_WT = "node:worker_threads";
  (import(/* webpackIgnore: true */ /* @vite-ignore */ NODE_WT) as Promise<{ parentPort: { on(e: string, f: (m: Msg) => void): void; postMessage(m: unknown): void } | null }>).then(({ parentPort }) => {
    parentPort?.on("message", (msg: Msg) => handle(msg, (m) => parentPort.postMessage(m)));
  }).catch(() => {});
}
