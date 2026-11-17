import type { FromWorker, InitConfig, RunReply, ToWorker } from "./protocol.js";
import type { CreateTinygradOptions, RunOptions, RunResult, RuntimeInfo } from "./types.js";

export const PYODIDE_VERSION = "314.0.8";
export const DEFAULT_TINYGRAD_VERSION = "0.14.0";
export const SUPPORTED_TINYGRAD_VERSIONS = ["0.14.0", "0.12.0"] as const;
export type TinygradVersion = (typeof SUPPORTED_TINYGRAD_VERSIONS)[number];
export const DEFAULT_PYODIDE_INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

/** URLs of the wheels shipped in dist/assets. Literal `new URL(..., import.meta.url)` calls let bundlers copy them. */
export function bundledWheelURL(version: TinygradVersion = DEFAULT_TINYGRAD_VERSION): string {
  switch (version) {
    case "0.14.0": return new URL("./assets/tinygrad-0.14.0-py3-none-any.whl", import.meta.url).href;
    case "0.12.0": return new URL("./assets/tinygrad-0.12.0-py3-none-any.whl", import.meta.url).href;
    default: throw new Error(`unsupported tinygrad version ${version as string}; supported: ${SUPPORTED_TINYGRAD_VERSIONS.join(", ")}`);
  }
}

interface Pending {
  resolve: (v: unknown) => void;
  reject: (e: Error) => void;
  kind: ToWorker["type"];
  stdout: string[];
  stderr: string[];
  onStdout?: (t: string) => void;
  onStderr?: (t: string) => void;
  timedOut?: boolean;
  interrupted?: boolean;
}

export class TinygradError extends Error {}

/** A tinygrad Python environment running in a dedicated Web Worker. Create with `createTinygrad()`. */
export class TinygradRuntime {
  private worker: Worker | null = null;
  private nextId = 1;
  private pending = new Map<number, Pending>();
  private interruptBuffer: Int32Array | undefined;
  private info: RuntimeInfo | null = null;
  private ready: Promise<RuntimeInfo>;
  private destroyed = false;
  private config: InitConfig;
  private workerURL: string | URL;

  private constructor(private opts: CreateTinygradOptions & { tinygradVersion?: TinygradVersion }) {
    this.workerURL = opts.workerURL ?? new URL("./worker.js", import.meta.url);
    this.config = {
      device: opts.device ?? "webgpu",
      pyodideIndexURL: new URL(opts.pyodideIndexURL ?? DEFAULT_PYODIDE_INDEX_URL, globalThis.location?.href).href,
      tinygradWheelURL: new URL(opts.tinygradWheelURL ?? bundledWheelURL(opts.tinygradVersion), globalThis.location?.href).href,
      packages: opts.packages ?? [],
      network: opts.network ?? "none",
      maxGpuMemoryBytes: opts.maxGpuMemoryBytes,
    };
    this.ready = this.start();
  }

  static async create(opts: CreateTinygradOptions & { tinygradVersion?: TinygradVersion } = {}): Promise<TinygradRuntime> {
    const rt = new TinygradRuntime(opts);
    await rt.ready;
    return rt;
  }

  private start(): Promise<RuntimeInfo> {
    this.info = null;
    const worker = new Worker(this.workerURL, { type: "module", name: "tinygrad-web" });
    this.worker = worker;
    worker.onmessage = (ev: MessageEvent<FromWorker>) => this.onMessage(ev.data);
    worker.onerror = (ev) => {
      ev.preventDefault?.();
      this.failAll(new TinygradError(`tinygrad-web worker error: ${ev.message || "failed to load worker script"}`));
    };
    if (typeof SharedArrayBuffer !== "undefined" && globalThis.crossOriginIsolated) {
      this.interruptBuffer = new Int32Array(new SharedArrayBuffer(4));
    }
    const config: InitConfig = { ...this.config, interruptBuffer: this.interruptBuffer };
    const initTimeout = this.opts.initTimeoutMs ?? 120_000;
    const p = this.request({ type: "init", id: 0, config }) as Promise<RuntimeInfo>;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const timeout = new Promise<never>((_, rej) => {
      timer = setTimeout(() => rej(new TinygradError(`tinygrad-web initialization timed out after ${initTimeout} ms`)), initTimeout);
    });
    return Promise.race([p, timeout]).then(
      (info) => { clearTimeout(timer); this.info = info; return info; },
      (e: Error) => { clearTimeout(timer); this.terminateWorker(); throw e; },
    );
  }

  private onMessage(m: FromWorker): void {
    if (m.type === "status") { this.opts.onStatus?.(m.message); return; }
    const p = this.pending.get(m.id);
    if (m.type === "stream") {
      if (!p) return;
      if (m.stream === "stdout") { p.stdout.push(m.text); p.onStdout?.(m.text); this.opts.onStdout?.(m.text); }
      else { p.stderr.push(m.text); p.onStderr?.(m.text); this.opts.onStderr?.(m.text); }
      return;
    }
    if (!p) return;
    this.pending.delete(m.id);
    if (m.ok) p.resolve(m.value); else p.reject(new TinygradError(m.error));
  }

  private request(msg: ToWorker, extra: Partial<Pending> = {}): Promise<unknown> {
    if (!this.worker) return Promise.reject(new TinygradError("tinygrad-web runtime has no worker (destroyed?)"));
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, kind: msg.type, stdout: [], stderr: [], ...extra });
      this.worker!.postMessage({ ...msg, id });
    });
  }

  private failAll(err: Error): void {
    const pending = [...this.pending.values()];
    this.pending.clear();
    for (const p of pending) p.reject(err);
  }

  private terminateWorker(): void {
    this.worker?.terminate();
    this.worker = null;
  }

  /**
   * Run Python code in the persistent environment. Globals persist across calls. Never rejects for Python errors:
   * check `result.ok` / `result.error`. Rejects only if the runtime was destroyed or failed to (re)start.
   */
  async run(code: string, options: RunOptions = {}): Promise<RunResult> {
    if (this.destroyed) throw new TinygradError("tinygrad-web runtime was destroyed");
    await this.ready;
    const holder: Partial<Pending> = { onStdout: options.onStdout, onStderr: options.onStderr };
    const id = this.nextId; // the id request() will assign
    const promise = this.request({ type: "run", id: 0, code }, holder);
    const entry = this.pending.get(id)!;
    const timeoutMs = options.timeoutMs ?? this.opts.timeoutMs ?? 0;
    let timer: ReturnType<typeof setTimeout> | undefined;
    if (timeoutMs > 0) timer = setTimeout(() => { entry.timedOut = true; void this.interrupt(); }, timeoutMs);
    try {
      const reply = (await promise) as RunReply;
      const stdout = entry.stdout.join(""), stderr = entry.stderr.join("");
      if (reply.ok) return { ok: true, status: "completed", stdout, stderr, result: reply.result, durationMs: reply.durationMs, gpu: reply.gpu };
      const interrupted = reply.error?.type === "KeyboardInterrupt";
      const error = entry.timedOut && interrupted
        ? { type: "TimeoutError", message: `execution exceeded ${timeoutMs} ms and was interrupted`, traceback: reply.error!.traceback }
        : reply.error!;
      return { ok: false, status: interrupted ? "interrupted" : "error", stdout, stderr, error, durationMs: reply.durationMs, gpu: reply.gpu };
    } catch (e) {
      if (entry.interrupted || entry.timedOut) {
        const message = entry.timedOut ? `execution exceeded ${timeoutMs} ms; the worker was terminated and restarted` :
          "execution was stopped by terminating the worker; the Python environment was restarted and its state lost";
        return {
          ok: false, status: "terminated", stdout: entry.stdout.join(""), stderr: entry.stderr.join(""),
          error: { type: entry.timedOut ? "TimeoutError" : "Terminated", message, traceback: "" }, durationMs: NaN, gpu: null,
        };
      }
      throw e;
    } finally {
      clearTimeout(timer);
    }
  }

  /**
   * Stop the running code.
   * - `"signal"` (cross-origin isolated pages): raises KeyboardInterrupt in Python; the environment survives. If Python
   *   does not stop within `graceMs` (e.g. stuck in a long GPU wait), falls back to terminating the worker.
   * - `"terminate"`: the worker is killed and a fresh environment is started (all Python state is lost).
   * Resolves with the mechanism that stopped execution, or `"idle"` when nothing was running.
   */
  async interrupt(graceMs = 3000): Promise<"signal" | "terminate" | "idle"> {
    const running = [...this.pending.values()].filter((p) => p.kind === "run");
    if (!running.length) return "idle";
    for (const p of running) p.interrupted = true;
    if (this.interruptBuffer) {
      const buf = this.interruptBuffer;
      Atomics.store(buf, 0, 2); // 2 == SIGINT
      const settled = await new Promise<boolean>((resolve) => {
        const t0 = Date.now();
        let lastSignal = t0;
        const poll = () => {
          if (running.every((p) => ![...this.pending.values()].includes(p))) resolve(true);
          else if (Date.now() - t0 > graceMs) resolve(false);
          else {
            // A KeyboardInterrupt raised inside a __del__ finalizer is printed and swallowed by Python; re-signal.
            if (Date.now() - lastSignal > 250 && Atomics.load(buf, 0) === 0) { Atomics.store(buf, 0, 2); lastSignal = Date.now(); }
            setTimeout(poll, 20);
          }
        };
        poll();
      });
      if (settled) return "signal";
    }
    await this.restart();
    return "terminate";
  }

  /** Kill the worker and start a fresh Python environment with the same options. */
  async restart(): Promise<RuntimeInfo> {
    if (this.destroyed) throw new TinygradError("tinygrad-web runtime was destroyed");
    this.terminateWorker();
    this.failAll(new TinygradError("tinygrad-web worker was terminated"));
    this.ready = this.start();
    return this.ready;
  }

  /** Clear user-defined Python globals without restarting (imports of tinygrad stay cached). */
  async reset(): Promise<void> {
    await this.ready;
    await this.request({ type: "reset", id: 0 });
  }

  /** Information about the device, versions, init timings and diagnostics. */
  getDeviceInfo(): RuntimeInfo {
    if (!this.info) throw new TinygradError("tinygrad-web runtime is not ready");
    return this.info;
  }

  /** Release GPU resources and terminate the worker. The runtime cannot be used afterwards. */
  async destroy(): Promise<void> {
    if (this.destroyed) return;
    this.destroyed = true;
    if (this.worker && this.info) {
      await Promise.race([this.request({ type: "dispose", id: 0 }).catch(() => {}), new Promise((r) => setTimeout(r, 1000))]);
    }
    this.terminateWorker();
    this.failAll(new TinygradError("tinygrad-web runtime was destroyed"));
  }
}
