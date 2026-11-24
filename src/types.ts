/** Which tinygrad device user code runs on by default. */
export type DeviceKind = "webgpu" | "cpu" | "auto";

/** How Python blocks on asynchronous WebGPU work (buffer readback, synchronize, pipeline compilation). */
export type SyncMode = "jspi";

export interface CreateTinygradOptions {
  /**
   * `"webgpu"` (default): tinygrad's WEBGPU device on the browser's WebGPU. Fails if WebGPU or JSPI is unavailable.
   * `"cpu"`: tinygrad's pure-Python `PYTHON` device (slow; for small tensors only).
   * `"auto"`: WebGPU when available, otherwise fall back to `"cpu"` (the reason is reported in `diagnostics`).
   */
  device?: DeviceKind;
  /** Base URL of a Pyodide distribution containing `pyodide.mjs` and the numpy wheel. Defaults to the jsDelivr CDN. */
  pyodideIndexURL?: string;
  /** URL of the tinygrad wheel to install. Defaults to the pinned wheel shipped with this package. */
  tinygradWheelURL?: string;
  /** URL of the worker script. Defaults to the worker shipped next to this module (`worker.js`). */
  workerURL?: string | URL;
  /** Extra Pyodide packages to load (names from the Pyodide distribution or wheel URLs). numpy is always loaded. */
  packages?: string[];
  /**
   * Network access for user code after initialization. `"none"` (default) removes fetch, XMLHttpRequest, WebSocket,
   * EventSource and importScripts from the worker scope once setup has finished. This is a best-effort policy, not a
   * security boundary; see README "Safety".
   */
  network?: "none" | "allow";
  /**
   * Make `import torch` work in user code (default `true`). The module is tinygrad.js's PyTorch-compatible shim: the commonly used
   * part of the PyTorch API (tensors, autograd, `torch.nn`, `torch.optim`, `torch.utils.data`, save/load) implemented on tinygrad
   * and running on WebGPU. It is not PyTorch; see docs/TORCH.md for coverage and differences. Needs tinygrad >= 0.14.
   * Set `false` to leave `torch` unimportable (or to provide your own package via `packages`).
   */
  torch?: boolean;
  /** Upper bound (bytes) for GPU buffer memory tinygrad may allocate. Exceeding it raises MemoryError in Python. */
  maxGpuMemoryBytes?: number;
  /** Default time limit for each `run()` in milliseconds. 0 = no limit (default). */
  timeoutMs?: number;
  /** Time limit for initialization in milliseconds. Default 120000. */
  initTimeoutMs?: number;
  /** Called with stdout/stderr text as it is produced (in addition to the buffered result). */
  onStdout?: (text: string) => void;
  onStderr?: (text: string) => void;
  /** Called with progress messages during initialization. */
  onStatus?: (message: string) => void;
}

export interface RunOptions {
  /** Time limit for this run in milliseconds; overrides `timeoutMs` from createTinygrad. 0 = no limit. */
  timeoutMs?: number;
  onStdout?: (text: string) => void;
  onStderr?: (text: string) => void;
}

export interface PythonError {
  /** Python exception type name, e.g. `ZeroDivisionError`, or `KeyboardInterrupt` after interrupt(). */
  type: string;
  message: string;
  /** Formatted Python traceback. */
  traceback: string;
}

export interface RunResult {
  ok: boolean;
  stdout: string;
  stderr: string;
  /** Set when the code raised, was interrupted, or timed out. */
  error?: PythonError;
  /** `"completed"`, `"error"` (Python exception), `"interrupted"` (KeyboardInterrupt), `"terminated"` (worker was killed). */
  status: "completed" | "error" | "interrupted" | "terminated";
  /** repr() of the value of the last expression, if any. */
  result?: string;
  /** Wall-clock milliseconds measured inside the worker. */
  durationMs: number;
  /** GPU counters accumulated during this run (null when not on WebGPU). */
  gpu: GpuStatsDelta | null;
}

export interface DeviceInfo {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
  /** True for software adapters such as SwiftShader: real WebGPU, but executed on the CPU. */
  isFallbackAdapter: boolean;
  features: string[];
  limits: Record<string, number>;
}

export interface RuntimeInfo {
  /** The tinygrad device user code runs on by default: `"WEBGPU"` or `"PYTHON"`. */
  tinygradDevice: "WEBGPU" | "PYTHON";
  /** WebGPU adapter details, or null when running on the CPU device. */
  gpu: DeviceInfo | null;
  tinygradVersion: string;
  pyodideVersion: string;
  pythonVersion: string;
  syncMode: SyncMode | null;
  /** Whether `interrupt()` can stop Python without killing the worker (needs SharedArrayBuffer / cross-origin isolation). */
  interruptMode: "signal" | "terminate";
  crossOriginIsolated: boolean;
  /** Notes about fallbacks or limitations detected during initialization. */
  diagnostics: string[];
  /** True when `import torch` resolves to tinygrad.js's PyTorch-compatible shim (see `CreateTinygradOptions.torch`). */
  torchShim: boolean;
  timings: InitTimings;
}

export interface InitTimings {
  /** Loading the Pyodide runtime (wasm + stdlib). */
  pyodideMs: number;
  /** Loading numpy + the tinygrad wheel (+ extra packages). */
  packagesMs: number;
  /** requestAdapter + requestDevice. */
  webgpuMs: number;
  /** `import tinygrad` after installing the backend. */
  tinygradImportMs: number;
  /** Total time from worker start until ready. */
  totalMs: number;
}

export interface GpuStats {
  kernelsCompiled: number;
  kernelsDispatched: number;
  submits: number;
  buffersAllocated: number;
  buffersLive: number;
  bytesAllocated: number;
  peakBytesAllocated: number;
  bytesUploaded: number;
  bytesDownloaded: number;
  readbacks: number;
  syncs: number;
  bindGroupCacheHits: number;
}

export type GpuStatsDelta = Pick<GpuStats, "kernelsCompiled" | "kernelsDispatched" | "bytesUploaded" | "bytesDownloaded" | "readbacks">;
