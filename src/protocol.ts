// Messages between the main-thread Runtime and the worker.
import type { DeviceKind, GpuStatsDelta, InitTimings, RuntimeInfo } from "./types.js";

export interface InitConfig {
  device: DeviceKind;
  pyodideIndexURL: string;
  tinygradWheelURL: string;
  packages: string[];
  network: "none" | "allow";
  /** Make `import torch` resolve to the PyTorch-compatible shim. */
  torch: boolean;
  maxGpuMemoryBytes?: number;
  /** Int32Array over a SharedArrayBuffer for Pyodide's interrupt mechanism, when cross-origin isolated. */
  interruptBuffer?: Int32Array;
}

export type ToWorker =
  | { type: "init"; id: number; config: InitConfig }
  | { type: "run"; id: number; code: string }
  | { type: "info"; id: number }
  | { type: "reset"; id: number }
  | { type: "dispose"; id: number };

export interface RunReply {
  ok: boolean;
  result?: string;
  error?: { type: string; message: string; traceback: string };
  durationMs: number;
  gpu: GpuStatsDelta | null;
}

export type FromWorker =
  | { type: "status"; message: string }
  | { type: "stream"; id: number; stream: "stdout" | "stderr"; text: string }
  | { type: "reply"; id: number; ok: true; value: RuntimeInfo | RunReply | null }
  | { type: "reply"; id: number; ok: false; error: string; timings?: Partial<InitTimings> };
