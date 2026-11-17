/**
 * tinygrad.js: run upstream tinygrad (Python) in the browser with Pyodide, executing kernels on the browser's WebGPU.
 *
 * ```ts
 * import { createTinygrad } from "tinygrad.js";
 * const runtime = await createTinygrad({ device: "webgpu" });
 * const result = await runtime.run("from tinygrad import Tensor\nprint((Tensor([1,2,3]) + 1).numpy())");
 * console.log(result.stdout);
 * await runtime.destroy();
 * ```
 */
import { TinygradRuntime, type TinygradVersion } from "./runtime.js";
import type { CreateTinygradOptions } from "./types.js";

export * from "./types.js";
export {
  TinygradRuntime, TinygradError, bundledWheelURL,
  PYODIDE_VERSION, DEFAULT_TINYGRAD_VERSION, SUPPORTED_TINYGRAD_VERSIONS, DEFAULT_PYODIDE_INDEX_URL,
  type TinygradVersion,
} from "./runtime.js";

/** Start a worker, load Pyodide + tinygrad, initialize the device. Resolves when ready to run code. */
export function createTinygrad(options: CreateTinygradOptions & { tinygradVersion?: TinygradVersion } = {}): Promise<TinygradRuntime> {
  return TinygradRuntime.create(options);
}

export interface WebGPUSupport {
  webgpu: boolean;
  adapter: boolean;
  jspi: boolean;
  crossOriginIsolated: boolean;
  /** Human-readable reason when `webgpu && adapter && jspi` is not all true. */
  reason?: string;
}

/**
 * Check what this browser supports, from the main thread. (The worker repeats the WebGPU check, since WebGPU in
 * workers is supported by the same browsers that support it on the main thread.)
 */
export async function checkSupport(): Promise<WebGPUSupport> {
  const gpu = (globalThis.navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } } | undefined)?.gpu;
  const jspi = typeof (WebAssembly as unknown as { Suspending?: unknown }).Suspending === "function";
  let adapter = false;
  if (gpu) { try { adapter = Boolean(await gpu.requestAdapter()); } catch { adapter = false; } }
  const reasons: string[] = [];
  if (!gpu) reasons.push("navigator.gpu is missing (WebGPU not supported or not a secure context)");
  else if (!adapter) reasons.push("no WebGPU adapter available (blocklisted GPU or disabled)");
  if (!jspi) reasons.push("WebAssembly JSPI is not supported (needed to read GPU results from Python)");
  return {
    webgpu: Boolean(gpu), adapter, jspi, crossOriginIsolated: Boolean(globalThis.crossOriginIsolated),
    reason: reasons.length ? reasons.join("; ") : undefined,
  };
}
