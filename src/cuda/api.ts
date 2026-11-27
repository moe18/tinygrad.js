/**
 * CUDA runtime API equivalents (cudaMalloc, cudaMemcpy, ...), bound to the current device context.
 *
 * Differences from native CUDA (see docs/CUDA.md): functions that wait for the device return Promises (browsers cannot
 * block the main thread); cudaMemcpy host->device returns once the data is captured; errors are thrown as CudaError (the
 * numeric cudaError_t codes are available as `err.code` and through cudaGetLastError); all streams share one in-order queue.
 */
import { CudaContext, CudaEvent, Stream, type HostArray } from "./runtime/context.js";
import { createCudaContext, type ContextOptions } from "./device.js";
import { CudaError, cudaGetErrorName, cudaGetErrorString } from "./runtime/errors.js";
import type { DevicePtr } from "./runtime/memory.js";

const contexts: CudaContext[] = [];
let current = -1;
let pending: Promise<CudaContext> | null = null;
let defaultOptions: ContextOptions = {};

/** Options for the default context created on first use (call before any other cuda* function). */
export function cudaConfigure(opts: ContextOptions): void { defaultOptions = opts; }

/** Create a device context and make it current. Returns its device index. */
export async function cudaInitDevice(opts: ContextOptions = defaultOptions): Promise<number> {
  const ctx = await createCudaContext(opts);
  contexts.push(ctx);
  current = contexts.length - 1;
  return current;
}

export async function cudaContext(): Promise<CudaContext> {
  if (current >= 0) return contexts[current];
  pending ??= createCudaContext(defaultOptions).then((ctx) => { contexts.push(ctx); current = contexts.length - 1; return ctx; });
  return pending;
}
function cur(): CudaContext {
  if (current < 0) throw new CudaError("cudaErrorInitializationError", "no CUDA context: await cudaInitDevice() (or any async cuda* call) first");
  return contexts[current];
}

export async function cudaGetDeviceCount(): Promise<number> { await cudaContext(); return contexts.length; }
export function cudaSetDevice(i: number): void {
  if (!contexts[i]) throw new CudaError("cudaErrorInvalidDevice", `no device ${i} (${contexts.length} created; create more with cudaInitDevice)`);
  current = i;
}
export function cudaGetDevice(): number { return current; }
export async function cudaGetDeviceProperties(i = current): Promise<ReturnType<CudaContext["getDeviceProperties"]>> {
  if (i < 0) await cudaContext();
  return contexts[i < 0 ? current : i].getDeviceProperties();
}
export async function cudaMalloc(bytes: number): Promise<DevicePtr> { return (await cudaContext()).malloc(bytes); }
export function cudaFree(p: DevicePtr): void { cur().free(p); }
export const cudaMemcpyKind = { cudaMemcpyHostToHost: 0, cudaMemcpyHostToDevice: 1, cudaMemcpyDeviceToHost: 2, cudaMemcpyDeviceToDevice: 3, cudaMemcpyDefault: 4 } as const;
export async function cudaMemcpy(dst: DevicePtr | HostArray, src: DevicePtr | HostArray, bytes: number, _kind?: number): Promise<void> {
  const ctx = await cudaContext();
  await ctx.memcpy(dst, src, bytes);
}
/** asynchronous copy: returns immediately; the copy is ordered on the device queue (await the result for device->host) */
export function cudaMemcpyAsync(dst: DevicePtr | HostArray, src: DevicePtr | HostArray, bytes: number, _kind?: number, _stream?: Stream): Promise<void> {
  return cur().memcpy(dst, src, bytes);
}
export async function cudaMemset(p: DevicePtr, value: number, bytes: number): Promise<void> { (await cudaContext()).memset(p, value, bytes); }
export async function cudaDeviceSynchronize(): Promise<number> {
  try { await (await cudaContext()).synchronize(); return 0; }
  catch (e) { if (e instanceof CudaError) throw e; throw e; }
}
export function cudaGetLastError(): number { return current < 0 ? 0 : cur().getLastError(); }
export function cudaPeekAtLastError(): number { return current < 0 ? 0 : cur().peekAtLastError(); }
export { cudaGetErrorName, cudaGetErrorString };
export function cudaStreamCreate(): Stream { return cur().streamCreate(); }
export async function cudaStreamSynchronize(s: Stream): Promise<void> { await s.synchronize(); }
export function cudaStreamDestroy(_s: Stream): void { /* streams hold no resources */ }
export function cudaEventCreate(): CudaEvent { return cur().eventCreate(); }
export function cudaEventRecord(ev: CudaEvent, s?: Stream): void { cur().eventRecord(ev, s); }
export async function cudaEventSynchronize(ev: CudaEvent): Promise<void> { await ev.synchronize(); }
/** milliseconds between two recorded events (completion wall-clock times; GPU kernel timestamps: launch({timing: true})) */
export async function cudaEventElapsedTime(a: CudaEvent, b: CudaEvent): Promise<number> { return cur().eventElapsedTime(a, b); }
export function cudaEventDestroy(_e: CudaEvent): void { /* nothing to free */ }
export async function cudaMemGetInfo(): Promise<{ free: number; total: number }> {
  const s = (await cudaContext()).backend.memoryStats();
  return { free: s.limit - s.bytesAllocated, total: s.limit };
}
export async function cudaDeviceReset(): Promise<void> {
  if (current < 0) return;
  contexts[current].destroy();
  contexts.splice(current, 1);
  current = contexts.length - 1;
  pending = null;
}
