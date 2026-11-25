/**
 * Kernel autotuning: time candidate launch configurations (block sizes, tile template arguments, ...) on this device and
 * cache the fastest per (key, device). Times use GPU timestamp queries when the device has them, otherwise synchronized
 * wall-clock time (reported as such).
 */
import type { CudaContext, LaunchParams, CudaModule } from "./runtime/context.js";

export interface TuneCandidate { name: string; params: LaunchParams }
export interface TuneResult { best: string; results: { name: string; ms: number; source: "gpu-timestamps" | "wall-clock"; error?: string }[]; cached: boolean }

const memoryCache = new Map<string, string>();
const STORAGE_KEY = "tinygrad.js-cuda-autotune";

function persisted(): Record<string, string> {
  try { return JSON.parse(globalThis.localStorage?.getItem(STORAGE_KEY) ?? "{}"); } catch { return {}; }
}
function persist(key: string, best: string): void {
  try {
    const all = persisted(); all[key] = best;
    globalThis.localStorage?.setItem(STORAGE_KEY, JSON.stringify(all));
  } catch { /* storage unavailable: in-memory only */ }
}

export async function autotune(ctx: CudaContext, module: CudaModule, key: string, candidates: TuneCandidate[], opts: { warmup?: number; iters?: number; force?: boolean } = {}): Promise<TuneResult> {
  const fullKey = `${key}|${module.hash}|${ctx.info().backend}|${ctx.info().name}`;
  const cachedBest = memoryCache.get(fullKey) ?? persisted()[fullKey];
  if (cachedBest && !opts.force && candidates.some((c) => c.name === cachedBest)) return { best: cachedBest, results: [], cached: true };
  const warmup = opts.warmup ?? 2, iters = opts.iters ?? 5;
  const results: TuneResult["results"] = [];
  for (const c of candidates) {
    try {
      for (let i = 0; i < warmup; i++) await module.launch(c.params);
      await ctx.synchronize();
      const t0 = performance.now();
      for (let i = 0; i < iters; i++) await module.launch({ ...c.params, timing: true });
      const r = await ctx.synchronize();
      const wall = (performance.now() - t0) / iters;
      const gpu = r.kernelTimes.filter((t) => t.ms !== null).map((t) => t.ms!);
      if (gpu.length) results.push({ name: c.name, ms: gpu.sort((a, b) => a - b)[Math.floor(gpu.length / 2)], source: "gpu-timestamps" });
      else results.push({ name: c.name, ms: wall, source: "wall-clock" });
    } catch (e) {
      results.push({ name: c.name, ms: Infinity, source: "wall-clock", error: (e as Error).message });
    }
  }
  const ok = results.filter((r) => Number.isFinite(r.ms));
  if (!ok.length) throw new Error(`autotune ${key}: every candidate failed: ${results.map((r) => `${r.name}: ${r.error}`).join("; ")}`);
  const best = ok.sort((a, b) => a.ms - b.ms)[0].name;
  memoryCache.set(fullKey, best);
  persist(fullKey, best);
  return { best, results, cached: false };
}
