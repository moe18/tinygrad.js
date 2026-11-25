/** Executes compiled CPU kernels: shared between the inline executor and the Web Worker / worker_threads pool. */
import { MATH_IMPORTS } from "./wasm-math.js";
import { formatPrintf } from "./encode.js";

export interface KernelFault { site: number; value: number; tid: number; bid: number }

export interface ExecModule {
  module: WebAssembly.Module;
  printf: { fmt: string; types: string[] }[];
}

export class KernelFaultError extends Error {
  constructor(readonly fault: KernelFault) { super(`kernel fault at site ${fault.site}`); }
}

export class Executor {
  private instances = new Map<number, { run: (args: number, scratch: number, worker: number) => void; out: string[] }>();
  constructor(readonly memory: WebAssembly.Memory) {}

  private instance(id: number, m: ExecModule) {
    let inst = this.instances.get(id);
    if (inst) return inst;
    const out: string[] = [];
    const memory = this.memory;
    const env: Record<string, unknown> = { memory };
    env.fail = (site: number, value: number, tid: number, bid: number) => { throw new KernelFaultError({ site, value: value >>> 0, tid, bid }); };
    env.printf = (id: number, ptr: number) => {
      const f = m.printf[id];
      const words: number[] = [];
      const dv = new DataView(memory.buffer);
      f.types.forEach((t, i) => {
        const a = ptr + i * 8;
        if (t === "i64" || t === "u64" || t === "f64") { words.push(dv.getUint32(a, true), dv.getUint32(a + 4, true)); }
        else if (t === "f32" || t === "f16" || t === "bf16") words.push(dv.getUint32(a, true));
        else words.push(dv.getUint32(a, true));
      });
      out.push(formatPrintf(f.fmt, f.types, words));
    };
    for (const [k, fn] of Object.entries(MATH_IMPORTS)) env[k] = fn;
    // only pass the imports the module declares
    const imports: Record<string, unknown> = {};
    for (const imp of WebAssembly.Module.imports(m.module)) imports[imp.name] = env[imp.name];
    const instance = new WebAssembly.Instance(m.module, { env: imports as WebAssembly.ModuleImports });
    inst = { run: instance.exports.run as (a: number, s: number, w: number) => void, out };
    if (this.instances.size > 256) this.instances.clear();
    this.instances.set(id, inst);
    return inst;
  }

  /** Run blocks until the shared counter is exhausted. Returns printf output; throws KernelFaultError. */
  run(id: number, m: ExecModule, args: number, scratch: number, worker: number): string[] {
    const inst = this.instance(id, m);
    inst.out.length = 0;
    try {
      inst.run(args, scratch, worker);
    } catch (e) {
      // tell the other workers to stop fetching blocks
      new Int32Array(this.memory.buffer)[(args + 16) >> 2] = 1;
      if (e instanceof KernelFaultError) throw e;
      const msg = e instanceof Error ? e.message : String(e);
      if (/divide by zero|integer overflow|remainder by zero/i.test(msg)) throw new KernelFaultError({ site: -2, value: 0, tid: -1, bid: -1 });
      throw e;
    }
    return inst.out.splice(0);
  }
}
