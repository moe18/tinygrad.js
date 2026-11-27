/** Compiler driver: parse a CUDA source once, then instantiate kernels (template arguments + launch specialization). */
import { CompileError, DiagnosticSink, fail, type Diagnostic } from "./diagnostics.js";
import { tokenize } from "./lexer.js";
import { preprocess, type PreprocessOptions } from "./preprocess.js";
import { parse, parseTypeString, parseExpressionString } from "./parser.js";
import type { FuncDecl, TemplateArg, Expr as AExpr } from "./ast.js";
import { Lowering, Program, type CompileOptions, type KernelSpec } from "./lower.js";
import type { Kernel } from "./ir.js";
import { optimize, type PassReport } from "./passes.js";
import { typeName } from "./types.js";

export const COMPILER_VERSION = "tinygrad.js-cuda 0.1.0";

export interface ModuleOptions extends CompileOptions, PreprocessOptions { filename?: string }

export interface KernelSignature {
  name: string;
  template: { kind: "type" | "value"; name: string }[];
  params: { name: string; type: string; pointer: boolean; isConst: boolean }[];
  line: number;
  launchBounds?: number[];
}

export class ParsedModule {
  readonly program: Program;
  readonly sink: DiagnosticSink;
  readonly kernels: KernelSignature[];
  constructor(readonly source: string, readonly options: ModuleOptions) {
    this.sink = new DiagnosticSink();
    const toks = preprocess(source, options.filename ?? "kernel.cu", options, this.sink);
    const tu = parse(toks, this.sink);
    this.program = new Program(tu, this.sink);
    this.kernels = this.program.kernels.map((fd) => ({
      name: fd.name,
      template: (fd.template ?? []).map((t) => ({ kind: t.kind, name: t.name })),
      params: fd.params.map((p) => ({ name: p.name, type: typeExprString(p.type), pointer: p.type.k === "ptr" || p.type.k === "array", isConst: isConstPtr(p.type) })),
      line: fd.loc.line,
      launchBounds: fd.launchBounds,
    }));
  }

  get warnings(): Diagnostic[] { return this.sink.warnings; }

  kernelDecl(name: string): FuncDecl {
    const fd = this.program.kernels.find((k) => k.name === name);
    if (!fd) {
      const names = this.program.kernels.map((k) => k.name);
      fail("name", `no __global__ function named '${name}'${names.length ? ` (kernels: ${names.join(", ")})` : " (the source defines no kernels)"}`);
    }
    return fd;
  }

  /** Split "name<args>" into the kernel name and its template arguments. */
  parseKernelRef(ref: string): { name: string; targs: TemplateArg[] } {
    const lt = ref.indexOf("<");
    if (lt < 0) return { name: ref.trim(), targs: [] };
    const name = ref.slice(0, lt).trim();
    const fd = this.kernelDecl(name);
    const toks = tokenize(ref.slice(lt), "<kernel name>");
    // parse "<a, b>" as a template argument list using the module's known type names
    const known = [...this.program.structs.keys(), ...this.program.typedefs.keys()];
    const inner = ref.slice(lt + 1, ref.lastIndexOf(">"));
    const parts = splitTopLevel(inner);
    void toks;
    const targs: TemplateArg[] = parts.map((p, i) => {
      const tp = fd.template?.[i];
      const ptoks = tokenize(p, "<template argument>");
      if (tp?.kind === "type") return { k: "type", type: parseTypeString(ptoks, this.sink, known) };
      return { k: "expr", expr: parseExpressionString(ptoks, this.sink) };
    });
    return { name, targs };
  }

  /** Lower one kernel to IR for a given block size (0 = unspecialized) and dynamic shared memory size. */
  instantiate(ref: string, spec: KernelSpec, opts: CompileOptions = {}): Kernel {
    const { name, targs } = this.parseKernelRef(ref);
    const fd = this.kernelDecl(name);
    const L = new Lowering(this.program, spec, { ...this.options, ...opts });
    const env = fd.template ? L.templateEnv(fd.template, targs, fd.loc) : new Map();
    if (!fd.template && targs.length) fail("type", `kernel '${name}' is not a template`);
    const display = fd.template ? `${name}<${[...env.values()].map((b) => (b.k === "type" ? typeName(b.t) : b.k === "const" ? String((b.e as { v: unknown }).v) : "?")).join(", ")}>` : name;
    const k = L.lowerKernel(fd, env, display);
    const reports = optimize(k, { mode: opts.mode ?? this.options.mode ?? "optimized", disable: (opts as { disablePasses?: string[] }).disablePasses });
    (k as Kernel & { passes?: PassReport[] }).passes = reports;
    return k;
  }

  /** Parse a launch string `kernel<<<grid, block, smem>>>(args...)` into its parts (for CudaModule.exec). */
  parseLaunch(text: string): AExpr & { k: "launch" } {
    const e = parseExpressionString(tokenize(text, "<launch>"), this.sink, this.program.kernels.filter((k) => k.template).map((k) => k.name));
    if (e.k !== "launch") fail("syntax", "expected a kernel launch: name<<<grid, block>>>(args)");
    return e;
  }
}

export function parseModule(source: string, options: ModuleOptions = {}): ParsedModule {
  try { return new ParsedModule(source, options); }
  catch (e) { if (e instanceof CompileError) throw e; throw new CompileError({ severity: "error", code: "internal", message: `internal compiler error: ${(e as Error).stack ?? e}` }); }
}

function splitTopLevel(s: string): string[] {
  const out: string[] = [];
  let depth = 0, cur = "";
  for (const c of s) {
    if (c === "<" || c === "(") depth++;
    if (c === ">" || c === ")") depth--;
    if (c === "," && depth === 0) { out.push(cur.trim()); cur = ""; continue; }
    cur += c;
  }
  if (cur.trim()) out.push(cur.trim());
  return out;
}

function typeExprString(t: import("./ast.js").TypeExpr): string {
  switch (t.k) {
    case "prim": return (t.isConst ? "const " : "") + t.name;
    case "named": return (t.isConst ? "const " : "") + t.name + (t.args ? `<${t.args.map((a) => (a.k === "type" ? typeExprString(a.type) : "…")).join(", ")}>` : "");
    case "ptr": return typeExprString(t.to) + "*" + (t.restrict ? " __restrict__" : "");
    case "ref": return typeExprString(t.to) + "&";
    case "array": return typeExprString(t.of) + "[]";
    case "auto": return "auto";
    case "decltype": return "decltype(...)";
  }
}
function isConstPtr(t: import("./ast.js").TypeExpr): boolean {
  return (t.k === "ptr" || t.k === "array") && Boolean(((t.k === "ptr" ? t.to : t.of) as { isConst?: boolean }).isConst);
}
