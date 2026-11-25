/**
 * Compiler diagnostics. Every error the CUDA compiler reports carries a source location and a stable code, so the
 * playground and tests can point at the offending line and tell unsupported features apart from user errors.
 */

export interface SourceLoc { file: string; line: number; col: number }

export type DiagnosticCode =
  | "syntax"          // malformed source
  | "preprocessor"    // #if/#define/#include problems
  | "type"            // type checking
  | "name"            // undeclared / redeclared names
  | "unsupported"     // valid CUDA we deliberately do not support (documented in docs/CUDA.md)
  | "backend"         // valid CUDA that the selected backend cannot execute (e.g. double on WebGPU)
  | "limit"           // compile resource limits (source size, inlining depth, unrolled size ...)
  | "internal";       // a compiler bug

export interface Diagnostic { severity: "error" | "warning"; code: DiagnosticCode; message: string; loc?: SourceLoc }

export class CompileError extends Error {
  readonly diagnostics: Diagnostic[];
  constructor(diags: Diagnostic | Diagnostic[]) {
    const list = Array.isArray(diags) ? diags : [diags];
    super(list.map(formatDiagnostic).join("\n"));
    this.name = "CompileError";
    this.diagnostics = list;
  }
  get code(): DiagnosticCode { return this.diagnostics[0]?.code ?? "internal"; }
  get loc(): SourceLoc | undefined { return this.diagnostics[0]?.loc; }
}

export function formatDiagnostic(d: Diagnostic): string {
  const where = d.loc ? `${d.loc.file}:${d.loc.line}:${d.loc.col}: ` : "";
  return `${where}${d.severity} [${d.code}]: ${d.message}`;
}

export function fail(code: DiagnosticCode, message: string, loc?: SourceLoc): never {
  throw new CompileError({ severity: "error", code, message, loc });
}

/** Collects warnings during a compilation. */
export class DiagnosticSink {
  readonly warnings: Diagnostic[] = [];
  warn(code: DiagnosticCode, message: string, loc?: SourceLoc): void {
    if (this.warnings.length < 200 && !this.warnings.some((w) => w.message === message && w.loc?.line === loc?.line))
      this.warnings.push({ severity: "warning", code, message, loc });
  }
}
