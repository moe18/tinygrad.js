/**
 * Token-based C preprocessor: #define (object-like, function-like, variadic, # and ##), #undef, #if/#ifdef/#ifndef/#elif/
 * #else/#endif, #include (built-in CUDA headers and user-supplied headers only: there is no file system), #pragma unroll,
 * #error/#warning, __FILE__/__LINE__.
 */
import { fail, type DiagnosticSink, type SourceLoc } from "./diagnostics.js";
import { parseNumber, tokenize, type Token } from "./lexer.js";
import { BUILTIN_HEADERS, HOST_ONLY_HEADERS } from "./headers.js";

export interface PreprocessOptions {
  /** extra macro definitions: `{ N: "1024", DEBUG: "" }` */
  defines?: Record<string, string>;
  /** contents of `#include "name"` files */
  headers?: Record<string, string>;
  /** value of __CUDA_ARCH__ (default 700) */
  cudaArch?: number;
  /** backend name, exposed as __TINYGRAD_JS_BACKEND__ ("wgsl" | "wasm") */
  backend?: string;
}

interface Macro { name: string; params: string[] | null; variadic: boolean; body: Token[]; builtin?: "file" | "line" }

const MAX_EXPANSIONS = 200_000;
const MAX_INCLUDE_DEPTH = 32;

export function preprocess(src: string, file: string, opts: PreprocessOptions, sink: DiagnosticSink): Token[] {
  const macros = new Map<string, Macro>();
  const def = (name: string, value: string) => {
    macros.set(name, { name, params: null, variadic: false, body: tokenize(value, "<built-in>").slice(0, -1) });
  };
  def("__CUDACC__", "1"); def("__NVCC__", "1"); def("__CUDA_ARCH__", String(opts.cudaArch ?? 700));
  def("__CUDACC_VER_MAJOR__", "12"); def("__CUDACC_VER_MINOR__", "0"); def("CUDART_VERSION", "12000");
  def("__cplusplus", "201703L"); def("__TINYGRAD_JS__", "1"); def("__TINYGRAD_JS_CUDA__", "1");
  def("__TINYGRAD_JS_BACKEND__", JSON.stringify(opts.backend ?? "generic"));
  if (opts.backend) def(`__TINYGRAD_JS_BACKEND_${opts.backend.toUpperCase()}__`, "1");
  macros.set("__FILE__", { name: "__FILE__", params: null, variadic: false, body: [], builtin: "file" });
  macros.set("__LINE__", { name: "__LINE__", params: null, variadic: false, body: [], builtin: "line" });
  for (const [k, v] of Object.entries(opts.defines ?? {})) {
    if (!/^[A-Za-z_]\w*$/.test(k)) fail("preprocessor", `invalid macro name in defines: '${k}'`);
    def(k, v);
  }
  const included = new Set<string>();
  let expansions = 0;
  const out: Token[] = [];

  // ---------------------------------------------------------------- macro expansion
  function expandList(toks: Token[]): Token[] {
    const res: Token[] = [];
    let i = 0;
    while (i < toks.length) {
      const t = toks[i];
      const m = t.k === "id" ? macros.get(t.v) : undefined;
      if (!m || t.hide?.has(t.v)) { res.push(t); i++; continue; }
      if (++expansions > MAX_EXPANSIONS) fail("limit", "macro expansion limit exceeded (recursive macros?)", t.loc);
      if (m.builtin) {
        res.push({ ...t, k: m.builtin === "file" ? "str" : "num", v: m.builtin === "file" ? t.loc.file : String(t.loc.line) });
        i++; continue;
      }
      if (m.params === null) {
        const hide = new Set(t.hide ?? []); hide.add(m.name);
        const body = m.body.map((b, j) => ({ ...b, loc: t.loc, space: j === 0 ? t.space : b.space, bol: false, hide: unionHide(b.hide, hide) }));
        res.push(...expandList(pasteTokens(body, t.loc)));
        i++; continue;
      }
      // function-like: needs '(' next
      if (toks[i + 1]?.v !== "(" || toks[i + 1].k !== "punct") { res.push(t); i++; continue; }
      const { args, end } = collectArgs(toks, i + 1, t.loc);
      if (m.variadic) {
        if (args.length < m.params.length) args.push(...Array(m.params.length - args.length).fill([]));
        const fixed = m.params.length;
        if (args.length > fixed + 1) {
          // fold extra args (with commas) into __VA_ARGS__
          const va: Token[] = [];
          args.slice(fixed).forEach((a, j) => { if (j) va.push({ k: "punct", v: ",", loc: t.loc, bol: false, space: false }); va.push(...a); });
          args.splice(fixed, args.length - fixed, va);
        }
        if (args.length === fixed) args.push([]);
      } else {
        if (m.params.length === 0 && args.length === 1 && args[0].length === 0) args.length = 0;
        if (args.length !== m.params.length) fail("preprocessor", `macro ${m.name} expects ${m.params.length} arguments, got ${args.length}`, t.loc);
      }
      const names = m.variadic ? [...m.params, "__VA_ARGS__"] : m.params;
      const hide = new Set(t.hide ?? []); hide.add(m.name);
      const substituted = substitute(m, names, args, t, hide);
      res.push(...expandList(substituted));
      i = end + 1;
    }
    return res;
  }

  function unionHide(a: Set<string> | undefined, b: Set<string>): Set<string> {
    if (!a) return b;
    const s = new Set(a); for (const x of b) s.add(x); return s;
  }

  function collectArgs(toks: Token[], open: number, loc: SourceLoc): { args: Token[][]; end: number } {
    const args: Token[][] = [[]];
    let depth = 0, j = open + 1;
    for (; j < toks.length; j++) {
      const a = toks[j];
      if (a.k === "eof") break;
      if (a.k === "punct" && (a.v === "(" || a.v === "[" || a.v === "{")) depth++;
      if (a.k === "punct" && (a.v === ")" || a.v === "]" || a.v === "}")) {
        if (depth === 0 && a.v === ")") return { args, end: j };
        depth--;
      }
      if (a.k === "punct" && a.v === "," && depth === 0) { args.push([]); continue; }
      args[args.length - 1].push(a);
    }
    fail("preprocessor", "unterminated macro argument list", loc);
  }

  function substitute(m: Macro, names: string[], args: Token[][], call: Token, hide: Set<string>): Token[] {
    const body = m.body;
    const res: Token[] = [];
    const expandedCache = new Map<number, Token[]>();
    const argIndex = (tok: Token | undefined) => (tok && tok.k === "id" ? names.indexOf(tok.v) : -1);
    for (let j = 0; j < body.length; j++) {
      const b = body[j];
      if (b.k === "punct" && b.v === "#" && argIndex(body[j + 1]) >= 0) {
        const a = args[argIndex(body[j + 1])];
        const text = a.map((x, q) => (q && x.space ? " " : "") + (x.k === "str" ? JSON.stringify(x.v) : x.k === "char" ? `'${x.v}'` : x.v)).join("");
        res.push({ k: "str", v: text, loc: call.loc, bol: false, space: b.space });
        j++; continue;
      }
      const ai = argIndex(b);
      if (ai >= 0) {
        const nextIsPaste = body[j + 1]?.v === "##";
        const prevIsPaste = body[j - 1]?.v === "##";
        let rep: Token[];
        if (nextIsPaste || prevIsPaste) rep = args[ai];
        else {
          if (!expandedCache.has(ai)) expandedCache.set(ai, expandList(args[ai]));
          rep = expandedCache.get(ai)!;
        }
        rep.forEach((x, q) => res.push({ ...x, loc: call.loc, space: q === 0 ? b.space : x.space }));
        if (rep.length === 0 && (nextIsPaste || prevIsPaste)) res.push({ k: "id", v: "", loc: call.loc, bol: false, space: false }); // placemarker
        continue;
      }
      res.push({ ...b, loc: call.loc });
    }
    const pasted = pasteTokens(res, call.loc).filter((x) => !(x.k === "id" && x.v === ""));
    if (pasted.length) pasted[0] = { ...pasted[0], space: call.space };
    return pasted.map((x) => ({ ...x, bol: false, hide: unionHide(x.hide, hide) }));
  }

  function pasteTokens(toks: Token[], loc: SourceLoc): Token[] {
    if (!toks.some((t) => t.k === "punct" && t.v === "##")) return toks;
    const res: Token[] = [];
    for (let j = 0; j < toks.length; j++) {
      if (toks[j].k === "punct" && toks[j].v === "##") {
        const l = res.pop(), r = toks[++j];
        if (!l || !r) fail("preprocessor", "'##' cannot appear at either end of a macro body", loc);
        const text = l.v + r.v;
        const re = text === "" ? [] : tokenize(text, loc.file).slice(0, -1);
        if (re.length > 1) fail("preprocessor", `pasting '${l.v}' and '${r.v}' does not give a valid token`, loc);
        res.push(re.length ? { ...re[0], loc, space: l.space, hide: l.hide } : { k: "id", v: "", loc, bol: false, space: false });
        continue;
      }
      res.push(toks[j]);
    }
    return res;
  }

  // ---------------------------------------------------------------- #if expressions
  function evalCondition(toks: Token[], loc: SourceLoc): boolean {
    // replace defined(X) / defined X before expansion
    const pre: Token[] = [];
    for (let j = 0; j < toks.length; j++) {
      const t = toks[j];
      if (t.k === "id" && t.v === "defined") {
        let name: Token | undefined;
        if (toks[j + 1]?.v === "(") { name = toks[j + 2]; j += 3; if (toks[j]?.v !== ")") fail("preprocessor", "expected ')' after defined(", loc); }
        else { name = toks[j + 1]; j++; }
        if (!name || name.k !== "id") fail("preprocessor", "expected identifier after 'defined'", loc);
        pre.push({ ...t, k: "num", v: macros.has(name.v) ? "1" : "0" });
        continue;
      }
      pre.push(t);
    }
    const e = expandList(pre).map((t) => (t.k === "id" ? (t.v === "true" ? { ...t, k: "num" as const, v: "1" } : { ...t, k: "num" as const, v: "0" }) : t));
    let p = 0;
    const peek = () => e[p];
    const take = (v?: string) => {
      const t = e[p];
      if (v !== undefined && (!t || t.v !== v)) fail("preprocessor", `expected '${v}' in #if expression`, t?.loc ?? loc);
      p++; return t;
    };
    const BIN: Record<string, number> = { "||": 1, "&&": 2, "|": 3, "^": 4, "&": 5, "==": 6, "!=": 6, "<": 7, ">": 7, "<=": 7, ">=": 7, "<<": 8, ">>": 8, "+": 9, "-": 9, "*": 10, "/": 10, "%": 10 };
    function unary(): bigint {
      const t = take();
      if (!t) fail("preprocessor", "incomplete #if expression", loc);
      if (t.k === "num") { const n = parseNumber(t.v, t.loc); if (n.isFloat) fail("preprocessor", "floating constant in #if", t.loc); return BigInt(n.value); }
      if (t.k === "char") return BigInt(t.v.charCodeAt(0) || 0);
      if (t.v === "(") { const v = ternary(); take(")"); return v; }
      if (t.v === "!") return unary() === 0n ? 1n : 0n;
      if (t.v === "~") return ~unary();
      if (t.v === "-") return -unary();
      if (t.v === "+") return unary();
      fail("preprocessor", `unexpected '${t.v}' in #if expression`, t.loc);
    }
    function binary(minPrec: number): bigint {
      let l = unary();
      for (;;) {
        const t = peek();
        const prec = t && t.k === "punct" ? BIN[t.v] : undefined;
        if (prec === undefined || prec < minPrec) return l;
        p++;
        const r = binary(prec + 1);
        switch (t.v) {
          case "||": l = l !== 0n || r !== 0n ? 1n : 0n; break;
          case "&&": l = l !== 0n && r !== 0n ? 1n : 0n; break;
          case "|": l |= r; break; case "^": l ^= r; break; case "&": l &= r; break;
          case "==": l = l === r ? 1n : 0n; break; case "!=": l = l !== r ? 1n : 0n; break;
          case "<": l = l < r ? 1n : 0n; break; case ">": l = l > r ? 1n : 0n; break;
          case "<=": l = l <= r ? 1n : 0n; break; case ">=": l = l >= r ? 1n : 0n; break;
          case "<<": l <<= r; break; case ">>": l >>= r; break;
          case "+": l += r; break; case "-": l -= r; break; case "*": l *= r; break;
          case "/": if (r === 0n) fail("preprocessor", "division by zero in #if", t.loc); l /= r; break;
          case "%": if (r === 0n) fail("preprocessor", "division by zero in #if", t.loc); l %= r; break;
        }
      }
    }
    function ternary(): bigint {
      const c = binary(1);
      if (peek()?.v === "?") { p++; const a = ternary(); take(":"); const b = ternary(); return c !== 0n ? a : b; }
      return c;
    }
    const v = ternary();
    if (p !== e.length) fail("preprocessor", `unexpected '${e[p].v}' in #if expression`, e[p].loc);
    return v !== 0n;
  }

  // ---------------------------------------------------------------- directives
  function processFile(text: string, fileName: string, depth: number): void {
    if (depth > MAX_INCLUDE_DEPTH) fail("limit", "#include nesting too deep");
    const toks = tokenize(text, fileName);
    // split into lines
    const stack: { active: boolean; taken: boolean; parentActive: boolean; loc: SourceLoc; sawElse: boolean }[] = [];
    const active = () => stack.length === 0 || stack[stack.length - 1].active;
    let pending: Token[] = [];
    const flush = () => { if (pending.length) { out.push(...expandList(pending)); pending = []; } };
    let i = 0;
    while (i < toks.length && toks[i].k !== "eof") {
      const t = toks[i];
      if (t.bol && t.k === "punct" && t.v === "#") {
        // directive line
        let j = i + 1;
        const line: Token[] = [];
        while (toks[j].k !== "eof" && !toks[j].bol) line.push(toks[j++]);
        i = j;
        const name = line[0]?.v ?? "";
        const rest = line.slice(1);
        const loc = t.loc;
        if (name === "if" || name === "ifdef" || name === "ifndef") {
          const parentActive = active();
          let cond = false;
          if (parentActive) {
            if (name === "if") cond = evalCondition(rest, loc);
            else {
              if (rest[0]?.k !== "id") fail("preprocessor", `#${name} needs a macro name`, loc);
              cond = macros.has(rest[0].v) === (name === "ifdef");
            }
          }
          stack.push({ active: parentActive && cond, taken: cond, parentActive, loc, sawElse: false });
          continue;
        }
        if (name === "elif") {
          const top = stack[stack.length - 1];
          if (!top || top.sawElse) fail("preprocessor", "#elif without #if", loc);
          if (top.taken || !top.parentActive) top.active = false;
          else { top.active = evalCondition(rest, loc); top.taken = top.active; }
          continue;
        }
        if (name === "else") {
          const top = stack[stack.length - 1];
          if (!top || top.sawElse) fail("preprocessor", "#else without #if", loc);
          top.sawElse = true;
          top.active = top.parentActive && !top.taken; top.taken = true;
          continue;
        }
        if (name === "endif") { if (!stack.pop()) fail("preprocessor", "#endif without #if", loc); continue; }
        if (!active()) continue;
        flush();
        switch (name) {
          case "": break; // null directive
          case "define": {
            const nt = rest[0];
            if (!nt || nt.k !== "id") fail("preprocessor", "#define needs a macro name", loc);
            if (nt.v === "defined") fail("preprocessor", "'defined' cannot be a macro name", loc);
            let params: string[] | null = null, variadic = false, bodyStart = 1;
            if (rest[1]?.v === "(" && !rest[1].space) {
              params = [];
              let k = 2;
              for (;;) {
                const pt = rest[k];
                if (!pt) fail("preprocessor", "unterminated macro parameter list", loc);
                if (pt.v === ")") { k++; break; }
                if (pt.v === "...") { variadic = true; k++; if (rest[k]?.v !== ")") fail("preprocessor", "'...' must be the last macro parameter", loc); k++; break; }
                if (pt.k !== "id") fail("preprocessor", `invalid macro parameter '${pt.v}'`, pt.loc);
                params.push(pt.v); k++;
                if (rest[k]?.v === ",") k++;
                else if (rest[k]?.v !== ")") fail("preprocessor", "expected ',' or ')' in macro parameters", loc);
              }
              bodyStart = k;
            }
            macros.set(nt.v, { name: nt.v, params, variadic, body: rest.slice(bodyStart) });
            break;
          }
          case "undef": if (rest[0]?.k === "id") macros.delete(rest[0].v); break;
          case "include": {
            let inc = rest;
            if (inc[0]?.k !== "str" && inc[0]?.v !== "<") inc = expandList(rest);
            let hname: string, system: boolean;
            if (inc[0]?.k === "str") { hname = inc[0].v; system = false; }
            else if (inc[0]?.v === "<") {
              const close = inc.findIndex((x) => x.v === ">");
              if (close < 0) fail("preprocessor", "expected '>' in #include", loc);
              hname = inc.slice(1, close).map((x, q) => (q && x.space ? " " : "") + x.v).join("");
              system = true;
            } else fail("preprocessor", "#include expects \"file\" or <file>", loc);
            if (!system && opts.headers && hname in opts.headers) {
              if (included.has("user:" + hname)) break;
              included.add("user:" + hname); // headers are included once (as if they had #pragma once / include guards)
              processFile(opts.headers[hname], hname, depth + 1);
            } else if (hname in BUILTIN_HEADERS) {
              if (included.has(hname)) break;
              included.add(hname);
              processFile(BUILTIN_HEADERS[hname], `<${hname}>`, depth + 1);
            } else if (HOST_ONLY_HEADERS.has(hname) || (system && /^(std|c[a-z]+$)|\.h$/.test(hname))) {
              sink.warn("preprocessor", `#include <${hname}>: host-only header ignored (only device code is compiled)`, loc);
            } else fail("preprocessor", `cannot find header '${hname}' (pass its contents in the 'headers' compile option)`, loc);
            break;
          }
          case "pragma": {
            if (rest[0]?.v === "unroll" || rest[0]?.v === "nounroll") {
              out.push({ k: "pragma", v: rest[0].v, loc, bol: false, space: true, args: expandList(rest.slice(1)) });
            } else if (rest[0]?.v !== "once") sink.warn("preprocessor", `#pragma ${rest.map((x) => x.v).join(" ")} ignored`, loc);
            break;
          }
          case "error": fail("preprocessor", `#error ${rest.map((x) => x.v).join(" ")}`, loc);
          case "warning": sink.warn("preprocessor", `#warning ${rest.map((x) => x.v).join(" ")}`, loc); break;
          case "line": break;
          default: fail("preprocessor", `unknown preprocessor directive #${name}`, loc);
        }
        continue;
      }
      if (active()) {
        // _Pragma("unroll") is rare; treat it as unsupported for clarity
        pending.push(t);
      }
      // expand pending tokens at line boundaries where no function-like macro call could be open
      i++;
      if (toks[i].bol && pending.length && !openCall(pending)) flush();
    }
    if (stack.length) fail("preprocessor", "unterminated #if", stack[stack.length - 1].loc);
    flush();
  }

  // a function-like macro name at the end of pending tokens, or unbalanced parens after one, means its call continues
  function openCall(toks: Token[]): boolean {
    let depth = 0;
    for (let j = toks.length - 1; j >= 0; j--) {
      const t = toks[j];
      if (t.v === ")") depth++;
      else if (t.v === "(") { depth--; if (depth < 0) return true; }
    }
    const last = toks[toks.length - 1];
    return last.k === "id" && macros.get(last.v)?.params != null;
  }

  processFile(src, file, 0);
  out.push({ k: "eof", v: "", loc: { file, line: 0, col: 0 }, bol: true, space: true });
  return out;
}
