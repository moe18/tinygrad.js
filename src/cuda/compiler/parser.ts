/**
 * Recursive-descent parser for the supported CUDA C/C++ subset. Host functions (no __device__/__global__) are skipped
 * without being parsed, so a .cu file with a host main() using std:: containers, printf and <<<>>> launches still compiles:
 * only device code is translated.
 */
import { fail, type DiagnosticSink, type SourceLoc } from "./diagnostics.js";
import type { Token } from "./lexer.js";
import type {
  EnumDecl, Expr, FuncDecl, Param, StructDecl, StructField, Stmt, TemplateArg, TemplateParam, TopDecl, TranslationUnit, TypeExpr, VarDecl,
} from "./ast.js";
import { BUILTIN_TYPE_NAMES } from "./types.js";

const PRIM_WORDS = new Set(["void", "bool", "char", "short", "int", "long", "float", "double", "signed", "unsigned", "__int64"]);
const QUALS = new Set(["const", "volatile", "__restrict__", "__restrict", "restrict", "register", "mutable"]);
const DECL_SPECS = new Set([
  "static", "extern", "inline", "__inline__", "__forceinline__", "__noinline__", "constexpr", "__device__", "__global__", "__host__",
  "__shared__", "__constant__", "__managed__", "virtual", "explicit", "friend", "__launch_bounds__", "__align__", "alignas", "__attribute__",
  "thread_local", "consteval", "__grid_constant__",
]);
const UNSUPPORTED_KEYWORDS: Record<string, string> = {
  goto: "goto is not supported (WGSL and the structured kernel IR have no unstructured jumps)",
  asm: "inline PTX assembly (asm) is not supported", __asm__: "inline PTX assembly is not supported",
  try: "exceptions are not supported in device code", throw: "exceptions are not supported in device code",
  new: "dynamic allocation (new) is not supported in device code", delete: "delete is not supported in device code",
  virtual: "virtual functions are not supported in device code", typeid: "RTTI is not supported", dynamic_cast: "dynamic_cast is not supported",
};

const BINARY_PREC: Record<string, number> = {
  "||": 3, "&&": 4, "|": 5, "^": 6, "&": 7, "==": 8, "!=": 8, "<": 9, ">": 9, "<=": 9, ">=": 9, "<<": 10, ">>": 10,
  "+": 11, "-": 11, "*": 12, "/": 12, "%": 12,
};
const ASSIGN_OPS = new Set(["=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="]);

export function parse(tokens: Token[], sink: DiagnosticSink): TranslationUnit {
  return new Parser(tokens, sink).translationUnit();
}

/** Parse a type name such as "float", "unsigned int" or "Vec<float, 4>" (used for template arguments given from JS). */
export function parseTypeString(tokens: Token[], sink: DiagnosticSink, knownTypes: Iterable<string> = []): TypeExpr {
  const p = new Parser(tokens, sink);
  for (const t of knownTypes) p.typeNames.add(t);
  const t = p.typeName();
  if (p.peek().k !== "eof") fail("syntax", `unexpected '${p.peek().v}' after type`, p.peek().loc);
  return t;
}

/** Parse an expression (used for the `kernel<<<grid, block>>>(args)` launch strings of CudaModule.exec). */
export function parseExpressionString(tokens: Token[], sink: DiagnosticSink, templates: Iterable<string> = []): Expr {
  const p = new Parser(tokens, sink);
  for (const t of templates) p.templates.add(t);
  const e = p.expression();
  if (p.peek().v === ";") p.next();
  if (p.peek().k !== "eof") fail("syntax", `unexpected '${p.peek().v}'`, p.peek().loc);
  return e;
}

class Parser {
  p = 0;
  readonly typeNames = new Set<string>(BUILTIN_TYPE_NAMES);
  readonly templates = new Set<string>();
  private pendingUnroll: number | "full" | "none" | undefined;
  constructor(private toks: Token[], private sink: DiagnosticSink) {}

  // ------------------------------------------------------------------ token helpers
  peek(o = 0): Token { return this.toks[Math.min(this.p + o, this.toks.length - 1)]; }
  next(): Token { return this.toks[this.p++]; }
  is(v: string, o = 0): boolean { const t = this.peek(o); return (t.k === "punct" || t.k === "id") && t.v === v; }
  accept(v: string): boolean { if (this.is(v)) { this.p++; return true; } return false; }
  expect(v: string, what?: string): Token {
    const t = this.peek();
    if (!this.is(v)) fail("syntax", `expected '${v}'${what ? ` ${what}` : ""} but found ${t.k === "eof" ? "end of input" : `'${t.v}'`}`, t.loc);
    return this.next();
  }
  ident(what = "identifier"): Token {
    const t = this.peek();
    if (t.k !== "id") fail("syntax", `expected ${what} but found '${t.v || "end of input"}'`, t.loc);
    return this.next();
  }
  /** consume one '>' (splitting '>>' / '>>>' / '>=' tokens when closing template argument lists) */
  closeAngle(): void {
    const t = this.peek();
    if (t.k === "punct" && t.v.startsWith(">")) {
      if (t.v === ">") { this.p++; return; }
      this.toks[this.p] = { ...t, v: t.v.slice(1) };
      return;
    }
    fail("syntax", `expected '>' but found '${t.v}'`, t.loc);
  }

  private skipAttributes(): void {
    for (;;) {
      if (this.is("[") && this.is("[", 1)) { // [[attribute]]
        let depth = 0;
        do { if (this.is("[")) depth++; if (this.is("]")) depth--; this.next(); } while (depth > 0 && this.peek().k !== "eof");
        continue;
      }
      if (this.is("__attribute__")) { this.next(); this.skipBalanced(); continue; }
      return;
    }
  }
  private skipBalanced(): void {
    const open = this.next();
    const close = open.v === "(" ? ")" : open.v === "{" ? "}" : "]";
    let depth = 1;
    while (depth > 0) {
      const t = this.next();
      if (t.k === "eof") fail("syntax", `unbalanced '${open.v}'`, open.loc);
      if (t.k === "punct" && t.v === open.v) depth++;
      else if (t.k === "punct" && t.v === close) depth--;
    }
  }

  // ------------------------------------------------------------------ types
  isTypeStart(o = 0): boolean {
    const t = this.peek(o);
    if (t.k !== "id") return false;
    if (PRIM_WORDS.has(t.v) || QUALS.has(t.v) || t.v === "auto" || t.v === "typename" || t.v === "struct" || t.v === "decltype" ||
        t.v === "union" || t.v === "class" || t.v === "enum") return true;
    if (this.typeNames.has(t.v)) return true;
    // qualified names: ns::Type
    if (this.is("::", o + 1) && this.peek(o + 2).k === "id") return this.isTypeStart(o + 2);
    return false;
  }

  /** decl-specifiers + base type. Returns specifiers seen. */
  private declSpecs(): { type: TypeExpr | null; specs: Set<string>; launchBounds?: number[]; align?: number } {
    const specs = new Set<string>();
    let launchBounds: number[] | undefined, align: number | undefined;
    let isConst = false, isVolatile = false;
    const prims: string[] = [];
    let named: TypeExpr | null = null;
    const start = this.peek().loc;
    for (;;) {
      this.skipAttributes();
      const t = this.peek();
      if (t.k !== "id") break;
      if (t.v in UNSUPPORTED_KEYWORDS && t.v === "virtual") fail("unsupported", UNSUPPORTED_KEYWORDS[t.v], t.loc);
      if (DECL_SPECS.has(t.v)) {
        this.next();
        if (t.v === "__launch_bounds__") {
          this.expect("(");
          launchBounds = [];
          do launchBounds.push(this.constInt(this.assignment())); while (this.accept(","));
          this.expect(")");
        } else if (t.v === "__align__" || t.v === "alignas") {
          this.expect("("); align = this.constInt(this.assignment()); this.expect(")");
        } else if (t.v === "__attribute__") { this.skipBalanced(); }
        else specs.add(t.v);
        continue;
      }
      if (QUALS.has(t.v)) { this.next(); if (t.v === "const") isConst = true; if (t.v === "volatile") isVolatile = true; continue; }
      if (named === null && PRIM_WORDS.has(t.v)) { this.next(); prims.push(t.v); continue; }
      if (prims.length || named) break;
      if (t.v === "auto") { this.next(); named = { k: "auto", loc: t.loc }; continue; }
      if (t.v === "decltype") {
        this.next(); this.expect("("); const e = this.expression(); this.expect(")");
        named = { k: "decltype", expr: e, loc: t.loc }; continue;
      }
      if (t.v === "typename") { this.next(); continue; }
      if (t.v === "struct" || t.v === "class" || t.v === "union" || t.v === "enum") {
        // elaborated type specifier: `struct Foo x;`
        if (this.peek(1).k === "id" && !this.is("{", 2)) { this.next(); const n = this.next(); named = { k: "named", name: n.v, loc: n.loc }; continue; }
        break;
      }
      if (this.isTypeStart()) { named = this.namedType(); continue; }
      break;
    }
    let type: TypeExpr | null = null;
    if (prims.length) type = { k: "prim", name: canonicalPrim(prims, start), loc: start, isConst, isVolatile };
    else if (named) type = { ...named, ...(isConst ? { isConst } : {}), ...(isVolatile ? { isVolatile } : {}) } as TypeExpr;
    else if (isConst || isVolatile) type = { k: "prim", name: "int", loc: start, isConst, isVolatile }; // `const x = 1` (implicit int) is rejected later
    if (type === null && (isConst || isVolatile)) return { type: null, specs, launchBounds, align };
    return { type, specs, launchBounds, align };
  }

  private namedType(): TypeExpr {
    let t = this.ident("type name");
    // strip namespace qualification: std::size_t, cooperative_groups::thread_block -> last component
    while (this.is("::")) { this.next(); t = this.ident("type name"); }
    let args: TemplateArg[] | undefined;
    if (this.is("<") && this.templates.has(t.v)) args = this.templateArgs();
    const res: TypeExpr = { k: "named", name: t.v, args, loc: t.loc };
    // nested name after template: Foo<T>::type
    if (this.is("::")) { this.next(); const inner = this.ident(); return { k: "named", name: `${t.v}::${inner.v}`, args, loc: t.loc }; }
    return res;
  }

  templateArgs(): TemplateArg[] {
    this.expect("<");
    const args: TemplateArg[] = [];
    if (this.peek().v.startsWith(">")) { this.closeAngle(); return args; }
    do {
      if (this.isTypeStart() && !this.looksLikeExprAfterType()) args.push({ k: "type", type: this.typeName() });
      else args.push({ k: "expr", expr: this.conditional(true) });
    } while (this.accept(","));
    this.closeAngle();
    return args;
  }
  private looksLikeExprAfterType(): boolean {
    // `N * 2` where N is a value: we only treat identifiers registered as type names as types, so a type start is a type
    return false;
  }

  /** abstract declarator: base type plus pointers / references / arrays, no name */
  typeName(): TypeExpr {
    const ds = this.declSpecs();
    if (!ds.type) fail("syntax", `expected a type but found '${this.peek().v}'`, this.peek().loc);
    let t = this.pointers(ds.type);
    while (this.is("[")) {
      const loc = this.next().loc;
      const size = this.is("]") ? null : this.expression();
      this.expect("]");
      t = { k: "array", of: t, size, loc };
    }
    return t;
  }

  private pointers(base: TypeExpr): TypeExpr {
    let t = base;
    for (;;) {
      if (this.is("*")) {
        const loc = this.next().loc;
        const pt: TypeExpr = { k: "ptr", to: t, loc };
        for (;;) {
          if (this.is("const")) { this.next(); pt.isConst = true; continue; }
          if (this.is("volatile")) { this.next(); continue; }
          if (this.is("__restrict__") || this.is("__restrict") || this.is("restrict")) { this.next(); pt.restrict = true; continue; }
          break;
        }
        t = pt; continue;
      }
      if (this.is("&")) { t = { k: "ref", to: t, loc: this.next().loc }; continue; }
      if (this.is("&&")) { t = { k: "ref", to: t, loc: this.next().loc, rvalue: true }; continue; }
      return t;
    }
  }

  /** declarator after the base type: pointers, optional name, array dims. */
  private declarator(base: TypeExpr, nameOptional = false): { name: string; type: TypeExpr; loc: SourceLoc } {
    let t = this.pointers(base);
    this.skipAttributes();
    if (this.is("(") && (this.is("*", 1) || this.is("&", 1))) fail("unsupported", "function pointers and parenthesized declarators are not supported", this.peek().loc);
    let name = "", loc = this.peek().loc;
    if (this.peek().k === "id" && !this.isTypeStart()) { const n = this.next(); name = n.v; loc = n.loc; }
    else if (this.peek().k === "id" && this.typeNames.has(this.peek().v) && !nameOptional) { const n = this.next(); name = n.v; loc = n.loc; }
    else if (!nameOptional) fail("syntax", `expected a name in declaration but found '${this.peek().v}'`, this.peek().loc);
    const dims: { size: Expr | null; loc: SourceLoc }[] = [];
    while (this.is("[")) {
      const l = this.next().loc;
      const size = this.is("]") ? null : this.expression();
      this.expect("]");
      dims.push({ size, loc: l });
    }
    // int a[2][3]: array of 2 arrays of 3
    for (let i = dims.length - 1; i >= 0; i--) t = { k: "array", of: t, size: dims[i].size, loc: dims[i].loc };
    this.skipAttributes();
    return { name, type: t, loc };
  }

  private constInt(e: Expr): number {
    if (e.k === "num" && /^\d+$/.test(e.text)) return Number(e.text);
    return -1; // resolved later by sema when needed (launch bounds / alignment are advisory)
  }

  // ------------------------------------------------------------------ top level
  translationUnit(): TranslationUnit {
    const decls: TopDecl[] = [];
    while (this.peek().k !== "eof") this.topDecl(decls, undefined);
    return { decls };
  }

  private topDecl(out: TopDecl[], template: TemplateParam[] | undefined): void {
    this.skipAttributes();
    const t = this.peek();
    if (t.k === "pragma") { this.next(); return; }
    if (this.accept(";")) return;
    if (t.v === "template") { this.next(); const tp = this.templateParams(); this.topDecl(out, tp); return; }
    if (t.v === "namespace") {
      this.next();
      if (this.peek().k === "id") this.next();
      if (this.accept("=")) { while (!this.accept(";")) this.next(); return; } // namespace alias
      this.expect("{");
      while (!this.is("}")) { if (this.peek().k === "eof") fail("syntax", "unterminated namespace", t.loc); this.topDecl(out, undefined); }
      this.expect("}");
      return;
    }
    if (t.v === "using") {
      this.next();
      if (this.accept("namespace")) { while (!this.accept(";")) this.next(); return; }
      const n = this.ident();
      if (this.accept("=")) {
        const ty = this.typeName(); this.expect(";");
        this.typeNames.add(n.v);
        if (template) this.templates.add(n.v);
        out.push({ k: "typedef", name: n.v, type: ty, template, loc: n.loc });
        return;
      }
      while (!this.accept(";")) this.next(); // using std::min;
      return;
    }
    if (t.v === "extern" && this.peek(1).k === "str") {
      this.next(); this.next();
      if (this.accept("{")) { while (!this.is("}")) this.topDecl(out, undefined); this.expect("}"); return; }
      this.topDecl(out, template); return;
    }
    if (t.v === "static_assert") {
      this.next(); this.expect("(");
      const e = this.assignment();
      let msg: string | undefined;
      if (this.accept(",")) msg = this.next().v;
      this.expect(")"); this.expect(";");
      out.push({ k: "static_assert", e, msg, loc: t.loc });
      return;
    }
    if (t.v === "typedef") { this.typedef(out); return; }
    if ((t.v === "struct" || t.v === "class" || t.v === "union") && (this.is("{", 1) || this.is("{", 2) || this.peek(1).v === "__align__" || this.peek(1).v === "alignas" || (this.is("<", 2) && template === undefined)
        || (this.peek(1).k === "id" && this.is(":", 2)))) {
      const sd = this.structDecl(template);
      out.push(sd);
      // struct S {...} s; at top level: a host variable unless it has a device specifier (unsupported)
      if (!this.accept(";")) fail("unsupported", "declaring variables together with a struct definition is not supported; declare them separately", this.peek().loc);
      return;
    }
    if (t.v === "struct" || t.v === "class" || t.v === "union") {
      if (this.peek(1).k === "id" && this.is(";", 2)) { this.next(); this.typeNames.add(this.next().v); this.next(); return; } // forward decl
    }
    if (t.v === "enum") { out.push(this.enumDecl()); this.expect(";"); return; }
    for (const [kw, why] of Object.entries(UNSUPPORTED_KEYWORDS)) if (t.v === kw) fail("unsupported", why, t.loc);

    // function or variable
    const start = this.p;
    const ds = this.declSpecs();
    if (!ds.type) {
      // constructor-like or unknown: skip as host code
      fail("syntax", `expected a declaration but found '${this.peek().v}'`, this.peek().loc);
    }
    const isDevice = ds.specs.has("__device__") || ds.specs.has("__global__");
    const isVarSpace = ds.specs.has("__shared__") || ds.specs.has("__constant__") || ds.specs.has("__managed__");
    const d = this.declaratorOrOperator(ds.type);
    if (this.is("(")) {
      // function
      const params = this.paramList();
      let isConstMethod = false;
      if (this.accept("const")) isConstMethod = true;
      while (this.is("noexcept") || this.is("override") || this.is("final")) { this.next(); if (this.is("(")) this.skipBalanced(); }
      let ret = d.type;
      if (this.accept("->")) ret = this.typeName(); // trailing return type (auto f() -> T)
      this.skipAttributes();
      const fd: FuncDecl = {
        k: "func", name: d.name, params, ret, global: ds.specs.has("__global__"), device: ds.specs.has("__device__") || ds.specs.has("__global__"),
        host: ds.specs.has("__host__") || !isDevice, inline: ds.specs.has("__forceinline__") ? "force" : ds.specs.has("__noinline__") ? "never" : "default",
        template, launchBounds: ds.launchBounds, loc: d.loc, isConstMethod,
      };
      if (d.specArgs) fd.specArgs = d.specArgs;
      if (template) this.templates.add(d.name);
      if (this.accept(";")) { if (isDevice || ds.specs.has("constexpr")) out.push(fd); return; } // prototype
      if (this.is("=")) { // = delete / = default
        this.next(); this.next(); this.expect(";"); return;
      }
      if (!this.is("{")) fail("syntax", `expected '{' or ';' after function declarator but found '${this.peek().v}'`, this.peek().loc);
      if (!isDevice && !ds.specs.has("constexpr")) {
        // host function: skip its body unparsed
        this.skipBalanced();
        out.push({ k: "host", name: d.name, loc: d.loc });
        return;
      }
      if (!isDevice) {
        // constexpr host function: usable in device code (as with nvcc --expt-relaxed-constexpr) if it parses
        const save = this.p;
        try { fd.body = this.block(); fd.device = true; out.push(fd); }
        catch { this.p = save; this.skipBalanced(); out.push({ k: "host", name: d.name, loc: d.loc }); }
        return;
      }
      fd.body = this.block();
      out.push(fd);
      return;
    }
    // variable(s)
    if (!isDevice && !isVarSpace && !ds.specs.has("constexpr") && !(ds.type.k !== "ref" && (ds.type as { isConst?: boolean }).isConst)) {
      // host variable: skip the rest of the declaration
      this.p = start;
      let depth = 0;
      for (;;) {
        const x = this.next();
        if (x.k === "eof") return;
        if (x.v === "{" || x.v === "(") depth++;
        if (x.v === "}" || x.v === ")") depth--;
        if (x.v === ";" && depth <= 0) break;
      }
      out.push({ k: "host", name: d.name, loc: d.loc });
      return;
    }
    let cur = d;
    for (;;) {
      const vd: VarDecl = {
        name: cur.name, type: cur.type, loc: cur.loc, isConstexpr: ds.specs.has("constexpr"), isStatic: ds.specs.has("static"),
        isExtern: ds.specs.has("extern"), align: ds.align,
        space: ds.specs.has("__shared__") ? "shared" : ds.specs.has("__constant__") ? "constant" : ds.specs.has("__managed__") ? "managed" : ds.specs.has("__device__") ? "device" : undefined,
      };
      if (this.accept("=")) vd.init = this.initializer();
      else if (this.is("{")) vd.init = this.initializer();
      out.push({ k: "var", decl: vd });
      if (!this.accept(",")) break;
      cur = this.declarator(ds.type);
    }
    this.expect(";", "after declaration");
  }

  private declaratorOrOperator(base: TypeExpr): { name: string; type: TypeExpr; loc: SourceLoc; specArgs?: TemplateArg[] } {
    const t = this.pointers(base);
    if (this.is("operator")) {
      const loc = this.next().loc;
      let op = "";
      if (this.is("(") && this.is(")", 1)) { this.next(); this.next(); op = "()"; }
      else if (this.is("[")) { this.next(); this.expect("]"); op = "[]"; }
      else op = this.next().v;
      return { name: "operator" + op, type: t, loc };
    }
    let name = this.ident("name");
    while (this.is("::")) { this.next(); name = this.ident("name"); } // out-of-line Foo::bar definitions are treated as free names
    let specArgs: TemplateArg[] | undefined;
    if (this.is("<") && this.templates.has(name.v)) specArgs = this.templateArgs();
    let ty = t;
    const dims: { size: Expr | null; loc: SourceLoc }[] = [];
    while (this.is("[")) {
      const l = this.next().loc;
      const size = this.is("]") ? null : this.expression();
      this.expect("]");
      dims.push({ size, loc: l });
    }
    for (let i = dims.length - 1; i >= 0; i--) ty = { k: "array", of: ty, size: dims[i].size, loc: dims[i].loc };
    this.skipAttributes();
    return { name: name.v, type: ty, loc: name.loc, specArgs };
  }

  private templateParams(): TemplateParam[] {
    this.expect("<");
    const ps: TemplateParam[] = [];
    if (this.peek().v.startsWith(">")) { this.closeAngle(); return ps; }
    do {
      if (this.is("typename") || this.is("class")) {
        this.next();
        if (this.accept("...")) fail("unsupported", "variadic templates are not supported", this.peek().loc);
        const n = this.ident("template parameter name");
        this.typeNames.add(n.v);
        const p: TemplateParam = { kind: "type", name: n.v };
        if (this.accept("=")) p.def = { k: "type", type: this.typeName() };
        ps.push(p);
      } else {
        const vt = this.typeName();
        const n = this.ident("template parameter name");
        const p: TemplateParam = { kind: "value", name: n.v, valueType: vt };
        if (this.accept("=")) p.def = { k: "expr", expr: this.conditional(true) };
        ps.push(p);
      }
    } while (this.accept(","));
    this.closeAngle();
    return ps;
  }

  private typedef(out: TopDecl[] | Stmt[]): void {
    const loc = this.expect("typedef").loc;
    let base: TypeExpr;
    if ((this.is("struct") || this.is("union")) && (this.is("{", 1) || this.is("{", 2))) {
      const sd = this.structDecl(undefined, true);
      (out as TopDecl[]).push(sd as never);
      base = { k: "named", name: sd.name, loc: sd.loc };
    } else if (this.is("enum") && (this.is("{", 1) || this.is("{", 2))) {
      const ed = this.enumDecl();
      (out as TopDecl[]).push(ed as never);
      base = { k: "prim", name: "int", loc };
    } else {
      const ds = this.declSpecs();
      if (!ds.type) fail("syntax", "expected a type after typedef", loc);
      base = ds.type;
    }
    do {
      const d = this.declarator(base);
      this.typeNames.add(d.name);
      (out as { k: string }[]).push({ k: "typedef", name: d.name, type: d.type, loc: d.loc } as never);
    } while (this.accept(","));
    this.expect(";");
  }

  private structDecl(template: TemplateParam[] | undefined, anonymousOk = false): StructDecl {
    const kw = this.next(); // struct/class/union
    let align: number | undefined;
    if (this.is("__align__") || this.is("alignas")) { this.next(); this.expect("("); align = this.constInt(this.assignment()); this.expect(")"); }
    this.skipAttributes();
    let name = "";
    if (this.peek().k === "id") name = this.next().v;
    else if (!anonymousOk) fail("syntax", "expected struct name", kw.loc);
    else name = `__anon_struct_${kw.loc.line}_${kw.loc.col}`;
    if (this.is("<")) this.templateArgs(); // partial specialization args (treated as the primary template)
    this.typeNames.add(name);
    if (template) this.templates.add(name);
    if (this.accept(":")) { // inheritance
      fail("unsupported", "struct inheritance is not supported", this.peek().loc);
    }
    this.expect("{");
    const fields: StructField[] = [];
    const methods: FuncDecl[] = [];
    while (!this.accept("}")) {
      if (this.peek().k === "eof") fail("syntax", "unterminated struct", kw.loc);
      if (this.accept(";")) continue;
      if ((this.is("public") || this.is("private") || this.is("protected")) && this.is(":", 1)) { this.next(); this.next(); continue; }
      if (this.is("typedef")) { const tmp: TopDecl[] = []; this.typedef(tmp); continue; }
      if (this.is("using")) { this.next(); const n = this.ident(); this.expect("="); this.typeName(); this.expect(";"); this.typeNames.add(n.v); continue; }
      if (this.is("static_assert")) { this.next(); this.skipBalanced(); this.expect(";"); continue; }
      let mtemplate: TemplateParam[] | undefined;
      if (this.is("template")) { this.next(); mtemplate = this.templateParams(); }
      // constructor: Name(...)
      const ctorStart = this.p;
      const specs = new Set<string>();
      while (this.peek().k === "id" && DECL_SPECS.has(this.peek().v)) specs.add(this.next().v);
      if (this.is(name) && this.is("(", 1)) {
        const loc = this.next().loc;
        const params = this.paramList();
        const inits: { name: string; args: Expr[] }[] = [];
        if (this.accept(":")) {
          do {
            const f = this.ident("member name");
            const open = this.next();
            if (open.v !== "(" && open.v !== "{") fail("syntax", "expected '(' in member initializer", open.loc);
            const args: Expr[] = [];
            const close = open.v === "(" ? ")" : "}";
            if (!this.is(close)) do args.push(this.assignment()); while (this.accept(","));
            this.expect(close);
            inits.push({ name: f.v, args });
          } while (this.accept(","));
        }
        const fd: FuncDecl = {
          k: "func", name: "__ctor", params, ret: { k: "prim", name: "void", loc }, global: false, device: specs.has("__device__"),
          host: specs.has("__host__") || !specs.has("__device__"), inline: "default", loc, memberOf: name, ctorInits: inits, template: mtemplate,
        };
        if (this.accept(";")) { methods.push(fd); continue; }
        if (this.is("=")) { this.next(); this.next(); this.expect(";"); continue; }
        fd.body = this.block();
        methods.push(fd);
        continue;
      }
      if (this.is("~")) fail("unsupported", "destructors are not supported in device code", this.peek().loc);
      this.p = ctorStart;
      const ds = this.declSpecs();
      if (!ds.type) fail("syntax", `unexpected '${this.peek().v}' in struct`, this.peek().loc);
      const d = this.declaratorOrOperator(ds.type);
      if (this.is("(")) {
        const params = this.paramList();
        const isConstMethod = this.accept("const");
        while (this.is("noexcept")) this.next();
        const fd: FuncDecl = {
          k: "func", name: d.name, params, ret: d.type, global: false, device: ds.specs.has("__device__"), host: ds.specs.has("__host__") || !ds.specs.has("__device__"),
          inline: ds.specs.has("__forceinline__") ? "force" : "default", loc: d.loc, memberOf: name, isConstMethod, isStatic: ds.specs.has("static"), template: mtemplate,
        };
        if (mtemplate) this.templates.add(d.name);
        if (this.accept(";")) { methods.push(fd); continue; }
        if (this.is("=")) { this.next(); this.next(); this.expect(";"); continue; }
        if (!fd.device && !ds.specs.has("constexpr")) { this.skipBalanced(); continue; } // host-only member
        fd.device = true;
        fd.body = this.block();
        methods.push(fd);
        continue;
      }
      let cur = d;
      for (;;) {
        if (this.is(":") ) fail("unsupported", "bit-fields are not supported", this.peek().loc);
        const f: StructField & { isStatic?: boolean } = { name: cur.name, type: cur.type, loc: cur.loc };
        if (ds.specs.has("static")) f.isStatic = true;
        if (this.accept("=")) f.init = this.initializer();
        else if (this.is("{")) f.init = this.initializer();
        fields.push(f);
        if (!this.accept(",")) break;
        cur = this.declarator(ds.type);
      }
      this.expect(";");
    }
    return { k: "struct", name, fields, methods, template, align, isUnion: kw.v === "union", loc: kw.loc };
  }

  private enumDecl(): EnumDecl {
    const loc = this.expect("enum").loc;
    if (this.is("class") || this.is("struct")) this.next();
    let name: string | undefined;
    if (this.peek().k === "id") { name = this.next().v; this.typeNames.add(name); }
    if (this.accept(":")) this.typeName();
    this.expect("{");
    const items: EnumDecl["items"] = [];
    while (!this.accept("}")) {
      const n = this.ident("enumerator");
      const it: EnumDecl["items"][number] = { name: n.v, loc: n.loc };
      if (this.accept("=")) it.value = this.conditional();
      items.push(it);
      if (!this.accept(",")) { this.expect("}"); break; }
    }
    return { k: "enum", name, items, loc };
  }

  private paramList(): Param[] {
    this.expect("(");
    const ps: Param[] = [];
    if (this.accept(")")) return ps;
    if (this.is("void") && this.is(")", 1)) { this.next(); this.next(); return ps; }
    do {
      if (this.is("...")) fail("unsupported", "variadic functions are not supported in device code", this.peek().loc);
      const ds = this.declSpecs();
      if (!ds.type) fail("syntax", `expected parameter type but found '${this.peek().v}'`, this.peek().loc);
      const d = this.declarator(ds.type, true);
      const p: Param = { name: d.name, type: d.type, loc: d.loc };
      if (this.accept("=")) p.def = this.assignment();
      ps.push(p);
    } while (this.accept(","));
    this.expect(")");
    return ps;
  }

  private initializer(): Expr {
    if (this.is("{")) {
      const loc = this.next().loc;
      const items: Expr[] = [];
      const designators: (string | null)[] = [];
      let hasDesig = false;
      while (!this.accept("}")) {
        let desig: string | null = null;
        if (this.is(".") && this.peek(1).k === "id" && this.is("=", 2)) { this.next(); desig = this.next().v; this.next(); hasDesig = true; }
        items.push(this.initializer());
        designators.push(desig);
        if (!this.accept(",")) { this.expect("}"); break; }
      }
      return hasDesig ? { k: "init", items, designators, loc } : { k: "init", items, loc };
    }
    return this.assignment();
  }

  // ------------------------------------------------------------------ statements
  block(): Stmt & { k: "block" } {
    const loc = this.expect("{").loc;
    const body: Stmt[] = [];
    while (!this.accept("}")) {
      if (this.peek().k === "eof") fail("syntax", "unterminated block: expected '}'", loc);
      body.push(this.statement());
    }
    return { k: "block", body, loc };
  }

  private takeUnroll(): number | "full" | "none" | undefined { const u = this.pendingUnroll; this.pendingUnroll = undefined; return u; }

  statement(): Stmt {
    this.skipAttributes();
    const t = this.peek();
    if (t.k === "pragma") {
      this.next();
      if (t.v === "nounroll") this.pendingUnroll = "none";
      else if (t.args && t.args.length) {
        const a = t.args.map((x) => x.v).join("");
        const n = /^\(?(\d+)\)?$/.exec(a);
        this.pendingUnroll = n ? (Number(n[1]) <= 1 ? "none" : Number(n[1])) : "full";
        if (!n) this.sink.warn("unsupported", `#pragma unroll ${a}: only integer literal factors are understood; unrolling fully when the trip count is constant`, t.loc);
      } else this.pendingUnroll = "full";
      // the pragma applies to the statement that follows (e.g. the body of an enclosing for)
      if (this.is("}")) return { k: "empty", loc: t.loc };
      return this.statement();
    }
    if (t.k === "id") {
      if (t.v in UNSUPPORTED_KEYWORDS) fail("unsupported", UNSUPPORTED_KEYWORDS[t.v], t.loc);
      switch (t.v) {
        case "if": {
          this.next();
          const constexpr = this.accept("constexpr");
          this.expect("(");
          const c = this.conditionExpr();
          this.expect(")");
          const th = this.statement();
          const f = this.accept("else") ? this.statement() : undefined;
          return { k: "if", c, t: th, f, loc: t.loc, constexpr };
        }
        case "for": {
          this.next();
          const unroll = this.takeUnroll();
          this.expect("(");
          let init: Stmt | undefined;
          if (!this.accept(";")) {
            if (this.isDeclStart()) init = this.declStatement();
            else { const e = this.expression(); this.expect(";"); init = { k: "expr", e, loc: e.loc }; }
          }
          if (this.is(":")) fail("unsupported", "range-based for loops are not supported", this.peek().loc);
          const c = this.is(";") ? undefined : this.conditionExpr();
          this.expect(";");
          const step = this.is(")") ? undefined : this.expression();
          this.expect(")");
          const body = this.statement();
          return { k: "for", init, c, step, body, loc: t.loc, unroll };
        }
        case "while": {
          this.next();
          const unroll = this.takeUnroll();
          this.expect("("); const c = this.conditionExpr(); this.expect(")");
          return { k: "while", c, body: this.statement(), loc: t.loc, unroll };
        }
        case "do": {
          this.next();
          const unroll = this.takeUnroll();
          const body = this.statement();
          this.expect("while"); this.expect("("); const c = this.expression(); this.expect(")"); this.expect(";");
          return { k: "do", c, body, loc: t.loc, unroll };
        }
        case "switch": {
          this.next(); this.expect("("); const c = this.expression(); this.expect(")");
          return { k: "switch", c, body: this.statement(), loc: t.loc };
        }
        case "case": { this.next(); const v = this.conditional(); this.expect(":"); return { k: "case", value: v, loc: t.loc }; }
        case "default": { if (this.is(":", 1)) { this.next(); this.next(); return { k: "case", value: null, loc: t.loc }; } break; }
        case "break": this.next(); this.expect(";"); return { k: "break", loc: t.loc };
        case "continue": this.next(); this.expect(";"); return { k: "continue", loc: t.loc };
        case "return": {
          this.next();
          if (this.accept(";")) return { k: "return", loc: t.loc };
          const e = this.is("{") ? this.initializer() : this.expression();
          this.expect(";");
          return { k: "return", e, loc: t.loc };
        }
        case "typedef": { const tmp: Stmt[] = []; this.typedef(tmp); return tmp.length === 1 ? tmp[0] : { k: "block", body: tmp, loc: t.loc }; }
        case "using": {
          this.next();
          if (this.accept("namespace")) { while (!this.accept(";")) this.next(); return { k: "empty", loc: t.loc }; }
          const n = this.ident();
          if (this.accept("=")) { const ty = this.typeName(); this.expect(";"); this.typeNames.add(n.v); return { k: "typedef", name: n.v, type: ty, loc: n.loc }; }
          while (!this.accept(";")) this.next();
          return { k: "empty", loc: t.loc };
        }
        case "static_assert": { this.next(); this.skipBalanced(); this.expect(";"); return { k: "empty", loc: t.loc }; }
        case "struct": case "union": case "class":
          if (this.is("{", 1) || this.is("{", 2)) { const sd = this.structDecl(undefined); this.expect(";"); return { k: "struct", decl: sd, loc: t.loc }; }
          break;
      }
      if (t.v.startsWith("__") || true) {
        // label `name:` (goto target) is unsupported
        if (this.is(":", 1) && !this.is("::", 1) && this.peek(1).v === ":" && t.v !== "default") fail("unsupported", "labels and goto are not supported", t.loc);
      }
    }
    if (this.is("{")) return this.block();
    if (this.accept(";")) return { k: "empty", loc: t.loc };
    if (this.isDeclStart()) return this.declStatement();
    const e = this.expression();
    this.expect(";", "after expression");
    return { k: "expr", e, loc: t.loc };
  }

  /** if/while conditions may declare a variable: `if (int x = f())` is not supported; plain expressions are */
  private conditionExpr(): Expr { return this.expression(); }

  private isDeclStart(): boolean {
    const t = this.peek();
    if (t.k !== "id") return false;
    if (DECL_SPECS.has(t.v) && t.v !== "__launch_bounds__") return true;
    if (!this.isTypeStart()) return false;
    if (PRIM_WORDS.has(t.v) || QUALS.has(t.v) || t.v === "auto" || t.v === "struct" || t.v === "typename" || t.v === "decltype") {
      // `float(x)` / `int{3}` at statement start is a functional cast expression
      return !((this.is("(", 1) || this.is("{", 1)) && (t.v === "float" || t.v === "int" || t.v === "double" || t.v === "bool" || t.v === "char" || t.v === "long" || t.v === "short" || t.v === "unsigned"));
    }
    // named type: a declaration if followed by a name, '*', '&', '<' (template), or '::'
    let o = 1;
    if (this.is("::", o)) return true;
    if (this.is("<", o) && this.templates.has(t.v)) {
      // skip template args
      let depth = 0;
      for (; this.peek(o).k !== "eof"; o++) {
        const v = this.peek(o).v;
        if (v === "<") depth++;
        else if (v === ">") { depth--; if (depth === 0) { o++; break; } }
        else if (v === ">>") { depth -= 2; if (depth <= 0) { o++; break; } }
        else if (v === ";" || v === "{") return false;
      }
      if (this.is("::", o)) return true;
    }
    const n = this.peek(o);
    return n.k === "id" || n.v === "*" || n.v === "&" || n.v === "&&";
  }

  private declStatement(): Stmt {
    const loc = this.peek().loc;
    const ds = this.declSpecs();
    if (!ds.type) fail("syntax", "expected a type in declaration", loc);
    const decls: VarDecl[] = [];
    do {
      const d = this.declarator(ds.type);
      const vd: VarDecl = {
        name: d.name, type: d.type, loc: d.loc, isStatic: ds.specs.has("static"), isConstexpr: ds.specs.has("constexpr"), isExtern: ds.specs.has("extern"),
        space: ds.specs.has("__shared__") ? "shared" : ds.specs.has("__constant__") ? "constant" : undefined, align: ds.align,
      };
      if (this.accept("=")) vd.init = this.initializer();
      else if (this.is("{")) vd.init = this.initializer();
      else if (this.is("(")) { // direct initialization: float x(1.0f); dim3 g(4, 1);
        const l = this.next().loc;
        const args: Expr[] = [];
        if (!this.is(")")) do args.push(this.assignment()); while (this.accept(","));
        this.expect(")");
        vd.init = { k: "construct", type: d.type, args, braces: false, loc: l };
      }
      decls.push(vd);
    } while (this.accept(","));
    this.expect(";", "after declaration");
    return { k: "decl", decls, loc };
  }

  // ------------------------------------------------------------------ expressions
  expression(): Expr {
    let e = this.assignment();
    while (this.is(",")) { const loc = this.next().loc; e = { k: "comma", l: e, r: this.assignment(), loc }; }
    return e;
  }

  assignment(): Expr {
    const l = this.conditional();
    const t = this.peek();
    if (t.k === "punct" && ASSIGN_OPS.has(t.v)) {
      this.next();
      const r = this.is("{") ? this.initializer() : this.assignment();
      return { k: "assign", op: t.v, l, r, loc: t.loc };
    }
    return l;
  }

  conditional(inTemplate = false): Expr {
    const c = this.binary(3, inTemplate);
    if (this.is("?")) {
      const loc = this.next().loc;
      const t = this.expression();
      this.expect(":");
      const f = inTemplate ? this.conditional(true) : this.assignment();
      return { k: "cond", c, t, f, loc };
    }
    return c;
  }

  private binary(minPrec: number, inTemplate: boolean): Expr {
    let l = this.unary();
    for (;;) {
      const t = this.peek();
      if (t.k !== "punct") return l;
      if (inTemplate && (t.v === ">" || t.v === ">>" || t.v === ">=" || t.v === ">>=")) return l;
      const prec = BINARY_PREC[t.v];
      if (prec === undefined || prec < minPrec) return l;
      this.next();
      const r = this.binary(prec + 1, inTemplate);
      l = { k: "binary", op: t.v, l, r, loc: t.loc };
    }
  }

  private unary(): Expr {
    const t = this.peek();
    if (t.k === "punct") {
      if (t.v === "++" || t.v === "--") { this.next(); return { k: "unary", op: t.v, e: this.unary(), loc: t.loc }; }
      if (t.v === "-" || t.v === "+" || t.v === "!" || t.v === "~" || t.v === "*" || t.v === "&") {
        this.next(); return { k: "unary", op: t.v, e: this.unary(), loc: t.loc };
      }
      if (t.v === "(" && this.isTypeStart(1) && this.isCast()) {
        this.next();
        const type = this.typeName();
        this.expect(")");
        if (this.is("{")) { const init = this.initializer(); return { k: "cast", type, e: init, style: "c", loc: t.loc }; }
        return { k: "cast", type, e: this.unary(), style: "c", loc: t.loc };
      }
    }
    if (t.k === "id") {
      if (t.v === "sizeof") {
        this.next();
        if (this.is("(") && this.isTypeStart(1)) { this.next(); const type = this.typeName(); this.expect(")"); return { k: "sizeof", type, loc: t.loc }; }
        if (this.is("...")) fail("unsupported", "sizeof... is not supported", t.loc);
        return { k: "sizeof", e: this.unary(), loc: t.loc };
      }
      if (t.v === "alignof" || t.v === "__alignof__") { this.next(); this.expect("("); const type = this.typeName(); this.expect(")"); return { k: "alignof", type, loc: t.loc }; }
      if (t.v in UNSUPPORTED_KEYWORDS) fail("unsupported", UNSUPPORTED_KEYWORDS[t.v], t.loc);
    }
    return this.postfix(this.primary());
  }

  private isCast(): boolean {
    // (Type) expr  vs  (expr): we already know the token after '(' starts a type; check that a ')' closes it with only
    // type-ish tokens in between
    let o = 1, depth = 0;
    for (; this.peek(o).k !== "eof"; o++) {
      const v = this.peek(o);
      if (v.v === "(") { depth++; continue; }
      if (v.v === ")") { if (depth === 0) break; depth--; continue; }
      if (v.k === "num" && depth === 0) {
        // array type inside a cast, e.g. (float(*)[4]) is unsupported; numbers only appear inside template args
        continue;
      }
    }
    // a cast must be followed by an operand
    const after = this.peek(o + 1);
    if (after.k === "eof") return false;
    if (after.k === "punct" && [")", ";", ",", "]", "}", "?", ":", "=", "==", "!=", "<", ">", "<=", ">=", "&&", "||", "|", "^", "/", "%", ".", "->", "<<", ">>"].includes(after.v)) {
      // `(T)` followed by an operator: `(x) + 1` where x is a type? only valid for unary ops; treat as not-a-cast
      return false;
    }
    return true;
  }

  private primary(): Expr {
    const t = this.peek();
    switch (t.k) {
      case "num": this.next(); return { k: "num", text: t.v, loc: t.loc };
      case "char": this.next(); return { k: "char", value: t.v.charCodeAt(0) || 0, loc: t.loc };
      case "str": {
        this.next();
        let v = t.v;
        while (this.peek().k === "str") v += this.next().v; // adjacent string literal concatenation
        return { k: "str", value: v, loc: t.loc };
      }
      case "punct":
        if (t.v === "(") { this.next(); const e = this.expression(); this.expect(")"); return e; }
        if (t.v === "{") return this.initializer();
        if (t.v === "::") { this.next(); return this.primary(); }
        if (t.v === "[") fail("unsupported", "lambdas are not supported in device code", t.loc);
        break;
      case "id": {
        if (t.v === "true" || t.v === "false") { this.next(); return { k: "bool", value: t.v === "true", loc: t.loc }; }
        if (t.v === "nullptr" || t.v === "NULL") { this.next(); return { k: "nullptr", loc: t.loc }; }
        if (t.v === "this") { this.next(); return { k: "this", loc: t.loc }; }
        if (t.v === "static_cast" || t.v === "reinterpret_cast" || t.v === "const_cast") {
          this.next();
          this.expect("<"); const type = this.typeName(); this.closeAngle();
          this.expect("("); const e = this.expression(); this.expect(")");
          return { k: "cast", type, e, style: t.v === "static_cast" ? "static" : t.v === "reinterpret_cast" ? "reinterpret" : "const", loc: t.loc };
        }
        if (this.isTypeStart() && !this.is("::", 1)) {
          // functional cast / construction: float(x), float4{...}, Vec<float,4>(...), unsigned(x)
          const save = this.p;
          const type = this.typeName();
          if (this.is("(")) {
            const loc = this.next().loc;
            const args: Expr[] = [];
            if (!this.is(")")) do args.push(this.assignment()); while (this.accept(","));
            this.expect(")");
            if (args.length === 1 && (type.k === "prim" || type.k === "named")) return { k: "cast", type, e: args[0], style: "functional", loc };
            return { k: "construct", type, args, braces: false, loc };
          }
          if (this.is("{")) {
            const init = this.initializer() as Expr & { k: "init" };
            return { k: "construct", type, args: init.items, braces: true, loc: t.loc };
          }
          this.p = save;
        }
        this.next();
        let name = t.v;
        // qualified names: std::min, ns::f -> last component (namespaces are flattened)
        while (this.is("::") && this.peek(1).k === "id") { this.next(); name = this.next().v; }
        let targs: TemplateArg[] | undefined;
        if (this.is("<") && this.templates.has(name)) {
          const save = this.p;
          try { targs = this.templateArgs(); } catch (e) { this.p = save; targs = undefined; }
          if (targs && this.is("::") && this.peek(1).k === "id") { this.next(); name = `${name}::${this.next().v}`; }
        }
        const id: Expr = { k: "ident", name, targs, loc: t.loc };
        if (this.is("<<<")) {
          const loc = this.next().loc;
          const config: Expr[] = [];
          do config.push(this.assignment()); while (this.accept(","));
          this.expect(">>>");
          this.expect("(");
          const args: Expr[] = [];
          if (!this.is(")")) do args.push(this.assignment()); while (this.accept(","));
          this.expect(")");
          return { k: "launch", callee: id, config, args, loc };
        }
        return id;
      }
    }
    fail("syntax", `unexpected ${t.k === "eof" ? "end of input" : `'${t.v}'`} in expression`, t.loc);
  }

  private postfix(e: Expr): Expr {
    for (;;) {
      const t = this.peek();
      if (t.k !== "punct") return e;
      if (t.v === "[") { this.next(); const i = this.expression(); this.expect("]"); e = { k: "index", base: e, index: i, loc: t.loc }; continue; }
      if (t.v === "(") {
        this.next();
        const args: Expr[] = [];
        if (!this.is(")")) do args.push(this.is("{") ? this.initializer() : this.assignment()); while (this.accept(","));
        this.expect(")");
        e = { k: "call", callee: e, args, loc: t.loc };
        continue;
      }
      if (t.v === "." || t.v === "->") {
        this.next();
        if (this.is("template")) this.next();
        let name: string;
        if (this.is("operator")) { this.next(); name = "operator" + this.next().v; }
        else if (this.is("~")) fail("unsupported", "explicit destructor calls are not supported", t.loc);
        else name = this.ident("member name").v;
        e = { k: "member", base: e, name, arrow: t.v === "->", loc: t.loc };
        continue;
      }
      if (t.v === "++" || t.v === "--") { this.next(); e = { k: "postfix", op: t.v, e, loc: t.loc }; continue; }
      return e;
    }
  }
}

function canonicalPrim(words: string[], loc: SourceLoc): string {
  const count = (w: string) => words.filter((x) => x === w).length;
  const unsigned = count("unsigned") > 0, signed = count("signed") > 0;
  if (unsigned && signed) fail("type", "both 'signed' and 'unsigned' in type", loc);
  const longs = count("long");
  if (words.includes("__int64")) return unsigned ? "unsigned long long" : "long long";
  if (words.includes("void")) return "void";
  if (words.includes("bool")) return "bool";
  if (words.includes("float")) return "float";
  if (words.includes("double")) return "double"; // long double is treated as double
  if (words.includes("char")) return unsigned ? "unsigned char" : signed ? "signed char" : "char";
  if (words.includes("short")) return unsigned ? "unsigned short" : "short";
  if (longs >= 2) return unsigned ? "unsigned long long" : "long long";
  if (longs === 1) return unsigned ? "unsigned long" : "long";
  return unsigned ? "unsigned int" : "int";
}
