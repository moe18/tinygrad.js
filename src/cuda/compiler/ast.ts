/** Abstract syntax tree of the supported CUDA C/C++ subset (see docs/CUDA.md for the grammar that is accepted). */
import type { SourceLoc } from "./diagnostics.js";

// ------------------------------------------------------------------ syntactic types (resolved by sema)
export type TypeExpr =
  | { k: "prim"; name: string; loc: SourceLoc; isConst?: boolean; isVolatile?: boolean }
  | { k: "named"; name: string; args?: TemplateArg[]; loc: SourceLoc; isConst?: boolean; isVolatile?: boolean }
  | { k: "ptr"; to: TypeExpr; loc: SourceLoc; isConst?: boolean; restrict?: boolean }
  | { k: "ref"; to: TypeExpr; loc: SourceLoc; rvalue?: boolean }
  | { k: "array"; of: TypeExpr; size: Expr | null; loc: SourceLoc }
  | { k: "auto"; loc: SourceLoc; isConst?: boolean }
  | { k: "decltype"; expr: Expr; loc: SourceLoc };

export type TemplateArg = { k: "type"; type: TypeExpr } | { k: "expr"; expr: Expr };

// ------------------------------------------------------------------ expressions
export type Expr =
  | { k: "num"; text: string; loc: SourceLoc }
  | { k: "char"; value: number; loc: SourceLoc }
  | { k: "str"; value: string; loc: SourceLoc }
  | { k: "bool"; value: boolean; loc: SourceLoc }
  | { k: "nullptr"; loc: SourceLoc }
  | { k: "ident"; name: string; targs?: TemplateArg[]; loc: SourceLoc }
  | { k: "binary"; op: string; l: Expr; r: Expr; loc: SourceLoc }
  | { k: "assign"; op: string; l: Expr; r: Expr; loc: SourceLoc }
  | { k: "unary"; op: "-" | "+" | "!" | "~" | "*" | "&" | "++" | "--"; e: Expr; loc: SourceLoc }
  | { k: "postfix"; op: "++" | "--"; e: Expr; loc: SourceLoc }
  | { k: "cond"; c: Expr; t: Expr; f: Expr; loc: SourceLoc }
  | { k: "call"; callee: Expr; args: Expr[]; loc: SourceLoc }
  | { k: "index"; base: Expr; index: Expr; loc: SourceLoc }
  | { k: "member"; base: Expr; name: string; arrow: boolean; loc: SourceLoc }
  | { k: "cast"; type: TypeExpr; e: Expr; style: "c" | "static" | "reinterpret" | "const" | "functional"; loc: SourceLoc }
  | { k: "construct"; type: TypeExpr; args: Expr[]; braces: boolean; loc: SourceLoc }
  | { k: "sizeof"; type?: TypeExpr; e?: Expr; loc: SourceLoc }
  | { k: "alignof"; type: TypeExpr; loc: SourceLoc }
  | { k: "comma"; l: Expr; r: Expr; loc: SourceLoc }
  | { k: "init"; items: Expr[]; designators?: (string | null)[]; loc: SourceLoc }
  | { k: "launch"; callee: Expr; config: Expr[]; args: Expr[]; loc: SourceLoc }
  | { k: "this"; loc: SourceLoc };

// ------------------------------------------------------------------ statements
export interface VarDecl {
  name: string;
  type: TypeExpr;
  init?: Expr;
  loc: SourceLoc;
  space?: "shared" | "constant" | "device" | "managed";
  isStatic?: boolean;
  isConstexpr?: boolean;
  isExtern?: boolean;
  align?: number;
}

export type Stmt =
  | { k: "block"; body: Stmt[]; loc: SourceLoc }
  | { k: "expr"; e: Expr; loc: SourceLoc }
  | { k: "decl"; decls: VarDecl[]; loc: SourceLoc }
  | { k: "if"; c: Expr; t: Stmt; f?: Stmt; loc: SourceLoc; constexpr?: boolean }
  | { k: "for"; init?: Stmt; c?: Expr; step?: Expr; body: Stmt; loc: SourceLoc; unroll?: number | "full" | "none" }
  | { k: "while"; c: Expr; body: Stmt; loc: SourceLoc; unroll?: number | "full" | "none" }
  | { k: "do"; c: Expr; body: Stmt; loc: SourceLoc; unroll?: number | "full" | "none" }
  | { k: "switch"; c: Expr; body: Stmt; loc: SourceLoc }
  | { k: "case"; value: Expr | null; loc: SourceLoc } // null = default
  | { k: "break"; loc: SourceLoc }
  | { k: "continue"; loc: SourceLoc }
  | { k: "return"; e?: Expr; loc: SourceLoc }
  | { k: "empty"; loc: SourceLoc }
  | { k: "typedef"; name: string; type: TypeExpr; loc: SourceLoc }
  | { k: "struct"; decl: StructDecl; loc: SourceLoc };

// ------------------------------------------------------------------ top level
export interface TemplateParam { kind: "type" | "value"; name: string; valueType?: TypeExpr; def?: TemplateArg }

export interface Param { name: string; type: TypeExpr; def?: Expr; loc: SourceLoc }

export interface FuncDecl {
  k: "func";
  name: string;
  params: Param[];
  ret: TypeExpr;
  body?: Stmt & { k: "block" };
  global: boolean;
  device: boolean;
  host: boolean;
  inline: "default" | "force" | "never";
  template?: TemplateParam[];
  /** explicit specialization arguments: template<> T f<int>(...) */
  specArgs?: TemplateArg[];
  launchBounds?: number[];
  /** member function: name of the struct, and whether it is const */
  memberOf?: string;
  isConstMethod?: boolean;
  isStatic?: boolean;
  /** constructor member initializers  : x(a), y(b) */
  ctorInits?: { name: string; args: Expr[] }[];
  loc: SourceLoc;
}

export interface StructField { name: string; type: TypeExpr; init?: Expr; loc: SourceLoc }
export interface StructDecl {
  k: "struct";
  name: string;
  fields: StructField[];
  methods: FuncDecl[];
  template?: TemplateParam[];
  align?: number;
  isUnion?: boolean;
  loc: SourceLoc;
}

export interface EnumDecl { k: "enum"; name?: string; items: { name: string; value?: Expr; loc: SourceLoc }[]; loc: SourceLoc }

export type TopDecl =
  | FuncDecl
  | StructDecl
  | EnumDecl
  | { k: "var"; decl: VarDecl }
  | { k: "typedef"; name: string; type: TypeExpr; template?: TemplateParam[]; loc: SourceLoc }
  | { k: "host"; name: string; loc: SourceLoc }
  | { k: "static_assert"; e: Expr; msg?: string; loc: SourceLoc };

export interface TranslationUnit { decls: TopDecl[] }
