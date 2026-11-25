/**
 * IR optimization passes. Every pass preserves the kernel's explicit parallel structure (threads, blocks, barriers,
 * shared memory); they only remove redundant work.
 *
 *   fold         constant folding + algebraic simplification (exact C semantics, see fold.ts)
 *   copyprop     constant / copy propagation of single-assignment variables
 *   dce          dead code elimination (unused variables, unreachable statements, constant branches, empty loops)
 *   cse          common subexpression elimination over let-bound values (loads only between memory writes/barriers)
 *   unroll       full unrolling of counted loops (constant trip count; #pragma unroll honoured, also factor N)
 *   licm         loop-invariant code motion of pure, non-trapping expressions
 *   promote      scalar replacement of local arrays whose accesses all have constant offsets (registers instead of memory)
 *   barriers     removal of redundant __syncthreads (adjacent, or with no memory access in between, or at kernel end)
 */
import {
  bin, cnst, exprChildren, i32, mapExpr, mapStmtExprs, readVar, someExpr, stmtExprs, vtEq, vtSize, walkStmts,
  type Expr, type Kernel, type Root, type Stmt, type Var,
} from "./ir.js";
import { exprEq, foldExpr, isConst } from "./fold.js";
import { fail } from "./diagnostics.js";

export interface PassOptions {
  mode: "optimized" | "direct";
  /** maximum statements produced by unrolling one loop */
  maxUnrolledStatements?: number;
  /** passes to skip (for debugging / benchmarks) */
  disable?: string[];
  /** don't hoist potentially trapping integer division (WebAssembly traps on division by zero) */
  noTrapHoist?: boolean;
}

export interface PassReport { name: string; changes: number }

export function optimize(k: Kernel, opts: PassOptions): PassReport[] {
  const reports: PassReport[] = [];
  const on = (n: string) => !(opts.disable ?? []).includes(n);
  const run = (name: string, f: () => number) => { if (on(name)) reports.push({ name, changes: f() }); };
  if (opts.mode === "direct") {
    // direct translation: only the folding needed for constants (template arguments, blockDim) and dead constant branches
    run("fold", () => foldPass(k, false));
    run("dce", () => dcePass(k, false));
    return reports;
  }
  for (let iter = 0; iter < 2; iter++) {
    run("fold", () => foldPass(k, true));
    run("copyprop", () => copyPropPass(k));
    run("dce", () => dcePass(k, true));
  }
  run("unroll", () => unrollPass(k, opts.maxUnrolledStatements ?? 4096));
  for (let iter = 0; iter < 2; iter++) {
    run("fold", () => foldPass(k, true));
    run("copyprop", () => copyPropPass(k));
    run("promote", () => promoteLocalsPass(k));
    run("cse", () => csePass(k));
    run("dce", () => dcePass(k, true));
  }
  run("licm", () => licmPass(k, opts.noTrapHoist ?? true));
  run("barriers", () => barrierPass(k));
  run("fold", () => foldPass(k, true));
  run("copyprop", () => copyPropPass(k));
  run("dce", () => dcePass(k, true));
  return reports;
}

// ------------------------------------------------------------------ utilities
function mapBody(body: Stmt[], f: (s: Stmt) => Stmt[] | Stmt): Stmt[] {
  const out: Stmt[] = [];
  for (const s0 of body) {
    let s = s0;
    if (s.s === "if") s = { ...s, then: mapBody(s.then, f), else: mapBody(s.else, f) };
    else if (s.s === "loop") s = { ...s, body: mapBody(s.body, f), cont: mapBody(s.cont, f) };
    const r = f(s);
    if (Array.isArray(r)) out.push(...r); else out.push(r);
  }
  return out;
}

function mapAllExprs(body: Stmt[], f: (e: Expr) => Expr): Stmt[] {
  return mapBody(body, (s) => mapStmtExprs(s, f));
}

interface VarInfo { defs: number; uses: number; defStmt?: Stmt & { s: "let" }; defList?: Stmt[]; defIndex?: number; defLoopDepth?: number }

function varInfo(body: Stmt[]): Map<Var, VarInfo> {
  const info = new Map<Var, VarInfo>();
  const get = (v: Var) => { let i = info.get(v); if (!i) info.set(v, (i = { defs: 0, uses: 0 })); return i; };
  const countUses = (e: Expr) => { someExpr(e, (x) => { if (x.op === "var") get(x.v).uses++; return false; }); };
  const visit = (list: Stmt[], depth: number) => {
    list.forEach((s, idx) => {
      for (const e of stmtExprs(s)) countUses(e);
      if (s.s === "let") { const i = get(s.v); i.defs++; i.defStmt = s; i.defList = list; i.defIndex = idx; i.defLoopDepth = depth; }
      if (s.s === "atomic" && s.result) get(s.result).defs++;
      if (s.s === "warp") get(s.result).defs++;
      if (s.s === "if") { visit(s.then, depth); visit(s.else, depth); }
      if (s.s === "loop") { visit(s.body, depth + 1); visit(s.cont, depth + 1); }
    });
  };
  visit(body, 0);
  return info;
}

function substVars(e: Expr, m: Map<Var, Expr>): Expr {
  if (!m.size) return e;
  return mapExpr(e, (x) => (x.op === "var" && m.has(x.v) ? m.get(x.v)! : x));
}

function exprSize(e: Expr): number { let n = 1; for (const c of exprChildren(e)) n += exprSize(c); return n; }

const isLeaf = (e: Expr) => e.op === "const" || e.op === "param" || e.op === "builtin" || e.op === "var";

// ------------------------------------------------------------------ fold
function foldPass(k: Kernel, simplify: boolean): number {
  let n = 0;
  k.body = mapAllExprs(k.body, (e) => { const f = foldExpr(e, simplify); if (f !== e) n++; return f; });
  return n;
}

// ------------------------------------------------------------------ copy / constant propagation
function copyPropPass(k: Kernel): number {
  const info = varInfo(k.body);
  const subst = new Map<Var, Expr>();
  // single-definition variables whose value is a leaf or a pure load-free expression of stable leaves
  const stable = (e: Expr): boolean => !someExpr(e, (x) => x.op === "load" || (x.op === "var" && (info.get(x.v)?.defs ?? 0) !== 1));
  for (const [v, i] of info) {
    if (i.defs !== 1 || !i.defStmt || i.uses === 0) continue;
    const e = i.defStmt.e;
    if (!dominatesUses(k.body, i, v)) continue;
    if (isLeaf(e) && (e.op !== "var" || info.get(e.v)?.defs === 1)) subst.set(v, e);
    else if (stable(e) && exprSize(e) <= 3 && i.uses === 1) subst.set(v, e);
  }
  // resolve chains (a = b; b = c)
  for (const [v, e] of subst) subst.set(v, resolveChain(e, subst, 0));
  if (!subst.size) return fuseLoadsIntoNext(k);
  k.body = mapAllExprs(k.body, (e) => substVars(e, subst));
  // the definitions are now dead: DCE removes them
  return subst.size + fuseLoadsIntoNext(k);
}

function resolveChain(e: Expr, m: Map<Var, Expr>, depth: number): Expr {
  if (depth > 32) return e;
  return mapExpr(e, (x) => (x.op === "var" && m.has(x.v) ? resolveChain(m.get(x.v)!, m, depth + 1) : x));
}

/** All uses of v are after its (only) definition in the same statement list, or nested in later statements. */
function dominatesUses(body: Stmt[], i: VarInfo, v: Var): boolean {
  let bad = false;
  const uses = (s: Stmt) => stmtExprs(s).some((e) => someExpr(e, (x) => x.op === "var" && x.v === v));
  const scan = (l: Stmt[], allowed: boolean) => {
    for (let j = 0; j < l.length && !bad; j++) {
      const s = l[j];
      const ok = allowed || (l === i.defList && j > i.defIndex!);
      if (!ok && uses(s)) bad = true;
      if (s.s === "if") { scan(s.then, ok); scan(s.else, ok); }
      if (s.s === "loop") { scan(s.body, ok); scan(s.cont, ok); }
    }
  };
  scan(body, false);
  return !bad;
}

function containsUse(l: Stmt[], v: Var): boolean {
  let found = false;
  walkStmts(l, (s) => { if (!found && stmtExprs(s).some((e) => someExpr(e, (x) => x.op === "var" && x.v === v))) found = true; });
  return found;
}

/** t = load(...); <next statement uses t once> with no write in between: put the load into its use. */
function fuseLoadsIntoNext(k: Kernel): number {
  const info = varInfo(k.body);
  let n = 0;
  const fuse = (list: Stmt[]): Stmt[] => {
    const out: Stmt[] = [];
    for (let j = 0; j < list.length; j++) {
      let s = list[j];
      if (s.s === "if") s = { ...s, then: fuse(s.then), else: fuse(s.else) };
      if (s.s === "loop") s = { ...s, body: fuse(s.body), cont: fuse(s.cont) };
      out.push(s);
    }
    for (let j = 0; j + 1 < out.length; j++) {
      const s = out[j], nx = out[j + 1];
      if (s.s !== "let") continue;
      const vi = info.get(s.v);
      if (!vi || vi.defs !== 1 || vi.uses !== 1) continue;
      if (nx.s === "if" || nx.s === "loop") continue;
      const uses = stmtExprs(nx).reduce((a, e) => a + countVar(e, s.v), 0);
      if (uses !== 1) continue;
      // nx must not assign a variable that s.e reads before using it (it can't: expressions are evaluated first)
      if (nx.s === "let" && someExpr(s.e, (x) => x.op === "var" && x.v === nx.v) && nx.v !== s.v) {
        // evaluation of nx's expression happens before its assignment: safe
      }
      const m = new Map([[s.v, s.e]]);
      out[j + 1] = mapStmtExprs(nx, (e) => substVars(e, m));
      out[j] = { s: "comment", text: "" };
      n++;
    }
    return out.filter((s) => !(s.s === "comment" && s.text === ""));
  };
  k.body = fuse(k.body);
  return n;
}

function countVar(e: Expr, v: Var): number {
  let c = 0;
  someExpr(e, (x) => { if (x.op === "var" && x.v === v) c++; return false; });
  return c;
}

// ------------------------------------------------------------------ dead code elimination
function dcePass(k: Kernel, aggressive: boolean): number {
  let n = 0;
  for (let iter = 0; iter < 8; iter++) {
    const before = n;
    const info = varInfo(k.body);
    const prune = (list: Stmt[]): Stmt[] => {
      const out: Stmt[] = [];
      for (const s0 of list) {
        let s = s0;
        if (s.s === "let" && aggressive && (info.get(s.v)?.uses ?? 0) === 0) { n++; continue; }
        if (s.s === "comment" && aggressive) continue;
        if (s.s === "if") {
          if (isConst(s.c)) { n++; out.push(...prune(s.c.v ? s.then : s.else)); if (endsWithJump(out)) break; continue; }
          s = { ...s, then: prune(s.then), else: prune(s.else) };
          if (!s.then.length && !s.else.length) { n++; continue; }
        }
        if (s.s === "loop") {
          s = { ...s, body: prune(s.body), cont: prune(s.cont) };
          // loop { break; } -> nothing;  loop { <no jumps>; break } -> its body
          if (s.body.length && s.body[0].s === "break") { n++; continue; }
          if (aggressive && s.body.length && s.body[s.body.length - 1].s === "break" && !hasJump(s.body.slice(0, -1), true) && !s.cont.length) {
            n++; out.push(...s.body.slice(0, -1)); continue;
          }
        }
        if (s.s === "atomic" && s.result && aggressive && (info.get(s.result)?.uses ?? 0) === 0) s = { ...s, result: null };
        out.push(s);
        if (s.s === "break" || s.s === "continue" || s.s === "return") break; // unreachable rest
      }
      return out;
    };
    k.body = prune(k.body);
    if (n === before) break;
  }
  // drop unused variables from the variable list
  const used = new Set<Var>();
  walkStmts(k.body, (s) => {
    if (s.s === "let") used.add(s.v);
    if (s.s === "atomic" && s.result) used.add(s.result);
    if (s.s === "warp") used.add(s.result);
    for (const e of stmtExprs(s)) someExpr(e, (x) => { if (x.op === "var") used.add(x.v); return false; });
  });
  k.vars = k.vars.filter((v) => used.has(v));
  // drop roots that are no longer referenced (kernel parameters always stay)
  const roots = new Set<Root>();
  walkStmts(k.body, (s) => {
    if (s.s === "store" || s.s === "atomic") roots.add(s.root);
    for (const e of stmtExprs(s)) someExpr(e, (x) => { if (x.op === "load") roots.add(x.root); return false; });
  });
  k.roots = k.roots.filter((r) => r.param !== undefined || roots.has(r) || r.dynamic);
  return n;
}

const endsWithJump = (l: Stmt[]) => l.length > 0 && ["break", "continue", "return"].includes(l[l.length - 1].s);

/** does the list contain a break/continue that targets the enclosing loop (or a return when includeReturn)? */
function hasJump(list: Stmt[], includeReturn: boolean): boolean {
  for (const s of list) {
    if (s.s === "break" || s.s === "continue") return true;
    if (s.s === "return" && includeReturn) return true;
    if (s.s === "if" && (hasJump(s.then, includeReturn) || hasJump(s.else, includeReturn))) return true;
    if (s.s === "loop" && includeReturn && containsReturn(s.body)) return true;
  }
  return false;
}
function containsReturn(l: Stmt[]): boolean {
  let f = false;
  walkStmts(l, (s) => { if (s.s === "return") f = true; });
  return f;
}

// ------------------------------------------------------------------ common subexpression elimination
function csePass(k: Kernel): number {
  const info = varInfo(k.body);
  let n = 0;
  type Avail = { e: Expr; v: Var; readsVars: Set<Var>; loadRoots: Set<Root> };
  const collect = (e: Expr) => {
    const vars = new Set<Var>(), roots = new Set<Root>();
    someExpr(e, (x) => { if (x.op === "var") vars.add(x.v); if (x.op === "load") roots.add(x.root); return false; });
    return { vars, roots };
  };
  const killedBy = (list: Stmt[]) => {
    const vars = new Set<Var>(), roots = new Set<Root>();
    let barrier = false;
    walkStmts(list, (s) => {
      if (s.s === "let") vars.add(s.v);
      if (s.s === "store" || s.s === "atomic") roots.add(s.root);
      if (s.s === "atomic" && s.result) vars.add(s.result);
      if (s.s === "warp") { vars.add(s.result); barrier = true; }
      if (s.s === "barrier" || s.s === "fence") barrier = true;
    });
    return { vars, roots, barrier };
  };
  const kill = (avail: Avail[], vars: Set<Var>, roots: Set<Root>, barrier: boolean) =>
    avail.filter((a) => ![...a.readsVars].some((v) => vars.has(v)) && !vars.has(a.v) && !(barrier && a.loadRoots.size) &&
      ![...a.loadRoots].some((r) => roots.has(r)));
  const replace = (e: Expr, avail: Avail[]): Expr => {
    if (!avail.length) return e;
    return mapExpr(e, (x) => {
      if (isLeaf(x)) return x;
      for (const a of avail) if (exprEq(a.e, x)) { n++; return readVar(a.v); }
      return x;
    });
  };
  const visit = (list: Stmt[], avail0: Avail[]): Stmt[] => {
    let avail = avail0;
    const out: Stmt[] = [];
    for (const s0 of list) {
      let s = mapStmtExprs(s0, (e) => replace(e, avail));
      if (s.s === "if") {
        s = { ...s, then: visit(s.then, avail), else: visit(s.else, avail) };
        const k1 = killedBy(s.then), k2 = killedBy(s.else);
        avail = kill(kill(avail, k1.vars, k1.roots, k1.barrier), k2.vars, k2.roots, k2.barrier);
      } else if (s.s === "loop") {
        const kl = killedBy([...s.body, ...s.cont]);
        const inner = kill(avail, kl.vars, kl.roots, kl.barrier);
        s = { ...s, body: visit(s.body, inner), cont: visit(s.cont, inner) };
        avail = kill(avail, kl.vars, kl.roots, kl.barrier);
      } else {
        const kl = killedBy([s]);
        avail = kill(avail, kl.vars, kl.roots, kl.barrier);
      }
      if (s.s === "let" && !isLeaf(s.e) && info.get(s.v)?.defs === 1) {
        const c = collect(s.e);
        if (!c.vars.has(s.v)) avail = [...avail, { e: s.e, v: s.v, readsVars: c.vars, loadRoots: c.roots }];
      }
      out.push(s);
    }
    return out;
  };
  k.body = visit(k.body, []);
  return n;
}

// ------------------------------------------------------------------ loop unrolling
interface Counted { v: Var; start: Expr; cmp: string; bound: Expr; step: number; body: Stmt[] }

/** Recognize `v = c0; loop { if (!(v < c1)) break; body } continuing { v = v + c2 }` with body not writing v. */
function matchCounted(prev: Stmt | undefined, loop: Stmt & { s: "loop" }): Counted | null {
  if (!prev || prev.s !== "let" || loop.breakIf || loop.cont.length !== 1) return null;
  const v = prev.v;
  const step = loop.cont[0];
  if (step.s !== "let" || step.v !== v) return null;
  const se = step.e;
  if (se.op !== "binop" || (se.fn !== "+" && se.fn !== "-" && se.fn !== "<<" && se.fn !== ">>" && se.fn !== "*") || se.a.op !== "var" || se.a.v !== v || se.b.op !== "const") return null;
  const first = loop.body[0];
  if (!first || first.s !== "if" || first.else.length || first.then.length !== 1 || first.then[0].s !== "break") return null;
  const c = first.c;
  // !(v < bound) folds to (v >= bound)
  if (c.op !== "binop" || c.a.op !== "var" || c.a.v !== v || c.b.op !== "const") return null;
  const body = loop.body.slice(1);
  let writes = false;
  walkStmts(body, (s) => { if ((s.s === "let" && s.v === v) || ((s.s === "atomic" || s.s === "warp") && (s as { result: Var | null }).result === v)) writes = true; });
  if (writes || hasJump(body, true) || containsLoopJumps(body)) return null;
  if (!isConst(prev.e)) return null;
  const stepK = Number(se.b.v);
  return { v, start: prev.e, cmp: c.fn, bound: c.b, step: se.fn === "-" ? -stepK : stepK, body, ...{ op: se.fn } } as Counted & { op: string };
}
function containsLoopJumps(body: Stmt[]): boolean {
  let f = false;
  walkStmts(body, (s) => { if (s.s === "return") f = true; });
  return f;
}

function unrollPass(k: Kernel, maxStmts: number): number {
  let n = 0;
  const count = (l: Stmt[]) => { let c = 0; walkStmts(l, () => c++); return c; };
  const visit = (list: Stmt[]): Stmt[] => {
    const out: Stmt[] = [];
    for (let j = 0; j < list.length; j++) {
      let s = list[j];
      if (s.s === "if") s = { ...s, then: visit(s.then), else: visit(s.else) };
      if (s.s === "loop") {
        s = { ...s, body: visit(s.body), cont: visit(s.cont) };
        if (s.unroll !== "none") {
          const m = matchCounted(out[out.length - 1], s) as (Counted & { op: string }) | null;
          if (m) {
            // trip count by simulation (exact for any of the recognized update forms)
            const iters: (number | bigint)[] = [];
            let x = (m.start as { v: number | bigint }).v;
            const bound = (m.bound as { v: number | bigint }).v;
            const exits = (val: number | bigint) => {
              switch (m.cmp) {
                case ">=": return val >= bound; case ">": return val > bound; case "<=": return val <= bound; case "<": return val < bound;
                case "==": return val === bound; case "!=": return val !== bound; default: return true;
              }
            };
            let ok = true;
            const stepE = (list[j] as Stmt & { s: "loop" }).cont[0] as Stmt & { s: "let" };
            while (!exits(x)) {
              iters.push(x);
              if (iters.length > 1024) { ok = false; break; }
              const nx = foldExpr(substVars(stepE.e, new Map([[m.v, cnst(m.v.t, x)]])), true);
              if (!isConst(nx)) { ok = false; break; }
              x = nx.v as number | bigint;
            }
            const bodySize = count(m.body);
            const full = s.unroll === "full" || typeof s.unroll === "number" || (iters.length <= 16 && bodySize * iters.length <= 256);
            if (ok && full && bodySize * iters.length <= maxStmts && (typeof s.unroll !== "number" || s.unroll >= iters.length)) {
              const renamable = localDefs(m.body, k.body);
              for (const it of iters) {
                const m2 = new Map<Var, Expr>([[m.v, cnst(m.v.t, it)]]);
                out.push(...cloneBody(m.body, m2, renamable, k));
              }
              out.push({ s: "let", v: m.v, e: cnst(m.v.t, x) });
              n++;
              continue;
            }
            if (ok && typeof s.unroll === "number" && s.unroll > 1 && iters.length > s.unroll) {
              // partial unroll by factor F: repeat (check; body; step) F times per iteration
              const F = s.unroll;
              const check = (s.body[0]) as Stmt;
              const step = s.cont[0];
              const inner: Stmt[] = [];
              for (let f = 0; f < F; f++) {
                if (f > 0) inner.push(step, check);
                else inner.push(check);
                inner.push(...m.body);
              }
              out.push({ ...s, body: inner, unroll: undefined });
              n++;
              continue;
            }
          }
        }
      }
      out.push(s);
    }
    return out;
  };
  k.body = visit(k.body);
  return n;
}

/**
 * Variables that are private to one iteration of a loop body: first appearance is a top-level `let` of the body and they
 * are not referenced outside the body. Each unrolled copy gets fresh ones, so later passes see single assignments.
 */
function localDefs(body: Stmt[], whole: Stmt[]): Set<Var> {
  const inBody = new Set<Stmt>();
  walkStmts(body, (s) => inBody.add(s));
  const firstSeen = new Map<Var, "def" | "use">();
  const note = (v: Var, how: "def" | "use") => { if (!firstSeen.has(v)) firstSeen.set(v, how); };
  for (const s of body) {
    // uses in a statement happen before its own definition
    walkStmts([s], (x) => {
      for (const e of stmtExprs(x)) someExpr(e, (y) => { if (y.op === "var") note(y.v, "use"); return false; });
      if (x.s === "let" && x === s) note(x.v, "def");
      else if (x.s === "let") note(x.v, "use"); // nested definitions: not renamed
      if ((x.s === "atomic" || x.s === "warp") && x.result) note(x.result, "use");
    });
  }
  const outside = new Set<Var>();
  walkStmts(whole, (s) => {
    if (inBody.has(s)) return;
    for (const e of stmtExprs(s)) someExpr(e, (y) => { if (y.op === "var") outside.add(y.v); return false; });
    if (s.s === "let") outside.add(s.v);
  });
  const res = new Set<Var>();
  for (const [v, how] of firstSeen) if (how === "def" && !outside.has(v)) res.add(v);
  return res;
}

/** Copy statements, substituting variables; per-iteration variables are renamed. */
function cloneBody(body: Stmt[], m: Map<Var, Expr>, renamable: Set<Var>, k: Kernel): Stmt[] {
  const ren = new Map(m);
  let id = 1 + Math.max(0, ...k.vars.map((v) => v.id));
  for (const v of renamable) {
    const nv: Var = { id: id++, name: `${v.name.replace(/_u\d+$/, "")}_u${id}`, t: v.t };
    k.vars.push(nv);
    ren.set(v, readVar(nv));
  }
  return mapBody(body, (s) => {
    const s2 = mapStmtExprs(s, (e) => foldExpr(substVars(e, ren), true));
    if (s2.s === "let" && renamable.has(s2.v)) return { ...s2, v: (ren.get(s2.v) as Expr & { op: "var" }).v };
    return s2;
  });
}

// ------------------------------------------------------------------ loop-invariant code motion
function licmPass(k: Kernel, noTrap: boolean): number {
  let n = 0;
  const counter = { next: 1 + Math.max(0, ...k.vars.map((v) => v.id)) };
  const visit = (list: Stmt[]): Stmt[] => {
    const out: Stmt[] = [];
    for (const s0 of list) {
      let s = s0;
      if (s.s === "if") s = { ...s, then: visit(s.then), else: visit(s.else) };
      if (s.s === "loop") {
        s = { ...s, body: visit(s.body), cont: visit(s.cont) };
        const assigned = new Set<Var>();
        walkStmts([...s.body, ...s.cont], (x) => {
          if (x.s === "let") assigned.add(x.v);
          if ((x.s === "atomic" || x.s === "warp") && x.result) assigned.add(x.result);
        });
        const hoisted: { e: Expr; v: Var }[] = [];
        const invariant = (e: Expr): boolean => !someExpr(e, (x) =>
          x.op === "load" || (x.op === "var" && assigned.has(x.v)) ||
          (noTrap && x.op === "binop" && (x.fn === "/" || x.fn === "%") && !x.t.s.startsWith("f")) ||
          x.op === "builtin" && (x.name === "lane" || x.name === "warp") && false);
        const hoist = (e: Expr): Expr => {
          if (isLeaf(e) || exprSize(e) < 3) {
            // still look inside
          }
          return mapExprTopDown(e, (x) => {
            if (isLeaf(x) || exprSize(x) < 3 || !invariant(x)) return null;
            // base + constant: hoist only the base (address offsets fold into the access for free)
            if (x.op === "binop" && (x.fn === "+" || x.fn === "-") && (x.b.op === "const" || x.a.op === "const")) return null;
            const have = hoisted.find((h) => exprEq(h.e, x));
            if (have) return readVar(have.v);
            const v: Var = { id: counter.next++, name: `inv_${counter.next - 1}`, t: x.t };
            k.vars.push(v);
            hoisted.push({ e: x, v });
            n++;
            return readVar(v);
          });
        };
        const rewrite = (l: Stmt[]): Stmt[] => l.map((x) => {
          let y = mapStmtExprs(x, hoist);
          if (y.s === "if") y = { ...y, then: rewrite(y.then), else: rewrite(y.else) };
          if (y.s === "loop") y = { ...y, body: rewrite(y.body), cont: rewrite(y.cont) };
          return y;
        });
        s = { ...s, body: rewrite(s.body), cont: rewrite(s.cont), breakIf: s.breakIf ? hoist(s.breakIf) : null };
        for (const h of hoisted) out.push({ s: "let", v: h.v, e: h.e });
      }
      out.push(s);
    }
    return out;
  };
  k.body = visit(k.body);
  return n;
}

/** Replace maximal subtrees for which f returns a replacement (top-down). */
function mapExprTopDown(e: Expr, f: (e: Expr) => Expr | null): Expr {
  const r = f(e);
  if (r) return r;
  switch (e.op) {
    case "unop": case "cast": case "bitcast": case "extract": return { ...e, a: mapExprTopDown(e.a, f) } as Expr;
    case "binop": return { ...e, a: mapExprTopDown(e.a, f), b: mapExprTopDown(e.b, f) };
    case "select": return { ...e, c: mapExprTopDown(e.c, f), a: mapExprTopDown(e.a, f), b: mapExprTopDown(e.b, f) };
    case "call": return { ...e, args: e.args.map((x) => mapExprTopDown(x, f)) };
    case "load": return { ...e, off: mapExprTopDown(e.off, f) };
    case "vec": return { ...e, items: e.items.map((x) => mapExprTopDown(x, f)) };
    default: return e;
  }
}

// ------------------------------------------------------------------ local array promotion
function promoteLocalsPass(k: Kernel): number {
  let n = 0;
  for (const root of k.roots) {
    if (root.space !== "local" || root.bytes <= 0 || root.bytes > 1024) continue;
    // all accesses at constant offsets with one consistent access type per offset
    const slots = new Map<number, Expr["t"]>();
    let ok = true;
    const check = (off: Expr, t: Expr["t"]) => {
      if (!isConst(off)) { ok = false; return; }
      const o = Number(off.v);
      const prev = slots.get(o);
      if (prev && !vtEq(prev, t)) ok = false;
      slots.set(o, t);
    };
    walkStmts(k.body, (s) => {
      if ((s.s === "store") && s.root === root) check(s.off, s.t);
      if (s.s === "atomic" && s.root === root) ok = false;
      for (const e of stmtExprs(s)) someExpr(e, (x) => { if (x.op === "load" && x.root === root) check(x.off, x.t); return false; });
    });
    // overlapping slots (e.g. a float4 and a float at offset 4) can't be promoted
    const sorted = [...slots.entries()].sort((a, b) => a[0] - b[0]);
    for (let i = 1; i < sorted.length; i++) if (sorted[i - 1][0] + vtSize(sorted[i - 1][1]) > sorted[i][0]) ok = false;
    if (!ok || !slots.size) continue;
    const vars = new Map<number, Var>();
    let id = 1 + Math.max(0, ...k.vars.map((v) => v.id));
    for (const [o, t] of slots) { const v: Var = { id: id++, name: `${root.name}_${o}`, t }; vars.set(o, v); k.vars.push(v); }
    const hasInit = root.init !== undefined;
    k.body = mapBody(k.body, (s) => {
      const s2 = mapStmtExprs(s, (e) => mapExpr(e, (x) => (x.op === "load" && x.root === root ? readVar(vars.get(Number((x.off as { v: number }).v))!) : x)));
      if (s2.s === "store" && s2.root === root) return { s: "let", v: vars.get(Number((s2.off as { v: number }).v))!, e: s2.e };
      return s2;
    });
    if (hasInit) fail("internal", "initialized local root");
    n++;
  }
  return n;
}

// ------------------------------------------------------------------ barrier elimination
function barrierPass(k: Kernel): number {
  let n = 0;
  const touchesMemory = (s: Stmt): boolean => {
    if (s.s === "store" || s.s === "atomic" || s.s === "warp" || s.s === "printf") return true;
    if (stmtExprs(s).some((e) => someExpr(e, (x) => x.op === "load" && x.root.space !== "local"))) return true;
    if (s.s === "if") return s.then.some(touchesMemory) || s.else.some(touchesMemory);
    if (s.s === "loop") return true; // conservative
    return false;
  };
  const visit = (list: Stmt[], atKernelEnd: boolean): Stmt[] => {
    const out: Stmt[] = [];
    let sinceBarrier = false, seenBarrier = false;
    for (const s0 of list) {
      let s = s0;
      if (s.s === "if") s = { ...s, then: visit(s.then, false), else: visit(s.else, false) };
      if (s.s === "loop") s = { ...s, body: visit(s.body, false), cont: visit(s.cont, false) };
      if (s.s === "barrier" && s.scope === "block") {
        if (seenBarrier && !sinceBarrier) { n++; continue; } // nothing happened since the last barrier
        seenBarrier = true; sinceBarrier = false;
        out.push(s); continue;
      }
      if (s.s === "if" || s.s === "loop") { seenBarrier = false; }
      if (touchesMemory(s)) sinceBarrier = true;
      out.push(s);
    }
    if (atKernelEnd) {
      // trailing barriers before the end of the kernel synchronize nothing
      while (out.length && out[out.length - 1].s === "barrier") { out.pop(); n++; }
    }
    return out;
  };
  k.body = visit(k.body, true);
  return n;
}

export { i32, bin };
