/** Tokenizer for CUDA C/C++ source (used both before and after preprocessing). */
import { fail, type SourceLoc } from "./diagnostics.js";

export type TokKind = "id" | "num" | "str" | "char" | "punct" | "pragma" | "eof";

export interface Token {
  k: TokKind;
  v: string;
  loc: SourceLoc;
  /** first token on its (logical) line: used to recognize preprocessor directives */
  bol: boolean;
  /** whitespace precedes the token: distinguishes `#define F(x)` from `#define F (x)` */
  space: boolean;
  /** macros that must not be expanded again in this token (C preprocessor hide set) */
  hide?: Set<string>;
  /** for "pragma" tokens: the pragma's argument tokens */
  args?: Token[];
}

const PUNCTS = [
  "<<<", ">>>", "...", "<<=", ">>=",
  "->", "++", "--", "<<", ">>", "<=", ">=", "==", "!=", "&&", "||", "+=", "-=", "*=", "/=", "%=", "&=", "^=", "|=", "::", "##",
  "#", "(", ")", "[", "]", "{", "}", ",", ";", ":", "?", ".", "+", "-", "*", "/", "%", "&", "|", "^", "~", "!", "=", "<", ">",
];

const isIdStart = (c: string) => (c >= "a" && c <= "z") || (c >= "A" && c <= "Z") || c === "_" || c === "$";
const isIdChar = (c: string) => isIdStart(c) || (c >= "0" && c <= "9");
const isDigit = (c: string) => c >= "0" && c <= "9";

export const MAX_SOURCE_BYTES = 4 * 1024 * 1024;

export function tokenize(src: string, file = "<source>"): Token[] {
  if (src.length > MAX_SOURCE_BYTES) fail("limit", `source is ${src.length} bytes; the limit is ${MAX_SOURCE_BYTES}`);
  const out: Token[] = [];
  let i = 0, line = 1, col = 1, bol = true, space = false;
  const n = src.length;
  const adv = (k = 1) => {
    for (let j = 0; j < k; j++) {
      if (src[i] === "\n") { line++; col = 1; } else col++;
      i++;
    }
  };
  while (i < n) {
    const c = src[i];
    // line continuation
    if (c === "\\" && (src[i + 1] === "\n" || (src[i + 1] === "\r" && src[i + 2] === "\n"))) { adv(src[i + 1] === "\r" ? 3 : 2); space = true; continue; }
    if (c === "\n") { adv(); bol = true; space = true; continue; }
    if (c === " " || c === "\t" || c === "\r" || c === "\f" || c === "\v") { adv(); space = true; continue; }
    if (c === "/" && src[i + 1] === "/") { while (i < n && src[i] !== "\n") { if (src[i] === "\\" && src[i + 1] === "\n") adv(); adv(); } space = true; continue; }
    if (c === "/" && src[i + 1] === "*") {
      const start: SourceLoc = { file, line, col };
      adv(2);
      while (i < n && !(src[i] === "*" && src[i + 1] === "/")) adv();
      if (i >= n) fail("syntax", "unterminated /* comment", start);
      adv(2); space = true; continue;
    }
    const loc: SourceLoc = { file, line, col };
    const push = (k: TokKind, v: string) => { out.push({ k, v, loc, bol, space }); bol = false; space = false; };
    if (isIdStart(c)) {
      let j = i; while (j < n && isIdChar(src[j])) j++;
      const v = src.slice(i, j);
      adv(j - i); push("id", v); continue;
    }
    if (isDigit(c) || (c === "." && isDigit(src[i + 1] ?? ""))) {
      // pp-number: digits, letters, '.', and e+/e-/p+/p-
      let j = i + 1;
      while (j < n) {
        const d = src[j];
        // e+/e- belong to a decimal exponent; in hex literals only p+/p- do (0x1e+1 is 0x1e + 1)
        const hex = /^0[xX]/.test(src.slice(i, j));
        if ((d === "+" || d === "-") && (hex ? /[pP]/ : /[eEpP]/).test(src[j - 1])) { j++; continue; }
        if (isIdChar(d) || d === "." || d === "'") { j++; continue; }
        break;
      }
      const v = src.slice(i, j).replace(/'/g, "");
      adv(j - i); push("num", v); continue;
    }
    if (c === '"' || c === "'" || ((c === "L" || c === "u" || c === "U") && (src[i + 1] === '"' || src[i + 1] === "'"))) {
      let j = i; if (c !== '"' && c !== "'") j++;
      const q = src[j]; j++;
      let v = "";
      while (j < n && src[j] !== q) {
        if (src[j] === "\n") fail("syntax", "unterminated string or character literal", loc);
        if (src[j] === "\\") {
          const e = src[j + 1];
          const esc: Record<string, string> = { n: "\n", t: "\t", r: "\r", "0": "\0", "\\": "\\", "'": "'", '"': '"', a: "\x07", b: "\b", f: "\f", v: "\v", "?": "?" };
          if (e === "x") {
            let k = j + 2, h = ""; while (/[0-9a-fA-F]/.test(src[k] ?? "") && h.length < 8) h += src[k++];
            v += String.fromCharCode(parseInt(h || "0", 16)); j = k; continue;
          }
          if (e >= "0" && e <= "7") {
            let k = j + 1, o = ""; while (src[k] >= "0" && src[k] <= "7" && o.length < 3) o += src[k++];
            v += String.fromCharCode(parseInt(o, 8)); j = k; continue;
          }
          v += esc[e] ?? e; j += 2; continue;
        }
        v += src[j++];
      }
      if (j >= n) fail("syntax", "unterminated string or character literal", loc);
      adv(j + 1 - i); push(q === '"' ? "str" : "char", v); continue;
    }
    let matched = "";
    for (const p of PUNCTS) if (src.startsWith(p, i)) { matched = p; break; }
    if (!matched) fail("syntax", `unexpected character '${c}'`, loc);
    adv(matched.length); push("punct", matched);
  }
  out.push({ k: "eof", v: "", loc: { file, line, col }, bol: true, space: true });
  return out;
}

export interface NumberLit { value: number | bigint; isFloat: boolean; type: string }

/** Parse a C numeric literal into its value and C type (int, unsigned int, long, ..., float, double). */
export function parseNumber(text: string, loc?: SourceLoc): NumberLit {
  let t = text.toLowerCase();
  const isHex = t.startsWith("0x");
  const isFloat = !isHex ? /[.e]/.test(t) || (/f$/.test(t) && /[.e]/.test(t)) : /p/.test(t);
  if (isFloat) {
    let type = "double";
    if (t.endsWith("f")) { type = "float"; t = t.slice(0, -1); }
    else if (t.endsWith("l")) { type = "double"; t = t.slice(0, -1); } // long double -> double (documented)
    let value: number;
    if (isHex) {
      const m = /^0x([0-9a-f]*)(?:\.([0-9a-f]*))?p([+-]?\d+)$/.exec(t);
      if (!m) fail("syntax", `invalid hexadecimal floating literal '${text}'`, loc);
      const mant = parseInt((m[1] || "0") + (m[2] || ""), 16) / Math.pow(16, (m[2] || "").length);
      value = mant * Math.pow(2, parseInt(m[3], 10));
    } else {
      if (!/^(\d+\.?\d*|\.\d+)(e[+-]?\d+)?$/.test(t)) fail("syntax", `invalid floating literal '${text}'`, loc);
      value = Number(t);
    }
    return { value, isFloat: true, type };
  }
  // integer
  const m = /^(0x[0-9a-f]+|0b[01]+|0[0-7]*|[1-9]\d*)(u?l{0,2}u?)$/.exec(t);
  if (!m || (m[2].match(/u/g)?.length ?? 0) > 1) {
    if (/^\d+f$/.test(t)) fail("syntax", `invalid literal '${text}' (write ${text.slice(0, -1)}.0f)`, loc);
    fail("syntax", `invalid integer literal '${text}'`, loc);
  }
  const digits = m[1];
  const v = digits.startsWith("0b") ? BigInt("0b" + digits.slice(2)) : digits.startsWith("0x") ? BigInt(digits) :
    digits.length > 1 && digits.startsWith("0") ? BigInt("0o" + digits.slice(1)) : BigInt(digits);
  const suf = m[2];
  const unsigned = suf.includes("u");
  const longs = (suf.match(/l/g) ?? []).length;
  const decimal = !digits.startsWith("0") || digits === "0";
  // C++ rules for the type of an integer literal (LP64: long is 64-bit, as in nvcc on Linux)
  let cands: string[];
  if (longs === 0) cands = unsigned ? ["unsigned int", "unsigned long"] : decimal ? ["int", "long"] : ["int", "unsigned int", "long", "unsigned long"];
  else if (longs === 1) cands = unsigned ? ["unsigned long"] : decimal ? ["long"] : ["long", "unsigned long"];
  else cands = unsigned ? ["unsigned long long"] : decimal ? ["long long"] : ["long long", "unsigned long long"];
  const max: Record<string, bigint> = {
    int: 0x7fffffffn, "unsigned int": 0xffffffffn, long: 0x7fffffffffffffffn, "unsigned long": 0xffffffffffffffffn,
    "long long": 0x7fffffffffffffffn, "unsigned long long": 0xffffffffffffffffn,
  };
  const type = cands.find((c) => v <= max[c]);
  if (!type) fail("syntax", `integer literal '${text}' is too large`, loc);
  return { value: type === "int" || type === "unsigned int" ? Number(v) : v, isFloat: false, type };
}
