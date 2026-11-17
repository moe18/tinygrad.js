#!/usr/bin/env node
// Download pinned third-party assets, verifying SHA-256 hashes.
//
//   node scripts/fetch-assets.mjs            -> tinygrad wheels into assets/
//   node scripts/fetch-assets.mjs --pyodide  -> also a self-hosted Pyodide subset into vendor/pyodide/
//
// The Pyodide subset (core runtime + numpy) is what tests, benchmarks and the demo use so they never depend on a CDN.
// Pass extra Pyodide package names after --pyodide to include them, e.g. `--pyodide sqlite3 micropip`.
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { PINS } from "./pins.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const cacheDir = path.join(root, ".cache");
fs.mkdirSync(cacheDir, { recursive: true });

const sha256 = (file) => crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");

async function download(url, dest, hash) {
  if (fs.existsSync(dest) && sha256(dest) === hash) return dest;
  console.log(`downloading ${url}`);
  // curl handles proxies/redirects robustly and streams large files to disk
  execFileSync("curl", ["-fsSL", "--retry", "4", "-o", dest + ".part", url], { stdio: "inherit" });
  const got = sha256(dest + ".part");
  if (got !== hash) throw new Error(`sha256 mismatch for ${url}: expected ${hash}, got ${got}`);
  fs.renameSync(dest + ".part", dest);
  return dest;
}

// tinygrad wheels (pure Python, MIT)
const assets = path.join(root, "assets");
fs.mkdirSync(assets, { recursive: true });
for (const w of PINS.tinygrad) await download(w.url, path.join(assets, w.file), w.sha256);
console.log(`tinygrad wheels ready in ${path.relative(root, assets)}/`);

const pi = process.argv.indexOf("--pyodide");
if (pi !== -1) {
  const extra = process.argv.slice(pi + 1).filter((a) => !a.startsWith("-"));
  const p = PINS.pyodide;
  const tarball = await download(p.url, path.join(cacheDir, path.basename(p.url)), p.sha256);
  const out = path.join(root, "vendor", "pyodide");
  fs.mkdirSync(out, { recursive: true });
  // resolve the packages to extract (with dependencies) from the lock file
  execFileSync("tar", ["-xjf", tarball, "-C", out, "--strip-components=1", "pyodide/pyodide-lock.json"]);
  const lock = JSON.parse(fs.readFileSync(path.join(out, "pyodide-lock.json"), "utf8"));
  const want = new Set();
  const visit = (name) => {
    const pkg = lock.packages[name];
    if (!pkg) throw new Error(`package ${name} is not in the Pyodide ${p.version} lock file`);
    if (want.has(pkg.file_name)) return;
    want.add(pkg.file_name);
    for (const d of pkg.depends) visit(d);
  };
  for (const name of ["numpy", ...extra]) visit(name);
  const files = [...p.coreFiles, ...want].map((f) => `pyodide/${f}`);
  execFileSync("tar", ["-xjf", tarball, "-C", out, "--strip-components=1", ...files], { stdio: "inherit" });
  console.log(`Pyodide ${p.version} subset (${files.length} files) ready in ${path.relative(root, out)}/`);
}
