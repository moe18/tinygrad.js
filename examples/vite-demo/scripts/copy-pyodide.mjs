// Copy the self-hosted Pyodide subset (created by `npm run fetch:assets -- --pyodide` in the repository root) into
// public/pyodide so the demo works offline. Set VITE_PYODIDE_URL to use a CDN instead.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
const here = path.dirname(fileURLToPath(import.meta.url));
const src = path.resolve(here, "../../../vendor/pyodide");
const dest = path.resolve(here, "../public/pyodide");
if (!fs.existsSync(src)) {
  console.warn(`[demo] ${src} not found; the demo will load Pyodide from the CDN (set VITE_PYODIDE_URL to override).`);
} else {
  fs.mkdirSync(dest, { recursive: true });
  for (const f of fs.readdirSync(src)) fs.copyFileSync(path.join(src, f), path.join(dest, f));
  console.log(`[demo] copied Pyodide to ${path.relative(process.cwd(), dest)}`);
}
