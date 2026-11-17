#!/usr/bin/env node
// Minimal static server for tests, benchmarks and local experiments (serves the repository root).
//   PORT=8080 COI=1 node scripts/serve.mjs   COI=1 adds COOP/COEP headers (cross-origin isolation -> SharedArrayBuffer)
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const port = Number(process.env.PORT ?? 8080);
const coi = process.env.COI === "1";
const types = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm",
  ".json": "application/json", ".map": "application/json", ".css": "text/css", ".zip": "application/zip", ".whl": "application/zip",
  ".py": "text/plain", ".ts": "text/plain", ".md": "text/plain" };

http.createServer((req, res) => {
  const url = new URL(req.url ?? "/", "http://x");
  let file = path.normalize(path.join(root, decodeURIComponent(url.pathname)));
  if (!file.startsWith(root)) { res.writeHead(403).end(); return; }
  if (fs.existsSync(file) && fs.statSync(file).isDirectory()) file = path.join(file, "index.html");
  fs.readFile(file, (err, data) => {
    if (coi) { res.setHeader("Cross-Origin-Opener-Policy", "same-origin"); res.setHeader("Cross-Origin-Embedder-Policy", "require-corp"); }
    res.setHeader("Cache-Control", "no-store");
    if (err) { res.writeHead(404, { "Content-Type": "text/plain" }).end("not found"); return; }
    res.writeHead(200, { "Content-Type": types[path.extname(file)] ?? "application/octet-stream" }).end(data);
  });
}).listen(port, "127.0.0.1", () => console.log(`serving ${root} on http://127.0.0.1:${port} (cross-origin isolated: ${coi})`));
