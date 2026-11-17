import { defineConfig } from "vite";

// COOP/COEP make the page cross-origin isolated, which enables SharedArrayBuffer and therefore a "soft" Stop button
// (KeyboardInterrupt) that keeps the Python environment alive. Without them, Stop terminates and restarts the worker.
const isolation = { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" };

export default defineConfig({
  // tinygrad-web locates its worker and tinygrad wheel with `new URL(..., import.meta.url)`; pre-bundling would break that.
  optimizeDeps: { exclude: ["tinygrad-web"] },
  server: { headers: isolation },
  preview: { headers: isolation },
  base: "./", // relocatable build: serve dist/ from any path
  build: { target: "es2022" },
});
