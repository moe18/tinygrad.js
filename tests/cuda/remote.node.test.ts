/**
 * Remote native CUDA (device "cuda_remote"): the TypeScript client against the reference worker gateway
 * (server/cuda-worker), started as a real process with real session tokens, CORS, limits and per-session child
 * processes. Machines running the test suite have no NVIDIA GPU, so the gateway is started with a test-only engine
 * (tests/cuda/remote/wasm_engine.py) that executes the kernels with the WebAssembly backend; everything between the
 * client and the engine (HTTP protocol, argument encoding, isolation, errors) is the production code path.
 */
import { test, expect } from "@playwright/test";
import { spawn, execFileSync, type ChildProcess } from "node:child_process";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { createCudaContext } from "../../src/cuda/device.js";
import { CudaError } from "../../src/cuda/runtime/errors.js";

const root = path.resolve(import.meta.dirname, "../..");
const workerDir = path.join(root, "server/cuda-worker");
// TGJS_CUDA_TEST_NATIVE=1 runs the same tests against the NVRTC + driver API engine (needs an NVIDIA GPU)
const NATIVE = Boolean(process.env.TGJS_CUDA_TEST_NATIVE);
const SECRET = "test-secret-0123456789abcdef0123456789abcdef";
const PORT = 18000 + Math.floor(Math.random() * 2000);
const endpoint = `http://127.0.0.1:${PORT}`;
const env = { ...process.env, PYTHONPATH: [workerDir, path.join(root, "tests/cuda/remote")].join(path.delimiter), TGJS_CUDA_TOKEN_SECRET: SECRET };
const token = (sub: string, ...extra: string[]) => execFileSync("python3", ["-m", "tgjs_cuda_worker", "issue-token", sub, ...extra], { env, encoding: "utf8" }).trim();

let server: ChildProcess;
test.describe.configure({ mode: "serial" });
test.beforeAll(async () => {
  server = spawn("python3", ["-m", "tgjs_cuda_worker", "serve", "--port", String(PORT), ...(NATIVE ? [] : ["--engine", "wasm_engine:WasmEngine"]), "--time-limit-ms", "8000",
    "--memory-bytes", String(1 << 20), "--max-sessions", "3", "--allowed-origin", "https://app.example"], { env, stdio: ["ignore", "pipe", "inherit"] });
  await new Promise<void>((resolve, reject) => {
    server.stdout!.on("data", (d: Buffer) => { if (d.toString().includes("listening")) resolve(); });
    server.on("exit", (c) => reject(new Error(`worker exited with ${c}`)));
  });
});
test.afterAll(() => { server?.kill("SIGTERM"); });

const remote = (tok?: string) => createCudaContext({ device: "cuda_remote", remote: { endpoint, token: tok } });

test("refuses to start without a token secret, and --no-auth only on loopback", () => {
  const run = (args: string[], e: NodeJS.ProcessEnv) => { try { execFileSync("python3", ["-m", "tgjs_cuda_worker", "serve", ...args], { env: e, encoding: "utf8", stdio: "pipe", timeout: 10_000 }); return ""; } catch (err) { return String((err as { stderr: string }).stderr); } };
  expect(run(["--port", "0"], { ...env, TGJS_CUDA_TOKEN_SECRET: "" })).toMatch(/TGJS_CUDA_TOKEN_SECRET .* is required/);
  expect(run(["--port", "0", "--host", "0.0.0.0", "--no-auth"], env)).toMatch(/only allowed on a loopback address/);
});

test("authentication: missing, forged and expired tokens are rejected", async () => {
  await expect(remote()).rejects.toThrow(/cudaErrorInitializationError: remote CUDA worker: missing bearer token/);
  const good = token("alice");
  await expect(remote(good.slice(0, -2) + "xx")).rejects.toThrow(/invalid session token: bad signature/);
  await expect(remote(token("alice", "--ttl", "-5"))).rejects.toThrow(/token expired/);
  // tokens issued by a Node.js backend (server/cuda-worker/issue-token.mjs) are accepted too
  const { issueSessionToken } = await import(pathToFileURL(path.join(workerDir, "issue-token.mjs")).href);
  const ctx = await remote(issueSessionToken(SECRET, "carol", { ttlS: 60, memoryBytes: 4096 }));
  expect(ctx.info().limits.memoryLimit).toBe(4096);
  ctx.destroy();
});

test("kernels run remotely: original CUDA source, pointer offsets, scalars, templates, printf, timing", async () => {
  const ctx = await remote(token("alice"));
  expect(ctx.selection.selected).toBe("cuda_remote");
  expect(ctx.info().name).toMatch(NATIVE ? /\(remote, sm_\d+\)/ : /\(remote, sm_70\)/);
  ctx.onPrintf = () => {};
  const m = ctx.compile(`
template <int SCALE> __global__ void axpy(float a, const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = SCALE * a * x[i] + y[i];
  if (i == 0) printf("n=%d a=%.1f\\n", n, a);
}
template __global__ void axpy<2>(float, const float*, float*, int);`);
  const x = ctx.tensor([1, 2, 3, 4, 5, 6, 7, 8]), y = ctx.tensor([10, 10, 10, 10, 10, 10, 10, 10]);
  const out: string[] = [];
  ctx.onPrintf = (s) => out.push(s);
  await m.launch({ kernel: "axpy<2>", grid: 1, block: 32, args: [0.5, x, y.ptr.add(8), 6], timing: true });
  const rep = await ctx.synchronize();
  expect(Array.from(await y.toArray() as Float32Array)).toEqual([10, 10, 11, 12, 13, 14, 15, 16]);
  expect(out.join("")).toBe("n=6 a=0.5\n");
  expect(rep.kernelTimes[0].ms).toBeGreaterThanOrEqual(0);
  ctx.memset(y.ptr, 0, 8);
  await ctx.memcpy(x.ptr.add(4), y.ptr, 4);
  expect(Array.from(await x.toArray() as Float32Array)).toEqual([1, 0, 3, 4, 5, 6, 7, 8]);
  ctx.destroy();
});

test("kernel faults are sticky: the session's context is unusable afterwards, as with native CUDA", async () => {
  const ctx = await remote(token("alice"));
  const m = ctx.compile("__global__ void k(float* a, int i) { a[i] = 1.0f; }");
  const a = ctx.zeros([4]);
  await m.launch({ grid: 1, block: 1, args: [a, 1 << 20] });
  const err = await ctx.synchronize().then(() => null, (e) => e);
  expect(err).toBeInstanceOf(CudaError);
  expect(String(err)).toMatch(/cudaErrorIllegalAddress/);
  await expect(a.toArray()).rejects.toThrow(/cudaErrorIllegalAddress/);
  ctx.destroy();
});

test("limits: per-session memory, per-launch time limit (the session process is killed), sessions", async () => {
  const t = token("bob");
  const ctx = await remote(t);
  expect(() => ctx.malloc(2 << 20)).toThrow(/cudaErrorMemoryAllocation/); // the client knows the session limit
  // the server enforces it too, whatever a client does
  const s = await (await fetch(`${endpoint}/v1/sessions`, { method: "POST", headers: { Authorization: `Bearer ${t}` }, body: "{}" })).json();
  const r = await fetch(`${endpoint}/v1/sessions/${s.session}/alloc`, { method: "POST", headers: { Authorization: `Bearer ${t}` }, body: JSON.stringify({ handle: "h1", bytes: 4 << 20 }) });
  expect(r.status).toBe(413);
  expect((await r.json()).error.name).toBe("cudaErrorMemoryAllocation");
  // other users cannot reach the session
  const other = await fetch(`${endpoint}/v1/sessions/${s.session}/alloc`, { method: "POST", headers: { Authorization: `Bearer ${token("mallory")}` }, body: JSON.stringify({ handle: "h1", bytes: 4 }) });
  expect(other.status).toBe(404);
  // validation
  const bad = await fetch(`${endpoint}/v1/sessions/${s.session}/launch`, { method: "POST", headers: { Authorization: `Bearer ${t}` }, body: JSON.stringify({ source: "x", kernel: "k", grid: [1], block: [2048], args: [] }) });
  expect((await bad.json()).error.message).toMatch(/block \[2048, 1, 1\] exceeds the CUDA limits/);
  await fetch(`${endpoint}/v1/sessions/${s.session}`, { method: "DELETE", headers: { Authorization: `Bearer ${t}` } });

  const m = ctx.compile("__global__ void spin(int* f) { while (f[0] == 0) { } }");
  const f = ctx.zeros([1], "int32");
  const t0 = Date.now();
  await m.launch({ grid: 1, block: 1, args: [f] });
  await expect(ctx.synchronize()).rejects.toThrow(/cudaErrorLaunchTimeout: remote: launch exceeded the time limit of 8000 ms; the session was terminated/);
  expect(Date.now() - t0).toBeLessThan(20_000);
  ctx.destroy();

  const ctxs = await Promise.all([remote(t), remote(t), remote(t), remote(t)].map((p) => p.catch((e) => e)));
  expect(ctxs.filter((c) => c instanceof Error).map(String)).toEqual([expect.stringMatching(/cudaErrorDevicesUnavailable: .*at its limit of 3 sessions/)]);
  for (const c of ctxs) if (!(c instanceof Error)) c.destroy();
});

test("CORS: only configured browser origins may call the worker", async () => {
  const pre = await fetch(`${endpoint}/v1/sessions`, { method: "OPTIONS", headers: { Origin: "https://app.example", "Access-Control-Request-Method": "POST" } });
  expect(pre.status).toBe(204);
  expect(pre.headers.get("access-control-allow-origin")).toBe("https://app.example");
  const evil = await fetch(`${endpoint}/v1/sessions`, { method: "POST", headers: { Origin: "https://evil.example", Authorization: `Bearer ${token("alice")}` }, body: "{}" });
  expect(evil.status).toBe(403);
  const health = await (await fetch(`${endpoint}/v1/health`)).json();
  expect(health).toMatchObject({ ok: true, maxSessions: 3 });
});
