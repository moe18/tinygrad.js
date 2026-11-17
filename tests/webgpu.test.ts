import { test, expect, openSession, type Session } from "./helpers.js";

test.describe.configure({ mode: "serial" });
let s: Session;
test.beforeAll(async ({ browser, tinygradVersion }) => { s = await openSession(browser, { tinygradVersion }); });
test.afterAll(async () => { await s?.close(); });

test("runtime reports a WebGPU device", async ({ tinygradVersion }) => {
  expect(s.info.tinygradDevice).toBe("WEBGPU");
  expect(s.info.gpu).not.toBeNull();
  expect(s.info.syncMode).toBe("jspi");
  expect(s.info.tinygradVersion).toBe(tinygradVersion);
  test.info().annotations.push({ type: "adapter", description: JSON.stringify(s.info.gpu) });
});

test("README example runs verbatim", async () => {
  const r = await s.run("from tinygrad import Tensor\na = Tensor.randn(100, 100)\nb = Tensor.randn(100, 100)\nc = a @ b\nprint(c.numpy())");
  expect(r.ok, JSON.stringify(r.error)).toBe(true);
  expect(r.stdout).toMatch(/^\[\[/);
  expect(r.gpu!.bytesDownloaded).toBe(100 * 100 * 4);
});

test("tensor data lives in a WebGPU buffer (read back directly through WebGPU, bypassing tinygrad)", async () => {
  const r = await s.run(`
import numpy as np, _tinygrad_web_bridge as bridge
from pyodide.ffi import run_sync
from tinygrad import Tensor
t = (Tensor([1.0, 2.0, 3.0, 4.0]) * 3 + 1).realize()
u = t.uop if hasattr(t, "uop") else t.lazydata
buf = u.base.buffer
handle = buf._buf
raw = run_sync(bridge.readBuffer(handle, buf.nbytes)).to_py()
print(type(handle).__name__, np.frombuffer(bytes(raw), dtype=np.float32).tolist())`);
  expect(r.ok, r.error?.traceback).toBe(true);
  expect(r.stdout.trim()).toBe("int [4.0, 7.0, 10.0, 13.0]");
});

test("kernels are compiled once and cached across runs", async () => {
  const code = "from tinygrad import Tensor\nprint((Tensor.arange(1000).float().reshape(10, 100).exp().sum(axis=1) * 0.5).numpy()[:2])";
  const first = await s.run(code);
  const second = await s.run(code);
  expect(first.ok && second.ok).toBe(true);
  expect(second.stdout).toBe(first.stdout);
  expect(second.gpu!.kernelsDispatched).toBeGreaterThan(0);
  expect(second.gpu!.kernelsCompiled).toBe(0);
});

test("allocation and deallocation of GPU buffers", async () => {
  const r = await s.run(`
import gc, tinygrad_web
from tinygrad import Tensor, Device
before = tinygrad_web.stats()["bytesAllocated"]
t = Tensor.ones(1024, 1024).contiguous().realize()
during = tinygrad_web.stats()["bytesAllocated"]
del t; gc.collect()
Device["WEBGPU"].allocator.free_cache()   # LRUAllocator keeps freed buffers for reuse until asked to release them
after = tinygrad_web.stats()["bytesAllocated"]
print(during - before >= 4 * 1024 * 1024, after <= before)`);
  expect(r.ok, r.error?.traceback).toBe(true);
  expect(r.stdout.trim()).toBe("True True");
});

test("WGSL compilation errors become Python exceptions", async () => {
  const r = await s.run("from tinygrad import Device\nDevice['WEBGPU'].compile('broken', 'fn broken( {')");
  expect(r.ok).toBe(false);
  expect(r.error!.type).toBe("RuntimeError");
  expect(r.error!.message).toMatch(/WGSL compilation of broken failed/);
});

test("Python exceptions in user code are reported with type and traceback", async () => {
  const r = await s.run("from tinygrad import Tensor\nTensor([1, 2, 3]).reshape(2, 2)");
  expect(r.ok).toBe(false);
  expect(r.status).toBe("error");
  expect(r.error!.traceback).toContain("<user-code>");
});

test("device loss is reported, and restart() recovers", async () => {
  const lost = await s.run("import _tinygrad_web_bridge as b\nb.device.destroy()\nfrom tinygrad import Tensor\n(Tensor([1.0]) + 1).numpy()");
  expect(lost.ok).toBe(false);
  expect(lost.error!.message).toMatch(/device lost|destroyed/i);
  const info = await s.page.evaluate(() => (window as any).rt.restart());
  expect(info.tinygradDevice).toBe("WEBGPU");
  const r = await s.run("from tinygrad import Tensor\nprint((Tensor([1.0]) + 1).numpy())");
  expect(r.ok, JSON.stringify(r.error)).toBe(true);
  expect(r.stdout.trim()).toBe("[2.]");
});

test("maxGpuMemoryBytes limits GPU allocations", async ({ browser, tinygradVersion }) => {
  const limited = await openSession(browser, { tinygradVersion, maxGpuMemoryBytes: 1 << 20 });
  try {
    const ok = await limited.run("from tinygrad import Tensor\nprint(Tensor.ones(1000).contiguous().sum().item())");
    expect(ok.stdout.trim()).toBe("1000.0");
    const r = await limited.run("from tinygrad import Tensor\nTensor.ones(1024, 1024).contiguous().realize()");
    expect(r.ok).toBe(false);
    expect(r.error!.type).toBe("MemoryError");
    // tinygrad 0.13+ re-raises allocation failures as its own MemoryError with ours as __cause__
    expect(r.error!.message + r.error!.traceback).toMatch(/memory limit exceeded/);
  } finally { await limited.close(); }
});
