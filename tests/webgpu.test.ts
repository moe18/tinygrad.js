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
import numpy as np, _tinygrad_js_bridge as bridge
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
import gc, tinygrad_js
from tinygrad import Tensor, Device
before = tinygrad_js.stats()["bytesAllocated"]
t = Tensor.ones(1024, 1024).contiguous().realize()
during = tinygrad_js.stats()["bytesAllocated"]
del t; gc.collect()
Device["WEBGPU"].allocator.free_cache()   # LRUAllocator keeps freed buffers for reuse until asked to release them
after = tinygrad_js.stats()["bytesAllocated"]
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
  const lost = await s.run("import _tinygrad_js_bridge as b\nb.device.destroy()\nfrom tinygrad import Tensor\n(Tensor([1.0]) + 1).numpy()");
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

test("generated WGSL avoids constructs that only Chrome's compiler accepts (bitcast<u32> of a negative literal)", async ({ tinygradVersion }) => {
  // Upstream's renderer writes negative uint32 constants as bitcast<u32>(-32), e.g. in the RNG kernel of a 16x2 tensor (the
  // first-layer weights of a width-16 MLP). Safari rejects that: "value -32 cannot be represented as 'u32'".
  const r = await s.run(`
import re
from tinygrad import Tensor, nn, Device
import tinygrad_js.webgpu as wg
assert wg.portable_wgsl("x+bitcast<u32>(-32)+bitcast<u32>(-1)+bitcast<u32>(7)+bitcast<i32>(-3)") == "x+4294967264u+4294967295u+bitcast<u32>(7)+bitcast<i32>(-3)"
sources, orig = [], wg.WebGpuDevice.compile
def recording(self, name, src):
  sources.append(src); return orig(self, name, src)
wg.WebGpuDevice.compile = recording
try:
  Tensor.manual_seed(1)
  layers = [nn.Linear(2, w) for w in (4, 8, 16, 32, 64)]
  Tensor.realize(*[t for l in layers for t in (l.weight, l.bias)])
  for n in (3, 16, 32, 100): Tensor.rand(n, 2).realize(); Tensor.randn(n).realize()
finally: wg.WebGpuDevice.compile = orig
bad = [s for s in sources if re.search(r"bitcast<u32>\\(\\s*-", s)]
print(len(sources), len(bad), sum("4294967264u" in s for s in sources), float(layers[2].weight.abs().max().item()) <= 2 ** -0.5 + 1e-6)
`);
  expect(r.ok, r.error?.traceback).toBe(true);
  const [compiled, bad, rewritten, inRange] = r.stdout.trim().split(" ");
  expect(Number(compiled)).toBeGreaterThan(0);
  expect(Number(bad), "kernels still containing bitcast<u32>(-N)").toBe(0);
  // tinygrad 0.12 renders that constant differently, so there is nothing to rewrite there
  if (tinygradVersion !== "0.12.0") expect(Number(rewritten), "the 16x2 RNG kernel was rewritten").toBeGreaterThan(0);
  expect(inRange).toBe("True");
});
