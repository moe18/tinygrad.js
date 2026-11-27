"""Gateway tests that need no GPU (a stub engine in host memory): request parsing before authentication, the per-session
memory quota under concurrent requests, closing a session while a request is in flight, and the #include check.
Linux / WSL only (session processes use pass_fds). Run from server/cuda-worker:  python3 -m unittest discover -s tests -v
TGJS_CUDA_TEST_NATIVE=1 also runs the NVRTC check against the real engine (needs an NVIDIA GPU and libnvrtc)."""
from __future__ import annotations
import json, os, signal, socket, struct, sys, threading, time, unittest, base64
import http.client

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ["PYTHONPATH"] = os.pathsep.join([HERE] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))
os.environ["STUB_ALLOC_SLEEP_S"] = "0.3"

from tgjs_cuda_worker import tokens
from tgjs_cuda_worker.errors import WorkerError
from tgjs_cuda_worker.server import Config, serve
from tgjs_cuda_worker.source import check_includes

SECRET = "test-secret-0123456789abcdef0123456789abcdef"

class GatewayTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cfg = Config(host="127.0.0.1", port=0, token_secret=SECRET, engine="stub_engine:StubEngine", max_sessions=4, memory_bytes=1000,
                 time_limit_ms=8000)
    cls.httpd, cls.gw = serve(cfg)
    cls.port = cls.httpd.server_address[1]
    threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
    cls.tok = tokens.issue(SECRET, "alice", 600)

  @classmethod
  def tearDownClass(cls):
    cls.httpd.shutdown(); cls.gw.shutdown()

  def req(self, method: str, path: str, body=None, raw: bytes | None = None, auth: bool = True) -> tuple[int, dict | bytes]:
    c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    c.request(method, path, body=data, headers={"Authorization": f"Bearer {self.tok}"} if auth else {})
    r = c.getresponse(); out = r.read(); c.close()
    return r.status, (json.loads(out) if r.getheader("Content-Type") == "application/json" else out)

  def session(self) -> str:
    st, out = self.req("POST", "/v1/sessions", {})
    self.assertEqual(st, 200, out)
    return out["session"]

  def raw_request(self, head: str, body: bytes = b"") -> tuple[bytes, bool]:
    """send a hand-written request; returns (response bytes, whether the server then closed the connection)"""
    s = socket.create_connection(("127.0.0.1", self.port), timeout=5)
    s.sendall(head.encode() + body)
    data = b""
    while b"\r\n\r\n" not in data: data += s.recv(65536)
    hdr, rest = data.split(b"\r\n\r\n", 1)
    n = int(next(l.split(b":")[1] for l in hdr.split(b"\r\n") if l.lower().startswith(b"content-length")))
    while len(rest) < n: rest += s.recv(65536)
    try: closed = s.recv(1) == b""
    except socket.timeout: closed = False
    s.close()
    return hdr + b"\r\n\r\n" + rest, closed

  # ------------------------------------------------------------------ request bodies
  def test_negative_content_length_is_refused(self):
    # int("-1") passed the size limit and rfile.read(-1) read until the client hung up, with no limit
    for cl in ("-1", "+5", "1_0", "0x10", "abc"):
      resp, closed = self.raw_request(f"POST /v1/sessions HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {self.tok}\r\nContent-Length: {cl}\r\n\r\n", b"{}")
      self.assertIn(b" 400 ", resp.split(b"\r\n")[0], cl)
      self.assertIn(b"invalid Content-Length", resp)
      self.assertTrue(closed, cl)

  def test_unauthenticated_body_is_not_read(self):
    # the client announces 100 MB and sends none of it: the 401 must come back at once (the body was read before auth)
    t0 = time.monotonic()
    resp, closed = self.raw_request("POST /v1/sessions HTTP/1.1\r\nHost: x\r\nContent-Length: 100000000\r\n\r\n")
    self.assertIn(b" 401 ", resp.split(b"\r\n")[0])
    self.assertLess(time.monotonic() - t0, 3)
    self.assertTrue(closed)  # the unread body must not be parsed as the next request

  def test_chunked_body_is_refused(self):
    resp, closed = self.raw_request(f"POST /v1/sessions HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {self.tok}\r\nTransfer-Encoding: chunked\r\n\r\n", b"2\r\n{}\r\n0\r\n\r\n")
    self.assertIn(b" 411 ", resp.split(b"\r\n")[0])
    self.assertTrue(closed)

  def test_keep_alive_still_works_for_good_requests(self):
    c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
    for _ in range(3):
      c.request("GET", "/v1/health"); r = c.getresponse(); r.read()
      self.assertEqual(r.status, 200)
    c.close()

  # ------------------------------------------------------------------ memory quota
  def test_concurrent_allocs_cannot_exceed_the_quota(self):
    sid = self.session()
    results: list[int] = []
    def alloc(i):
      st, _ = self.req("POST", f"/v1/sessions/{sid}/alloc", {"handle": f"h{i}", "bytes": 400})
      results.append(st)
    ts = [threading.Thread(target=alloc, args=(i,)) for i in range(6)]
    for t in ts: t.start()
    for t in ts: t.join()
    self.assertEqual(sorted(results), [200, 200, 413, 413, 413, 413])  # 1000-byte limit: two allocations of 400
    s = self.gw.sessions[sid]
    self.assertEqual(s.used, 800)
    # the same handle twice at once: one wins, the other is refused (not a 500)
    results.clear()
    ts = [threading.Thread(target=lambda: results.append(self.req("POST", f"/v1/sessions/{sid}/free", {"handle": "h0"})[0])) for _ in range(2)]
    for t in ts: t.start()
    for t in ts: t.join()
    self.assertEqual(sorted(results), [200, 400])
    self.assertEqual(s.used, 400)
    self.req("DELETE", f"/v1/sessions/{sid}")

  def test_failed_alloc_releases_its_reservation(self):
    sid = self.session()
    st, out = self.req("POST", f"/v1/sessions/{sid}/alloc", {"handle": "a", "bytes": 600})
    self.assertEqual(st, 200)
    st, out = self.req("POST", f"/v1/sessions/{sid}/alloc", {"handle": "a", "bytes": 10})
    self.assertEqual(st, 400)
    st, out = self.req("POST", f"/v1/sessions/{sid}/alloc", {"handle": "b", "bytes": 400})
    self.assertEqual(st, 200)
    self.assertEqual(self.gw.sessions[sid].used, 1000)
    self.req("DELETE", f"/v1/sessions/{sid}")

  # ------------------------------------------------------------------ closing sessions
  def test_kill_waits_for_the_request_in_flight_before_closing_fds(self):
    sid = self.session()
    s = self.gw.sessions[sid]
    s.lock.acquire()  # stands for a request between send() and recv()
    killer = threading.Thread(target=s.kill); killer.start()
    time.sleep(0.3)
    os.fstat(s.wfd); os.fstat(s.rfd)  # still open: their numbers cannot have been handed to another session
    self.assertIsNotNone(s.proc.wait(5))  # but the child was signalled at once
    s.lock.release(); killer.join(5)
    self.assertFalse(killer.is_alive())
    with self.assertRaises(OSError): os.fstat(s.wfd)
    with self.assertRaises(WorkerError): s.call({"op": "read", "handle": "x", "offset": 0, "bytes": 0})
    self.req("DELETE", f"/v1/sessions/{sid}")

  def test_closing_a_session_whose_child_died_mid_request(self):
    a = self.session()
    s = self.gw.sessions[a]
    out: list = []
    def slow():
      out.append(self.req("POST", f"/v1/sessions/{a}/launch", {"source": "__global__ void sleep(int){}", "kernel": "sleep", "grid": [1], "block": [1],
                                                          "args": [{"scalar": {"ctype": "int", "bytes": base64.b64encode(struct.pack("<i", 3000)).decode()}}]}))
    t = threading.Thread(target=slow); t.start()
    time.sleep(0.5)
    os.killpg(s.proc.pid, signal.SIGKILL)  # the session process dies on its own (OOM, CPU limit) during the request
    st, _ = self.req("DELETE", f"/v1/sessions/{a}")
    self.assertIn(st, (200, 404))
    t.join(10)
    self.assertEqual(out[0][0], 200)
    self.assertIn("error", out[0][1])  # the launch reports the failure; it never reads another session's pipe
    b = self.session()  # likely gets the fd numbers A had
    self.assertEqual(self.req("POST", f"/v1/sessions/{b}/alloc", {"handle": "x", "bytes": 4})[0], 200)
    self.assertEqual(self.req("PUT", f"/v1/sessions/{b}/mem/x?offset=0", raw=b"abcd")[0], 200)
    self.assertEqual(self.req("GET", f"/v1/sessions/{b}/mem/x?offset=0&bytes=4"), (200, b"abcd"))
    self.req("DELETE", f"/v1/sessions/{b}")

  def test_launch_with_a_file_include_is_refused(self):
    sid = self.session()
    st, out = self.req("POST", f"/v1/sessions/{sid}/launch", {"source": '#include "/proc/self/environ"\n__global__ void k(){}', "kernel": "k", "grid": [1], "block": [1], "args": []})
    self.assertEqual(st, 400)
    self.assertEqual(out["error"]["name"], "cudaErrorInvalidSource")
    self.req("DELETE", f"/v1/sessions/{sid}")

class IncludeCheckTest(unittest.TestCase):
  OK = [
    "#include <cuda_fp16.h>\n", '#include "cuda_bf16.h"\n', "#include <cuda/std/cstdint>\n", "#  include <mma.h> // tensor cores\n",
    "#include <cooperative_groups.h>\n#include <cooperative_groups/reduce.h>\n",
    'const char* s = "#include </etc/passwd>";\n',                     # in a string: not a directive
    "// #include </etc/passwd>\n", "/* #include </etc/passwd> */\n",  # in comments
    "#define INC 1\n#if INC\n#endif\n", "int a = 5 % 3;\n",
  ]
  BAD = [
    '#include "/proc/self/environ"\n', "#include </etc/passwd>\n", '#include "../../../etc/passwd"\n', "#include <cuda/../../../../etc/hosts>\n",
    "#include_next </etc/passwd>\n", "#import </etc/passwd>\n", "#embed </etc/passwd>\n",
    '#define F "/proc/self/environ"\n#include F\n',                    # macro-computed name
    "#/**/include </etc/passwd>\n",                                  # comment between # and the directive
    "# /* spans\nlines */ include </etc/passwd>\n",
    "#inc\\\nlude </etc/passwd>\n", "#include \\\n</etc/passwd>\n",   # line splices
    "%:include </etc/passwd>\n",                                     # digraph
    'const char* s = "unterminated\n#include </etc/passwd>\n',       # an unterminated literal ends at the newline
    'auto r = R"x(\n#include </etc/passwd>\n)x";\n',                  # raw-string contents are checked too (refusing is safe)
    "#include <a\\b>\n", "#include <>\n", "#include\n", '#include "a" "b"\n',
  ]

  def test_allowed(self):
    for src in self.OK:
      with self.subTest(src=src): check_includes(src)

  def test_refused(self):
    for src in self.BAD:
      with self.subTest(src=src):
        with self.assertRaises(WorkerError) as e: check_includes(src)
        self.assertEqual(e.exception.name, "cudaErrorInvalidSource")

  def test_kernel_library_passes(self):
    path = os.environ.get("TGJS_KERNEL_SOURCES")  # JSON {name: source}, written by tests/cuda/nvcc.node.test.ts-style export
    if not path: self.skipTest("set TGJS_KERNEL_SOURCES to a JSON file of the kernel library sources")
    for name, src in json.load(open(path)).items():
      with self.subTest(name=name): check_includes(src)

@unittest.skipUnless(os.environ.get("TGJS_CUDA_TEST_NATIVE"), "set TGJS_CUDA_TEST_NATIVE=1 to test the NVRTC engine (needs an NVIDIA GPU)")
class NativeIncludeTest(unittest.TestCase):
  def test_nvrtc_engine_refuses_host_files_and_compiles_headers(self):
    from tgjs_cuda_worker.cuda import CudaEngine
    e = CudaEngine()
    try:
      with self.assertRaises(WorkerError) as err: e._compile('#include "/proc/self/environ"\n__global__ void k(){}', "k", {})
      self.assertIn("only allows headers", err.exception.message)
      e._compile("#include <cuda_fp16.h>\n__global__ void k(__half* x){ x[0] = __float2half(1.f); }", "k", {})
    finally: e.close()

if __name__ == "__main__": unittest.main()
