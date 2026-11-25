"""HTTP gateway for remote native CUDA execution (the protocol of src/cuda/runtime/remote.ts). Standard library only.

  POST   /v1/sessions                                   -> { session, device, limits }
  POST   /v1/sessions/:s/alloc        { handle, bytes }
  PUT    /v1/sessions/:s/mem/:handle?offset=N            body: raw bytes
  GET    /v1/sessions/:s/mem/:handle?offset=N&bytes=M    -> raw bytes
  POST   /v1/sessions/:s/free         { handle }
  POST   /v1/sessions/:s/copy         { dst, dstOffset, src, srcOffset, bytes }
  POST   /v1/sessions/:s/memset       { handle, offset, value, bytes }
  POST   /v1/sessions/:s/launch       { source, kernel, options, grid, block, sharedMem, args, timing } -> { ms, printf, error? }
  DELETE /v1/sessions/:s
  GET    /v1/health

Every request needs "Authorization: Bearer <session token>" (tokens.py) unless the server runs with --no-auth, which is only
allowed on a loopback address. Sessions belong to the token's subject; other subjects cannot see them.
"""
from __future__ import annotations
import json, os, re, secrets, signal, subprocess, sys, threading, time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs
from . import tokens
from .errors import WorkerError
from .protocol import send, recv, ChildTimeout, ChildDied

HANDLE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

@dataclass
class Config:
  host: str = "127.0.0.1"
  port: int = 8787
  token_secret: str | None = None
  no_auth: bool = False
  allowed_origins: list[str] = field(default_factory=list)
  engine: str = "tgjs_cuda_worker.cuda:CudaEngine"
  max_sessions: int = 4
  memory_bytes: int = 2 << 30           # per session
  time_limit_ms: int = 10_000           # per launch (and per memory operation)
  session_ttl_s: int = 600              # idle sessions are closed
  max_source_bytes: int = 512 << 10
  max_body_bytes: int = 512 << 20
  max_launches: int = 100_000           # per session
  printf_bytes: int = 1 << 20           # per launch
  startup_timeout_s: float = 60.0
  run_as_uid: int | None = None
  run_as_gid: int | None = None

  @staticmethod
  def from_env(**over) -> "Config":
    e = os.environ.get
    c = Config(
      host=e("TGJS_CUDA_HOST", "127.0.0.1"), port=int(e("TGJS_CUDA_PORT", "8787")), token_secret=e("TGJS_CUDA_TOKEN_SECRET"),
      allowed_origins=[o.strip() for o in e("TGJS_CUDA_ALLOWED_ORIGINS", "").split(",") if o.strip()],
      engine=e("TGJS_CUDA_ENGINE", "tgjs_cuda_worker.cuda:CudaEngine"), max_sessions=int(e("TGJS_CUDA_MAX_SESSIONS", "4")),
      memory_bytes=int(e("TGJS_CUDA_MEMORY_BYTES", str(2 << 30))), time_limit_ms=int(e("TGJS_CUDA_TIME_LIMIT_MS", "10000")),
      session_ttl_s=int(e("TGJS_CUDA_SESSION_TTL_S", "600")), max_launches=int(e("TGJS_CUDA_MAX_LAUNCHES", "100000")),
      run_as_uid=int(e("TGJS_CUDA_RUN_AS_UID")) if e("TGJS_CUDA_RUN_AS_UID") else None,
      run_as_gid=int(e("TGJS_CUDA_RUN_AS_GID")) if e("TGJS_CUDA_RUN_AS_GID") else None)
    for k, v in over.items():
      if v is not None: setattr(c, k, v)
    return c

  def validate(self) -> None:
    if self.no_auth:
      if self.host not in ("127.0.0.1", "::1", "localhost"): raise SystemExit("--no-auth is only allowed on a loopback address")
    elif not self.token_secret or len(self.token_secret) < 32:
      raise SystemExit("TGJS_CUDA_TOKEN_SECRET (at least 32 characters) is required; see README.md (or use --no-auth on localhost)")

class Session:
  """One isolated child process (its own CUDA context) per session."""
  def __init__(self, cfg: Config, owner: str, memory_bytes: int, time_limit_ms: int):
    self.id = secrets.token_urlsafe(18)
    self.owner, self.cfg = owner, cfg
    self.memory_bytes, self.time_limit_ms = memory_bytes, time_limit_ms
    self.allocs: dict[str, int] = {}
    self.used = 0
    self.launches = 0
    self.lock = threading.Lock()
    self.last_used = time.monotonic()
    self.dead: WorkerError | None = None
    self._killed = False
    p2c_r, p2c_w = os.pipe(); c2p_r, c2p_w = os.pipe()
    child_cfg = {"engine": cfg.engine, "printfBytes": cfg.printf_bytes, "runAsUid": cfg.run_as_uid, "runAsGid": cfg.run_as_gid,
                 "cpuSeconds": max(60, cfg.session_ttl_s * 4)}
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_VISIBLE_DEVICES", "TGJS_CUDA_NVRTC", "PYTHONPATH", "LANG", "HOME")}
    env["PYTHONPATH"] = os.pathsep.join([os.path.dirname(os.path.dirname(os.path.abspath(__file__)))] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    self.proc = subprocess.Popen([sys.executable, "-s", "-m", "tgjs_cuda_worker.child", str(p2c_r), str(c2p_w), json.dumps(child_cfg)],
                                 pass_fds=(p2c_r, c2p_w), stdin=subprocess.DEVNULL, env=env, start_new_session=True)
    os.close(p2c_r); os.close(c2p_w)
    self.wfd, self.rfd = p2c_w, c2p_r
    try: hello, _ = recv(self.rfd, time.monotonic() + cfg.startup_timeout_s)
    except (ChildTimeout, ChildDied):
      self.kill(); raise WorkerError("cudaErrorDevicesUnavailable", "the CUDA session process failed to start", 503)
    if not hello.get("ok"):
      self.kill(); err = hello.get("error", {})
      raise WorkerError(err.get("name", "cudaErrorNoDevice"), err.get("message", "no CUDA device"), int(hello.get("status", 503)))
    self.device = hello["device"]

  def call(self, header: dict, payload: bytes = b"", timeout_ms: int | None = None) -> tuple[dict, bytes]:
    with self.lock:
      self.last_used = time.monotonic()
      if self.dead: raise self.dead
      try:
        send(self.wfd, header, payload)
        reply, body = recv(self.rfd, time.monotonic() + (timeout_ms or self.time_limit_ms) / 1000)
      except ChildTimeout:
        self.dead = WorkerError("cudaErrorLaunchTimeout", f"{header.get('op')} exceeded the time limit of {timeout_ms or self.time_limit_ms} ms; the session was terminated", 408)
        self.kill(); raise self.dead
      except ChildDied:
        self.dead = WorkerError("cudaErrorLaunchFailure", "the CUDA session process exited unexpectedly; the session was terminated", 500)
        self.kill(); raise self.dead
      if not reply.get("ok"):
        err = reply.get("error", {})
        e = WorkerError(err.get("name", "cudaErrorUnknown"), err.get("message", "error"), int(reply.get("status", 500)))
        if reply.get("sticky"):
          self.dead = WorkerError(e.name, f"{e.message} (the CUDA context is no longer usable; the session was terminated)", 410)
          self.kill()
        e.printf = reply.get("printf", "")  # type: ignore[attr-defined]
        raise e
      return reply, body

  def kill(self) -> None:
    if self._killed: return  # file descriptors must be closed exactly once (their numbers are reused)
    self._killed = True
    # the child leads its own process group (start_new_session): kill the group, so nothing it started survives
    try: os.killpg(self.proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
      try: self.proc.kill()
      except Exception: pass
    for fd in (self.wfd, self.rfd):
      try: os.close(fd)
      except OSError: pass
    try: self.proc.wait(5)
    except Exception: pass

  def close(self) -> None:
    if self.proc.poll() is None and not self.dead:
      try: self.call({"op": "close"}, timeout_ms=5000)
      except WorkerError: pass
    self.kill()

class Gateway:
  def __init__(self, cfg: Config):
    self.cfg = cfg
    self.sessions: dict[str, Session] = {}
    self.starting: list[str] = []
    self.lock = threading.Lock()
    self._stop = threading.Event()
    threading.Thread(target=self._reaper, daemon=True).start()

  def _reaper(self) -> None:
    while not self._stop.wait(5):
      now = time.monotonic()
      with self.lock:
        idle = [s for s in self.sessions.values() if now - s.last_used > self.cfg.session_ttl_s and not s.lock.locked()]
        for s in idle: del self.sessions[s.id]
      for s in idle: s.close()

  def shutdown(self) -> None:
    self._stop.set()
    with self.lock: ss = list(self.sessions.values()); self.sessions.clear()
    for s in ss: s.close()

  def auth(self, header: str | None) -> dict:
    if self.cfg.no_auth: return {"sub": "local"}
    if not header or not header.startswith("Bearer "): raise WorkerError("cudaErrorInitializationError", "missing bearer token", 401)
    try: return tokens.verify(self.cfg.token_secret or "", header[7:].strip())
    except tokens.InvalidToken as e: raise WorkerError("cudaErrorInitializationError", f"invalid session token: {e}", 401)

  def create(self, claims: dict, body: dict) -> dict:
    mem = min(self.cfg.memory_bytes, int(claims.get("memoryBytes", self.cfg.memory_bytes)), int(body.get("memoryLimit") or self.cfg.memory_bytes))
    tl = min(self.cfg.time_limit_ms, int(claims.get("timeLimitMs", self.cfg.time_limit_ms)), int(body.get("timeLimitMs") or self.cfg.time_limit_ms))
    with self.lock:  # reserve the slot before starting the (slow) session process, so concurrent requests cannot overshoot
      if len(self.sessions) + len(self.starting) >= self.cfg.max_sessions: raise WorkerError("cudaErrorDevicesUnavailable", f"the worker is at its limit of {self.cfg.max_sessions} sessions", 503)
      mine = sum(1 for s in self.sessions.values() if s.owner == claims["sub"]) + self.starting.count(claims["sub"])
      if "maxSessions" in claims and mine >= int(claims["maxSessions"]): raise WorkerError("cudaErrorDevicesUnavailable", f"session limit of {claims['maxSessions']} reached for this user", 429)
      self.starting.append(claims["sub"])
    try: s = Session(self.cfg, claims["sub"], mem, tl)
    finally:
      with self.lock: self.starting.remove(claims["sub"])
    with self.lock: self.sessions[s.id] = s
    return {"session": s.id, "device": s.device, "limits": {"memoryBytes": mem, "timeLimitMs": tl}}

  def get(self, claims: dict, sid: str) -> Session:
    with self.lock: s = self.sessions.get(sid)
    if s is None or not secrets.compare_digest(s.owner, claims["sub"]): raise WorkerError("cudaErrorInvalidResourceHandle", "no such session (expired or closed)", 404)
    return s

  def delete(self, claims: dict, sid: str) -> None:
    s = self.get(claims, sid)
    with self.lock: self.sessions.pop(sid, None)
    s.close()

  # ---------------------------------------------------------------- operations
  def op(self, claims: dict, sid: str, action: str, body: dict | None, raw: bytes, query: dict) -> tuple[dict | bytes, int]:
    s = self.get(claims, sid)
    if action == "alloc":
      h, n = _handle(body.get("handle")), _int(body.get("bytes"), "bytes", 0)
      if h in s.allocs: raise WorkerError("cudaErrorInvalidValue", f"allocation {h!r} exists")
      if s.used + n > s.memory_bytes: raise WorkerError("cudaErrorMemoryAllocation", f"allocating {n} bytes exceeds the session memory limit of {s.memory_bytes} bytes ({s.used} in use)", 413)
      s.call({"op": "alloc", "handle": h, "bytes": n})
      s.allocs[h] = n; s.used += n
      return {}, 200
    if action == "free":
      h = _handle(body.get("handle"))
      if h not in s.allocs: raise WorkerError("cudaErrorInvalidDevicePointer", f"unknown allocation {h!r}")
      s.call({"op": "free", "handle": h})
      s.used -= s.allocs.pop(h)
      return {}, 200
    if action == "mem_put":
      h, off = _handle(query.get("handle")), _int(query.get("offset", 0), "offset", 0)
      _bounds(s, h, off, len(raw))
      s.call({"op": "write", "handle": h, "offset": off}, raw)
      return {}, 200
    if action == "mem_get":
      h, off, n = _handle(query.get("handle")), _int(query.get("offset", 0), "offset", 0), _int(query.get("bytes"), "bytes", 0)
      _bounds(s, h, off, n)
      _, data = s.call({"op": "read", "handle": h, "offset": off, "bytes": n})
      return data, 200
    if action == "copy":
      d, so = _handle(body.get("dst")), _handle(body.get("src"))
      n, do_, sof = _int(body.get("bytes"), "bytes", 0), _int(body.get("dstOffset", 0), "dstOffset", 0), _int(body.get("srcOffset", 0), "srcOffset", 0)
      _bounds(s, d, do_, n); _bounds(s, so, sof, n)
      s.call({"op": "copy", "dst": d, "dstOffset": do_, "src": so, "srcOffset": sof, "bytes": n})
      return {}, 200
    if action == "memset":
      h, off, n = _handle(body.get("handle")), _int(body.get("offset", 0), "offset", 0), _int(body.get("bytes"), "bytes", 0)
      _bounds(s, h, off, n)
      s.call({"op": "memset", "handle": h, "offset": off, "value": _int(body.get("value", 0), "value", 0) & 0xFF, "bytes": n})
      return {}, 200
    if action == "launch": return self.launch(s, body), 200
    raise WorkerError("cudaErrorInvalidValue", f"unknown operation {action!r}", 404)

  def launch(self, s: Session, b: dict) -> dict:
    src, kernel = b.get("source"), b.get("kernel")
    if not isinstance(src, str) or not src: raise WorkerError("cudaErrorInvalidSource", "missing kernel source")
    if len(src.encode()) > self.cfg.max_source_bytes: raise WorkerError("cudaErrorInvalidSource", f"kernel source exceeds {self.cfg.max_source_bytes} bytes", 413)
    if not isinstance(kernel, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_:<>, ()*&+\-.]{0,255}", kernel): raise WorkerError("cudaErrorInvalidValue", "invalid kernel name")
    grid, block = _dims(b.get("grid"), "grid", 2**31 - 1), _dims(b.get("block"), "block", 1024)
    if block[0] * block[1] * block[2] > 1024: raise WorkerError("cudaErrorInvalidConfiguration", "more than 1024 threads per block")
    smem = _int(b.get("sharedMem", 0), "sharedMem", 0)
    args = b.get("args")
    if not isinstance(args, list) or len(args) > 256: raise WorkerError("cudaErrorInvalidValue", "args must be a list of at most 256 kernel arguments")
    for a in args:
      if isinstance(a, dict) and isinstance(a.get("ptr"), dict):
        h = _handle(a["ptr"].get("handle")); off = _int(a["ptr"].get("offset", 0), "offset", 0)
        if h not in s.allocs or off > s.allocs[h]: raise WorkerError("cudaErrorInvalidDevicePointer", f"argument points outside allocation {h!r}")
      elif not (isinstance(a, dict) and isinstance(a.get("scalar"), dict) and isinstance(a["scalar"].get("bytes"), str) and len(a["scalar"]["bytes"]) <= 4096):
        raise WorkerError("cudaErrorInvalidValue", "each argument must be {ptr: {handle, offset}} or {scalar: {ctype, bytes}}")
    opts = b.get("options") if isinstance(b.get("options"), dict) else {}
    if s.launches >= self.cfg.max_launches: raise WorkerError("cudaErrorLaunchOutOfResources", f"session launch limit of {self.cfg.max_launches} reached", 429)
    s.launches += 1
    try:
      r, _ = s.call({"op": "launch", "source": src, "kernel": kernel, "options": {"defines": opts.get("defines")}, "grid": grid, "block": block,
                     "sharedMem": smem, "args": args, "timing": bool(b.get("timing"))})
      return {"ms": r.get("ms"), "printf": r.get("printf", "")}
    except WorkerError as e:
      # kernel faults are reported in the launch result (like cudaGetLastError after a launch); the client makes them sticky
      return {"printf": getattr(e, "printf", ""), "error": {"name": e.name, "message": e.message}}

def _int(v, what: str, lo: int) -> int:
  try: n = int(v)
  except (TypeError, ValueError): raise WorkerError("cudaErrorInvalidValue", f"{what} must be an integer")
  if isinstance(v, float) and v != n or n < lo or n > 2**53: raise WorkerError("cudaErrorInvalidValue", f"{what} out of range")
  return n

def _handle(v) -> str:
  if not isinstance(v, str) or not HANDLE.match(v): raise WorkerError("cudaErrorInvalidValue", "invalid allocation handle")
  return v

def _bounds(s: Session, h: str, off: int, n: int) -> None:
  if h not in s.allocs: raise WorkerError("cudaErrorInvalidDevicePointer", f"unknown allocation {h!r}")
  if off + n > s.allocs[h]: raise WorkerError("cudaErrorInvalidValue", f"{n} bytes at offset {off} exceed the allocation of {s.allocs[h]} bytes")

def _dims(v, what: str, hi: int) -> list[int]:
  if isinstance(v, int): v = [v]
  if not isinstance(v, list) or not 1 <= len(v) <= 3: raise WorkerError("cudaErrorInvalidConfiguration", f"{what} must be 1-3 positive integers")
  d = [_int(x, what, 1) for x in v] + [1] * (3 - len(v))
  if d[0] > hi or d[1] > (65535 if what == "grid" else 1024) or d[2] > (65535 if what == "grid" else 64): raise WorkerError("cudaErrorInvalidConfiguration", f"{what} {d} exceeds the CUDA limits")
  return d

ROUTES = [
  ("POST", re.compile(r"^/v1/sessions/([^/]+)/(alloc|free|copy|memset|launch)$")),
  ("PUT", re.compile(r"^/v1/sessions/([^/]+)/mem/([^/]+)$")),
  ("GET", re.compile(r"^/v1/sessions/([^/]+)/mem/([^/]+)$")),
]

def make_handler(gw: Gateway):
  cfg = gw.cfg
  class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "tgjs-cuda-worker/1"
    def log_message(self, fmt, *a):
      if os.environ.get("TGJS_CUDA_LOG"): super().log_message(fmt, *a)

    def _cors(self) -> bool:
      origin = self.headers.get("Origin")
      if not origin: return True
      if "*" in cfg.allowed_origins or origin in cfg.allowed_origins:
        self._origin = origin; return True
      return False

    def _send(self, status: int, body: dict | bytes) -> None:
      data = body if isinstance(body, bytes) else json.dumps(body).encode()
      self.send_response(status)
      self.send_header("Content-Type", "application/octet-stream" if isinstance(body, bytes) else "application/json")
      self.send_header("Content-Length", str(len(data)))
      self.send_header("Cache-Control", "no-store")
      self.send_header("X-Content-Type-Options", "nosniff")
      if getattr(self, "_origin", None):
        self.send_header("Access-Control-Allow-Origin", self._origin); self.send_header("Vary", "Origin")
      self.end_headers()
      self.wfile.write(data)

    def do_OPTIONS(self):
      if not self._cors(): return self._send(403, {"error": {"name": "cudaErrorInitializationError", "message": "origin not allowed"}})
      self.send_response(204)
      if getattr(self, "_origin", None):
        self.send_header("Access-Control-Allow-Origin", self._origin); self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
      self.send_header("Content-Length", "0")
      self.end_headers()

    def _body(self) -> bytes:
      n = int(self.headers.get("Content-Length") or 0)
      if n > cfg.max_body_bytes:
        self.close_connection = True
        raise WorkerError("cudaErrorMemoryAllocation", f"request body exceeds {cfg.max_body_bytes} bytes", 413)
      return self.rfile.read(n) if n else b""

    def _handle(self, method: str):
      try:
        if not self._cors(): raise WorkerError("cudaErrorInitializationError", "origin not allowed", 403)
        u = urlsplit(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        raw = self._body()
        if method == "GET" and u.path == "/v1/health":
          with gw.lock: n = len(gw.sessions)
          return self._send(200, {"ok": True, "sessions": n, "maxSessions": cfg.max_sessions})
        claims = gw.auth(self.headers.get("Authorization"))
        body = None
        if method == "POST":
          try: body = json.loads(raw or b"{}")
          except ValueError: raise WorkerError("cudaErrorInvalidValue", "request body must be JSON")
          if not isinstance(body, dict): raise WorkerError("cudaErrorInvalidValue", "request body must be a JSON object")
        if method == "POST" and u.path == "/v1/sessions": return self._send(200, gw.create(claims, body))
        m = re.fullmatch(r"/v1/sessions/([^/]+)", u.path)
        if method == "DELETE" and m: gw.delete(claims, m.group(1)); return self._send(200, {})
        for meth, rx in ROUTES:
          m = rx.match(u.path)
          if meth == method and m:
            if method == "POST": out, st = gw.op(claims, m.group(1), m.group(2), body, b"", q)
            else:
              q["handle"] = m.group(2)
              out, st = gw.op(claims, m.group(1), "mem_put" if method == "PUT" else "mem_get", None, raw, q)
            return self._send(st, out)
        raise WorkerError("cudaErrorInvalidValue", f"no route for {method} {u.path}", 404)
      except WorkerError as e:
        self._send(e.status, e.to_json())
      except Exception as e:  # never leak a traceback to clients
        sys.stderr.write(f"internal error: {type(e).__name__}: {e}\n")
        self._send(500, {"error": {"name": "cudaErrorUnknown", "message": "internal worker error"}})

    def do_GET(self): self._handle("GET")
    def do_POST(self): self._handle("POST")
    def do_PUT(self): self._handle("PUT")
    def do_DELETE(self): self._handle("DELETE")
  return Handler

def serve(cfg: Config) -> tuple[ThreadingHTTPServer, Gateway]:
  cfg.validate()
  gw = Gateway(cfg)
  httpd = ThreadingHTTPServer((cfg.host, cfg.port), make_handler(gw))
  httpd.daemon_threads = True
  return httpd, gw
