"""A session's child process: owns one CUDA context and executes the gateway's requests (python -m tgjs_cuda_worker.child).

Isolation: every session gets its own process (and so its own CUDA context); the gateway kills it when a launch exceeds the
time limit or the session ends, which also releases all of its GPU memory. The process runs in an empty temporary
directory with a scrubbed environment and resource limits, and drops to an unprivileged uid when configured. Device-side
printf output, which the CUDA driver writes to stdout, is captured through a temporary file and returned per launch.
"""
from __future__ import annotations
import base64, ctypes, importlib, json, os, sys, tempfile, traceback
from .protocol import send, recv, ChildDied
from .errors import WorkerError

def _sandbox(cfg: dict) -> None:
  import resource
  def lim(r, v):
    try: resource.setrlimit(r, (v, v))
    except (ValueError, OSError): pass
  lim(resource.RLIMIT_CORE, 0)
  lim(resource.RLIMIT_NOFILE, 256)
  lim(resource.RLIMIT_FSIZE, int(cfg.get("printfBytes", 1 << 20)) * 4 + (1 << 20))
  if cfg.get("cpuSeconds"): lim(resource.RLIMIT_CPU, int(cfg["cpuSeconds"]))
  os.chdir(tempfile.mkdtemp(prefix="tgjs-cuda-"))
  os.umask(0o077)
  uid = cfg.get("runAsUid")
  if uid is not None and os.getuid() == 0:
    os.setgroups([]); os.setgid(int(cfg.get("runAsGid", uid))); os.setuid(int(uid))

def main() -> None:
  rfd, wfd = int(sys.argv[1]), int(sys.argv[2])
  cfg = json.loads(sys.argv[3])
  # stdout is where the CUDA driver writes device printf: capture it in a file we read after each launch
  cap = tempfile.TemporaryFile()
  os.dup2(cap.fileno(), 1)
  libc = ctypes.CDLL(None)
  printf_pos = 0
  def take_printf() -> str:
    nonlocal printf_pos
    libc.fflush(None)
    cap.seek(printf_pos); data = cap.read(); printf_pos = cap.tell()
    limit = int(cfg.get("printfBytes", 1 << 20))
    return data[:limit].decode(errors="replace") + ("\n[printf output truncated]\n" if len(data) > limit else "")

  _sandbox(cfg)
  try:
    mod, cls = cfg["engine"].split(":")
    engine = getattr(importlib.import_module(mod), cls)()
    send(wfd, {"ok": True, "device": engine.device_info()})
  except WorkerError as e:
    send(wfd, {"ok": False, **e.to_json(), "status": e.status}); return
  except Exception as e:
    send(wfd, {"ok": False, "error": {"name": "cudaErrorNoDevice", "message": f"{type(e).__name__}: {e}"}, "status": 503}); return

  while True:
    try: req, payload = recv(rfd)
    except ChildDied: break
    op = req.get("op")
    try:
      out: dict = {"ok": True}; body = b""
      if op == "alloc": engine.alloc(req["handle"], int(req["bytes"]))
      elif op == "free": engine.free(req["handle"])
      elif op == "write": engine.write(req["handle"], int(req["offset"]), payload)
      elif op == "read": body = engine.read(req["handle"], int(req["offset"]), int(req["bytes"]))
      elif op == "copy": engine.copy(req["dst"], int(req["dstOffset"]), req["src"], int(req["srcOffset"]), int(req["bytes"]))
      elif op == "memset": engine.memset(req["handle"], int(req["offset"]), int(req["value"]), int(req["bytes"]))
      elif op == "launch":
        args = []
        for a in req["args"]:
          if "ptr" in a: args.append({"ptr": {"handle": str(a["ptr"]["handle"]), "offset": int(a["ptr"]["offset"])}})
          else: args.append({"scalar": {"ctype": a["scalar"].get("ctype"), "bytes": base64.b64decode(a["scalar"]["bytes"], validate=True)}})
        try: out.update(engine.launch(req["source"], req["kernel"], req.get("options") or {}, req["grid"], req["block"], int(req.get("sharedMem", 0)), args, bool(req.get("timing"))))
        finally: out["printf"] = take_printf()
      elif op == "close":
        engine.close(); send(wfd, out); break
      else: raise WorkerError("cudaErrorInvalidValue", f"unknown operation {op!r}")
      send(wfd, out, body)
    except WorkerError as e:
      send(wfd, {"ok": False, **e.to_json(), "status": e.status, "sticky": bool(getattr(e, "sticky", False)), "printf": out.get("printf", "")})
    except Exception as e:
      traceback.print_exc(file=sys.stderr)
      send(wfd, {"ok": False, "error": {"name": "cudaErrorUnknown", "message": f"{type(e).__name__}: {e}"}, "status": 500, "sticky": True})

if __name__ == "__main__":
  main()
