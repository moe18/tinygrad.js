"""python -m tgjs_cuda_worker [serve|issue-token|probe] ..."""
from __future__ import annotations
import argparse, json, os, signal, sys

def main(argv: list[str] | None = None) -> None:
  ap = argparse.ArgumentParser(prog="tgjs_cuda_worker", description=__doc__)
  sub = ap.add_subparsers(dest="cmd")
  s = sub.add_parser("serve", help="run the HTTP gateway (default)")
  s.add_argument("--host"); s.add_argument("--port", type=int)
  s.add_argument("--allowed-origin", action="append", dest="allowed_origins", help="browser origin allowed by CORS (repeatable)")
  s.add_argument("--max-sessions", type=int); s.add_argument("--memory-bytes", type=int); s.add_argument("--time-limit-ms", type=int)
  s.add_argument("--session-ttl-s", type=int); s.add_argument("--engine")
  s.add_argument("--no-auth", action="store_true", help="disable token checks (loopback addresses only; for local development)")
  t = sub.add_parser("issue-token", help="issue a session token with TGJS_CUDA_TOKEN_SECRET (for testing; your backend issues tokens in production)")
  t.add_argument("sub"); t.add_argument("--ttl", type=int, default=900); t.add_argument("--memory-bytes", type=int); t.add_argument("--time-limit-ms", type=int); t.add_argument("--max-sessions", type=int)
  sub.add_parser("probe", help="initialize CUDA in this process and print the device description")
  a = ap.parse_args(argv)
  if a.cmd == "issue-token":
    from .tokens import issue
    secret = os.environ.get("TGJS_CUDA_TOKEN_SECRET")
    if not secret: sys.exit("TGJS_CUDA_TOKEN_SECRET is not set")
    print(issue(secret, a.sub, a.ttl, memoryBytes=a.memory_bytes, timeLimitMs=a.time_limit_ms, maxSessions=a.max_sessions)); return
  if a.cmd == "probe":
    from .cuda import CudaEngine
    from .errors import WorkerError
    try: print(json.dumps(CudaEngine().device_info(), indent=2)); return
    except WorkerError as e: sys.exit(f"{e.name}: {e.message}")
  from .server import Config, serve
  cfg = Config.from_env(**({} if a.cmd is None else {k: getattr(a, k) for k in ("host", "port", "allowed_origins", "max_sessions", "memory_bytes", "time_limit_ms", "session_ttl_s", "engine")}))
  if a.cmd and a.no_auth: cfg.no_auth = True
  httpd, gw = serve(cfg)
  def stop(*_): gw.shutdown(); os._exit(0)
  signal.signal(signal.SIGTERM, stop)
  print(f"tgjs-cuda-worker listening on http://{cfg.host}:{cfg.port} (auth {'OFF' if cfg.no_auth else 'on'}, {cfg.max_sessions} sessions, {cfg.memory_bytes} bytes/session)", flush=True)
  try: httpd.serve_forever()
  finally: gw.shutdown()

if __name__ == "__main__":
  main()
