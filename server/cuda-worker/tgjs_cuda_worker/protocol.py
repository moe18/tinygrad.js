"""Framing between the gateway and a session's child process: <u32 header length><u32 payload length><json header><payload>."""
from __future__ import annotations
import json, os, select, struct, time

class ChildTimeout(Exception): pass
class ChildDied(Exception): pass

def send(fd: int, header: dict, payload: bytes = b"") -> None:
  h = json.dumps(header, separators=(",", ":")).encode()
  _write_all(fd, struct.pack("<II", len(h), len(payload)) + h + payload)

def recv(fd: int, deadline: float | None = None) -> tuple[dict, bytes]:
  hl, pl = struct.unpack("<II", _read_exact(fd, 8, deadline))
  header = json.loads(_read_exact(fd, hl, deadline))
  return header, _read_exact(fd, pl, deadline) if pl else b""

def _write_all(fd: int, data: bytes) -> None:
  view = memoryview(data)
  while view:
    try: n = os.write(fd, view)
    except BrokenPipeError as e: raise ChildDied("session process exited") from e
    view = view[n:]

def _read_exact(fd: int, n: int, deadline: float | None) -> bytes:
  chunks, got = [], 0
  while got < n:
    if deadline is not None:
      left = deadline - time.monotonic()
      if left <= 0 or not select.select([fd], [], [], left)[0]: raise ChildTimeout()
    b = os.read(fd, min(n - got, 1 << 20))
    if not b: raise ChildDied("session process exited")
    chunks.append(b); got += len(b)
  return b"".join(chunks)
