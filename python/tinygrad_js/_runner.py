"""Executes user code for the worker: persistent globals, structured errors, tracebacks limited to user frames."""
from __future__ import annotations
import sys, time, traceback

USER_FILENAME = "<user-code>"
user_globals: dict = {"__name__": "__main__"}

def reset_globals() -> None:
  user_globals.clear()
  user_globals["__name__"] = "__main__"

def _format_tb(e:BaseException) -> str:
  tb = e.__traceback__
  # drop the frames of this runner and of pyodide's eval machinery until the first user frame
  while tb is not None and tb.tb_frame.f_code.co_filename != USER_FILENAME: tb = tb.tb_next
  return "".join(traceback.format_exception(type(e), e, tb))

async def run(code:str) -> dict:
  from pyodide.code import eval_code_async
  st = time.perf_counter()
  try:
    result = await eval_code_async(code, user_globals, filename=USER_FILENAME)
    out = {"ok": True, "result": None if result is None else repr(result)}
  except BaseException as e:  # KeyboardInterrupt (interrupt()) and SystemExit are reported, not propagated
    out = {"ok": False, "type": type(e).__name__, "message": str(e), "traceback": _format_tb(e)}
  finally:
    sys.stdout.flush(); sys.stderr.flush()
  out["duration_ms"] = (time.perf_counter() - st) * 1000
  return out
