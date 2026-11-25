"""Checks on user CUDA source before NVRTC sees it.

NVRTC's preprocessor opens whatever an #include names, on the worker's filesystem, and its error log quotes the lines it
could not parse. `#include "/proc/<gateway pid>/environ"` would send the gateway's environment (with
TGJS_CUDA_TOKEN_SECRET) back to the client in the compile error, and any readable host file could be read the same way.
So only plain relative header names are allowed: `<name>` or `"name"` made of [A-Za-z0-9_.+-/], not absolute, no "..".
They resolve against NVRTC's built-in headers and the CUDA toolkit include directory (the session process runs in an
empty directory, so a quoted name finds nothing else). An include whose name comes from a macro is refused, because its
target cannot be checked before preprocessing.

The check errs on the side of refusing: line splices (backslash-newline) are joined, comments become one space, the
digraph %: counts as #, and literals are kept as they are (they are only tracked so that "/*" inside a string does not
start a comment), so a directive-like line inside a raw string is checked too.
"""
from __future__ import annotations
import re
from .errors import WorkerError

_FILE_DIRECTIVES = ("include_next", "include", "import", "embed")
_DIRECTIVE = re.compile(r"[ \t\f\v]*(?:#|%:)[ \t\f\v]*(" + "|".join(_FILE_DIRECTIVES) + r")(?![A-Za-z0-9_])[ \t\f\v]*(.*)")
_NAME = re.compile(r"[A-Za-z0-9_+\-][A-Za-z0-9_.+\-/]*")
_RAW_PREFIX = re.compile(r"(?:u8|u|U|L)?R\"([^()\\ \t\n]{0,16})\(")

def _without_comments(src: str) -> str:
  s = src.replace("\r\n", "\n").replace("\r", "\n")
  s = re.sub(r"\\[ \t]*\n", "", s)  # line splices (GCC also accepts spaces between the backslash and the newline)
  out: list[str] = []
  i, n = 0, len(s)
  while i < n:
    c = s[i]
    if c == "/" and s.startswith("//", i):
      j = s.find("\n", i)
      i = n if j < 0 else j
      out.append(" ")
    elif c == "/" and s.startswith("/*", i):
      j = s.find("*/", i + 2)
      i = n if j < 0 else j + 2
      out.append(" ")
    elif c in "uULR" and (i == 0 or not (s[i - 1].isalnum() or s[i - 1] == "_")) and (m := _RAW_PREFIX.match(s, i)):
      end = s.find(")" + m.group(1) + '"', m.end())
      j = n if end < 0 else end + len(m.group(1)) + 2
      out.append(s[i:j]); i = j
    elif c in "\"'":
      # a literal ends at its closing quote or, unterminated, at the end of the line (where the preprocessor resumes)
      j = i + 1
      while j < n and s[j] != c and s[j] != "\n":
        j += 2 if s[j] == "\\" else 1
      j = j + 1 if j < n and s[j] == c else j
      out.append(s[i:j]); i = j
    else:
      out.append(c); i += 1
  return "".join(out)

def _header_name(rest: str) -> str | None:
  rest = rest.strip()
  if len(rest) >= 2 and ((rest[0] == "<" and rest[-1] == ">") or (rest[0] == '"' and rest[-1] == '"')):
    return rest[1:-1]
  return None

def check_includes(src: str) -> None:
  """Raise WorkerError (cudaErrorInvalidSource) for an #include (or #include_next / #import / #embed) that could read a
  file other than a CUDA header."""
  for line in _without_comments(src).split("\n"):
    m = _DIRECTIVE.match(line)
    if not m: continue
    name = _header_name(m.group(2))
    if name is None or name.startswith("/") or ".." in name.split("/") or not _NAME.fullmatch(name):
      shown = m.group(2).strip()[:80] or "(nothing)"
      raise WorkerError("cudaErrorInvalidSource",
                        f"#{m.group(1)} {shown}: the remote worker only allows headers by plain relative name, "
                        "e.g. #include <cuda_fp16.h> (no absolute paths, '..' or macro-computed names)")
