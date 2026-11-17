#!/usr/bin/env python3
"""Generate CPU reference results with *native* upstream tinygrad.

  python -m venv .venv && .venv/bin/pip install tinygrad==0.14.0 numpy
  DEV=CPU .venv/bin/python tests/reference/generate.py   # writes tests/reference/cpu-<version>.json
"""
import json, os, pathlib, subprocess, sys, importlib.metadata
here = pathlib.Path(__file__).resolve().parent
progs = here.parent / "programs"
prelude = (progs / "_prelude.py").read_text()
version = importlib.metadata.version("tinygrad")
out = {"tinygrad": version, "device": os.environ.get("DEV", "CPU"), "results": {}}
for p in sorted(progs.glob("*.py")):
  if p.name.startswith("_"): continue
  r = subprocess.run([sys.executable, "-c", prelude + "\n" + p.read_text()], capture_output=True, text=True, env={**os.environ, "DEV": out["device"]})
  if r.returncode: sys.exit(f"{p.name} failed:\n{r.stderr}")
  line = next(l for l in r.stdout.splitlines() if l.startswith("RESULT_JSON:"))
  out["results"][p.stem] = json.loads(line[len("RESULT_JSON:"):])
  print("ok", p.name)
(here / f"cpu-{version}.json").write_text(json.dumps(out))
