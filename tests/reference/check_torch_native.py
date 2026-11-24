#!/usr/bin/env python3
"""Run the torch test programs on the tinygrad.js torch shim with *native* tinygrad and compare with real PyTorch's results.

  python -m venv .venv && .venv/bin/pip install tinygrad==0.14.0 numpy      # no torch in this environment
  DEV=CPU .venv/bin/python tests/reference/check_torch_native.py [program ...]

The reference (tests/reference/torch.json) comes from tests/reference/generate_torch.py. The browser tests (tests/torch.test.ts)
make the same comparison on WebGPU; this script is the fast loop while working on the shim.
Also writes tests/fixtures/shim_checkpoint.pt with the shim's torch.save, which generate_torch.py loads with real PyTorch.
"""
import json, math, os, pathlib, subprocess, sys
here = pathlib.Path(__file__).resolve().parent
root = here.parent.parent
sys.path.insert(0, str(root / "python"))
progs = here.parent / "torch_programs"
prelude = (progs / "_prelude.py").read_text()
BOOT = f"import sys; sys.path.insert(0, {str(root / 'python')!r}); import tinygrad_js; assert tinygrad_js.install_torch()\n"
RTOL, ATOL = 2e-3, 2e-4   # float32 results; training runs accumulate rounding differences over their steps

def compare(got, want, where, errs):
  if isinstance(want, dict) and "data" in want and "shape" in want:
    if not isinstance(got, dict) or got.get("shape") != want["shape"] or got.get("dtype") != want["dtype"]:
      return errs.append(f"{where}: {str(got)[:80]} != shape {want['shape']} {want['dtype']}")
    for i, (a, b) in enumerate(zip(got["data"], want["data"])):
      if isinstance(b, str) or isinstance(a, str):
        if a != b: return errs.append(f"{where}[{i}]: {a} != {b}")
      elif abs(a - b) > ATOL + RTOL * abs(b): return errs.append(f"{where}[{i}]: {a} != {b}")
  elif isinstance(want, dict):
    for k in want:
      if k != "device": compare(got.get(k) if isinstance(got, dict) else None, want[k], f"{where}.{k}", errs)
  elif isinstance(want, list):
    if not isinstance(got, list) or len(got) != len(want): return errs.append(f"{where}: {str(got)[:80]} != {str(want)[:80]}")
    for i, (a, b) in enumerate(zip(got, want)): compare(a, b, f"{where}[{i}]", errs)
  elif isinstance(want, float):
    if not isinstance(got, (int, float)) or isinstance(got, bool) or abs(got - want) > ATOL + RTOL * abs(want): errs.append(f"{where}: {got} != {want}")
  elif got != want or type(got) is not type(want): errs.append(f"{where}: {got!r} != {want!r}")

ref = json.loads((here / "torch.json").read_text())
names = sys.argv[1:] or sorted(ref["results"])
failed = 0
for name in names:
  r = subprocess.run([sys.executable, "-c", BOOT + prelude + "\n" + (progs / f"{name}.py").read_text()], capture_output=True, text=True,
                     env={**os.environ, "DEV": os.environ.get("DEV", "CPU")})
  line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT_JSON:")), None)
  if line is None: print(f"FAIL {name}\n{r.stdout[-2000:]}{r.stderr[-3000:]}"); failed += 1; continue
  errs: list[str] = []
  compare(json.loads(line[len("RESULT_JSON:"):]), ref["results"][name], name, errs)
  print(("ok  " if not errs else "FAIL") + f" {name}" + "".join(f"\n     {e}" for e in errs[:20]))
  failed += bool(errs)

# a checkpoint written by the shim, for generate_torch.py to load with real PyTorch
import tinygrad_js
tinygrad_js.install_torch()
import torch
m = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2))
with torch.no_grad():
  m[0].weight.copy_(torch.tensor([[0.5, -1.0, 2.0], [1.5, 0.25, -0.75]])); m[0].bias.copy_(torch.tensor([0.1, -0.2]))
  m[1].running_mean.copy_(torch.tensor([0.3, 0.4])); m[1].num_batches_tracked.fill_(7)
torch.save({"model": m.state_dict(), "epoch": 3, "ids": torch.arange(4), "nested": {"t": torch.tensor([[1, 2], [3, 4]], dtype=torch.int32).t()}},
           here.parent / "fixtures" / "shim_checkpoint.pt")
print(f"reference: PyTorch {ref['torch']}; wrote tests/fixtures/shim_checkpoint.pt")
sys.exit(1 if failed else 0)
