#!/usr/bin/env python3
"""Generate reference results for the torch shim with *real PyTorch*.

  python -m venv .venv-torch && .venv-torch/bin/pip install torch numpy
  .venv-torch/bin/python tests/reference/generate_torch.py     # writes tests/reference/torch.json

Runs every tests/torch_programs/*.py under real PyTorch. The same programs then run unmodified on the tinygrad.js torch shim
(natively: tests/reference/check_torch_native.py; in the browser on WebGPU: tests/torch.test.ts) and must give the same results.
Also checks that a checkpoint written by the shim's torch.save (tests/fixtures/shim_checkpoint.pt) loads in real PyTorch.
"""
import json, pathlib, subprocess, sys
import torch
assert not getattr(torch, "__tinygrad_js_shim__", False), "run this with real PyTorch, not the shim"
here = pathlib.Path(__file__).resolve().parent
progs = here.parent / "torch_programs"
prelude = (progs / "_prelude.py").read_text()
out = {"torch": torch.__version__, "results": {}}
for p in sorted(progs.glob("*.py")):
  if p.name.startswith("_"): continue
  r = subprocess.run([sys.executable, "-c", prelude + "\n" + p.read_text()], capture_output=True, text=True)
  if r.returncode: sys.exit(f"{p.name} failed:\n{r.stderr}")
  line = next(l for l in r.stdout.splitlines() if l.startswith("RESULT_JSON:"))
  out["results"][p.stem] = json.loads(line[len("RESULT_JSON:"):])
  print("ok", p.name)
(here / "torch.json").write_text(json.dumps(out))

ckpt = here.parent / "fixtures" / "shim_checkpoint.pt"
if ckpt.exists():
  sd = torch.load(ckpt, weights_only=True)
  m = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2))
  m.load_state_dict(sd["model"])
  assert sd["epoch"] == 3 and sd["ids"].tolist() == [0, 1, 2, 3] and sd["ids"].dtype == torch.int64, sd
  assert torch.allclose(m[0].weight, torch.tensor([[0.5, -1.0, 2.0], [1.5, 0.25, -0.75]])) and int(m[1].num_batches_tracked) == 7
  assert m[1].num_batches_tracked.shape == () and sd["nested"]["t"].tolist() == [[1, 3], [2, 4]]
  print("ok shim_checkpoint.pt loads in real PyTorch", torch.__version__)
