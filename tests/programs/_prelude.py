# Shared helpers for test programs. The programs run both natively (reference: upstream tinygrad, DEV=CPU)
# and in the browser (tinygrad.js, DEV=WEBGPU). They print one line "RESULT_JSON:{...}".
import json
import numpy as np
from tinygrad import Tensor, Device
from tinygrad.helpers import Context

def emit(**kw):
  def conv(v):
    if isinstance(v, Tensor): v = v.numpy()
    if isinstance(v, np.ndarray): return {"shape": list(v.shape), "dtype": str(v.dtype), "data": v.astype(np.float64).ravel().tolist()}
    if isinstance(v, (list, tuple)): return [conv(x) for x in v]
    if isinstance(v, (np.floating, np.integer)): return v.item()
    return v
  print("RESULT_JSON:" + json.dumps({"device": Device.DEFAULT, **{k: conv(v) for k, v in kw.items()}}))

def training_context():
  # tinygrad <= 0.12: Tensor.train(); >= 0.13: Context(TRAINING=1)
  return Tensor.train() if hasattr(Tensor, "train") else Context(TRAINING=1)

def param(data):
  # tinygrad <= 0.12 only tracks gradients for requires_grad=True tensors; 0.13+ removed the argument (all tensors can)
  import inspect
  return Tensor(data, requires_grad=True) if "requires_grad" in inspect.signature(Tensor.__init__).parameters else Tensor(data)
