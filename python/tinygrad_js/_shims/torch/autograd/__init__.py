"""torch.autograd: grad(), grad-mode context managers. Custom autograd Functions are not supported."""
from __future__ import annotations
from .._tensor import Tensor, no_grad, enable_grad, set_grad_enabled, inference_mode, _mk, _diff_slice, _live

def grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, only_inputs=True, allow_unused=None, is_grads_batched=False,
         materialize_grads=False):
  """Gradients of ``outputs`` with respect to ``inputs``. With ``create_graph=True`` the result can be differentiated again."""
  single = isinstance(inputs, Tensor)
  inputs = [inputs] if single else list(inputs)
  outs = [outputs] if isinstance(outputs, Tensor) else list(outputs)
  gos = [None] * len(outs) if grad_outputs is None else ([grad_outputs] if isinstance(grad_outputs, Tensor) else list(grad_outputs))
  total = [None] * len(inputs)
  for out, go in zip(outs, gos):
    if not out.requires_grad: raise RuntimeError("element 0 of tensors does not require grad and does not have a grad_fn")
    if go is None and out.numel() != 1: raise RuntimeError("grad can be implicitly created only for scalar outputs")
    root = out._t if go is not None or out._t.shape == () else out._t.reshape(())
    reach = _diff_slice(root.uop)
    used = [i for i, w in enumerate(inputs) if w._t.uop in reach]
    if len(used) != len(inputs) and not allow_unused and not materialize_grads:
      raise RuntimeError("One of the differentiated Tensors appears to not have been used in the graph. Set allow_unused=True if this is the desired behavior.")
    gs = root.gradient(*[inputs[i]._t for i in used], gradient=None if go is None else go._t) if used else []
    for i, g in zip(used, gs):
      if not create_graph: g = g.detach()
      total[i] = g if total[i] is None else total[i] + g
    if not (retain_graph if retain_graph is not None else create_graph):
      for w in _live:
        if w.requires_grad and not w.is_leaf and w._tt.uop in reach: w._freed = True
  res = tuple(None if g is None and not materialize_grads else _mk(g if g is not None else inputs[i]._t.zeros_like(), create_graph, not create_graph)
              for i, g in enumerate(total))
  return res

def backward(tensors, grad_tensors=None, retain_graph=None, create_graph=False, inputs=None):
  ts = [tensors] if isinstance(tensors, Tensor) else list(tensors)
  gs = [None] * len(ts) if grad_tensors is None else ([grad_tensors] if isinstance(grad_tensors, Tensor) else list(grad_tensors))
  for n, (t, g) in enumerate(zip(ts, gs)): t.backward(g, retain_graph=True if n < len(ts) - 1 else retain_graph, create_graph=create_graph, inputs=inputs)

class Function:
  def __init_subclass__(cls, **kw):
    raise NotImplementedError("custom torch.autograd.Function is not supported by the tinygrad.js torch shim: tinygrad derives gradients "
                              "from the forward computation, so write the forward pass with tensor operations instead")

def Variable(data, requires_grad=False): return data.requires_grad_(requires_grad) if requires_grad else data
class set_detect_anomaly:
  def __init__(self, mode=True, check_nan=True): pass
  def __enter__(self): return self
  def __exit__(self, *a): return False
detect_anomaly = set_detect_anomaly

class _Functional:
  @staticmethod
  def jacobian(func, inputs, create_graph=False, **k):
    from .. import stack
    x = inputs if inputs.requires_grad else inputs.detach().requires_grad_(True)
    y = func(x)
    flat = y.reshape(-1)
    rows = [grad(flat[i], x, retain_graph=True, create_graph=create_graph)[0] for i in range(flat.shape[0])]
    return stack(rows).reshape(*y.shape, *x.shape)
functional = _Functional()
