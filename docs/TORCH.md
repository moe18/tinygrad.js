# `import torch` in tinygrad.js

tinygrad.js ships a **PyTorch-compatible API on top of tinygrad**. In the browser runtime, `import torch` just works and runs on
WebGPU:

```python
import torch, torch.nn as nn

model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1))
opt = torch.optim.Adam(model.parameters(), lr=0.01)
for step in range(100):
  opt.zero_grad()
  loss = nn.functional.mse_loss(model(x), y)
  loss.backward()
  opt.step()
```

**This is not PyTorch.** PyTorch itself has no WebAssembly build. The module is a reimplementation of the commonly used part of the
`torch` API (about 3,300 lines of Python in [`python/tinygrad_js/_shims/torch`](../python/tinygrad_js/_shims/torch)) that
translates calls to upstream tinygrad. It identifies itself: `torch.__version__ == "0.1.0+tinygrad.js"` and
`torch.__tinygrad_js_shim__ is True`.

The target is: **tutorial-level PyTorch code runs unmodified and gives the same numbers as PyTorch.** Libraries built on PyTorch
internals (`torchvision`, `transformers`, `lightning`, `timm`, ...) do not run.

## How it is tested

[`tests/torch_programs/`](../tests/torch_programs) contains eleven plain PyTorch scripts. Each runs unmodified three ways:

1. under **real PyTorch** (2.15), producing the reference `tests/reference/torch.json` (`tests/reference/generate_torch.py`);
2. on the shim with native tinygrad on CPU (`tests/reference/check_torch_native.py`);
3. on the shim **in the browser on WebGPU** (`tests/torch.test.ts`, Playwright).

Results must agree with PyTorch's to rtol 2e-3 / atol 2e-4 (float32; training runs accumulate rounding differences over their steps).
Because the shim's random number streams differ from PyTorch's, the programs set weights and data from fixed values wherever numbers
are compared, and check random initializers statistically.

| program | what it covers |
|---|---|
| `tensor_basics` | creation, dtype inference, arithmetic, broadcasting, ~40 reductions, indexing (slices, index tensors, boolean masks), views, `cat`/`stack`/`split`, sort/topk, `gather`/`scatter`, `einsum`, in-place operations, `item`/`tolist`/`numpy`, `repr` |
| `autograd` | `backward`, gradient accumulation, `no_grad`/`detach`/`set_grad_enabled`, `retain_graph`, `torch.autograd.grad`, second derivatives, vector-Jacobian products, leaf rules, gradients of 23 operations, `.data` |
| `polynomial` | the "Learning PyTorch with Examples" tutorial: manual gradients, autograd, `nn.Sequential`, `optim`, a custom `nn.Module` |
| `nn_layers` | `Linear`, `Conv1d/2d`, `ConvTranspose2d`, pooling, `BatchNorm` (train and eval), `LayerNorm`, `GroupNorm`, `Embedding`, 25 activations, 30+ loss configurations and their gradients, `MultiheadAttention`, `TransformerEncoder/DecoderLayer`, `RNN`/`GRU`/`LSTM`, default initialization |
| `optimizers` | `SGD` (momentum, Nesterov, dampening, weight decay), `Adam`, `AdamW`, `RMSprop`, `Adagrad`, `Adadelta`, parameter groups, 11 LR schedulers, gradient clipping, closures |
| `classifier` | the "Quickstart" structure: `Dataset`, `DataLoader`, model class, train and test loops |
| `cnn` | the CIFAR-10 tutorial CNN, and a small ResNet with BatchNorm and in-place residual adds |
| `gpt` | a nanoGPT-style transformer (model code follows nanoGPT's `model.py`): weight tying, AdamW parameter groups, per-iteration LR schedule, gradient clipping, `F.scaled_dot_product_attention` and the manual masked-attention path, top-k generation |
| `patterns` | DCGAN-style alternating updates, DQN target networks (`deepcopy`, soft updates, EMA through `.data`), REINFORCE with `torch.distributions`, fine-tuning with frozen layers, forward hooks, module introspection and `repr` |
| `save_load` | `torch.save`/`torch.load` round trips (state dicts, optimizer state, whole modules), and loading a checkpoint written by real PyTorch |
| `snippets` | short learner-style code: linear regression from scratch, sinusoidal positional encoding, a VAE loss, a character LSTM, k-means, and 40 one-line API uses |

Checkpoints are interchangeable in both directions: `save_load` reads a file written by real PyTorch, and `generate_torch.py` loads a
file written by the shim (`tests/fixtures/shim_checkpoint.pt`) with real PyTorch.

## What is implemented

* **`torch`**: `Tensor` with ~300 methods (in-place variants included), the creation functions, dtypes, `torch.device`, grad modes
  (`no_grad`, `enable_grad`, `set_grad_enabled`, `inference_mode`), `manual_seed`, `save`/`load`, `torch.autograd.grad`,
  `torch.linalg` norms, `torch.distributions` (`Normal`, `Uniform`, `Bernoulli`, `Categorical`). Tensor methods that are not
  defined explicitly fall through to the tinygrad method of the same name.
* **`torch.nn`**: `Module` (parameters, buffers, `state_dict`, `load_state_dict`, `train`/`eval`, `apply`, forward hooks, `repr`),
  `Parameter`, containers (`Sequential`, `ModuleList`, `ModuleDict`, `ParameterList`, `ParameterDict`), linear, convolution,
  pooling, normalization, dropout, embedding, activation, loss, recurrent, attention and transformer encoder/decoder layers,
  `nn.functional`, `nn.init`, `nn.utils.clip_grad_norm_`/`clip_grad_value_`. Layers use PyTorch's parameter names, shapes and
  default initialization distributions, so `state_dict`s are compatible.
* **`torch.optim`**: `SGD`, `Adam`, `AdamW`, `RMSprop`, `Adagrad`, `Adadelta` with PyTorch's update rules, parameter groups,
  `state_dict`; `lr_scheduler`: `StepLR`, `MultiStepLR`, `ExponentialLR`, `CosineAnnealingLR`, `CosineAnnealingWarmRestarts`,
  `LambdaLR`, `LinearLR`, `ConstantLR`, `PolynomialLR`, `OneCycleLR` (learning rate only), `SequentialLR`, `ReduceLROnPlateau`.
* **`torch.utils.data`**: `Dataset`, `TensorDataset`, `Subset`, `ConcatDataset`, `random_split`, samplers, `DataLoader`
  (single process), `default_collate`.

## Differences from PyTorch

**Devices.** There is one device: tinygrad's default (WebGPU in the browser). Every device argument is accepted and ignored,
`torch.cuda.is_available()` is `False`, and `tensor.device` reports `webgpu` (or `cpu` on the CPU fallback). The usual
`device = "cuda" if torch.cuda.is_available() else "cpu"` therefore picks `"cpu"` and still runs on the GPU.

**64-bit types.** WebGPU has no `float64` or `int64`. On WebGPU `torch.long` tensors are stored as `int32` and `torch.double`
tensors as `float32`. Dtypes that are stored the same way compare equal there (`labels.dtype == torch.long` is `True`), but
`tensor.dtype` prints as `torch.int32`, integers overflow at 2³¹, and `float64` precision is not available.

**Random numbers.** `torch.manual_seed(s)` makes runs reproducible, but the streams are tinygrad's, so random values (initial weights,
dropout masks, `randn`, shuffles) differ from PyTorch's for the same seed. Distributions are the same.

**Lazy evaluation.** Operations build a tinygrad graph. It is computed when a value is read (`item()`, `numpy()`, `print`, `float()`)
or storage is mutated (`optimizer.step()`, in-place updates of parameters). At such a point *everything* that is still pending is
computed, so a tensor computed before `optimizer.step()` keeps its pre-step value, as in PyTorch. Consequences:
* an error in an operation (for example a shape mismatch inside a kernel) can surface later than in PyTorch, at the next read;
* reading a value before `backward()` computes the forward pass a second time during `backward()`. Reading the loss after
  `optimizer.step()` costs nothing extra.

**In-place operations and views.** In-place operations on parameters write into the same GPU buffer. On other tensors they rebind
the tensor to a new value, which matches PyTorch for the tensor itself, but **a view taken earlier does not see the change**
(`v = x[0]; x += 1` leaves `v` unchanged). `Parameter(t)`, `t.detach()` and `state_dict()` values are snapshots rather than aliases;
`param.data` and `param.grad` do write through.

**Speed.** Each kernel is scheduled by Python running in WebAssembly, and every parameter and optimizer-state tensor is updated by
its own kernel, so a plain training step is slow compared with PyTorch. Measured in the test environment (SwiftShader, a CPU
emulating a GPU, so real GPUs will differ): a step of an MLP (2→32→1, batch 256, SGD with momentum) takes 64 ms; a step of the
two-layer nanoGPT-style model (about 30 parameter tensors, AdamW) dispatches ~420 kernels and takes about 5 s.
`tinygrad_js.run_training(step, steps, args=(X, Y))` runs an ordinary PyTorch-style step function under tinygrad's `TinyJit`,
which replays the recorded kernels without rerunning Python: the same MLP step takes 6.8 ms with identical losses (tested for SGD,
Adam and a learning-rate schedule changed between calls). See the README for `run_training`'s rules: Python inside the step runs
only while recording, so keep data-dependent Python control flow and `.item()` calls out of the step.

**Not implemented** (each raises a clear error instead of silently doing something else):
* custom `torch.autograd.Function`, tensor hooks (`register_hook`), backward hooks;
* `torch.jit` (script/trace are pass-throughs), `torch.compile` (returns the model unchanged), `torch.fx`, ONNX export;
* sparse tensors, complex numbers, quantization, most of `torch.linalg` (decompositions, solvers), `torch.fft`;
* multiprocessing data loading (`num_workers` is ignored), distributed training, mixed precision (`autocast` is a no-op);
* packed sequences, `nn.Transformer` (compose `TransformerEncoder`/`TransformerDecoder`), convolution padding modes other than
  zeros, `adaptive_*_pool` with an output size that does not divide the input size;
* `torch.load` of the legacy (pre-1.6, non-zip) checkpoint format.

## Using it

**In the browser** it is on by default: `createTinygrad()` → `runtime.run("import torch; ...")`. Pass `torch: false` to
`createTinygrad` to leave `torch` unimportable. `runtime.getDeviceInfo().torchShim` says whether it is active. It needs
tinygrad ≥ 0.14 (the default); with `tinygradVersion: "0.12.0"` `import torch` raises `ModuleNotFoundError` and the reason is
in `diagnostics`.

**Natively** (for example to grade the same code on a server without installing PyTorch):

```python
import tinygrad_js
tinygrad_js.install_torch()      # returns False and does nothing if a real torch is installed
import torch
```

`torch.save` writes to Pyodide's in-memory file system in the browser; pass an `io.BytesIO` to get the bytes out.

## Extending it

The shim is ordinary Python. To add an operation, add a method to `Tensor` in `_tensor.py` (wrap the tinygrad result with
`_wrap(result, *inputs)` so that `requires_grad` propagates), add a line that uses it to a program in `tests/torch_programs/`,
regenerate the reference with real PyTorch, and run `tests/reference/check_torch_native.py`.
