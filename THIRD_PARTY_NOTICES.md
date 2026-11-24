# Third-party notices

tinygrad.js's own code is MIT licensed (see LICENSE). The npm package redistributes the following third-party software.

## tinygrad (redistributed: `dist/assets/tinygrad-*.whl`, unmodified wheels from PyPI)

Source: https://github.com/tinygrad/tinygrad. The wheels are byte-identical to PyPI's (SHA-256 pinned in `scripts/pins.mjs`)
and contain their own license file.

```
Copyright (c) 2024, the tiny corp

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## Loaded at runtime, not redistributed by the npm package

These are fetched by the browser from the configured Pyodide URL (by default the jsDelivr CDN). `npm run fetch:assets -- --pyodide`
downloads them into `vendor/` for self-hosting and tests.

* **Pyodide** 314.0.8: Mozilla Public License 2.0, https://github.com/pyodide/pyodide (includes CPython, PSF License).
* **NumPy** 2.4.6 (Pyodide build): BSD-3-Clause, https://github.com/numpy/numpy.

## The `torch` module (`python/tinygrad_js/_shims/torch`)

tinygrad.js's `import torch` is an independent implementation of part of PyTorch's public Python API on top of tinygrad. It contains
no PyTorch source code and is not affiliated with or endorsed by the PyTorch project. PyTorch is a trademark of The Linux Foundation.

## Test programs (in the repository only, not in the npm package)

* `tests/torch_programs/gpt.py`: the model classes are adapted from nanoGPT's `model.py`, https://github.com/karpathy/nanoGPT (MIT):

  ```
  MIT License

  Copyright (c) 2022 Andrej Karpathy

  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
  ```

* `tests/torch_programs/polynomial.py`, `classifier.py`, `cnn.py` and the GAN part of `patterns.py` follow the structure of the PyTorch
  tutorials ("Learning PyTorch with Examples", "Quickstart", "Training a Classifier", "DCGAN Tutorial"),
  https://github.com/pytorch/tutorials (BSD-3-Clause, Copyright (c) 2017-2022, Pytorch contributors).
* `tests/reference/torch.json` holds numbers computed by PyTorch (BSD-3-Clause), which is only needed to regenerate that file.

## Development-only dependencies

esbuild (MIT), TypeScript (Apache-2.0), @webgpu/types (BSD-3-Clause), @playwright/test (Apache-2.0), Vite (MIT, demo only).
