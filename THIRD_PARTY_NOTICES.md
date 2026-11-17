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

## Development-only dependencies

esbuild (MIT), TypeScript (Apache-2.0), @webgpu/types (BSD-3-Clause), @playwright/test (Apache-2.0), Vite (MIT, demo only).
