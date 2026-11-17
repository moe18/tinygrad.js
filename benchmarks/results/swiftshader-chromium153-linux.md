# Benchmark: swiftshader-chromium153-linux

- date: 2026-11-17T03:52:45.002Z
- browser: Chromium 153.0.8010.12
- WebGPU adapter: google / swiftshader  (fallback adapter: true)
- host CPU: Intel(R) Xeon(R) Processor @ 2.10GHz × 4, linux 6.18.44-fc-v553
- tinygrad 0.14.0, Pyodide 314.0.8, Python 3.14.2

## Initialization (median of 5 fresh workers)

| Pyodide load | numpy + tinygrad wheel | WebGPU adapter+device | import tinygrad + backend | total |
|---|---|---|---|---|
| 1894 ms | 455 ms | 11 ms | 659 ms | 3159 ms |

WebGPU init runs concurrently with Pyodide loading, so it is not on the critical path.

## First kernel (compile + run) vs warm run

| where | first | warm |
|---|---|---|
| browser WebGPU | 53.99 ms | 5.283 ms |
| browser CPU (interpreted) | 44.01 ms | 13.248 ms |
| native CPU (tinygrad 0.14.0) | 49.41 ms | 0.952 ms |

## Elementwise `x*y+x` (realize + synchronize)

| where | elements | time | GB/s |
|---|---|---|---|
| browser WebGPU | 1024 | 5.165 ms | 0.002 |
| browser WebGPU | 65536 | 5.170 ms | 0.152 |
| browser WebGPU | 1048576 | 13.255 ms | 0.949 |
| browser WebGPU | 4194304 | 43.672 ms | 1.152 |
| browser CPU (interpreted) | 1024 | 21.265 ms | 0.001 |
| browser CPU (interpreted) | 16384 | 182.055 ms | 0.001 |
| native CPU (tinygrad 0.14.0) | 1024 | 1.369 ms | 0.009 |
| native CPU (tinygrad 0.14.0) | 65536 | 1.068 ms | 0.736 |
| native CPU (tinygrad 0.14.0) | 1048576 | 2.032 ms | 6.192 |
| native CPU (tinygrad 0.14.0) | 4194304 | 4.123 ms | 12.208 |

## Matmul n×n (realize + synchronize, no readback)

| where | n | time | GFLOP/s |
|---|---|---|---|
| browser WebGPU | 64 | 8.880 ms | 0.059 |
| browser WebGPU | 128 | 7.955 ms | 0.527 |
| browser WebGPU | 256 | 15.585 ms | 2.153 |
| browser WebGPU | 512 | 47.835 ms | 5.612 |
| browser WebGPU | 1024 | 363.165 ms | 5.913 |
| browser CPU (interpreted) | 16 | 705.170 ms | 0.000 |
| browser CPU (interpreted) | 32 | 3691.090 ms | 0.000 |
| native CPU (tinygrad 0.14.0) | 64 | 2.022 ms | 0.259 |
| native CPU (tinygrad 0.14.0) | 128 | 2.500 ms | 1.678 |
| native CPU (tinygrad 0.14.0) | 256 | 3.962 ms | 8.469 |
| native CPU (tinygrad 0.14.0) | 512 | 13.612 ms | 19.720 |
| native CPU (tinygrad 0.14.0) | 1024 | 546.601 ms | 3.929 |

## Readback (`.numpy()`)

| where | bytes | time | MB/s |
|---|---|---|---|
| browser WebGPU | 4096 | 3.338 ms | 1.2 |
| browser WebGPU | 4194304 | 7.987 ms | 525.1 |
| browser WebGPU | 16777216 | 21.527 ms | 779.4 |
| native CPU (tinygrad 0.14.0) | 4096 | 0.203 ms | 20.2 |
| native CPU (tinygrad 0.14.0) | 4194304 | 1.233 ms | 3400.8 |
| native CPU (tinygrad 0.14.0) | 16777216 | 3.659 ms | 4584.9 |

## MLP training (2→64→1, batch 256, Adam)

| where | TinyJit | ms/step | loss first → last |
|---|---|---|---|
| browser WebGPU | no | 36.68 | 0.2607 → 0.1057 |
| browser WebGPU | yes | 4.35 | 0.2502 → 0.0892 |
| browser CPU (interpreted) | no | 4963.16 | 0.2214 → 0.1887 |
| native CPU (tinygrad 0.14.0) | no | 29.76 | 0.2607 → 0.1057 |
| native CPU (tinygrad 0.14.0) | yes | 0.51 | 0.2502 → 0.0892 |

