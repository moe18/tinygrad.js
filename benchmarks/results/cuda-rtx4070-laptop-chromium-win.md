# CUDA benchmarks: rtx4070-laptop-chromium-win

- date: 2026-11-25T18:11:58.917Z
- browser: Chromium 153.0.8010.12, cross-origin isolated: true
- WebGPU: nvidia lovelace (software fallback adapter: false; timestamps: true; subgroups: [32,32])
- CPU: 13th Gen Intel(R) Core(TM) i9-13900H × 20 (win32 10.0.26300); WASM pool: 8 threads (scalar WebAssembly, no SIMD)

Times in ms per launch. **exec**: steady state, launches queued back to back, as many as fill ~200 ms, then one synchronize (median). **latency**: one launch + synchronize. **compile**: extra time of the first launch (lowering, codegen, pipeline creation; ~0 when an earlier workload already compiled the same kernel for the same block size). **gpu**: GPU timestamp (when supported). Results are verified against host references (max rel. error) before being reported.

| workload | variant | exec | latency | gpu | compile | upload | download | GFLOP/s | GB/s | max rel err |
|---|---|---|---|---|---|---|---|---|---|---|
| vector_add n=4194304 | CUDA→WGSL optimized | 0.204 | 2.69 | 0.196 | 24.1 | 35.28 | 11.91 | – | 246.9 | 0.0e+0 |
| vector_add n=4194304 | CUDA→WGSL direct | 0.206 | 2.43 | 0.203 | 8.4 | 15.65 | 7.94 | – | 244.2 | 0.0e+0 |
| vector_add n=4194304 | CUDA→WASM 8 threads | 1.952 | 1.90 | – | 44.3 | 20.62 | 5.77 | – | 25.8 | 0.0e+0 |
| vector_add n=4194304 | CUDA→WASM 1 thread | 7.061 | 6.79 | – | 3.3 | 17.30 | 3.24 | – | 7.1 | 0.0e+0 |
| vector_add n=4194304 | handwritten WGSL | 0.209 | 2.32 | – | 5.6 | 16.21 | 6.63 | – | 240.6 | 0.0e+0 |
| matmul_naive 256 | CUDA→WGSL optimized | 0.039 | 2.58 | 0.037 | 11.4 | 4.19 | 2.76 | 850.9 | – | 2.6e-7 |
| matmul_naive 256 | CUDA→WGSL direct | 0.039 | 2.50 | 0.039 | 10.2 | 3.87 | 2.66 | 855.4 | – | 2.6e-7 |
| matmul_naive 256 | CUDA→WASM 8 threads | 6.656 | 6.16 | – | 65.7 | 0.50 | 6.09 | 5.0 | – | 2.6e-7 |
| matmul_naive 256 | CUDA→WASM 1 thread | 31.924 | 32.39 | – | 15.4 | 0.44 | 0.08 | 1.1 | – | 2.6e-7 |
| matmul_tiled<16> 256 | CUDA→WGSL optimized | 0.039 | 2.53 | 0.041 | 24.2 | 3.24 | 2.60 | 851.3 | – | 2.6e-7 |
| matmul_tiled<16> 256 | CUDA→WGSL direct | 0.038 | 2.49 | 0.038 | 22.1 | 3.47 | 2.54 | 876.9 | – | 2.6e-7 |
| matmul_tiled<16> 256 | CUDA→WASM 8 threads | 5.413 | 6.03 | – | 50.2 | 0.52 | 5.08 | 6.2 | – | 2.6e-7 |
| matmul_tiled<16> 256 | CUDA→WASM 1 thread | 23.862 | 23.43 | – | 6.8 | 0.44 | 0.05 | 1.4 | – | 2.6e-7 |
| matmul_tiled<16> 256 | handwritten WGSL | 0.029 | 2.43 | – | 11.8 | 2.81 | 2.88 | 1165.8 | – | 2.6e-7 |
| matmul_regtile<64,64,8,4,4> 256 | CUDA→WGSL optimized | 0.037 | 2.54 | 0.039 | 78.8 | 3.17 | 3.08 | 906.9 | – | 2.6e-7 |
| matmul_regtile<64,64,8,4,4> 256 | CUDA→WGSL direct | 0.037 | 3.44 | 0.039 | 33.3 | 3.14 | 2.59 | 898.1 | – | 2.6e-7 |
| matmul_regtile<64,64,8,4,4> 256 | CUDA→WASM 8 threads | 2.384 | 2.17 | – | 23.2 | 0.40 | 2.06 | 14.1 | – | 2.6e-7 |
| matmul_regtile<64,64,8,4,4> 256 | CUDA→WASM 1 thread | 10.146 | 10.54 | – | 17.2 | 0.27 | 0.05 | 3.3 | – | 2.6e-7 |
| matmul_naive 512 | CUDA→WGSL optimized | 0.263 | 3.33 | 0.225 | 9.1 | 6.17 | 3.73 | 1019.0 | – | 3.2e-7 |
| matmul_naive 512 | CUDA→WGSL direct | 0.261 | 2.72 | 0.263 | 7.0 | 3.51 | 2.72 | 1027.2 | – | 3.2e-7 |
| matmul_naive 512 | CUDA→WASM 8 threads | 55.751 | 50.75 | – | 0.1 | 1.75 | 57.82 | 4.8 | – | 3.2e-7 |
| matmul_tiled<16> 512 | CUDA→WGSL optimized | 0.264 | 2.55 | 0.260 | 2.5 | 3.68 | 3.15 | 1016.3 | – | 3.2e-7 |
| matmul_tiled<16> 512 | CUDA→WGSL direct | 0.256 | 2.56 | 0.243 | 1.3 | 4.76 | 3.25 | 1049.3 | – | 3.2e-7 |
| matmul_tiled<16> 512 | CUDA→WASM 8 threads | 42.561 | 39.54 | – | 0.0 | 1.22 | 44.63 | 6.3 | – | 3.2e-7 |
| matmul_tiled<16> 512 | handwritten WGSL | 0.174 | 2.34 | – | 1.0 | 3.86 | 3.33 | 1540.2 | – | 3.2e-7 |
| matmul_regtile<64,64,8,4,4> 512 | CUDA→WGSL optimized | 0.114 | 2.78 | 0.114 | 14.0 | 3.56 | 2.80 | 2357.4 | – | 3.2e-7 |
| matmul_regtile<64,64,8,4,4> 512 | CUDA→WGSL direct | 0.114 | 2.62 | 0.114 | 0.4 | 3.41 | 3.76 | 2350.3 | – | 3.2e-7 |
| matmul_regtile<64,64,8,4,4> 512 | CUDA→WASM 8 threads | 18.211 | 23.09 | – | 26.5 | 1.71 | 25.09 | 14.7 | – | 3.2e-7 |
| reduce_sum<256> n=4194304 | CUDA→WGSL optimized | 0.046 | 2.69 | 0.045 | 21.1 | 12.44 | 3.55 | – | 361.6 | 4.5e-6 |
| reduce_sum<256> n=4194304 | CUDA→WGSL direct | 0.047 | 2.87 | 0.046 | 12.5 | 8.28 | 2.47 | – | 357.6 | 3.9e-6 |
| reduce_sum<256> n=4194304 | CUDA→WASM 8 threads | 2.048 | 2.65 | – | 17.7 | 5.77 | 2.32 | – | 8.2 | 4.2e-6 |
| reduce_sum<256> n=4194304 | CUDA→WASM 1 thread | 12.807 | 13.38 | – | 0.9 | 6.42 | 12.85 | – | 1.3 | 5.9e-6 |
| reduce_sum<256> n=4194304 | handwritten WGSL | 0.029 | 2.39 | – | 21.2 | 9.13 | 2.65 | – | 572.2 | 5.0e-6 |
| softmax_rows<128> 1024x1024 | CUDA→WGSL optimized | 0.018 | 2.96 | 0.016 | 24.9 | 5.61 | 3.55 | – | 478.6 | 2.0e-7 |
| softmax_rows<128> 1024x1024 | CUDA→WGSL direct | 0.017 | 2.51 | 0.015 | 26.6 | 3.35 | 3.44 | – | 479.6 | 2.0e-7 |
| softmax_rows<128> 1024x1024 | CUDA→WASM 8 threads | 6.125 | 5.61 | – | 57.5 | 3.29 | 5.43 | – | 1.4 | 2.0e-7 |
| softmax_rows<128> 1024x1024 | CUDA→WASM 1 thread | 28.936 | 28.97 | – | 6.2 | 2.57 | 0.83 | – | 0.3 | 2.0e-7 |
| layernorm_fwd<128> 1024x1024 | CUDA→WGSL optimized | 0.016 | 2.81 | 0.014 | 24.9 | 6.89 | 4.54 | – | 512.9 | 1.8e-7 |
| layernorm_fwd<128> 1024x1024 | CUDA→WGSL direct | 0.016 | 2.39 | 0.016 | 19.5 | 3.56 | 3.44 | – | 510.2 | 1.8e-7 |
| layernorm_fwd<128> 1024x1024 | CUDA→WASM 8 threads | 1.280 | 1.30 | – | 17.1 | 2.96 | 2.19 | – | 6.6 | 1.8e-7 |
| layernorm_fwd<128> 1024x1024 | CUDA→WASM 1 thread | 5.945 | 6.26 | – | 2.7 | 2.63 | 0.74 | – | 1.4 | 1.8e-7 |
| sdpa_naive causal H=8 T=512 D=16 | CUDA→WGSL optimized | 1.306 | 2.54 | 1.287 | 33.6 | 4.27 | 2.54 | 51.4 | – | 1.7e-7 |
| sdpa_naive causal H=8 T=512 D=16 | CUDA→WGSL direct | 1.295 | 2.46 | 1.277 | 34.5 | 2.74 | 3.52 | 51.8 | – | 1.7e-7 |
| sdpa_naive causal H=8 T=512 D=16 | CUDA→WASM 8 threads | 26.163 | 26.01 | – | 255.2 | 0.42 | 29.88 | 2.6 | – | 1.7e-7 |
| flash_attention_fwd<32,16,16> causal H=8 T=512 D=16 | CUDA→WGSL optimized | 0.088 | 2.60 | 0.089 | 69.2 | 3.30 | 2.43 | 766.8 | – | 1.3e-7 |
| flash_attention_fwd<32,16,16> causal H=8 T=512 D=16 | CUDA→WGSL direct | 0.088 | 2.96 | 0.088 | 42.8 | 3.25 | 3.07 | 766.0 | – | 1.3e-7 |
| flash_attention_fwd<32,16,16> causal H=8 T=512 D=16 | CUDA→WASM 8 threads | 12.390 | 12.08 | – | 23.0 | 0.41 | 10.63 | 5.4 | – | 1.3e-7 |

## CUDA kernels on tinygrad tensors vs tinygrad operations (tinygrad 0.14.0)

Per call: calls realized back to back for ~200 ms, then one synchronize (median); both produce the same result (checked).

| device | workload | tinygrad.cuda kernel | tinygrad op | ratio (op / kernel) | max rel err |
|---|---|---|---|---|---|
| WEBGPU | vector add n=1048576 | 1.846 | 1.569 | 0.85× | 0.0e+0 |
| WEBGPU | matmul 256 (matmul_tiled<16>) | 2.802 | 4.669 | 1.67× | 0.0e+0 |
| WEBGPU | matmul 256 (matmul_regtile) | 3.058 | 4.700 | 1.54× | 0.0e+0 |
| WEBGPU | matmul 512 (matmul_tiled<16>) | 3.878 | 4.943 | 1.27× | 0.0e+0 |
| WEBGPU | matmul 512 (matmul_regtile) | 3.805 | 6.126 | 1.61× | 0.0e+0 |
| WEBGPU | softmax 1024x1024 | 2.231 | 4.974 | 2.23× | 2.0e-7 |
| PYTHON | vector add n=16384 | 1.876 | 137.530 | 73.32× | 0.0e+0 |
| PYTHON | matmul 64 (matmul_tiled<16>) | 2.749 | 797.870 | 290.21× | 3.3e-7 |
| PYTHON | softmax 16x256 | 2.262 | 212.680 | 94.03× | 1.6e-7 |

A ratio above 1× means the CUDA kernel was faster on this machine; it says nothing about other hardware. On a software WebGPU adapter (SwiftShader) GPU numbers measure a CPU emulation of a GPU.
