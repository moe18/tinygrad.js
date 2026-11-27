# CUDA benchmarks: swiftshader-chromium-linux

- date: 2026-11-25T06:46:25.837Z
- browser: Chromium 158.0.8127.36, cross-origin isolated: true
- WebGPU: google swiftshader (software fallback adapter: true; timestamps: true; subgroups: [4,4])
- CPU: Intel(R) Xeon(R) Processor @ 2.10GHz × 4 (linux 6.18.44-fc-v880); WASM pool: 4 threads (scalar WebAssembly, no SIMD)

Times in ms per launch. **exec**: steady state, launches queued back to back (median). **latency**: one launch + synchronize. **compile**: extra time of the first launch (lowering, codegen, pipeline creation; ~0 when an earlier workload already compiled the same kernel for the same block size). **gpu**: GPU timestamp (when supported). Results are verified against host references (max rel. error) before being reported.

| workload | variant | exec | latency | gpu | compile | upload | download | GFLOP/s | GB/s | max rel err |
|---|---|---|---|---|---|---|---|---|---|---|
| vector_add n=4194304 | CUDA→WGSL optimized | 36.746 | 37.58 | 34.673 | 18.7 | 51.19 | 62.77 | – | 1.4 | 0.0e+0 |
| vector_add n=4194304 | CUDA→WGSL direct | 41.562 | 42.21 | 40.703 | 17.7 | 22.24 | 60.65 | – | 1.2 | 0.0e+0 |
| vector_add n=4194304 | CUDA→WASM 4 threads | 5.183 | 4.73 | – | 35.6 | 48.77 | 11.71 | – | 9.7 | 0.0e+0 |
| vector_add n=4194304 | CUDA→WASM 1 thread | 8.655 | 8.87 | – | 3.3 | 50.03 | 6.99 | – | 5.8 | 0.0e+0 |
| vector_add n=4194304 | handwritten WGSL | 31.956 | 34.94 | – | 6.5 | 23.35 | 20.25 | – | 1.6 | 0.0e+0 |
| matmul_naive 256 | CUDA→WGSL optimized | 63.028 | 62.46 | 60.765 | 20.0 | 3.89 | 74.09 | 0.5 | – | 2.6e-7 |
| matmul_naive 256 | CUDA→WGSL direct | 84.389 | 85.28 | 83.471 | 14.2 | 5.38 | 85.46 | 0.4 | – | 2.6e-7 |
| matmul_naive 256 | CUDA→WASM 4 threads | 15.923 | 16.57 | – | 38.9 | 0.56 | 16.07 | 2.1 | – | 2.6e-7 |
| matmul_naive 256 | CUDA→WASM 1 thread | 44.750 | 44.34 | – | 15.0 | 0.33 | 0.12 | 0.7 | – | 2.6e-7 |
| matmul_tiled<16> 256 | CUDA→WGSL optimized | 162.887 | 158.16 | 156.822 | 165.4 | 3.29 | 159.77 | 0.2 | – | 2.6e-7 |
| matmul_tiled<16> 256 | CUDA→WGSL direct | 173.802 | 166.69 | 165.146 | 222.7 | 3.17 | 243.53 | 0.2 | – | 2.6e-7 |
| matmul_tiled<16> 256 | CUDA→WASM 4 threads | 11.600 | 13.51 | – | 81.2 | 0.41 | 12.23 | 2.9 | – | 2.6e-7 |
| matmul_tiled<16> 256 | CUDA→WASM 1 thread | 31.550 | 31.56 | – | 5.9 | 0.38 | 0.08 | 1.1 | – | 2.6e-7 |
| matmul_tiled<16> 256 | handwritten WGSL | 132.173 | 128.10 | – | 108.9 | 1.12 | 2.74 | 0.3 | – | 2.6e-7 |
| matmul_regtile<64,64,8,4,4> 256 | CUDA→WGSL optimized | 41.430 | 43.46 | 42.392 | 423.9 | 2.98 | 43.58 | 0.8 | – | 2.6e-7 |
| matmul_regtile<64,64,8,4,4> 256 | CUDA→WGSL direct | 157.116 | 162.10 | 160.176 | 262.2 | 1.30 | 180.40 | 0.2 | – | 2.6e-7 |
| matmul_regtile<64,64,8,4,4> 256 | CUDA→WASM 4 threads | 5.200 | 5.84 | – | 49.2 | 0.35 | 5.21 | 6.5 | – | 2.6e-7 |
| matmul_regtile<64,64,8,4,4> 256 | CUDA→WASM 1 thread | 12.542 | 12.39 | – | 18.1 | 0.35 | 0.08 | 2.7 | – | 2.6e-7 |
| matmul_naive 512 | CUDA→WGSL optimized | 505.344 | 497.87 | 496.087 | 0.0 | 6.40 | 466.12 | 0.5 | – | 3.2e-7 |
| matmul_naive 512 | CUDA→WGSL direct | 656.996 | 663.24 | 661.882 | 0.0 | 4.71 | 686.44 | 0.4 | – | 3.2e-7 |
| matmul_naive 512 | CUDA→WASM 4 threads | 128.837 | 122.21 | – | 10.6 | 2.38 | 149.64 | 2.1 | – | 3.2e-7 |
| matmul_tiled<16> 512 | CUDA→WGSL optimized | 1103.392 | 1068.96 | 1067.718 | 36.6 | 4.40 | 1081.50 | 0.2 | – | 3.2e-7 |
| matmul_tiled<16> 512 | CUDA→WGSL direct | 1220.266 | 1281.54 | 1280.178 | 0.0 | 4.22 | 1200.54 | 0.2 | – | 3.2e-7 |
| matmul_tiled<16> 512 | CUDA→WASM 4 threads | 88.400 | 83.96 | – | 24.3 | 2.06 | 86.16 | 3.0 | – | 3.2e-7 |
| matmul_tiled<16> 512 | handwritten WGSL | 822.378 | 838.80 | – | 0.5 | 2.50 | 3.61 | 0.3 | – | 3.2e-7 |
| matmul_regtile<64,64,8,4,4> 512 | CUDA→WGSL optimized | 280.933 | 278.46 | 277.053 | 22.8 | 2.58 | 288.68 | 1.0 | – | 3.2e-7 |
| matmul_regtile<64,64,8,4,4> 512 | CUDA→WGSL direct | 1049.952 | 1080.76 | 1079.067 | 0.0 | 4.86 | 1006.79 | 0.3 | – | 3.2e-7 |
| matmul_regtile<64,64,8,4,4> 512 | CUDA→WASM 4 threads | 39.003 | 37.71 | – | 20.9 | 2.65 | 36.68 | 6.9 | – | 3.2e-7 |
| reduce_sum<256> n=4194304 | CUDA→WGSL optimized | 69.384 | 67.66 | 66.009 | 68.4 | 19.84 | 64.14 | – | 0.2 | 5.6e-6 |
| reduce_sum<256> n=4194304 | CUDA→WGSL direct | 80.619 | 78.20 | 75.639 | 83.0 | 18.83 | 99.67 | – | 0.2 | 5.0e-6 |
| reduce_sum<256> n=4194304 | CUDA→WASM 4 threads | 1.970 | 1.95 | – | 14.7 | 18.77 | 2.97 | – | 8.5 | 6.7e-6 |
| reduce_sum<256> n=4194304 | CUDA→WASM 1 thread | 5.065 | 5.29 | – | 4.0 | 12.56 | 5.93 | – | 3.3 | 5.9e-6 |
| reduce_sum<256> n=4194304 | handwritten WGSL | 57.977 | 61.02 | – | 99.3 | 12.94 | 0.44 | – | 0.3 | 5.0e-6 |
| softmax_rows<128> 1024x1024 | CUDA→WGSL optimized | 151.427 | 142.38 | 140.571 | 116.1 | 5.97 | 138.04 | – | 0.1 | 2.5e-7 |
| softmax_rows<128> 1024x1024 | CUDA→WGSL direct | 171.883 | 170.25 | 168.954 | 77.0 | 7.55 | 179.26 | – | 0.0 | 2.5e-7 |
| softmax_rows<128> 1024x1024 | CUDA→WASM 4 threads | 16.857 | 16.55 | – | 94.0 | 4.66 | 20.42 | – | 0.5 | 2.0e-7 |
| softmax_rows<128> 1024x1024 | CUDA→WASM 1 thread | 50.210 | 44.32 | – | 9.2 | 5.84 | 2.12 | – | 0.2 | 2.0e-7 |
| layernorm_fwd<128> 1024x1024 | CUDA→WGSL optimized | 128.057 | 116.40 | 115.026 | 116.9 | 4.52 | 148.37 | – | 0.1 | 1.8e-7 |
| layernorm_fwd<128> 1024x1024 | CUDA→WGSL direct | 118.155 | 111.47 | 110.365 | 129.6 | 7.41 | 141.95 | – | 0.1 | 1.8e-7 |
| layernorm_fwd<128> 1024x1024 | CUDA→WASM 4 threads | 2.832 | 2.80 | – | 33.4 | 6.13 | 5.04 | – | 3.0 | 1.8e-7 |
| layernorm_fwd<128> 1024x1024 | CUDA→WASM 1 thread | 7.070 | 6.93 | – | 10.6 | 5.16 | 1.90 | – | 1.2 | 1.8e-7 |
| sdpa_naive causal H=8 T=512 D=16 | CUDA→WGSL optimized | 281.967 | 282.43 | 281.123 | 48.9 | 3.81 | 309.15 | 0.2 | – | 1.5e-7 |
| sdpa_naive causal H=8 T=512 D=16 | CUDA→WGSL direct | 293.152 | 296.14 | 294.208 | 43.3 | 3.34 | 310.44 | 0.2 | – | 1.5e-7 |
| sdpa_naive causal H=8 T=512 D=16 | CUDA→WASM 4 threads | 59.115 | 58.70 | – | 30.0 | 0.84 | 59.24 | 1.1 | – | 1.7e-7 |
| flash_attention_fwd<32,16,16> causal H=8 T=512 D=16 | CUDA→WGSL optimized | 54.001 | 58.57 | 57.238 | 237.5 | 3.26 | 84.27 | 1.2 | – | 1.2e-7 |
| flash_attention_fwd<32,16,16> causal H=8 T=512 D=16 | CUDA→WGSL direct | 155.292 | 158.48 | 157.321 | 181.5 | 3.65 | 144.26 | 0.4 | – | 1.2e-7 |
| flash_attention_fwd<32,16,16> causal H=8 T=512 D=16 | CUDA→WASM 4 threads | 27.688 | 27.00 | – | 145.4 | 0.55 | 28.29 | 2.4 | – | 1.3e-7 |

## CUDA kernels on tinygrad tensors vs tinygrad operations (tinygrad 0.14.0)

Per call: realize + synchronize, median; both produce the same result (checked).

| device | workload | tinygrad.cuda kernel | tinygrad op | ratio (op / kernel) | max rel err |
|---|---|---|---|---|---|
| WEBGPU | vector add n=1048576 | 14.908 | 10.260 | 0.69× | 0.0e+0 |
| WEBGPU | matmul 256 (matmul_tiled<16>) | 196.522 | 16.622 | 0.08× | 0.0e+0 |
| WEBGPU | matmul 256 (matmul_regtile) | 54.253 | 17.855 | 0.33× | 0.0e+0 |
| WEBGPU | matmul 512 (matmul_tiled<16>) | 1118.267 | 109.948 | 0.10× | 0.0e+0 |
| WEBGPU | matmul 512 (matmul_regtile) | 304.980 | 86.815 | 0.28× | 0.0e+0 |
| WEBGPU | softmax 1024x1024 | 152.567 | 26.725 | 0.18× | 3.0e-7 |
| PYTHON | vector add n=16384 | 4.235 | 288.965 | 68.23× | 0.0e+0 |
| PYTHON | matmul 64 (matmul_tiled<16>) | 4.195 | 1187.065 | 282.97× | 4.4e-7 |
| PYTHON | softmax 16x256 | 4.680 | 317.005 | 67.74× | 1.6e-7 |

A ratio above 1× means the CUDA kernel was faster on this machine; it says nothing about other hardware. On a software WebGPU adapter (SwiftShader) GPU numbers measure a CPU emulation of a GPU.
