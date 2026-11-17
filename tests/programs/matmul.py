results = {}
for n in (1, 7, 64, 128, 130):
  a = (Tensor.arange(n * n).reshape(n, n).float() % 13 - 6) / 7
  b = (Tensor.arange(n * n).reshape(n, n).float() % 11 - 5) / 3
  results[f"mm_{n}"] = (a @ b)
Tensor.manual_seed(1337)
x, y = Tensor.randn(128, 128), Tensor.randn(128, 128)
results["randn_x"] = x
# matmul correctness on random inputs, independent of RNG precision: compare against numpy on the same inputs
results["randn_mm_maxerr_vs_numpy"] = float(np.abs((x @ y).numpy() - x.numpy().astype(np.float64) @ y.numpy().astype(np.float64)).max())
results["batched"] = Tensor.arange(2*3*4).reshape(2, 3, 4).float() @ Tensor.arange(2*4*5).reshape(2, 4, 5).float()
results["matvec"] = Tensor.arange(12).reshape(3, 4).float() @ Tensor([1.0, -2.0, 3.0, 0.5])
results["spec_shape"] = list((Tensor.randn(128, 128) @ Tensor.randn(128, 128)).shape)
emit(**results)
