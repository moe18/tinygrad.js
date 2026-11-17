import numpy as np
out = {}
for n in (1, 3, 5, 1023, 4096, 1 << 18):
  arr = (np.arange(n, dtype=np.float32) * 0.5 - 3)
  t = Tensor(arr)
  back = t.numpy()
  out[f"f32_{n}"] = bool(np.array_equal(back, arr))
for dt in ("int32", "uint8", "int8", "int16", "bool"):
  arr = (np.arange(37) % 7 - 3).astype(dt) if dt != "bool" else (np.arange(37) % 3 == 0)
  out[f"rt_{dt}"] = bool(np.array_equal(Tensor(arr).numpy(), arr))
  out[f"op_{dt}"] = (Tensor(arr) + Tensor(arr)).numpy().tolist() if dt != "bool" else (Tensor(arr) | Tensor(arr[::-1].copy())).numpy().tolist()
big = Tensor.ones(512, 512).contiguous()
out["big_sum"] = big.sum().item()
x = Tensor([1.0, 2.0, 3.0]).realize()
x.assign(x * 10).realize()
out["assign"] = x.numpy().tolist()
out["item"] = Tensor([3.5]).item()
out["tolist"] = Tensor([[1, 2], [3, 4]]).tolist()
emit(**out)
