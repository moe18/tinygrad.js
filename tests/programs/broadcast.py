col = Tensor([[1.0], [2.0], [3.0]])
row = Tensor([[10.0, 20.0, 30.0, 40.0]])
m = Tensor.arange(24).reshape(2, 3, 4).float()
emit(outer_add=col + row, outer_mul=col * row, scalar=m * 2.5 - 1, rowwise=m - m.mean(axis=2, keepdim=True),
     three_d=m + Tensor([1.0, 2.0, 3.0, 4.0]), expand=Tensor([1.0, 2.0]).reshape(2, 1, 1).expand(2, 3, 4) * m,
     softmax=m.softmax(axis=-1), norm=(m / m.square().sum(axis=-1, keepdim=True).sqrt()))
