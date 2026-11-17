a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])
f = Tensor([[0.5, -1.25, 3.0], [2.0, 0.1, -7.5]])
g = Tensor([[1.5, 2.0, -0.5], [4.0, 0.3, 2.5]])
emit(add=a + b, mul=a * b, sub=a - b, int_div=b // a, fadd=f + g, fmul=f * g, fdiv=f / g,
     exp=f.exp(), log=g.abs().log(), sqrt=g.abs().sqrt(), relu=f.relu(), sigmoid=f.sigmoid(), tanh=f.tanh(),
     neg=-f, pow=g.abs() ** 1.5, maximum=f.maximum(g), where=(f > g).where(f, g), cmp=(f < g),
     sum=f.sum(), mean=f.mean(axis=1), max=f.max(axis=0), argmax=f.argmax(axis=1), cumsum=g.cumsum(axis=1))
