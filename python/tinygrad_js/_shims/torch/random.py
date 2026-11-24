from ._tensor import manual_seed, seed, default_generator
def get_rng_state(): raise NotImplementedError("torch.random.get_rng_state is not supported by the tinygrad.js torch shim")
def initial_seed(): return default_generator.initial_seed()
