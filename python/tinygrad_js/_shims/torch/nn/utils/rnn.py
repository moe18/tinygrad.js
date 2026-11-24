"""torch.nn.utils.rnn: pad_sequence. Packed sequences are not supported."""
def pad_sequence(sequences, batch_first=False, padding_value=0.0, padding_side="right"):
  from ... import stack
  from .. import functional as F
  n = max(s.shape[0] for s in sequences)
  def pad(s):
    extra = n - s.shape[0]
    spec = [0, 0] * (s.ndim - 1) + ([0, extra] if padding_side == "right" else [extra, 0])
    return F.pad(s, spec, value=padding_value) if extra else s
  out = stack([pad(s) for s in sequences])
  return out if batch_first else out.transpose(0, 1)
def pack_padded_sequence(*a, **k): raise NotImplementedError("packed sequences are not supported by the tinygrad.js torch shim")
pad_packed_sequence = pack_sequence = pack_padded_sequence
