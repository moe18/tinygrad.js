"""torch.utils.data: datasets, samplers and a single-process DataLoader."""
from __future__ import annotations
import math
import numpy as np
from ... import _tensor
from ..._tensor import Tensor, _mk, _coerce

class Dataset:
  def __getitem__(self, index): raise NotImplementedError
  def __add__(self, other): return ConcatDataset([self, other])
class IterableDataset(Dataset):
  def __iter__(self): raise NotImplementedError

class TensorDataset(Dataset):
  def __init__(self, *tensors):
    if any(t.shape[0] != tensors[0].shape[0] for t in tensors): raise ValueError("Size mismatch between tensors")
    self.tensors = tensors
    self._host = None
  def __getitem__(self, index): return tuple(t[index] for t in self.tensors)
  def __len__(self): return self.tensors[0].shape[0]
  def _batch(self, indices):
    # batches are gathered on the host and uploaded: indexing a GPU tensor with an index tensor costs O(dataset) per batch in tinygrad
    if self._host is None: self._host = [t.detach().numpy() for t in self.tensors]
    return tuple(_mk(_coerce(np.ascontiguousarray(a[indices]))) for a in self._host)

class Subset(Dataset):
  def __init__(self, dataset, indices): self.dataset, self.indices = dataset, list(indices)
  def __getitem__(self, idx): return self.dataset[self.indices[idx]]
  def __len__(self): return len(self.indices)
  def _batch(self, indices):
    if not hasattr(self.dataset, "_batch"): return None
    return self.dataset._batch([self.indices[i] for i in indices])
class ConcatDataset(Dataset):
  def __init__(self, datasets): self.datasets = list(datasets)
  def __len__(self): return sum(len(d) for d in self.datasets)
  def __getitem__(self, idx):
    if idx < 0: idx += len(self)
    for d in self.datasets:
      if idx < len(d): return d[idx]
      idx -= len(d)
    raise IndexError(idx)

def random_split(dataset, lengths, generator=None):
  n = len(dataset)
  if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
    counts = [int(math.floor(n * f)) for f in lengths]
    for i in range(n - sum(counts)): counts[i % len(counts)] += 1
    lengths = counts
  if sum(lengths) != n: raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
  perm = _tensor._host_rng.permutation(n).tolist()
  out, start = [], 0
  for l in lengths: out.append(Subset(dataset, perm[start:start + l])); start += l
  return out

class Sampler:
  def __init__(self, data_source=None): self.data_source = data_source
class SequentialSampler(Sampler):
  def __iter__(self): return iter(range(len(self.data_source)))
  def __len__(self): return len(self.data_source)
class RandomSampler(Sampler):
  def __init__(self, data_source, replacement=False, num_samples=None, generator=None):
    self.data_source, self.replacement, self._n = data_source, replacement, num_samples
  def __iter__(self):
    n = len(self.data_source)
    if self.replacement: return iter(_tensor._host_rng.integers(0, n, size=len(self)).tolist())
    return iter(_tensor._host_rng.permutation(n).tolist()[:len(self)])
  def __len__(self): return self._n if self._n is not None else len(self.data_source)
class SubsetRandomSampler(Sampler):
  def __init__(self, indices, generator=None): self.indices = list(indices)
  def __iter__(self): return iter([self.indices[i] for i in _tensor._host_rng.permutation(len(self.indices)).tolist()])
  def __len__(self): return len(self.indices)
class WeightedRandomSampler(Sampler):
  def __init__(self, weights, num_samples, replacement=True, generator=None):
    w = np.asarray(weights.tolist() if isinstance(weights, Tensor) else weights, dtype=np.float64)
    self.p, self.num_samples, self.replacement = w / w.sum(), num_samples, replacement
  def __iter__(self): return iter(_tensor._host_rng.choice(len(self.p), size=self.num_samples, replace=self.replacement, p=self.p).tolist())
  def __len__(self): return self.num_samples
class BatchSampler(Sampler):
  def __init__(self, sampler, batch_size, drop_last): self.sampler, self.batch_size, self.drop_last = sampler, batch_size, drop_last
  def __iter__(self):
    batch = []
    for i in self.sampler:
      batch.append(i)
      if len(batch) == self.batch_size: yield batch; batch = []
    if batch and not self.drop_last: yield batch
  def __len__(self):
    n = len(self.sampler)
    return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size

def default_collate(batch):
  from ... import stack, tensor
  e = batch[0]
  if isinstance(e, Tensor): return stack(batch, 0)
  if isinstance(e, np.ndarray): return _mk(_coerce(np.stack(batch)))
  if isinstance(e, (np.floating, np.integer, np.bool_)): return _mk(_coerce(np.array(batch)))
  if isinstance(e, float): return tensor(batch, dtype=_tensor.float64)
  if isinstance(e, int): return tensor(batch)
  if isinstance(e, (str, bytes)): return list(batch)
  if isinstance(e, dict): return {k: default_collate([b[k] for b in batch]) for k in e}
  if isinstance(e, tuple) and hasattr(e, "_fields"): return type(e)(*(default_collate(list(s)) for s in zip(*batch)))
  if isinstance(e, (tuple, list)): return [default_collate(list(s)) for s in zip(*batch)]
  raise TypeError(f"default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found {type(e)}")

class DataLoader:
  """Iterates a dataset in batches. Always single-process: ``num_workers``, ``pin_memory`` etc. are accepted and ignored."""
  def __init__(self, dataset, batch_size=1, shuffle=None, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False,
               drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None, generator=None, *, prefetch_factor=None,
               persistent_workers=False, pin_memory_device="", in_order=True):
    self.dataset, self.batch_size, self.drop_last, self.collate_fn, self.num_workers = dataset, batch_size, drop_last, collate_fn, num_workers
    self._iterable = isinstance(dataset, IterableDataset)
    if batch_sampler is not None: self.batch_sampler, self.sampler = batch_sampler, None
    elif self._iterable: self.batch_sampler = self.sampler = None
    else:
      self.sampler = sampler if sampler is not None else (RandomSampler(dataset) if shuffle else SequentialSampler(dataset))
      self.batch_sampler = BatchSampler(self.sampler, batch_size, drop_last) if batch_size is not None else None
  def __len__(self):
    if self._iterable: raise TypeError("DataLoader over an IterableDataset has no len()")
    return len(self.batch_sampler) if self.batch_sampler is not None else len(self.sampler)
  def __iter__(self):
    collate = self.collate_fn or default_collate
    if self._iterable:
      batch = []
      for item in self.dataset:
        if self.batch_size is None: yield item; continue
        batch.append(item)
        if len(batch) == self.batch_size: yield collate(batch); batch = []
      if batch and not self.drop_last: yield collate(batch)
      return
    if self.batch_sampler is None:
      for i in self.sampler: yield self.dataset[i]
      return
    fast = self.collate_fn is None and hasattr(self.dataset, "_batch")
    for indices in self.batch_sampler:
      out = self.dataset._batch(indices) if fast else None
      yield list(out) if out is not None else collate([self.dataset[i] for i in indices])

def get_worker_info(): return None
