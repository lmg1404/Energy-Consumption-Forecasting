"""
Utils to help out with repeated loads in the project lifecycle
"""

import numpy as np

# essentially https://goldinlocks.github.io/Time-Series-Cross-Validation/
class BlockingTimeSeriesSplit:
  """
  Class that is a custom time series split not supported by existing libraries
  """
  def __init__(self, n_splits: int = 5, val_size: float = 0.2):
    self.n_splits = n_splits
    self.val_size = val_size

  def split(self, X, y=None) -> np.array:
    block_size = len(X) // self.n_splits
    indices = np.arange(len(X))
    for i in range(self.n_splits):
      start = i * block_size
      stop = start + block_size
      split = stop - int(self.val_size * (stop-start))
      yield indices[start:split], indices[split:stop]