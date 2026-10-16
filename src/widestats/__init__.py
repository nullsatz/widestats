"""GPU statistics for wide data, on PyTorch.

For data with tens of thousands of variables, where the matrix of every
pair is the problem:

- `cor_pairs`: the strongly correlated pairs (every pair over a threshold,
  or each variable's top k), without forming the correlation matrix.
- `cor_perm`: permutation p-values for every correlation, and p-values
  adjusted for testing them all (Westfall and Young's maxT).
- `ttest_perm`: a t test of every feature between two groups, with
  permutation and maxT p-values.
- `granger`: Granger causality F tests for every pair of time series.
- `mutual_info`: the B-spline mutual information (Daub et al. 2004) of
  every pair of variables, which detects any kind of dependence.

Each runs on NVIDIA and AMD GPUs (``"cuda"``), Apple GPUs (``"mps"``,
float32 only) or the CPU, choosing the first available unless given
``device=``; float32 by default, float64 with ``dtype="float64"``. Inputs
are matrices with the samples in rows (NumPy arrays, anything
`numpy.asarray` accepts, or torch tensors); results are NumPy arrays.

These are the functions of the R package gputools
(https://github.com/nullsatz/gputools) that Python lacks on a GPU, and are
checked against NumPy and SciPy the same way.
"""

from ._core import pick_device
from .granger import Granger, granger
from .mi import mutual_info
from .pairs import Pairs, cor_pairs
from .perm import CorPerm, TTestPerm, cor_perm, ttest_perm

__all__ = [
    "CorPerm",
    "Granger",
    "Pairs",
    "TTestPerm",
    "cor_pairs",
    "cor_perm",
    "granger",
    "mutual_info",
    "pick_device",
    "ttest_perm",
]

__version__ = "0.1.0.dev0"
