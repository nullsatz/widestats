"""Statistics for wide data on a GPU, built on PyTorch.

Wide data has far more variables than samples: 20,000 genes in 100
patients, 1,000 time series at 1,000 time points. Many questions about such
data are about every pair of variables, and the number of pairs grows with
the square of the number of variables. widestats answers those questions
with a few large matrix products on the GPU, without building the full
matrix of pairs in memory.

- `cor_pairs`: the strongly correlated pairs, every pair over a threshold
  or each variable's top k, without forming the correlation matrix.
- `cor_perm`: permutation p-values for every correlation, and p-values
  adjusted for testing them all (Westfall and Young's maxT).
- `ttest_perm`: a t test of every feature between two groups, with
  permutation and maxT p-values.
- `granger`: Granger causality F tests for every ordered pair of time
  series.
- `mutual_info`: the B-spline mutual information (Daub et al. 2004) of
  every pair of variables, which also finds nonlinear dependence.

Each runs on NVIDIA and AMD GPUs (``"cuda"``), Apple GPUs (``"mps"``,
float32 only) or the CPU, using the first it finds unless given
``device=``; float32 by default, float64 with ``dtype="float64"``
(`granger` always computes in float64). Inputs are matrices with the
samples in rows: NumPy arrays, anything `numpy.asarray` accepts, or torch
tensors. Results are NumPy arrays, or small result objects holding several.

These are the functions of the R package gputools
(https://github.com/nullsatz/gputools) that Python did not yet have on a
GPU. Each is tested against NumPy, SciPy, statsmodels or the published
method. The README (https://github.com/nullsatz/widestats) has examples
and benchmarks.
"""

from importlib.metadata import version as _version

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

__version__ = _version("widestats")   # the one in pyproject.toml
