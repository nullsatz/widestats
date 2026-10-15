"""GPU statistics for wide data, on PyTorch: the strongly correlated pairs
among tens of thousands of variables without the correlation matrix, and
permutation tests of every correlation or every feature with Westfall and
Young's maxT adjustment. Runs on NVIDIA and AMD GPUs (cuda), Apple GPUs
(mps, float32 only) and the CPU."""

from ._core import pick_device
from .pairs import Pairs, cor_pairs
from .perm import CorPerm, TTestPerm, cor_perm, ttest_perm

__all__ = [
           "CorPerm",
           "Pairs",
           "TTestPerm",
           "cor_pairs",
           "cor_perm",
           "pick_device",
           "ttest_perm",
]
__version__ = "0.1.0.dev0"
