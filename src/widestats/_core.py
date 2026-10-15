"""Shared machinery: device, precision, inputs, and column preparation.

Every function in widestats chooses the device and the precision, turns
its inputs into matrices, and prepares columns the same way.

Columns are prepared on the CPU in float64 (centring, scaling and ranking
lose precision if done in float32) and only then moved to the device in
the precision it computes in.
"""

from __future__ import annotations

import numpy as np
import torch


def pick_device(device=None) -> torch.device:
    """Choose the device to compute on.

    Parameters
    ----------
    device : str or torch.device, optional
        A device to use as given, such as ``"cuda"``, ``"cuda:1"``,
        ``"mps"`` or ``"cpu"``.

    Returns
    -------
    torch.device
        `device`, or if it is None the first available of CUDA (NVIDIA
        GPUs, and AMD GPUs under ROCm builds of PyTorch), MPS (Apple GPUs)
        and the CPU.

    Examples
    --------
    >>> import widestats as ws
    >>> ws.pick_device("cpu")
    device(type='cpu')
    """
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(dtype, device: torch.device) -> torch.dtype:
    """Choose the precision to compute in.

    Parameters
    ----------
    dtype : {None, "float32", "single", "float64", "double"} or torch.dtype
        None means float32, which most GPUs run many times faster than
        float64 and which loses nothing that matters for most statistics.
    device : torch.device
        Where the computation runs.

    Returns
    -------
    torch.dtype
        ``torch.float32`` or ``torch.float64``.

    Raises
    ------
    ValueError
        For any other precision, or float64 on an Apple GPU (``mps``),
        which has none.
    """
    dt = {None: torch.float32, "float32": torch.float32, "single": torch.float32,
          "float64": torch.float64, "double": torch.float64}.get(dtype, dtype)
    if dt not in (torch.float32, torch.float64):
        raise ValueError("dtype must be float32 or float64")
    if dt == torch.float64 and device.type == "mps":
        raise ValueError("Apple GPUs (mps) have no float64; use float32 or "
                         "device='cpu'")
    return dt


def as_matrix(x, name: str = "x") -> np.ndarray:
    """Convert an input to a samples x variables matrix.

    Parameters
    ----------
    x : array_like or torch.Tensor
        A matrix, or a vector (taken as a single variable).
    name : str
        What to call `x` in an error message.

    Returns
    -------
    ndarray, shape (n_samples, n_variables)
        `x` in float64.

    Raises
    ------
    ValueError
        If `x` has more than two dimensions or any missing (NaN) value.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"{name} must be a matrix (samples x variables)")
    if np.isnan(x).any():
        raise ValueError(f"{name} must have no missing values")
    return x


def standardise(x: np.ndarray) -> np.ndarray:
    """Centre each column and scale it to unit length.

    Then the Pearson correlation of two columns is their dot product, and
    a correlation matrix is one matrix product.

    Parameters
    ----------
    x : ndarray, shape (n, p)

    Returns
    -------
    ndarray, shape (n, p)
        ``(x - mean) / |x - mean|`` column by column, in float64. A
        constant column becomes NaN, so its correlations are NaN, as
        `numpy.corrcoef` gives them.

    Examples
    --------
    >>> import numpy as np
    >>> z = standardise(np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 7.0]]))
    >>> np.allclose((z * z).sum(axis=0), 1)
    True
    """
    z = x - x.mean(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = z / np.sqrt((z * z).sum(axis=0))
    return z


def rank_columns(x: np.ndarray) -> np.ndarray:
    """Rank the values in each column, ties given their average rank.

    The same ranks as `scipy.stats.rankdata` (and R's ``rank``) give;
    Spearman's coefficient is Pearson's on them.

    Parameters
    ----------
    x : ndarray, shape (n, p)
        No missing values.

    Returns
    -------
    ndarray, shape (n, p)
        Ranks from 1 to n, in float64.

    Notes
    -----
    After sorting a column, the copies of a value occupy positions
    ``first .. last - 1``, found for every value at once by binary search
    (`torch.searchsorted`), and so get ranks ``first + 1 .. last``, whose
    average is ``(first + 1 + last) / 2``.

    Examples
    --------
    >>> import numpy as np
    >>> rank_columns(np.array([[10.0], [30.0], [20.0], [30.0]])).ravel()
    array([1. , 3.5, 2. , 3.5])
    """
    t = torch.from_numpy(np.ascontiguousarray(x.T))     # variables x samples
    s, _ = torch.sort(t, dim=1)
    first = torch.searchsorted(s, t, side="left")
    last = torch.searchsorted(s, t, side="right")
    return ((first + 1 + last).to(torch.float64) / 2).numpy().T


def to_device(z: np.ndarray, device: torch.device, dtype: torch.dtype):
    """Copy a NumPy array to the device, in the precision given."""
    return torch.from_numpy(np.ascontiguousarray(z)).to(device=device,
                                                       dtype=dtype)


def rounding_allowance(dtype: torch.dtype) -> float:
    """How far apart, relatively, two statistics may be by rounding alone.

    A permuted statistic is "at least as extreme" as the observed one when
    ``|permuted| >= |observed| * (1 - allowance)``. Without the allowance,
    a permutation that leaves the statistic mathematically unchanged (the
    identity, or a reordering within groups) could fall a rounding error
    short and not count, making a p-value too small.

    Parameters
    ----------
    dtype : torch.dtype
        The precision computed in.

    Returns
    -------
    float
        1e-10 in float64, 1e-5 in float32.
    """
    return 1e-10 if dtype == torch.float64 else 1e-5
