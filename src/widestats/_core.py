"""Device and dtype choice, input conversion, and the column preparation
every correlation here starts from."""

from __future__ import annotations

import numpy as np
import torch


def pick_device(device=None) -> torch.device:
    """The device to compute on: as given, or the first of CUDA (NVIDIA or
    AMD), MPS (Apple) and the CPU that is available."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(dtype, device: torch.device) -> torch.dtype:
    """float32 by default, as most GPUs run it many times faster; float64
    on request, except on Apple GPUs, which have none."""
    dt = {None: torch.float32, "float32": torch.float32, "single": torch.float32,
          "float64": torch.float64, "double": torch.float64}.get(dtype, dtype)
    if dt not in (torch.float32, torch.float64):
        raise ValueError("dtype must be float32 or float64")
    if dt == torch.float64 and device.type == "mps":
        raise ValueError("Apple GPUs (mps) have no float64; use float32 or "
                         "device='cpu'")
    return dt


def as_matrix(x, name: str = "x") -> np.ndarray:
    """x as a 2-d float64 NumPy array, samples in rows; no missing values."""
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
    """Each column centred and scaled to unit length, in float64, so that
    the correlation of two columns is their dot product. A constant column
    becomes NaN and correlates with nothing."""
    z = x - x.mean(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = z / np.sqrt((z * z).sum(axis=0))
    return z


def rank_columns(x: np.ndarray) -> np.ndarray:
    """Each column ranked, ties given the average of the ranks they span,
    as scipy.stats.rankdata (and R's rank) do. Spearman's coefficient is
    Pearson's on these ranks."""
    t = torch.from_numpy(np.ascontiguousarray(x.T))     # variables x samples
    s, _ = torch.sort(t, dim=1)
    # tied values occupy positions first .. last - 1 of the sorted column,
    # ranks first + 1 .. last: their average
    first = torch.searchsorted(s, t, side="left")
    last = torch.searchsorted(s, t, side="right")
    return ((first + 1 + last).to(torch.float64) / 2).numpy().T


def to_device(z: np.ndarray, device: torch.device, dtype: torch.dtype):
    return torch.from_numpy(np.ascontiguousarray(z)).to(device=device,
                                                       dtype=dtype)


def rounding_allowance(dtype: torch.dtype) -> float:
    """How much a statistic may differ from an exactly equal one by rounding
    alone: a permuted statistic within this (relatively) of the observed one
    counts as at least as extreme."""
    return 1e-10 if dtype == torch.float64 else 1e-5
