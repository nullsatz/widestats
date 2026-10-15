"""Permutation tests with Westfall and Young's single-step maxT adjustment:
every test, under every shuffle of the samples, at once. A thousand
permutations cost a thousand times the test, which is the work a GPU is for.

p-values are (1 + b) / (1 + B), b of the B permutations reaching the
observed statistic (Phipson and Smyth 2010); the adjusted p-value counts the
permutations whose largest statistic over all the tests reaches it, which
controls the family-wise error rate (Westfall and Young 1993). The
permutations are drawn from `seed` on the CPU, so a seed gives the same ones
on any device."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import stats

from ._core import (as_matrix, pick_device, pick_dtype, rank_columns,
                    rounding_allowance, standardise, to_device)


def _permutations(n: int, B: int, seed) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed if seed is not None
                                      else int(torch.seed() % 2**63))
    return torch.argsort(torch.rand(B, n, generator=g), dim=1)


def _adjust(observed: np.ndarray, maxima: np.ndarray, eps: float, B: int):
    """maxT: for each |observed| statistic, (1 + the permutations whose
    maximum reaches it) / (1 + B)."""
    m = np.sort(maxima)
    reach = B - np.searchsorted(m, np.abs(observed) * (1 - eps), side="left")
    adj = (reach + 1) / (B + 1)
    return np.where(np.isnan(observed), np.nan, adj)


@dataclass
class CorPerm:
    """Correlations (x columns by y columns) with their permutation and maxT
    adjusted p-values; with y omitted the diagonal is NaN."""
    r: np.ndarray
    p_value: np.ndarray
    p_adjusted: np.ndarray
    permutations: int


def cor_perm(x, y=None, *, B=1000, method="pearson", seed=None,
             batch=None, device=None, dtype=None) -> CorPerm:
    """Permutation test of every correlation between the columns of x and
    those of y (or among the columns of x): the rows of y are shuffled B
    times. Each permutation's correlations are one matrix product of the
    standardised columns, which shuffling rows leaves standardised; `batch`
    permutations go at once (by default, as many as fit in ~256 MB)."""
    if int(B) != B or B < 1:
        raise ValueError("B must be a positive whole number")
    if method not in ("pearson", "spearman"):
        raise ValueError('method must be "pearson" or "spearman"')
    x = as_matrix(x, "x")
    self_ = y is None
    y = x if self_ else as_matrix(y, "y")
    n = x.shape[0]
    if y.shape[0] != n:
        raise ValueError("x and y have different numbers of rows")
    if n < 3:
        raise ValueError("need at least 3 rows")
    dev = pick_device(device)
    dt = pick_dtype(dtype, dev)
    eps = rounding_allowance(dt)
    if method == "spearman":
        x = rank_columns(x)
        y = x if self_ else rank_columns(y)
    zx = to_device(standardise(x), dev, dt)
    zy = to_device(standardise(y), dev, dt)
    p, q = zx.shape[1], zy.shape[1]

    r0 = zx.T @ zy
    test = ~torch.isnan(r0)
    if self_:
        test &= ~torch.eye(p, dtype=torch.bool, device=dev)
    r0 = torch.where(test, r0, torch.nan)
    bar = r0.abs() * (1 - eps)
    counts = torch.zeros((p, q), dtype=torch.int64, device=dev)
    maxima = torch.empty(B, dtype=torch.float64)

    perms = _permutations(n, B, seed).to(dev)
    if batch is None:
        batch = max(1, (1 << 26) // max(1, p * q + n * q))
    for b0 in range(0, B, batch):
        pb = perms[b0:b0 + batch]                       # b x n
        rb = zx.T @ zy[pb]                              # b x p x q
        rb = torch.where(test, rb.abs(), 0)             # no-tests count as 0
        counts += (rb >= bar).sum(dim=0)
        maxima[b0:b0 + len(pb)] = rb.amax(dim=(1, 2)).cpu().to(torch.float64)

    r = r0.cpu().to(torch.float64).numpy()
    p_value = np.where(np.isnan(r), np.nan,
                       (counts.cpu().numpy() + 1) / (B + 1))
    return CorPerm(np.clip(r, -1, 1), p_value,
                   _adjust(r, maxima.numpy(), eps, B), int(B))


@dataclass
class TTestPerm:
    """Two-group t tests of each column (feature): the group means, t,
    degrees of freedom and two-sided p-value as scipy.stats.ttest_ind gives
    them, and with B > 0 the permutation and maxT adjusted p-values."""
    mean1: np.ndarray
    mean2: np.ndarray
    statistic: np.ndarray
    df: np.ndarray
    p_value: np.ndarray
    p_perm: np.ndarray | None = None
    p_adjusted: np.ndarray | None = None


def ttest_perm(x, group, *, equal_var=False, B=0, seed=None, batch=None,
               device=None, dtype=None) -> TTestPerm:
    """Welch's (or with equal_var, Student's) t test of every column of x
    (samples x features) between the two groups of samples `group` names,
    the first group (in sorted order) minus the second, as
    scipy.stats.ttest_ind(x[g1], x[g2], equal_var=...) gives them; computed
    in float64 on the CPU, being too little work for a GPU.

    With B > 0 the group labels are shuffled B times, and every feature's t
    under every labelling comes from two matrix products on the device:
    each labelling's group-1 sums and sums of squares, G'x and G'x^2, G the
    samples x labellings indicator matrix."""
    x = as_matrix(x, "x")
    n, p = x.shape
    group = np.asarray(group)
    if group.shape != (n,):
        raise ValueError("group must have one entry per row of x")
    levels = np.unique(group)
    if len(levels) != 2:
        raise ValueError("group must have exactly two levels")
    in1 = group == levels[0]
    n1 = int(in1.sum())
    n2 = n - n1
    if n1 < 2 or n2 < 2:
        raise ValueError("each group needs at least 2 samples")
    if int(B) != B or B < 0:
        raise ValueError("B must be a non-negative whole number")

    res = stats.ttest_ind(x[in1], x[~in1], equal_var=equal_var, axis=0)
    out = TTestPerm(x[in1].mean(axis=0), x[~in1].mean(axis=0),
                    res.statistic, np.asarray(res.df, dtype=float) *
                    np.ones(p), res.pvalue)
    if B == 0:
        return out

    dev = pick_device(device)
    dt = pick_dtype(dtype, dev)
    eps = rounding_allowance(dt)
    # centring and scaling each feature changes no t, and keeps the sums of
    # squares in float32 from cancelling
    sd = x.std(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (x - x.mean(axis=0)) / sd
    zt = to_device(z, dev, dt)
    z2 = zt * zt
    s, q = zt.sum(dim=0), z2.sum(dim=0)

    labels = torch.zeros((B + 1, n), dtype=dt)
    labels[0] = torch.from_numpy(in1)
    perms = _permutations(n, B, seed)
    labels[1:] = torch.from_numpy(in1)[perms].to(dt)
    labels = labels.to(dev)

    def tstat(g):                                    # g: labellings x n
        a1, b1 = g @ zt, g @ z2
        a2, b2 = s - a1, q - b1
        m1, m2 = a1 / n1, a2 / n2
        ss1 = (b1 - a1 * m1).clamp_min(0)
        ss2 = (b2 - a2 * m2).clamp_min(0)
        if equal_var:
            v = (ss1 + ss2) / (n - 2)
            se2 = v * (1 / n1 + 1 / n2)
        else:
            se2 = ss1 / (n1 - 1) / n1 + ss2 / (n2 - 1) / n2
        return torch.where(se2 > 0, (m1 - m2) / se2.sqrt(), torch.nan)

    t0 = tstat(labels[:1])[0]
    usable = ~torch.isnan(t0)
    bar = t0.abs() * (1 - eps)
    counts = torch.zeros(p, dtype=torch.int64, device=dev)
    maxima = torch.empty(B, dtype=torch.float64)
    if batch is None:
        batch = max(1, (1 << 25) // p)
    for b0 in range(1, B + 1, batch):
        tb = tstat(labels[b0:b0 + batch]).abs()
        tb = torch.where(usable & ~torch.isnan(tb), tb, 0)
        counts += (tb >= bar).sum(dim=0)
        maxima[b0 - 1:b0 - 1 + len(tb)] = tb.amax(dim=1).cpu().to(torch.float64)

    t0 = t0.cpu().to(torch.float64).numpy()
    out.p_perm = np.where(np.isnan(t0), np.nan,
                          (counts.cpu().numpy() + 1) / (B + 1))
    out.p_adjusted = _adjust(t0, maxima.numpy(), eps, B)
    return out
