"""Permutation tests of many statistics at once, with maxT adjustment.

Westfall and Young's single-step maxT adjusts the p-values for testing them
all.

A permutation test repeats the whole computation for every shuffle of the
samples, so a thousand permutations cost a thousand times the test: the
work a GPU is for. Here every test runs under every permutation at once, as
matrix products on the device.

Conventions shared by the functions here:

- A test's permutation p-value is ``(1 + b) / (1 + B)``, where ``b`` of the
  ``B`` permutations gave a statistic at least as extreme as the observed
  one (two-sided). Counting the observed data as one of the permutations
  keeps it from ever being 0 (Phipson and Smyth 2010).
- Its maxT adjusted p-value counts instead the permutations whose largest
  statistic over *all* the tests reaches it. That controls the family-wise
  error rate, and since the tests share their permutations it keeps the
  dependence between them (Westfall and Young 1993).
- A permuted statistic within rounding of the observed one counts as
  reaching it (see `widestats._core.rounding_allowance`).
- The permutations are drawn from `seed` on the CPU and then moved to the
  device, so a seed gives the same permutations, and the same p-values, on
  any device.

References
----------
Westfall, P. H. and Young, S. S. (1993). *Resampling-Based Multiple
Testing*. Wiley.

Phipson, B. and Smyth, G. K. (2010). Permutation p-values should never be
zero. *Statistical Applications in Genetics and Molecular Biology* 9, 39.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import stats

from ._core import (
    as_matrix,
    pick_device,
    pick_dtype,
    rank_columns,
    rounding_allowance,
    standardise,
    to_device,
)


def _permutations(n: int, B: int, seed) -> torch.Tensor:
    """Draw `B` random permutations of ``range(n)`` from `seed`.

    Returns a ``(B, n)`` tensor on the CPU; with no `seed`, fresh entropy.
    Sorting uniform draws gives each permutation with equal probability.
    """
    g = torch.Generator().manual_seed(seed if seed is not None
                                      else int(torch.seed() % 2**63))
    return torch.argsort(torch.rand(B, n, generator=g), dim=1)


def _adjust(observed: np.ndarray, maxima: np.ndarray, eps: float, B: int):
    """Westfall-Young single-step maxT adjusted p-values.

    Parameters
    ----------
    observed : ndarray
        The observed statistics, any shape; NaN marks a test not made.
    maxima : ndarray, shape (B,)
        Each permutation's largest ``|statistic|`` over all the tests.
    eps : float
        The relative rounding allowance.
    B : int
        The number of permutations.

    Returns
    -------
    ndarray
        For each observed statistic, ``(1 + #{b : maxima[b] >= |observed|})
        / (1 + B)``, NaN where `observed` is.
    """
    m = np.sort(maxima)
    reach = B - np.searchsorted(m, np.abs(observed) * (1 - eps), side="left")
    adj = (reach + 1) / (B + 1)
    return np.where(np.isnan(observed), np.nan, adj)


@dataclass
class CorPerm:
    """Permutation tests of every correlation, as `cor_perm` returns them.

    The arrays are ``p x q``: a row per column of ``x``, a column per
    column of ``y`` (of ``x`` when ``y`` was omitted, and then the diagonal,
    a variable with itself, is NaN, as is any correlation of a constant
    column).

    Attributes
    ----------
    r : ndarray, shape (p, q)
        The correlations.
    p_value : ndarray, shape (p, q)
        Their two-sided permutation p-values.
    p_adjusted : ndarray, shape (p, q)
        Their maxT adjusted p-values, for testing all of them at once.
    permutations : int
        The number of permutations, ``B``. The smallest possible p-value is
        ``1 / (B + 1)``.
    """

    r: np.ndarray
    p_value: np.ndarray
    p_adjusted: np.ndarray
    permutations: int


def cor_perm(x, y=None, *, B=1000, method="pearson", seed=None,
             batch=None, device=None, dtype=None) -> CorPerm:
    """Test every correlation by permutation, with maxT adjusted p-values.

    The correlations are between the columns of `x` and those of `y`, or
    among the columns of `x`.

    The rows of `y` (of a copy of `x`, if `y` is omitted) are shuffled `B`
    times, breaking any association while keeping each variable's
    distribution, and each correlation's p-value is how often a shuffle
    gives one as strong. No normality is assumed, unlike the t test of
    `cor_pairs` with ``test=True``, which matters for small samples and
    skewed data.

    Parameters
    ----------
    x : array_like, shape (n_samples, p)
        The variables, one per column, with no missing values.
    y : array_like, shape (n_samples, q), optional
        A second set of variables, the same samples in the same rows.
    B : int
        The number of permutations.
    method : {"pearson", "spearman"}
        Pearson's coefficient, or Spearman's (Pearson's on the ranks).
    seed : int, optional
        Seeds the permutations, for reproducible p-values.
    batch : int, optional
        How many permutations to compute at once; by default as many as fit
        in about 256 MB. It has no effect on the result.
    device : str or torch.device, optional
        Where to compute; by default the first available of CUDA, MPS and
        the CPU.
    dtype : {"float32", "float64"}, optional
        The precision the device computes in; float32 by default.

    Returns
    -------
    CorPerm
        The correlations and their permutation and adjusted p-values.

    Raises
    ------
    ValueError
        If `B` is not a positive whole number, or the data have missing
        values or fewer than 3 rows.

    See Also
    --------
    ttest_perm : the same for two-group t tests of every feature.

    Notes
    -----
    Shuffling the rows of a standardised column leaves it standardised, so
    each permutation's correlation matrix is one product ``zx.T @ zy[perm]``
    of the standardised columns, and a batch of permutations one batched
    product. The whole ``p x q`` matrix is held on the device, a few times
    over; for wider data, split the columns of `y` (the maxT adjustment is
    then within each part).

    Examples
    --------
    >>> import numpy as np
    >>> import widestats as ws
    >>> rng = np.random.default_rng(0)
    >>> x = rng.exponential(size=(15, 40))       # 15 samples, skewed
    >>> y = rng.exponential(size=(15, 30))
    >>> y[:, 0] += 6 * x[:, 0]                   # one real association
    >>> res = ws.cor_perm(x, y, B=999, seed=1, device="cpu")
    >>> float(res.p_value[0, 0]), float(res.p_adjusted[0, 0])
    (0.001, 0.001)
    >>> bool(np.nanmin(res.p_adjusted[1:, 1:]) > 0.05)
    True
    """
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
    # which entries are tests: not a constant column, nor a variable with
    # itself
    test = ~torch.isnan(r0)
    if self_:
        test &= ~torch.eye(p, dtype=torch.bool, device=dev)
    r0 = torch.where(test, r0, torch.nan)
    # what a permuted |r| must reach to count
    bar = r0.abs() * (1 - eps)
    counts = torch.zeros((p, q), dtype=torch.int64, device=dev)
    maxima = torch.empty(B, dtype=torch.float64)

    perms = _permutations(n, B, seed).to(dev)
    if batch is None:
        batch = max(1, (1 << 26) // max(1, p * q + n * q))
    for b0 in range(0, B, batch):
        pb = perms[b0:b0 + batch]                       # b x n
        rb = zx.T @ zy[pb]                              # b x p x q
        # entries that are no test count as 0, so they neither reach a bar
        # nor become a permutation's maximum
        rb = torch.where(test, rb.abs(), 0)
        counts += (rb >= bar).sum(dim=0)
        maxima[b0:b0 + len(pb)] = rb.amax(dim=(1, 2)).cpu().to(torch.float64)

    r = r0.cpu().to(torch.float64).numpy()
    p_value = np.where(np.isnan(r), np.nan,
                       (counts.cpu().numpy() + 1) / (B + 1))
    return CorPerm(np.clip(r, -1, 1), p_value,
                   _adjust(r, maxima.numpy(), eps, B), int(B))


@dataclass
class TTestPerm:
    """Two-group t tests of every feature, as `ttest_perm` returns them.

    Each attribute has one entry per column (feature) of ``x``.

    Attributes
    ----------
    mean1, mean2 : ndarray
        The feature's mean in the first group (in sorted order of the group
        labels) and in the second.
    statistic : ndarray
        The t statistic, of the first group's mean minus the second's.
    df : ndarray
        Its degrees of freedom (Welch's, or ``n - 2`` pooled).
    p_value : ndarray
        Its two-sided p-value from the t distribution.
    p_perm : ndarray or None
        With ``B > 0``, the two-sided permutation p-value.
    p_adjusted : ndarray or None
        With ``B > 0``, the maxT adjusted p-value, for testing every
        feature at once.

    A constant feature, or one constant within both groups, has no test:
    its statistic is NaN, as are its p-values.
    """

    mean1: np.ndarray
    mean2: np.ndarray
    statistic: np.ndarray
    df: np.ndarray
    p_value: np.ndarray
    p_perm: np.ndarray | None = None
    p_adjusted: np.ndarray | None = None


def ttest_perm(x, group, *, equal_var=False, B=0, seed=None, batch=None,
               device=None, dtype=None) -> TTestPerm:
    """Test every feature for a difference in mean between two groups.

    Optionally with permutation and maxT adjusted p-values.

    The t tests are those of ``scipy.stats.ttest_ind(x[g1], x[g2],
    equal_var=equal_var)``, one per column, computed in float64 on the CPU:
    a t test per feature is too little work to be worth a GPU. With
    ``B > 0``, the group labels are shuffled `B` times and every feature's
    t computed under every shuffle on the device, giving p-values that
    assume no normality, and maxT adjusted ones that control the
    family-wise error rate over all the features.

    Parameters
    ----------
    x : array_like, shape (n_samples, n_features)
        The data, a row per sample (cell, patient, ...), a column per
        feature (gene, ...), with no missing values.
    group : array_like, shape (n_samples,)
        Each sample's group: exactly two distinct labels, at least two
        samples each. The first label in sorted order is group 1.
    equal_var : bool
        Student's t with the variances pooled, rather than Welch's (the
        default), as in `scipy.stats.ttest_ind`.
    B : int
        The number of permutations; 0 (the default) for the t tests alone.
    seed : int, optional
        Seeds the permutations, for reproducible p-values.
    batch : int, optional
        How many labellings to compute at once; by default as many as fit
        in about 128 MB. It has no effect on the result.
    device : str or torch.device, optional
        Where to compute the permutations; by default the first available
        of CUDA, MPS and the CPU.
    dtype : {"float32", "float64"}, optional
        The precision the device computes in; float32 by default.

    Returns
    -------
    TTestPerm
        The tests, and with ``B > 0`` their permutation p-values.

    Raises
    ------
    ValueError
        If `group` does not have one entry per sample and exactly two
        labels with at least two samples each, `B` is negative, or the
        data have missing values.

    See Also
    --------
    cor_perm : the same for every correlation.

    Notes
    -----
    Each labelling's group-1 sums and sums of squares for every feature are
    rows of ``G @ x`` and ``G @ x**2``, where ``G`` is the labellings x
    samples 0/1 indicator matrix; group 2's follow from the totals. So all
    the permutations' t statistics come from two matrix products, a batch
    of labellings at a time. Each feature is first centred and scaled,
    which changes no t, so that its sums of squares do not cancel in
    float32.

    Examples
    --------
    >>> import numpy as np
    >>> import widestats as ws
    >>> rng = np.random.default_rng(0)
    >>> genes = rng.standard_normal((12, 2000))       # 12 samples
    >>> group = np.repeat(["control", "treated"], 6)
    >>> genes[group == "treated", :3] += 5            # 3 genes respond
    >>> res = ws.ttest_perm(genes, group, B=5000, seed=1, device="cpu")
    >>> np.flatnonzero(res.p_adjusted < 0.05)
    array([0, 1, 2])
    """
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
    # squares in float32 from cancelling (a constant feature becomes NaN)
    sd = x.std(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (x - x.mean(axis=0)) / sd
    zt = to_device(z, dev, dt)
    z2 = zt * zt
    s, q = zt.sum(dim=0), z2.sum(dim=0)

    # row 0 is the observed labelling, rows 1..B the permuted ones
    labels = torch.zeros((B + 1, n), dtype=dt)
    labels[0] = torch.from_numpy(in1)
    perms = _permutations(n, B, seed)
    labels[1:] = torch.from_numpy(in1)[perms].to(dt)
    labels = labels.to(dev)

    def tstat(g):
        """Compute every feature's t under each labelling in `g`.

        `g` is labellings x samples, 1 for group 1; the result is
        labellings x features.
        """
        a1, b1 = g @ zt, g @ z2                      # group 1 sums, squares
        a2, b2 = s - a1, q - b1
        m1, m2 = a1 / n1, a2 / n2
        # within-group sums of squared deviations; rounding can take an
        # exactly-zero one below zero
        ss1 = (b1 - a1 * m1).clamp_min(0)
        ss2 = (b2 - a2 * m2).clamp_min(0)
        if equal_var:
            v = (ss1 + ss2) / (n - 2)
            se2 = v * (1 / n1 + 1 / n2)
        else:
            se2 = ss1 / (n1 - 1) / n1 + ss2 / (n2 - 1) / n2
        return torch.where(se2 > 0, (m1 - m2) / se2.sqrt(), torch.nan)

    # the observed t is recomputed here, in the device's precision, so that
    # the permuted ones are compared with it on equal terms
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
