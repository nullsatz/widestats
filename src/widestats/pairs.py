"""Strongly correlated pairs among many variables, without the matrix.

With tens of thousands of variables the matrix of every pair is the
obstacle: 50,000 variables make a 10 GB matrix in float64, though usually
only a few thousand pairs are wanted. `cor_pairs` computes the matrix a tile
at a time on the device, as the matrix product of standardised columns, and
keeps only the pairs that pass, so memory grows with the tile and the
answer rather than with the number of pairs.
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
    standardise,
    to_device,
)


@dataclass
class Pairs:
    """Selected pairs of variables, as `cor_pairs` returns them.

    Each attribute is a NumPy array with one entry per pair.

    Attributes
    ----------
    i : ndarray of int
        The pair's column of ``x`` (0-based).
    j : ndarray of int
        The pair's column of ``y``, or of ``x`` when ``y`` was omitted.
    r : ndarray of float
        Their correlation.
    t : ndarray of float or None
        With ``test=True``, the t statistic of ``r`` on ``n - 2`` degrees of
        freedom.
    p_value : ndarray of float or None
        With ``test=True``, its two-sided p-value.
    """

    i: np.ndarray
    j: np.ndarray
    r: np.ndarray
    t: np.ndarray | None = None
    p_value: np.ndarray | None = None

    def __len__(self) -> int:
        """Return the number of pairs."""
        return len(self.r)

    def to_frame(self):
        """Return the pairs as a pandas DataFrame, a row per pair.

        pandas is not a dependency of widestats; this needs it installed.

        Returns
        -------
        pandas.DataFrame
            Columns ``i``, ``j``, ``r``, and ``t`` and ``p_value`` if they
            were computed.
        """
        import pandas as pd
        cols = {"i": self.i, "j": self.j, "r": self.r}
        if self.t is not None:
            cols.update(t=self.t, p_value=self.p_value)
        return pd.DataFrame(cols)


def cor_pairs(x, y=None, *, threshold=None, top=None, method="pearson",
              absolute=True, test=False, limit=10_000_000, tile=2048,
              device=None, dtype=None) -> Pairs:
    """Find the strongly correlated pairs without forming the matrix.

    The pairs are among the columns of `x`, or between those of `x` and
    those of `y`.
    Give exactly one of `threshold`, for every pair at least that strongly
    correlated, or `top`, for each column's strongest partners.

    Parameters
    ----------
    x : array_like, shape (n_samples, p)
        The variables, one per column, with no missing or infinite values.
    y : array_like, shape (n_samples, q), optional
        A second set of variables, the same samples in the same rows. If
        omitted, the columns of `x` are paired with each other.
    threshold : float, optional
        Keep every pair with ``|r| >= threshold`` (``r >= threshold`` if
        `absolute` is false). With `y` omitted, each pair appears once,
        with ``i < j``.
    top : int, optional
        Keep, for each column of `y`, the `top` columns of `x` most
        strongly correlated with it. With `y` omitted, a column is never
        its own partner, and a pair in both columns' lists appears twice.
    method : {"pearson", "spearman"}
        Pearson's coefficient, or Spearman's: Pearson's on the ranks, ties
        given their average rank.
    absolute : bool
        Measure strength by ``|r|``, so that strong negative correlations
        count (the default), or by ``r``, keeping positive ones only.
    test : bool
        Also return each pair's t statistic and two-sided p-value, as
        `scipy.stats.pearsonr` computes them.
    limit : int
        The most pairs a threshold may return. A threshold that lets more
        through raises an error, rather than returning a result of
        unexpected size.
    tile : int
        The device computes the matrix in blocks of ``tile x tile``. It
        sets the memory used and has no effect on the result.
    device : str or torch.device, optional
        Where to compute: ``"cuda"``, ``"mps"``, ``"cpu"``, ... By default
        the first available of CUDA, MPS and the CPU.
    dtype : {"float32", "float64"}, optional
        The precision the device computes in; float32 by default. Results
        are float64 either way.

    Returns
    -------
    Pairs
        The pairs, ordered by `j`, and then by `i` for a threshold or from
        the strongest for `top`.

    Raises
    ------
    ValueError
        If both or neither of `threshold` and `top` are given, the data
        have missing or infinite values or fewer than 3 rows, or more than `limit`
        pairs pass the threshold.

    See Also
    --------
    cor_perm : permutation p-values for every correlation.

    Notes
    -----
    Each column is centred and scaled to unit length (in float64, on the
    CPU), so that the correlation of two columns is their dot product, and
    the correlation matrix is ``zx.T @ zy``. A constant column has no
    correlations (they are NaN) and is never returned.

    Examples
    --------
    >>> import numpy as np
    >>> import widestats as ws
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal((50, 1000))
    >>> x[:, 7] = x[:, 3] + 0.1 * rng.standard_normal(50)
    >>> found = ws.cor_pairs(x, threshold=0.9, device="cpu")
    >>> found.i, found.j
    (array([3]), array([7]))
    >>> round(float(found.r[0]), 3)
    0.996
    """
    if (threshold is None) == (top is None):
        raise ValueError("give exactly one of threshold and top")
    if top is not None and (int(top) != top or top < 1):
        raise ValueError("top must be a positive whole number")
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

    if method == "spearman":
        x = rank_columns(x)
        y = x if self_ else rank_columns(y)
    zx = to_device(standardise(x), dev, dt)
    zy = zx if self_ else to_device(standardise(y), dev, dt)

    if threshold is not None:
        i, j, r = _threshold(zx, zy, self_, float(threshold), absolute,
                             tile, limit)
    else:
        i, j, r = _top(zx, zy, self_, int(top), absolute, tile)
    # rounding can put a perfect correlation a hair outside [-1, 1]
    r = np.clip(r, -1.0, 1.0)

    out = Pairs(i, j, r)
    if test:
        out.t, out.p_value = _t_test(r, n)
    return out


def _strength(r: torch.Tensor, absolute: bool) -> torch.Tensor:
    """Return how strong each correlation is, for selection.

    That is ``|r|``, or ``r`` if not `absolute`, with NaN (a constant
    column) as minus infinity so that it is never picked.
    """
    s = r.abs() if absolute else r
    return torch.nan_to_num(s, nan=-torch.inf)


def _threshold(zx, zy, self_, threshold, absolute, tile, limit):
    """Every pair at or above `threshold`, a tile at a time.

    Parameters
    ----------
    zx, zy : torch.Tensor, shape (n, p) and (n, q)
        Standardised columns on the device; ``zy is zx`` for `x` with
        itself (`self_`), when only pairs with ``i < j`` are kept.
    threshold, absolute, tile, limit
        As in `cor_pairs`.

    Returns
    -------
    i, j, r : ndarray
        The pairs, sorted by `j` then `i` (the device finds them in no
        particular order).
    """
    p, q = zx.shape[1], zy.shape[1]
    found, count = [], 0
    for j0 in range(0, q, tile):
        j1 = min(j0 + tile, q)
        for i0 in range(0, p, tile):
            # for x with itself, a tile whose every row i is at or below the
            # tile's last column j holds no pair with i < j, nor do later ones
            if self_ and i0 >= j1 - 1:
                break
            i1 = min(i0 + tile, p)
            r = zx[:, i0:i1].T @ zy[:, j0:j1]
            keep = _strength(r, absolute) >= threshold
            if self_:
                ii = torch.arange(i0, i1, device=r.device)[:, None]
                jj = torch.arange(j0, j1, device=r.device)[None, :]
                keep &= ii < jj
            a, b = keep.nonzero(as_tuple=True)
            count += len(a)
            if count > limit:
                raise ValueError(
                    f"more than {limit} pairs pass the threshold; raise the "
                    "threshold, or raise limit if you want them all")
            # to the CPU before float64: Apple GPUs (mps) cannot convert
            found.append(((a + i0).cpu(), (b + j0).cpu(),
                          r[a, b].cpu().to(torch.float64)))
    if not found:
        return (np.array([], dtype=np.int64),) * 2 + (np.array([]),)
    i, j, r = (torch.cat(c).numpy() for c in zip(*found, strict=True))
    order = np.lexsort((i, j))
    return i[order], j[order], r[order]


def _top(zx, zy, self_, k, absolute, tile):
    """Find each column of `zy`'s `k` strongest partners, a tile at a time.

    The partners are columns of `zx`.

    For each block of `zy` columns, the strongest `k` seen so far are kept
    on the device and merged with each new tile of rows, so a column's
    candidates never exceed ``k + tile``. Arguments are as in `_threshold`.

    Returns
    -------
    i, j, r : ndarray
        The pairs, by `j` and then from the strongest; ties go to the lower
        `i`. A column with fewer than `k` usable partners (NaN columns are
        unusable) contributes fewer.
    """
    p, q = zx.shape[1], zy.shape[1]
    k = min(k, p - self_)                 # for x with itself, not j itself
    out_i, out_j, out_r = [], [], []
    for j0 in range(0, q, tile):
        j1 = min(j0 + tile, q)
        tb = j1 - j0
        # each column's running best: strengths, x columns, correlations
        best_s = torch.full((0, tb), -torch.inf, device=zx.device,
                            dtype=zx.dtype)
        best_i = torch.empty((0, tb), dtype=torch.long, device=zx.device)
        best_r = torch.empty((0, tb), device=zx.device, dtype=zx.dtype)
        for i0 in range(0, p, tile):
            i1 = min(i0 + tile, p)
            r = zx[:, i0:i1].T @ zy[:, j0:j1]
            s = _strength(r, absolute)
            ii = torch.arange(i0, i1, device=r.device)[:, None].expand_as(r)
            if self_:
                jj = torch.arange(j0, j1, device=r.device)[None, :]
                s = s.masked_fill(ii == jj, -torch.inf)
            # merge the tile into the running best and keep the strongest k.
            # Sorting by x column first, then stably by strength, sends ties
            # to the lower x column whatever order the tiles came in.
            cs = torch.cat([best_s, s])
            ci = torch.cat([best_i, ii])
            cr = torch.cat([best_r, r])
            order = torch.sort(ci, dim=0, stable=True).indices
            cs, ci, cr = (t.gather(0, order) for t in (cs, ci, cr))
            top = torch.sort(cs, dim=0, descending=True, stable=True).indices
            top = top[:min(k, len(cs))]
            best_s, best_i, best_r = (t.gather(0, top) for t in (cs, ci, cr))
        # minus infinity marks a slot no usable partner filled
        usable = torch.isfinite(best_s)
        cols = torch.arange(j0, j1, device=zx.device)[None, :].expand_as(best_i)
        # transpose so the pairs come out column by column
        sel = usable.T.reshape(-1)
        out_i.append(best_i.T.reshape(-1)[sel].cpu())
        out_j.append(cols.T.reshape(-1)[sel].cpu())
        out_r.append(best_r.T.reshape(-1)[sel].cpu().to(torch.float64))
    return tuple(torch.cat(c).numpy() for c in (out_i, out_j, out_r))


def _t_test(r: np.ndarray, n: int):
    """Compute the t statistics and two-sided p-values of correlations.

    For correlations `r` from `n` samples, on ``n - 2`` degrees of freedom,
    as `scipy.stats.pearsonr` computes them; ``|r| = 1`` gives an infinite
    t and a p-value of 0.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        t = r * np.sqrt((n - 2) / (1 - r * r))
    return t, 2 * stats.t.sf(np.abs(t), n - 2)
