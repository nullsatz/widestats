"""The strongly correlated pairs among many variables, without the
correlation matrix: it is computed a tile at a time on the device and
filtered there, so 50,000 variables (a 10 GB matrix in float64) take a few
hundred MB."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import stats

from ._core import (as_matrix, pick_device, pick_dtype, rank_columns,
                    standardise, to_device)


@dataclass
class Pairs:
    """Pairs of variables: x column i, y column j (0-based), their
    correlation r, and with test=True its t statistic and two-sided
    p-value."""
    i: np.ndarray
    j: np.ndarray
    r: np.ndarray
    t: np.ndarray | None = None
    p_value: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.r)

    def to_frame(self):
        """As a pandas DataFrame (pandas is not a dependency)."""
        import pandas as pd
        cols = {"i": self.i, "j": self.j, "r": self.r}
        if self.t is not None:
            cols.update(t=self.t, p_value=self.p_value)
        return pd.DataFrame(cols)


def cor_pairs(x, y=None, *, threshold=None, top=None, method="pearson",
              absolute=True, test=False, limit=10_000_000, tile=2048,
              device=None, dtype=None) -> Pairs:
    """Correlated pairs among the columns of x, or between those of x and y.

    Give one of
      threshold: every pair with |r| >= threshold (r >= threshold if not
                 absolute); for x with itself each pair once, i < j;
      top:       each column of y's `top` most strongly correlated columns
                 of x, strongest first; for x with itself a column is not
                 its own partner.
    method is "pearson" or "spearman". A threshold that lets more than
    `limit` pairs through is an error rather than a result of unexpected
    size. `tile` sets the block size on the device and has no effect on the
    result. Returns Pairs ordered by j, then by i (threshold) or from the
    strongest (top).
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
    p, q = zx.shape[1], zy.shape[1]

    if threshold is not None:
        i, j, r = _threshold(zx, zy, self_, float(threshold), absolute,
                             tile, limit)
    else:
        i, j, r = _top(zx, zy, self_, int(top), absolute, tile)
    r = np.clip(r, -1.0, 1.0)

    out = Pairs(i, j, r)
    if test:
        out.t, out.p_value = _t_test(r, n)
    return out


def _strength(r: torch.Tensor, absolute: bool) -> torch.Tensor:
    s = r.abs() if absolute else r
    return torch.nan_to_num(s, nan=-torch.inf)    # NaN is never picked


def _threshold(zx, zy, self_, threshold, absolute, tile, limit):
    p, q = zx.shape[1], zy.shape[1]
    found, count = [], 0
    for j0 in range(0, q, tile):
        j1 = min(j0 + tile, q)
        for i0 in range(0, p, tile):
            if self_ and i0 >= j1 - 1:
                break                           # no i < j left
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
            found.append(((a + i0).cpu(), (b + j0).cpu(),
                          r[a, b].cpu().to(torch.float64)))
    if not found:
        return (np.array([], dtype=np.int64),) * 2 + (np.array([]),)
    i, j, r = (torch.cat(c).numpy() for c in zip(*found))
    order = np.lexsort((i, j))
    return i[order], j[order], r[order]


def _top(zx, zy, self_, k, absolute, tile):
    p, q = zx.shape[1], zy.shape[1]
    k = min(k, p - self_)
    out_i, out_j, out_r = [], [], []
    for j0 in range(0, q, tile):
        j1 = min(j0 + tile, q)
        tb = j1 - j0
        # each column's running best k: strengths, x columns, correlations
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
            # merge this tile into the running best, keep the strongest k;
            # ties go to the lower x column, as the stable sort leaves them
            cs = torch.cat([best_s, s])
            ci = torch.cat([best_i, ii])
            cr = torch.cat([best_r, r])
            order = torch.sort(ci, dim=0, stable=True).indices
            cs, ci, cr = (t.gather(0, order) for t in (cs, ci, cr))
            top = torch.sort(cs, dim=0, descending=True, stable=True).indices
            top = top[:min(k, len(cs))]
            best_s, best_i, best_r = (t.gather(0, top) for t in (cs, ci, cr))
        usable = torch.isfinite(best_s)            # a column of NaNs has none
        cols = torch.arange(j0, j1, device=zx.device)[None, :].expand_as(best_i)
        sel = usable.T.reshape(-1)                 # column by column
        out_i.append(best_i.T.reshape(-1)[sel].cpu())
        out_j.append(cols.T.reshape(-1)[sel].cpu())
        out_r.append(best_r.T.reshape(-1)[sel].cpu().to(torch.float64))
    return tuple(torch.cat(c).numpy() for c in (out_i, out_j, out_r))


def _t_test(r: np.ndarray, n: int):
    """The t statistic and two-sided p-value of correlations from n
    samples, as scipy.stats.pearsonr's: on n - 2 degrees of freedom."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = r * np.sqrt((n - 2) / (1 - r * r))
    return t, 2 * stats.t.sf(np.abs(t), n - 2)
