"""Mutual information between every pair of variables, by B-splines.

The estimator of Daub et al. (2004): each variable's range is divided into
`bins` bins, and each value is spread over neighbouring bins with B-spline
weights instead of falling into one, which makes the estimate smoother and
less sensitive to where the bin edges fall. With every variable's weights
side by side, one matrix product gives the joint distribution of every pair
at once.

References
----------
Daub, C. O., Steuer, R., Selbig, J. and Kloska, S. (2004). Estimating
mutual information using B-spline functions -- an improved similarity
measure for analysing gene expression data. *BMC Bioinformatics* 5, 118.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import BSpline

from ._core import as_matrix, pick_device, pick_dtype, to_device


def bspline_weights(x: np.ndarray, bins: int, order: int) -> np.ndarray:
    """Compute each value's B-spline weights over the bins, per variable.

    Parameters
    ----------
    x : ndarray, shape (n, p)
        The variables, one per column.
    bins : int
        The number of bins, ``M`` in Daub et al.
    order : int
        The spline order, ``k`` in Daub et al.: 1 puts each value wholly in
        one bin (plain binning), 2 spreads it linearly over two, 3
        (quadratically) over three, and so on. The polynomial degree is
        ``order - 1``.

    Returns
    -------
    ndarray, shape (n, p, bins)
        Non-negative weights, each value's summing to 1.

    Notes
    -----
    Each variable is scaled linearly onto the spline domain
    ``[0, bins - order + 1]``, its minimum at 0 and its maximum at the far
    end, with the knot vector ``0, ..., 0, 1, 2, ..., bins - order, bins -
    order + 1, ..., bins - order + 1`` (each end repeated `order` times), as
    in Daub et al. A constant variable is put wholly at 0.
    """
    n, p = x.shape
    span = bins - order + 1
    knots = np.r_[np.zeros(order), np.arange(1, span), np.full(order, span)]
    lo, hi = x.min(axis=0), x.max(axis=0)
    width = np.where(hi > lo, hi - lo, 1.0)        # a constant variable: at 0
    z = (x - lo) / width * span
    w = np.empty((n, p, bins))
    for i in range(p):
        w[:, i] = BSpline.design_matrix(z[:, i], knots, order - 1).toarray()
    return w


def _mi_bits(joint: torch.Tensor) -> torch.Tensor:
    """Compute the mutual information of each joint distribution, in bits.

    `joint` is ``(p, bins, q, bins)``, a ``bins x bins`` joint distribution
    per pair. The marginals are taken from the joint itself (its row and
    column sums), and ``sum P log2(P / (P_a P_b))`` computed directly:
    the same as ``H_a + H_b - H_ab``, but without subtracting large
    entropies, and consistent even when the joint carries rounding error
    of its own. ``0 log 0`` counts as 0.
    """
    rows = joint.sum(dim=3, keepdim=True)          # p x bins x q x 1
    cols = joint.sum(dim=1, keepdim=True)          # p x 1 x q x bins
    # in logs: rows * cols can underflow to 0 in float32 for a nearly empty
    # bin while the joint entry stays positive. Where the entry is positive
    # its row and column sums are at least as large, so every log is finite.
    tiny = torch.finfo(joint.dtype).tiny
    log2 = lambda t: torch.log2(t.clamp_min(tiny))  # noqa: E731
    terms = torch.where(joint > 0,
                        joint * (log2(joint) - log2(rows) - log2(cols)), 0)
    return terms.sum(dim=(1, 3))


def mutual_info(x, y=None, *, bins=10, order=3, tile=None, device=None,
                dtype=None) -> np.ndarray:
    """Estimate the mutual information between every pair of variables.

    The B-spline estimator of Daub et al. (2004), in bits, as gputools'
    ``gpuMi`` computes it. Unlike a correlation it detects any kind of
    dependence, not only linear (or monotone) association.

    Parameters
    ----------
    x : array_like, shape (n_samples, p)
        The variables, one per column, with no missing or infinite values.
    y : array_like, shape (n_samples, q), optional
        A second set of variables, the same samples in the same rows. If
        omitted, the columns of `x` are paired with each other.
    bins : int
        The number of bins each variable's range is divided into. Daub et
        al. find 10 to work well.
    order : int
        The B-spline order, from 1 (plain binning) to `bins`; 3, which
        Daub et al. recommend, spreads each value quadratically over three
        bins.
    tile : int, optional
        How many variables of each set to pair at once; by default as many
        as fit in a few hundred MB. It has no effect on the result.
    device : str or torch.device, optional
        Where to compute; by default the first that works of CUDA, MPS and
        the CPU.
    dtype : {"float32", "float64"}, optional
        The precision the device computes in; float32 by default.

    Returns
    -------
    ndarray, shape (p, q)
        The mutual information of each column of `x` with each column of
        `y` (of `x`), in bits. A constant variable shares no information:
        0. With `y` omitted, the diagonal is each variable's information
        with itself: its entropy for ``order=1``, and less for higher
        orders, whose spread weights blur a variable's joint distribution
        with itself.

    Raises
    ------
    ValueError
        If `bins` and `order` are not whole numbers with ``1 <= order <=
        bins``, or the data have missing or infinite values or fewer than 2
        rows.

    Notes
    -----
    With ``w_i(s, a)`` the weight of sample ``s`` of variable ``i`` in bin
    ``a``, the bin probabilities are ``P_i(a) = mean_s w_i(s, a)`` and the
    joint ones ``P_ij(a, b) = mean_s w_i(s, a) w_j(s, b)``, so that ``MI =
    H_i + H_j - H_ij`` with ``H = -sum P log2 P``. Stacking the weights of
    many variables side by side, ``W_x' W_y / n`` holds the joint
    distribution of every pair, a ``bins x bins`` block per pair. Since
    each sample's weights sum to 1, a block's row and column sums are the
    two marginals, and MI is computed from the block alone as ``sum P_ij
    log2(P_ij / (P_i P_j))``: the same quantity, without subtracting large
    entropies to get a small difference, and consistent with any rounding
    in the product.

    Examples
    --------
    >>> import numpy as np
    >>> import widestats as ws
    >>> rng = np.random.default_rng(0)
    >>> u = rng.uniform(-1, 1, 2000)
    >>> x = np.column_stack([u, u ** 2, rng.uniform(-1, 1, 2000)])
    >>> mi = ws.mutual_info(x, device="cpu")
    >>> bool(mi[0, 1] > 10 * mi[0, 2])  # u and u**2: dependent, uncorrelated
    True
    >>> bool(abs(np.corrcoef(u, u ** 2)[0, 1]) < 0.05)
    True
    """
    if int(bins) != bins or int(order) != order or not 1 <= order <= bins:
        raise ValueError("bins and order must be whole numbers with "
                         "1 <= order <= bins")
    bins, order = int(bins), int(order)
    x = as_matrix(x, "x")
    self_ = y is None
    y = x if self_ else as_matrix(y, "y")
    n = x.shape[0]
    if y.shape[0] != n:
        raise ValueError("x and y have different numbers of rows")
    if n < 2:
        raise ValueError("need at least 2 rows")
    dev = pick_device(device)
    dt = pick_dtype(dtype, dev)
    p, q = x.shape[1], y.shape[1]

    wx = to_device(bspline_weights(x, bins, order).reshape(n, p * bins),
                   dev, dt)
    wy = wx if self_ else to_device(
        bspline_weights(y, bins, order).reshape(n, q * bins), dev, dt)

    if tile is None:
        tile = max(1, int(np.sqrt((1 << 24) / bins**2)))
    mi = torch.empty((p, q), dtype=torch.float64)
    for i0 in range(0, p, tile):
        i1 = min(i0 + tile, p)
        for j0 in range(0, q, tile):
            j1 = min(j0 + tile, q)
            # every pair's joint distribution in this tile: a bins x bins
            # block of W_x' W_y / n per pair
            joint = (wx[:, i0 * bins:i1 * bins].T
                     @ wy[:, j0 * bins:j1 * bins]) / n
            joint = joint.reshape(i1 - i0, bins, j1 - j0, bins)
            mi[i0:i1, j0:j1] = _mi_bits(joint).cpu().to(torch.float64)
    mi = mi.numpy()
    # a constant variable shares no information: exactly 0, not rounding
    mi[np.ptp(x, axis=0) == 0, :] = 0.0
    mi[:, np.ptp(y, axis=0) == 0] = 0.0
    # an estimate of 0 can round just below it
    return np.maximum(mi, 0.0)
