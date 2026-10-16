"""Granger causality F tests for every pair of time series.

Whether series ``i`` Granger-causes series ``j`` at lag ``L`` is the F test
of two least-squares fits of ``j``: the restricted one, on an intercept and
``j``'s own ``L`` past values, and the unrestricted one, adding ``i``'s
``L`` past values. Fitting both for every pair is ``p * q`` regressions;
here each effect's own past is projected out once, after which every
cause's contribution to every effect comes from one matrix product and a
small ``L x L`` solve per pair.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import stats

from ._core import (
    as_matrix,
    float64_solve_works,
    pick_device,
    pick_dtype,
    to_device,
)


@dataclass
class Granger:
    """Granger causality tests of every pair of series.

    As `granger` returns them.

    The arrays are ``p x q``: a row per cause (a column of ``x``), a column
    per effect (a column of ``y``, or of ``x`` when ``y`` was omitted, and
    then the diagonal, a series against itself, is NaN).

    Attributes
    ----------
    f : ndarray, shape (p, q)
        The F statistics.
    p_value : ndarray, shape (p, q)
        Their p-values, from the F distribution on ``df`` degrees of
        freedom.
    lag : int
        The number of past values of each series in the models.
    df : tuple of int
        The F test's degrees of freedom, ``(lag, T - 2 * lag - 1)`` for
        ``T = n - lag`` usable time points.
    """

    f: np.ndarray
    p_value: np.ndarray
    lag: int
    df: tuple[int, int]


def granger(x, y=None, *, lag=1, tile=None, device=None, dtype=None) -> Granger:
    """Test every series for Granger causality of every other.

    For each ordered pair, does the past of the cause improve the
    prediction of the effect beyond the effect's own past? The answer is the
    F test that ``statsmodels``' ``grangercausalitytests`` reports as
    ``ssr_ftest``, and gputools' ``gpuGranger``.

    Parameters
    ----------
    x : array_like, shape (n_times, p)
        The series, one per column, in time order down the rows, with no
        missing values. They are the causes, and with `y` omitted the
        effects too.
    y : array_like, shape (n_times, q), optional
        The effects, the same times in the same rows.
    lag : int
        How many past values of each series the models use.
    tile : int, optional
        How many effects to compute at once; by default as many as fit in
        a few hundred MB. It has no effect on the result.
    device : str or torch.device, optional
        Where to compute; by default the first that works of CUDA, MPS and
        the CPU.
    dtype : {"float32", "float64"}, optional
        The precision the device computes in; float32 by default.

    Returns
    -------
    Granger
        F statistics and p-values, causes by effects.

    Raises
    ------
    ValueError
        If `lag` is not a positive whole number, the series are too short
        for it (fewer than ``3 * lag + 2`` times), or the data have missing
        or infinite values.

    Notes
    -----
    With ``T = n - lag`` usable times, the restricted model of effect ``j``
    has ``lag + 1`` coefficients and the unrestricted ``2 * lag + 1``, so
    ``F = ((RSS_r - RSS_u) / lag) / (RSS_u / (T - 2 * lag - 1))``.

    By the Frisch-Waugh theorem, ``RSS_r - RSS_u = g' G^{-1} g``, where,
    after projecting the effect's intercept and own lags (the matrix
    ``Q_j``, from a QR decomposition) out of everything, ``g = L_i' e_j``
    is the cause's lags against the effect's residual and ``G = L_i' L_i -
    (Q_j' L_i)' (Q_j' L_i)`` the cause's residual cross-products. ``L_i'
    e_j`` for all pairs is one matrix product, and so is ``Q_j' L_i``. The
    ``lag x lag`` systems, one per pair, are formed and solved in float64
    as a batch on the device (CUDA, CPU), or on the CPU for Apple GPUs,
    which have no float64. Each series is first centred and scaled, which
    changes no F.

    A series is not tested against itself (its lags are already in the
    restricted model), and a constant series has no tests: NaN.

    Examples
    --------
    >>> import numpy as np
    >>> import widestats as ws
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal((300, 4))
    >>> x[1:, 2] += 0.8 * x[:-1, 0]          # series 0 drives series 2
    >>> res = ws.granger(x, lag=2, device="cpu")
    >>> np.argwhere(res.p_value < 1e-6)
    array([[0, 2]])
    >>> res.df
    (2, 293)
    """
    if int(lag) != lag or lag < 1:
        raise ValueError("lag must be a positive whole number")
    lag = int(lag)
    x = as_matrix(x, "x")
    self_ = y is None
    y = x if self_ else as_matrix(y, "y")
    n = x.shape[0]
    if y.shape[0] != n:
        raise ValueError("x and y have different numbers of rows")
    times = n - lag
    df2 = times - 2 * lag - 1
    if df2 < 1:
        raise ValueError(f"the series are too short for lag {lag}: need at "
                         f"least {3 * lag + 2} times")
    dev = pick_device(device)
    dt = pick_dtype(dtype, dev)
    p, q = x.shape[1], y.shape[1]

    def scaled(v):
        # centring and scaling a series changes no F; a constant one is NaN
        v = v - v.mean(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return v / np.sqrt((v * v).sum(axis=0))

    def lagged(v):
        # times x series x lags: column k - 1 is the value k steps back
        return np.stack([v[lag - k:n - k] for k in range(1, lag + 1)], axis=2)

    xs, ys = scaled(x), (None if self_ else scaled(y))
    ys = xs if self_ else ys
    lx = to_device(lagged(xs).reshape(times, p * lag), dev, dt)     # T x p*L
    ly = to_device(lagged(ys), dev, dt)                              # T x q x L
    target = to_device(ys[lag:], dev, dt)                            # T x q
    # The lag x lag systems G g = gain are solved in float64, as G is a
    # difference that can cancel: on the device where it can (CUDA, CPU),
    # else (Apple GPUs) on the CPU. Each cause's own cross-products L_i' L_i
    # come exactly from the float64 data.
    solve_dev = dev if float64_solve_works(dev.type) else torch.device("cpu")
    lx64 = lagged(xs)                                                # T x p x L
    g0 = torch.from_numpy(np.einsum("tik,til->ikl", lx64, lx64)).to(solve_dev)

    if tile is None:
        tile = max(1, (1 << 24) // max(1, p * lag * (2 * lag + 1)))
    gain = torch.empty((p, q), dtype=torch.float64)
    rss_r = torch.empty(q, dtype=torch.float64)
    ones = torch.ones((times, 1), device=dev, dtype=dt)
    for j0 in range(0, q, tile):
        j1 = min(j0 + tile, q)
        # each effect's restricted design, [1, own lags], orthonormalised
        design = torch.cat([ones.expand(j1 - j0, times, 1),
                            ly[:, j0:j1].permute(1, 0, 2)], dim=2)
        qmat = torch.linalg.qr(design).Q                       # b x T x (L+1)
        yb = target[:, j0:j1].T                                # b x T
        resid = yb - (qmat @ (qmat.transpose(1, 2) @ yb[:, :, None]))[:, :, 0]
        rss_r[j0:j1] = (resid * resid).sum(dim=1).cpu().to(torch.float64)
        # cause lags against effect residuals: L' e (the residual is already
        # orthogonal to Q_j, so nothing of Q_j' L enters g)
        g = (lx.T @ resid.T).reshape(p, lag, j1 - j0).permute(2, 0, 1)
        c = (qmat.transpose(1, 2) @ lx).reshape(j1 - j0, lag + 1, p, lag)
        # to the CPU first when float64 is not on the device (mps)
        c = c.to(solve_dev).to(torch.float64)
        g = g.to(solve_dev).to(torch.float64)
        gmat = g0[None] - torch.einsum("bcik,bcil->bikl", c, c)
        sol, info = torch.linalg.solve_ex(gmat, g[..., None])
        gb = (g * sol[..., 0]).sum(dim=2)                      # b x p
        gb = torch.where(info == 0, gb, torch.nan)             # singular G
        gain[:, j0:j1] = gb.T.cpu()

    rss_r = rss_r.numpy()
    gain = gain.numpy()
    if self_:
        np.fill_diagonal(gain, np.nan)
    # rounding can take a gain of 0 just below 0, or one equal to the
    # restricted RSS just above it
    gain = np.clip(gain, 0, rss_r[None, :])
    with np.errstate(invalid="ignore", divide="ignore"):
        rss_u = rss_r[None, :] - gain
        f = (gain / lag) / (rss_u / df2)
    return Granger(f, stats.f.sf(f, lag, df2), lag, (lag, df2))
