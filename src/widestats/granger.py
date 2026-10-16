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

import warnings
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


def _orthonormal_columns(a: torch.Tensor):
    """Orthonormalise the few columns of each matrix in a batch.

    Returns an orthonormal basis ``Q`` (``b x T x m``, same device and
    precision as `a`) of the columns of each ``T x m`` matrix in `a`, and
    each matrix's condition number (float64, on the CPU).

    Cholesky-QR2 in float64: ``R`` from the Cholesky factor of the small
    Gram matrix ``A'A`` and ``Q = A R^{-1}``, done twice. With ``m`` a
    handful of columns it is all batched matrix products and tiny
    factorisations, much faster on a GPU than a batched Householder QR (one
    small kernel per matrix), and in float64 as accurate for condition
    numbers up to about 1e7. Matrices beyond that (series integrated twice
    or more give 1e8 to 1e10) are factored by Householder QR instead. Done
    on the device where it has float64 (CUDA, the CPU), else on the CPU. A
    constant series' design gives NaN.
    """
    work = a.device if float64_solve_works(a.device.type) \
        else torch.device("cpu")
    a64 = a.to(work).to(torch.float64)
    q, cond, bad = a64, None, torch.zeros(a.shape[0], dtype=torch.bool,
                                          device=work)
    for _ in range(2):
        chol, info = torch.linalg.cholesky_ex(q.transpose(1, 2) @ q)
        if cond is None:
            # cond(A) = cond(R) for A = Q R with Q orthonormal; infinite
            # where the factorisation failed or the design is NaN
            ok = (info == 0) & torch.isfinite(chol).flatten(1).all(dim=1)
            eye = torch.eye(chol.shape[1], dtype=chol.dtype, device=work)
            cond = torch.where(ok, torch.linalg.cond(
                torch.where(ok[:, None, None], chol, eye)), torch.inf)
        bad |= (info != 0) | ~(cond < 1e7)
        # Q R = A with R = chol', so Q = A R^{-1}: solve X R = A for X
        q = torch.linalg.solve_triangular(chol.transpose(1, 2), q,
                                          upper=True, left=False)
    if bad.any():
        # Householder for the ill-conditioned ones (a constant series' NaN
        # design stays NaN)
        qq, rr = torch.linalg.qr(a64[bad])
        q[bad] = qq
        finite = torch.isfinite(rr).flatten(1).all(dim=1)
        eye = torch.eye(rr.shape[1], dtype=rr.dtype, device=work)
        cond[bad] = torch.where(finite, torch.linalg.cond(
            torch.where(finite[:, None, None], rr, eye)), torch.inf)
    # to the precision first: Apple GPUs (mps) cannot hold float64
    return q.to(a.dtype).to(a.device), cond.cpu()


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
        The precision the device computes in: by default float64 where the
        device has it (CUDA, the CPU) and float32 on Apple GPUs. float32 is
        faster but can be badly off for long, persistent series (by over
        10% in F for random walks of 50,000 steps) or when the cause
        explains nearly all of the effect; see Notes.

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

    Warns
    -----
    RuntimeWarning
        If some pairs could not be resolved in the precision computed in
        (see Notes); their tests are NaN.

    Notes
    -----
    With ``T = n - lag`` usable times, the restricted model of effect ``j``
    has ``lag + 1`` coefficients and the unrestricted ``2 * lag + 1``, so
    ``F = ((RSS_r - RSS_u) / lag) / (RSS_u / (T - 2 * lag - 1))``.

    By the Frisch-Waugh theorem, ``RSS_r - RSS_u = g' G^{-1} g``, where,
    after projecting the effect's restricted design ``[1, own lags]`` (an
    orthonormal basis ``Q_j`` of it, by Cholesky-QR2 in float64) out of
    everything,
    ``g = L_i' e_j`` is the cause's lags against the effect's residual and
    ``G = L_i' L_i - (Q_j' L_i)' (Q_j' L_i)`` the cause's residual
    cross-products. ``L' e`` for all pairs is one matrix product, and so is
    ``Q_j' L``; the ``lag x lag`` systems are formed and solved in float64
    as a batch, on the device where it can (CUDA, CPU) or on the CPU for
    Apple GPUs, which have no float64. Each series is first centred and
    scaled, which changes no F.

    A cause whose lags are almost exactly a linear function of the effect's
    own past (nearly a copy of the effect) leaves residual cross-products
    too small to resolve beside ``L'L`` in the precision computed in: such
    pairs are NaN, with a warning. float32 resolves a residual variance down
    to about 1e-4 of the cause's, float64 to about 1e-12.

    So are all the tests of an effect whose own lags are nearly collinear:
    a restricted design with condition number above 1e5. White noise and
    stationary series are near 1, random walks and near-unit-root series
    at most a few hundred, but a series integrated twice or more reaches
    1e7 to 1e11, where the cancellation in ``G`` is beyond this method.
    Difference such series first, as Granger tests assume stationary series
    anyway.

    Short of that, float32 still loses accuracy wherever the residuals are
    small beside the series, as ``RSS_r - RSS_u`` then cancels: for long,
    strongly persistent series, and when the cause explains nearly all of
    the effect. That is why float64 is the default where the device has
    it; on Apple GPUs (float32 only), use ``device="cpu"`` for such data.

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
    if dtype is None:
        # float64 wherever the device has it: in float32 the F statistics of
        # long, persistent series (random walks) can be off by over 10%
        dtype = "float64" if float64_solve_works(dev.type) else "float32"
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
        # a tile holds b x T x (L+1) designs and b x (L+1) x p*L products
        tile = max(1, (1 << 24) // max(1, p * lag * (2 * lag + 1),
                                       times * (lag + 1)))
    gain = torch.empty((p, q), dtype=torch.float64)
    rss_r = torch.empty(q, dtype=torch.float64)
    weak = torch.zeros((p, q), dtype=torch.bool)
    tol = 1e-12 if dt == torch.float64 else 1e-4
    ones = torch.ones((times, 1), device=dev, dtype=dt)
    for j0 in range(0, q, tile):
        j1 = min(j0 + tile, q)
        b = j1 - j0
        # each effect's restricted design D = [1, own lags], orthonormalised
        design = torch.cat([ones.expand(b, times, 1),
                            ly[:, j0:j1].permute(1, 0, 2)], dim=2)  # b x T x m
        qmat, cond = _orthonormal_columns(design)                  # b x T x m
        yb = target[:, j0:j1].T                                     # b x T
        resid = yb - (qmat @ (qmat.transpose(1, 2) @ yb[:, :, None]))[:, :, 0]
        rss_r[j0:j1] = (resid * resid).sum(dim=1).cpu().to(torch.float64)
        # cause lags against effect residuals: g = L' e (the residual is
        # already orthogonal to Q_j, so nothing of Q_j' L enters g)
        g = (lx.T @ resid.T).reshape(p, lag, b).permute(2, 0, 1)
        c = (qmat.transpose(1, 2) @ lx).reshape(b, lag + 1, p, lag)
        # to the CPU first when float64 is not on the device (mps)
        g = g.to(solve_dev).to(torch.float64)                       # b x p x L
        c = c.to(solve_dev).to(torch.float64)
        # the causes' residual cross-products G = L'L - (Q'L)'(Q'L)
        gmat = g0[None] - torch.einsum("bcik,bcil->bikl", c, c)    # b x p x L x L
        sol, info = torch.linalg.solve_ex(gmat, g[..., None])
        gb = (g * sol[..., 0]).sum(dim=2)                           # b x p
        # A cause whose lags are almost a linear function of the effect's
        # own past leaves residual cross-products too small to resolve in
        # this precision beside L'L: G and the gain would be rounding noise.
        # An effect whose own lags are nearly collinear (condition above
        # 1e5: series integrated twice or more; random walks and
        # near-unit-root series stay below about 300) leaves G = L'L - C'C
        # to cancellation this method cannot control: all its tests are NaN.
        hopeless = (cond > 1e5).to(solve_dev)
        resolved = (torch.diagonal(gmat, dim1=2, dim2=3)
                    / torch.diagonal(g0, dim1=1, dim2=2)[None]).amin(dim=2)
        short = (resolved < tol) | hopeless[:, None]
        weak[:, j0:j1] = short.T.cpu()
        gb = torch.where((info == 0) & ~short, gb, torch.nan)
        gain[:, j0:j1] = gb.T.cpu()

    rss_r = rss_r.numpy()
    gain = gain.numpy()
    weak = weak.numpy()
    if self_:
        np.fill_diagonal(gain, np.nan)
        np.fill_diagonal(weak, False)       # not a test anyway
    if weak.any():
        warnings.warn(
            f"{int(weak.sum())} pair(s) are too close to resolve in "
            f"{str(dt).replace('torch.', '')}: a cause almost exactly a "
            "linear function of the effect's own past, or an effect whose "
            "own lags are nearly collinear (a series integrated twice or "
            "more: difference it); their tests are NaN"
            + ("; dtype='float64' resolves more" if dt == torch.float32
               else ""), RuntimeWarning, stacklevel=2)
    # rounding can take a gain of 0 just below 0, or one equal to the
    # restricted RSS just above it
    gain = np.clip(gain, 0, rss_r[None, :])
    with np.errstate(invalid="ignore", divide="ignore"):
        rss_u = rss_r[None, :] - gain
        f = (gain / lag) / (rss_u / df2)
    return Granger(f, stats.f.sf(f, lag, df2), lag, (lag, df2))
