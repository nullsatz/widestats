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
)

# Condition numbers above which a test is NaN: of the effect's restricted
# design, and of a cause's residual cross-products G. Measured against least
# squares (1,152 pairs: white noise, AR(0.9999), random walks with and
# without drift, trends, I(2) with and without drift, I(3), mixed panels;
# n 300-30,000; lag 1-8): realistic series stay below 5e4 and 3e5 with
# errors under 1e-8; within these limits every pair was within 1e-4;
# beyond them errors reach 100%.
EFFECT_LIMIT = 1e10
CAUSE_LIMIT = 1e12
# A cause whose lags keep less than this fraction of their squared length
# once the effect's own past is projected out is (to rounding) a linear
# function of that past: nothing is left to test.
LEFT_LIMIT = 1e-20
# Cholesky-QR2 for designs better conditioned than this (as accurate here
# as Householder, measured, and much faster on GPUs); Householder QR for
# the rest, where the Cholesky factorisation itself becomes unreliable.
CHOLESKY_QR_LIMIT = 1e7
# The fast form of a pair's test, G = L'L - C'C and g = L'e - C'(Q'e) with
# C = Q'L, subtracts quantities of the size of the cause's lags to leave
# what is beside the effect's own past. Its error relative to the explicit
# form's grows with cond(G) / left, and, as F compares the gain with what
# is left of the restricted RSS, with 1 + F * lag / df2. Measured over
# 1,501 resolved pairs of the sweep panels (white noise, AR(0.9999),
# random walks with and without drift, trends, I(2) with and without
# drift, I(3), mixed, causal, near-copies; n 300-30,000; lag 1-8), the
# error is at most about 20 eps times that product: below this limit it
# stayed under 2.2e-10 (bound 4.4e-9), and plain random walks, AR, trends
# and white noise all pass; above it the pair is recomputed explicitly.
FAST_SCORE_LIMIT = 1e6


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
        bad |= (info != 0) | ~(cond < CHOLESKY_QR_LIMIT)
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


def _granger_core(x, y, self_, lag, tile, dev, fast=None):
    """Compute the pieces of every pair's Granger test, in float64.

    Returns ``gain = RSS_r - RSS_u`` (causes x effects), ``RSS_r`` per
    effect, the condition number of each effect's restricted design, that
    of each pair's residual cross-products ``G`` (NaN for a constant
    series, infinite for exactly collinear lags), and the fraction of each
    cause's lags (squared length) left once the effect's own past is
    projected out. See `granger`.

    `fast` chooses the form: by default the fast one for the pairs it
    resolves (see `FAST_SCORE_LIMIT`) and the explicit residual lags for
    the rest; True the fast form for every pair, False the explicit one
    (both for measuring one against the other).
    """
    n = x.shape[0]
    times = n - lag
    # float64 throughout, on the device if it has float64, else (Apple GPUs)
    # on the CPU: in float32 these F statistics can be badly off (over 10%
    # for long random walks)
    wd = dev if float64_solve_works(dev.type) else torch.device("cpu")
    p, q = x.shape[1], y.shape[1]

    def scaled(v):
        # centring and scaling a series changes no F; a constant one is NaN
        v = v - v.mean(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return v / np.sqrt((v * v).sum(axis=0))

    def lagged(v):
        # times x series x lags: column k - 1 is the value k steps back
        return np.stack([v[lag - k:n - k] for k in range(1, lag + 1)], axis=2)

    def on(a):
        return torch.from_numpy(np.ascontiguousarray(a)).to(wd)

    xs = scaled(x)
    ys = xs if self_ else scaled(y)
    lx = on(lagged(xs).reshape(times, p * lag))                    # T x p*L
    ly = on(lagged(ys))                                             # T x q x L
    target = on(ys[lag:])                                           # T x q

    if tile is None:
        # a tile holds b x T x p*L residual lags
        tile = max(1, (1 << 24) // max(1, times * p * lag))
    gain = torch.empty((p, q), dtype=torch.float64)
    rss_r = torch.empty(q, dtype=torch.float64)
    effect_cond = torch.empty(q, dtype=torch.float64)
    cause_cond = torch.empty((p, q), dtype=torch.float64)
    left = torch.empty((p, q), dtype=torch.float64)
    # each cause's lag columns' squared lengths, to measure what is left,
    # and their cross-products L'L per cause, for the fast form
    lx4 = lx.reshape(times, p, lag)
    l2 = (lx * lx).sum(dim=0).reshape(p, lag)
    ll = torch.einsum("tik,til->ikl", lx4, lx4)                     # p x L x L
    # the intercept column at unit length, like the scaled lag columns, so
    # that the design's condition number measures collinearity, not scale
    ones = torch.full((times, 1), times ** -0.5, dtype=torch.float64,
                      device=wd)
    eye = torch.eye(lag, dtype=torch.float64, device=wd)

    def solve_pairs(gmat, g):
        # gain g' G^{-1} g, G's condition number (NaN for a constant
        # series, infinite for exactly collinear lags), and how much of
        # the cause's lags is left beside the effect's own past
        sol, info = torch.linalg.solve_ex(gmat, g[..., None])
        gb = (g * sol[..., 0]).sum(dim=-1)
        gb = torch.where(info == 0, gb, torch.nan)
        finite = torch.isfinite(gmat).flatten(-2).all(dim=-1)
        ev = torch.linalg.eigvalsh(torch.where(finite[..., None, None],
                                               gmat, eye))          # ascending
        cond = torch.where(ev[..., 0] > 0, ev[..., -1] / ev[..., 0],
                           torch.inf)
        cond = torch.where(finite, cond, torch.nan)
        return gb, cond

    for j0 in range(0, q, tile):
        j1 = min(j0 + tile, q)
        b = j1 - j0
        # each effect's restricted design D = [1, own lags], orthonormalised
        design = torch.cat([ones.expand(b, times, 1),
                            ly[:, j0:j1].permute(1, 0, 2)], dim=2)  # b x T x m
        qmat, effect_cond[j0:j1] = _orthonormal_columns(design)
        yb = target[:, j0:j1].T                                     # b x T
        resid = yb - (qmat @ (qmat.transpose(1, 2) @ yb[:, :, None]))[:, :, 0]
        rss_b = (resid * resid).sum(dim=1)                          # b
        rss_r[j0:j1] = rss_b.cpu()
        qt = qmat.transpose(1, 2)                                   # b x m x T
        if fast is False:
            lft = torch.zeros((b, p), dtype=torch.float64, device=wd)
            redo = torch.ones((b, p), dtype=torch.bool, device=wd)
            gb = torch.empty((b, p), dtype=torch.float64, device=wd)
            cond = torch.empty((b, p), dtype=torch.float64, device=wd)
        else:
            # the fast form: everything from the few m x pL numbers Q'L and
            # the big product L'e. G = L'L - C'C and g = L'e - C'(Q'e)
            # with C = Q'L: in exact arithmetic M'M and M'e, but computed
            # by subtracting quantities of the size of L to leave what is
            # beside the effect's own past, which the explicit M does not
            c = (qt @ lx).reshape(b, -1, p, lag)                    # b x m x p x L
            qe = qt @ resid[:, :, None]                             # b x m x 1
            gmat = ll[None] - torch.einsum("bmik,bmil->bikl", c, c)
            g = (torch.einsum("tik,bt->bik", lx4, resid)
                 - torch.einsum("bmik,bm->bik", c, qe[..., 0]))
            gb, cond = solve_pairs(gmat, g)
            lft = (torch.diagonal(gmat, dim1=2, dim2=3) / l2[None]).amin(dim=2)
            # the pairs the fast form cannot be trusted on: see
            # FAST_SCORE_LIMIT. (A NaN score, a constant series, has no test
            # either way; a gain beyond the restricted RSS, rounding, is
            # amplified as far as the comparison can be.)
            amp = rss_b[:, None] / (rss_b[:, None] - gb.clamp(min=0))
            amp = torch.where(amp > 0, amp, torch.inf)
            # nothing left, or less than nothing (cancellation), is also
            # unresolved: a negative left would make the score negative
            score = torch.where(lft > 0, cond / lft * amp, torch.inf)
            redo = ~(score < FAST_SCORE_LIMIT) & ~torch.isnan(score)
            if fast is True:
                redo = torch.zeros_like(redo)
        if redo.any():
            # the residual lags M = L - Q Q'L formed explicitly, for the
            # flagged (effect, cause) pairs only, then g = M'e and G = M'M.
            # (L'e is the same in exact arithmetic, but e's rounding-sized
            # part along the effect's own past, times L, swamps it when
            # the effect is nearly predicted by its past.)
            bj, ci = torch.nonzero(redo, as_tuple=True)
            lsel = lx4[:, ci].permute(1, 0, 2)                      # k x T x L
            qsel = qmat[bj]                                         # k x T x m
            m = lsel - qsel @ (qsel.transpose(1, 2) @ lsel)
            esel = resid[bj]                                        # k x T
            g = torch.einsum("ktl,kt->kl", m, esel)
            gmat = torch.einsum("kta,ktl->kal", m, m)
            gb[bj, ci], cond[bj, ci] = solve_pairs(gmat, g)
            lft[bj, ci] = (torch.diagonal(gmat, dim1=1, dim2=2)
                           / l2[ci]).amin(dim=1)
        gain[:, j0:j1] = gb.T.cpu()
        cause_cond[:, j0:j1] = cond.T.cpu()
        left[:, j0:j1] = lft.T.cpu()

    return gain.numpy(), rss_r.numpy(), effect_cond.numpy(), \
        cause_cond.numpy(), left.numpy()


def granger(x, y=None, *, lag=1, tile=None, device=None) -> Granger:
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
        If some pairs could not be resolved in float64 (see Notes); their
        tests are NaN.

    Notes
    -----
    With ``T = n - lag`` usable times, the restricted model of effect ``j``
    has ``lag + 1`` coefficients and the unrestricted ``2 * lag + 1``, so
    ``F = ((RSS_r - RSS_u) / lag) / (RSS_u / (T - 2 * lag - 1))``.

    By the Frisch-Waugh theorem, ``RSS_r - RSS_u = g' G^{-1} g``, where,
    after projecting the effect's restricted design ``[1, own lags]`` out
    of everything (by an orthonormal basis ``Q_j`` of it: Cholesky-QR2,
    or Householder QR for designs too ill-conditioned for it), ``M = L -
    Q_j Q_j' L`` holds each cause's lags with the effect's own past
    removed, ``g = M' e_j`` is their product with the effect's residual,
    and ``G = M' M``. All of it is computed in float64, as batches of
    matrix products on the device where it has float64 (CUDA, the CPU),
    else on the CPU (Apple GPUs). Each series is first centred and scaled,
    which changes no F.

    For most pairs ``G`` and ``g`` come from a few small products, ``G =
    L'L - C'C`` and ``g = L'e - C'(Q'e)`` with ``C = Q_j' L``, which cost
    far less than forming ``M``. That form subtracts numbers of the size of
    the cause's lags to leave what is beside the effect's own past, so its
    rounding error grows with ``cond(G) / left`` (and with ``1 + F lag /
    df2``, as a large F compares the gain with a small remainder); where
    that product exceeds 1e6, which keeps the error below about 1e-9
    (measured), the pair is recomputed with ``M`` formed explicitly.
    Plain random walks, autoregressions, trends and white noise take the
    fast form for every pair; series integrated twice or more, and
    near-copies of the effect, mostly the explicit one.

    A test is NaN, with a warning, when it cannot be resolved in float64:
    when the effect's restricted design (intercept and lags, each at unit
    length) has condition number above 1e10, or a cause's ``G`` above
    1e12, or when less than 1e-20 of a cause's lags (in squared length) is
    left beside the effect's own past (a cause that is, to rounding, a
    linear function of it). White noise, near-unit-root series, random
    walks with or without drift and trends stay far below (5e4 and 3e5 at
    most, measured), with F accurate to about 1e-8; series integrated two
    or three times can exceed them at long lags, and should be differenced
    first, as Granger tests assume stationary series anyway. Exactly
    periodic series (a pure sinusoid) have collinear lags too.

    A cause that is very nearly a lagged copy of the effect (within noise
    of 1e-6 of its scale) gives an F of order 1e15, accurate only to about
    1%; its p-value is 0 either way.

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
    gain, rss_r, effect_cond, cause_cond, left = _granger_core(
        x, y, self_, lag, tile, pick_device(device))
    weak = (np.isfinite(effect_cond) & (effect_cond > EFFECT_LIMIT))[None, :] \
        | (cause_cond > CAUSE_LIMIT) | (left < LEFT_LIMIT)
    weak &= np.isfinite(gain) | np.isinf(cause_cond)
    if self_:
        np.fill_diagonal(gain, np.nan)
        np.fill_diagonal(weak, False)       # not a test anyway
    gain[weak] = np.nan
    if weak.any():
        warnings.warn(
            f"{int(weak.sum())} pair(s) are too ill-conditioned to resolve "
            "in float64: a cause almost exactly a linear function of the "
            "effect's own past, or nearly collinear lags (a series "
            "integrated twice or more: difference it; or an exactly "
            "periodic one); their tests are NaN",
            RuntimeWarning, stacklevel=2)
    # rounding can take a gain of 0 just below 0, or one equal to the
    # restricted RSS just above it
    gain = np.clip(gain, 0, rss_r[None, :])
    with np.errstate(invalid="ignore", divide="ignore"):
        rss_u = rss_r[None, :] - gain
        f = (gain / lag) / (rss_u / df2)
    return Granger(f, stats.f.sf(f, lag, df2), lag, (lag, df2))
