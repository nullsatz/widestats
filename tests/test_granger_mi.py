"""Granger tests and mutual information against independent references.

Granger: both least-squares fits done by `numpy.linalg.lstsq` for every
pair, on the raw series, and statsmodels' own test where it is installed.
Mutual information: the B-spline weights by the Cox-de Boor recursion
written out from Daub et al. (2004), independent of SciPy's.
"""

import warnings

import numpy as np
import pytest
import torch
from scipy import stats

import widestats as ws

DEVICES = [("cpu", "float64"), ("cpu", "float32")]
if torch.cuda.is_available() and ws._core.device_works("cuda"):
    DEVICES += [("cuda", "float64"), ("cuda", "float32")]
if torch.backends.mps.is_available() and ws._core.device_works("mps"):
    DEVICES += [("mps", "float32")]


def rtol(dtype):
    return 1e-8 if dtype == "float64" else 2e-3


# --- Granger ----------------------------------------------------------------

def granger_ref(cause, effect, lag):
    """F and p of `cause` Granger-causing `effect`, by two lstsq fits."""
    n = len(effect)
    rows = np.arange(lag, n)
    target = effect[rows]

    def lags(v):
        return np.column_stack([v[rows - k] for k in range(1, lag + 1)])

    def rss(a):
        return np.sum((target - a @ np.linalg.lstsq(a, target,
                                                    rcond=None)[0])**2)

    restricted = np.column_stack([np.ones(len(rows)), lags(effect)])
    unrestricted = np.column_stack([restricted, lags(cause)])
    rss_r, rss_u = rss(restricted), rss(unrestricted)
    df2 = len(rows) - 2 * lag - 1
    f = ((rss_r - rss_u) / lag) / (rss_u / df2)
    return f, stats.f.sf(f, lag, df2)


@pytest.fixture
def series():
    rng = np.random.default_rng(40)
    x = rng.standard_normal((120, 6)).cumsum(axis=0) * 0.1 + 5  # wandering
    x[2:, 3] += 0.7 * x[:-2, 1]                                 # 1 -> 3
    x[1:, 5] -= 0.5 * x[:-1, 0]                                 # 0 -> 5
    y = rng.standard_normal((120, 4))
    y[1:, 2] += 0.9 * x[:-1, 4]                                 # x4 -> y2
    return x, y


@pytest.mark.parametrize("device,dtype", DEVICES)
@pytest.mark.parametrize("lag", [1, 2, 3])
@pytest.mark.parametrize("self_", [True, False])
@pytest.mark.parametrize("tile", [None, 1, 3])
def test_granger_matches_lstsq(series, device, dtype, lag, self_, tile):
    x, y = series
    yy = None if self_ else y
    effects = x if self_ else y
    # granger computes in float64 on any device (the CPU for Apple GPUs)
    got = ws.granger(x, yy, lag=lag, tile=tile, device=device)
    f = np.full(got.f.shape, np.nan)
    p = np.full(got.f.shape, np.nan)
    for i in range(x.shape[1]):
        for j in range(effects.shape[1]):
            if not (self_ and i == j):
                f[i, j], p[i, j] = granger_ref(x[:, i], effects[:, j], lag)
    np.testing.assert_allclose(got.f, f, rtol=1e-8)
    np.testing.assert_allclose(got.p_value, p, rtol=1e-8, atol=1e-12)
    assert got.df == (lag, x.shape[0] - lag - 2 * lag - 1)
    assert got.lag == lag
    if self_:
        assert np.all(np.isnan(np.diag(got.f)))


def test_granger_matches_statsmodels(series):
    tsa = pytest.importorskip("statsmodels.tsa.stattools")
    x, _ = series
    got = ws.granger(x, lag=2, device="cpu")
    for i, j in [(1, 3), (0, 5), (2, 4)]:
        # statsmodels tests whether the second column causes the first
        res = tsa.grangercausalitytests(x[:, [j, i]], maxlag=[2])
        f, p, _, _ = res[2][0]["ssr_ftest"]
        assert got.f[i, j] == pytest.approx(f, rel=1e-8)
        assert got.p_value[i, j] == pytest.approx(p, rel=1e-8)


@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
def test_granger_constant_series_and_refusals(series):
    x, _ = series
    x = x.copy()
    x[:, 2] = 3.0
    got = ws.granger(x, lag=1, device="cpu")
    assert np.all(np.isnan(got.f[2])) and np.all(np.isnan(got.f[:, 2]))
    assert np.isfinite(got.f[0, 1])
    with pytest.raises(ValueError, match="positive"):
        ws.granger(x, lag=0)
    with pytest.raises(ValueError, match="positive"):
        ws.granger(x, lag=1.5)
    with pytest.raises(ValueError, match="too short"):
        ws.granger(x[:4], lag=1)
    with pytest.raises(ValueError, match="rows"):
        ws.granger(x, x[:50])


# --- mutual information ------------------------------------------------------

def daub_weights(v, bins, order):
    """Compute one variable's B-spline weights, from Daub et al. (2004).

    Scaled onto [0, bins - order + 1]; Cox-de Boor recursion on the
    clamped knots.
    """
    span = bins - order + 1
    knots = np.r_[np.zeros(order), np.arange(1, span), np.full(order, span)]
    lo, hi = v.min(), v.max()
    z = (v - lo) / (hi - lo) * span if hi > lo else np.zeros_like(v)

    def basis(i, k):
        if k == 1:
            inside = (knots[i] <= z) & (z < knots[i + 1])
            # the right end of the domain belongs to the last non-empty
            # interval
            last = ((z == knots[-1]) & (knots[i + 1] == knots[-1])
                    & (knots[i] < knots[i + 1]))
            return (inside | last).astype(float)
        out = np.zeros_like(z)
        if knots[i + k - 1] > knots[i]:
            out += ((z - knots[i]) / (knots[i + k - 1] - knots[i])
                    * basis(i, k - 1))
        if knots[i + k] > knots[i + 1]:
            out += ((knots[i + k] - z) / (knots[i + k] - knots[i + 1])
                    * basis(i + 1, k - 1))
        return out

    return np.column_stack([basis(i, order) for i in range(bins)])


def entropy_bits(prob):
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob))


def daub_mi(u, v, bins, order):
    wu, wv = daub_weights(u, bins, order), daub_weights(v, bins, order)
    return (entropy_bits(wu.mean(axis=0)) + entropy_bits(wv.mean(axis=0))
            - entropy_bits(wu.T @ wv / len(u)))


@pytest.fixture
def mi_data():
    rng = np.random.default_rng(41)
    u = rng.uniform(-1, 1, 300)
    x = np.column_stack([u, u**2, np.sin(3 * u), rng.standard_normal(300),
                         np.round(rng.uniform(0, 4, 300))])   # ties too
    y = np.column_stack([np.exp(u), rng.uniform(size=300), -u])
    return x, y


@pytest.mark.parametrize("bins,order", [(10, 3), (5, 1), (6, 2), (8, 4),
                                        (4, 4)])
def test_bspline_weights_match_cox_de_boor(mi_data, bins, order):
    x, _ = mi_data
    w = ws.mi.bspline_weights(x, bins, order)
    for i in range(x.shape[1]):
        np.testing.assert_allclose(w[:, i], daub_weights(x[:, i], bins,
                                                         order), atol=1e-12)
    np.testing.assert_allclose(w.sum(axis=2), 1.0)


@pytest.mark.parametrize("device,dtype", DEVICES)
@pytest.mark.parametrize("bins,order", [(10, 3), (5, 1), (6, 2)])
@pytest.mark.parametrize("self_", [True, False])
@pytest.mark.parametrize("tile", [None, 2])
def test_mutual_info_matches_daub(mi_data, device, dtype, bins, order,
                                  self_, tile):
    x, y = mi_data
    yy = x if self_ else y
    got = ws.mutual_info(x, None if self_ else y, bins=bins, order=order,
                         tile=tile, device=device, dtype=dtype)
    want = np.array([[daub_mi(x[:, i], yy[:, j], bins, order)
                      for j in range(yy.shape[1])]
                     for i in range(x.shape[1])])
    np.testing.assert_allclose(got, np.maximum(want, 0), rtol=rtol(dtype),
                               atol=1e-10 if dtype == "float64" else 1e-5)
    if self_:
        # a variable with itself: its entropy for plain binning; less for
        # higher orders, whose weights blur the joint distribution
        ent = np.array([entropy_bits(daub_weights(x[:, i], bins,
                                                  order).mean(axis=0))
                        for i in range(x.shape[1])])
        if order == 1:
            np.testing.assert_allclose(np.diag(got), ent, rtol=rtol(dtype))
        else:
            assert np.all(np.diag(got) < ent)


def test_mutual_info_constant_variable_and_refusals(mi_data):
    x, _ = mi_data
    x = x.copy()
    x[:, 3] = 2.0
    got = ws.mutual_info(x, device="cpu", dtype="float64")
    np.testing.assert_allclose(got[3], 0.0, atol=1e-12)
    np.testing.assert_allclose(got[:, 3], 0.0, atol=1e-12)
    assert np.all(got >= 0)
    for bins, order in [(3, 4), (5, 0), (5.5, 2), (5, 2.5)]:
        with pytest.raises(ValueError, match="order"):
            ws.mutual_info(x, bins=bins, order=order)
    with pytest.raises(ValueError, match="2 rows"):
        ws.mutual_info(x[:1])
    with pytest.raises(ValueError, match="rows"):
        ws.mutual_info(x, x[:10])


# --- regressions from the code review -----------------------------------------

def test_granger_resolves_a_cause_nearly_equal_to_the_effect():
    # cause = effect + 1e-6 noise: once silently wrong (F ~ 3e-6) in
    # float32, then flagged; with the residual lags formed explicitly in
    # float64, the 1e-6 that is left is resolved accurately
    rng = np.random.default_rng(50)
    e = rng.standard_normal(200).cumsum()
    x = np.column_stack([e, e + 1e-6 * rng.standard_normal(200),
                         rng.standard_normal(200)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = ws.granger(x, lag=2, device="cpu")
    for i, j in [(1, 0), (2, 0)]:
        want, _ = granger_ref(x[:, i], x[:, j], 2)
        assert got.f[i, j] == pytest.approx(want, rel=1e-6)


def test_granger_cause_equal_to_an_effect_is_nan():
    rng = np.random.default_rng(51)
    x = rng.standard_normal((150, 3)).cumsum(axis=0)
    # the cause is the effect: nothing of its lags is left beside them
    with pytest.warns(RuntimeWarning, match="ill-conditioned"):
        got = ws.granger(x, x[:, [1]], lag=2, device="cpu")
    assert np.isnan(got.f[1, 0]) and np.isfinite(got.f[0, 0])


def test_granger_long_series():
    # long series: nothing T x T is formed (the QR once was, on Apple GPUs)
    rng = np.random.default_rng(52)
    x = rng.standard_normal((50_000, 3)).cumsum(axis=0)
    x[1:, 2] += 0.05 * x[:-1, 0]
    got = ws.granger(x, lag=3, device="cpu")
    want, _ = granger_ref(x[:, 0], x[:, 2], 3)
    assert got.f[0, 2] == pytest.approx(want, rel=1e-6)


def test_mutual_info_constant_variable_is_exactly_zero_in_float32(mi_data):
    # once a rounding error of ~1e-7 in float32, against a documented 0
    x, y = mi_data
    for k in range(x.shape[1]):
        xk = x.copy()
        xk[:, k] = -7.0
        got = ws.mutual_info(xk, y, device="cpu", dtype="float32")
        assert np.all(got[k] == 0.0), k


def test_granger_is_accurate_on_long_random_walks():
    # float32 was off by >10% in F on long random walks; granger computes
    # in float64, accurate to lstsq's precision
    rng = np.random.default_rng(54)
    x = rng.standard_normal((50_000, 2)).cumsum(axis=0)
    got = ws.granger(x, lag=1, device="cpu")
    want, _ = granger_ref(x[:, 0], x[:, 1], 1)
    assert got.f[0, 1] == pytest.approx(want, rel=1e-6)


def test_mutual_info_nearly_empty_bin_is_finite_in_float32():
    # one sample just entering an otherwise empty bin: the product of the
    # bin's marginals underflowed in float32 and MI came out infinite
    rng = np.random.default_rng(0)
    n = 1000
    z = rng.uniform(0, 0.5, n)
    z[0], z[1], z[2] = 0, 8, 4 + 1.5e-10
    v = z / 8
    x = np.column_stack([v, v, rng.standard_normal(n)])
    got = ws.mutual_info(x, device="cpu", dtype="float32")
    want = ws.mutual_info(x, device="cpu", dtype="float64")
    assert np.all(np.isfinite(got))
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("lag", [1, 3, 5])
@pytest.mark.parametrize("n", [500, 50_000])
def test_orthonormal_columns_matches_householder(lag, n):
    # Cholesky-QR2 against Householder QR on random-walk lag designs, whose
    # columns are nearly collinear: the same projection, orthonormal Q
    from widestats.granger import _orthonormal_columns
    rng = np.random.default_rng(55)
    walks = rng.standard_normal((n + lag, 3)).cumsum(axis=0)
    a = np.stack([np.column_stack([np.ones(n)] + [w[lag - k:n + lag - k]
                                                  for k in range(1, lag + 1)])
                  for w in walks.T])
    t = torch.tensor(a)
    q, cond = _orthonormal_columns(t)
    assert torch.all(cond < 1e5)            # random walks: well conditioned
    qq = torch.linalg.qr(t).Q
    eye = torch.eye(lag + 1, dtype=torch.float64).expand(3, -1, -1)
    # Q'Q sums n products, so its rounding grows with n (about n * eps);
    # 1e-12 was just exceeded at n = 50,000 on Linux's BLAS (1.08e-12)
    torch.testing.assert_close(q.transpose(1, 2) @ q, eye, atol=1e-10, rtol=0)
    # same span: projecting a vector onto either basis gives the same result
    v = torch.tensor(rng.standard_normal((3, n, 1)))
    torch.testing.assert_close(q @ (q.transpose(1, 2) @ v),
                               qq @ (qq.transpose(1, 2) @ v),
                               atol=1e-9, rtol=0)


def test_orthonormal_columns_singular_and_nan_designs():
    from widestats.granger import _orthonormal_columns
    a = torch.ones((3, 10, 2), dtype=torch.float64)
    a[1, :, 1] = torch.arange(10.0)                       # a good design
    a[2, :, 1] = torch.nan                                # a constant series
    q, cond = _orthonormal_columns(a)
    assert cond[0] > 1e12                  # two equal columns: singular
    assert torch.isfinite(q[1]).all() and cond[1] < 100
    assert torch.isnan(q[2]).any() and torch.isinf(cond[2])


@pytest.mark.parametrize("bins", [5, 10, 20, 30])
def test_bspline_weights_float32_output_from_float64_data(bins):
    # discrete data puts values exactly on knots; the weights are computed
    # in float64 from the float64 data and only then cast, so no value
    # moves bin (computing in float32 once moved them, MI off ~0.35 bits)
    from widestats.mi import _bspline_weights
    x = np.tile(np.arange(bins + 1) * 0.1, 3)[:, None]
    for order in range(1, min(bins, 4) + 1):
        w64 = _bspline_weights(torch.tensor(x), bins, order)
        w32 = _bspline_weights(torch.tensor(x), bins, order, torch.float32)
        assert w32.dtype == torch.float32 and w64.dtype == torch.float64
        np.testing.assert_allclose(w32.numpy(), w64.numpy(), atol=1e-6)


def test_mutual_info_discrete_data_float32_matches_float64():
    rng = np.random.default_rng(56)
    v = rng.integers(0, 31, 2000) * 0.1
    x = np.column_stack([v, v, rng.integers(0, 31, 2000) * 0.1])
    a = ws.mutual_info(x, bins=30, order=1, device="cpu", dtype="float32")
    b = ws.mutual_info(x, bins=30, order=1, device="cpu", dtype="float64")
    np.testing.assert_allclose(a, b, atol=1e-4)


@pytest.mark.parametrize("k,n,lag", [(2, 5000, 5), (2, 50_000, 5),
                                     (3, 5000, 5), (3, 300, 2)])
def test_granger_on_integrated_series_is_never_silently_wrong(k, n, lag):
    # series integrated twice or three times: the effect's own lags are
    # nearly collinear (condition up to 1e12), and float64 once returned
    # F = 1.7 where the answer is 2.88. Every pair must now be either
    # flagged (NaN, with a warning) or within 0.1% of least squares.
    rng = np.random.default_rng(1)
    x = rng.standard_normal((n, 3))
    for _ in range(k):
        x = x.cumsum(axis=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = ws.granger(x, lag=lag, device="cpu")
    off = ~np.eye(3, dtype=bool)
    if np.isnan(got.f[off]).any():
        assert any("difference it" in str(w.message) for w in caught)
    for i, j in zip(*np.nonzero(off & np.isfinite(got.f)), strict=True):
        want, _ = granger_ref(x[:, i], x[:, j], lag)
        assert got.f[i, j] == pytest.approx(want, rel=1e-3), (i, j)


def test_granger_integrated_cases_check_some_pairs():
    # the test above is not vacuous: in these cases some pairs survive
    rng = np.random.default_rng(1)
    x = rng.standard_normal((300, 3)).cumsum(axis=0).cumsum(axis=0)
    got = ws.granger(x, lag=2, device="cpu")
    assert np.isfinite(got.f[~np.eye(3, dtype=bool)]).any()


@pytest.mark.parametrize("kind", ["drift", "trend"])
def test_granger_does_not_flag_drifting_walks_or_trends(kind):
    # these were once all NaN: the intercept's scale had inflated the
    # condition number past the cut. They are accurate and kept.
    rng = np.random.default_rng(58)
    n = 50_000
    if kind == "drift":
        x = (1 + rng.standard_normal((n, 3))).cumsum(axis=0)
    else:
        x = 0.01 * np.arange(n)[:, None] + rng.standard_normal((n, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("error")       # no "too close" warning
        got = ws.granger(x, lag=3, device="cpu")
    want, _ = granger_ref(x[:, 0], x[:, 1], 3)
    assert got.f[0, 1] == pytest.approx(want, rel=1e-4)


def test_granger_constant_series_gives_no_warning():
    rng = np.random.default_rng(59)
    x = rng.standard_normal((200, 3)).cumsum(axis=0)
    x[:, 1] = 4.0
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with np.errstate(invalid="ignore", divide="ignore"):
            got = ws.granger(x, lag=2, device="cpu")
    assert np.isnan(got.f[1]).all() and np.isnan(got.f[:, 1]).all()
    assert np.isfinite(got.f[0, 2])


def test_orthonormal_columns_falls_back_for_ill_conditioned_designs():
    from widestats.granger import _orthonormal_columns
    rng = np.random.default_rng(57)
    n = 5000
    w = rng.standard_normal(n + 5).cumsum().cumsum().cumsum()    # I(3)
    a = np.column_stack([np.ones(n)] + [w[5 - k:n + 5 - k]
                                        for k in range(1, 6)])[None]
    q, cond = _orthonormal_columns(torch.tensor(a))
    assert cond[0] > 1e7                        # Cholesky-QR2 would fail
    qq = torch.linalg.qr(torch.tensor(a)).Q
    torch.testing.assert_close(q @ q.transpose(1, 2),
                               qq @ qq.transpose(1, 2), atol=1e-8, rtol=0)


@pytest.mark.parametrize("panel,n,lag", [("mixed", 30_000, 4),
                                         ("I2 drift", 30_000, 2)])
def test_granger_mixed_and_drifting_integrated_panels(panel, n, lag):
    # once 2-7% off without a flag: I(2) causes acting on I(1) effects
    # (only the effect's conditioning was checked), and I(2) walks with
    # drift. Every pair must be within 1e-4 of least squares, or flagged.
    rng = np.random.default_rng(60)
    if panel == "mixed":
        x = np.column_stack([rng.standard_normal(n).cumsum(),
                             rng.standard_normal(n).cumsum().cumsum(),
                             rng.standard_normal(n).cumsum(),
                             rng.standard_normal(n).cumsum().cumsum()])
    else:
        x = (1 + rng.standard_normal((n, 3))).cumsum(axis=0).cumsum(axis=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = ws.granger(x, lag=lag, device="cpu")
    p = x.shape[1]
    off = ~np.eye(p, dtype=bool)
    if np.isnan(got.f[off]).any():
        assert any("ill-conditioned" in str(w.message) for w in caught)
    checked = 0
    for i, j in zip(*np.nonzero(off & np.isfinite(got.f)), strict=True):
        want, _ = granger_ref(x[:, i], x[:, j], lag)
        assert got.f[i, j] == pytest.approx(want, rel=1e-4), (i, j)
        checked += 1
    assert checked > 0
