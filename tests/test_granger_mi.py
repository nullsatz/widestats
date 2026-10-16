"""Granger tests and mutual information against independent references.

Granger: both least-squares fits done by `numpy.linalg.lstsq` for every
pair, on the raw series, and statsmodels' own test where it is installed.
Mutual information: the B-spline weights by the Cox-de Boor recursion
written out from Daub et al. (2004), independent of SciPy's.
"""

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
    got = ws.granger(x, yy, lag=lag, tile=tile, device=device, dtype=dtype)
    f = np.full(got.f.shape, np.nan)
    p = np.full(got.f.shape, np.nan)
    for i in range(x.shape[1]):
        for j in range(effects.shape[1]):
            if not (self_ and i == j):
                f[i, j], p[i, j] = granger_ref(x[:, i], effects[:, j], lag)
    np.testing.assert_allclose(got.f, f, rtol=rtol(dtype))
    np.testing.assert_allclose(got.p_value, p, rtol=rtol(dtype),
                               atol=1e-12 if dtype == "float64" else 1e-6)
    assert got.df == (lag, x.shape[0] - lag - 2 * lag - 1)
    assert got.lag == lag
    if self_:
        assert np.all(np.isnan(np.diag(got.f)))


def test_granger_matches_statsmodels(series):
    tsa = pytest.importorskip("statsmodels.tsa.stattools")
    x, _ = series
    got = ws.granger(x, lag=2, device="cpu", dtype="float64")
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
    got = ws.granger(x, lag=1, device="cpu", dtype="float64")
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

@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_granger_flags_a_cause_too_close_to_the_effect(dtype):
    # cause = effect + 1e-6 noise: float32 once returned F ~ 3e-6 (p ~ 1)
    # silently; the residual cross-products are below what it resolves
    rng = np.random.default_rng(50)
    e = rng.standard_normal(200).cumsum()
    x = np.column_stack([e, e + 1e-6 * rng.standard_normal(200),
                         rng.standard_normal(200)])
    with pytest.warns(RuntimeWarning, match="too close to resolve"):
        got = ws.granger(x, lag=2, device="cpu", dtype=dtype)
    assert np.isnan(got.f[1, 0])
    # an ordinary pair is unaffected
    want, _ = granger_ref(x[:, 2], x[:, 0], 2)
    assert got.f[2, 0] == pytest.approx(want, rel=rtol(dtype))


def test_granger_cause_equal_to_an_effect_is_nan():
    rng = np.random.default_rng(51)
    x = rng.standard_normal((150, 3)).cumsum(axis=0)
    with pytest.warns(RuntimeWarning, match="too close to resolve"):
        got = ws.granger(x, x[:, [1]], lag=2, device="cpu", dtype="float64")
    assert np.isnan(got.f[1, 0]) and np.isfinite(got.f[0, 0])


def test_granger_long_series():
    # long series: nothing T x T is formed (the QR once was, on Apple GPUs)
    rng = np.random.default_rng(52)
    x = rng.standard_normal((50_000, 3)).cumsum(axis=0)
    x[1:, 2] += 0.05 * x[:-1, 0]
    got = ws.granger(x, lag=3, device="cpu", dtype="float32")
    want, _ = granger_ref(x[:, 0], x[:, 2], 3)
    assert got.f[0, 2] == pytest.approx(want, rel=rtol("float32"))


def test_mutual_info_constant_variable_is_exactly_zero_in_float32(mi_data):
    # once a rounding error of ~1e-7 in float32, against a documented 0
    x, y = mi_data
    for k in range(x.shape[1]):
        xk = x.copy()
        xk[:, k] = -7.0
        got = ws.mutual_info(xk, y, device="cpu", dtype="float32")
        assert np.all(got[k] == 0.0), k
