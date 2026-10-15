"""widestats against NumPy and SciPy.

The same pairs, correlations, t tests and permutation p-values, on the CPU
in float64 and float32 and on whatever GPU is present.
"""

import numpy as np
import pytest
import torch
from scipy import stats

import widestats as ws
from widestats.perm import _permutations

DEVICES = [("cpu", "float64"), ("cpu", "float32")]
if torch.cuda.is_available():
    DEVICES += [("cuda", "float64"), ("cuda", "float32")]
if torch.backends.mps.is_available():
    DEVICES += [("mps", "float32")]


def tol(dtype):
    return 1e-10 if dtype == "float64" else 1e-4


def ref_cor(x, y, method):
    if method == "spearman":
        x = np.apply_along_axis(stats.rankdata, 0, x)
        y = np.apply_along_axis(stats.rankdata, 0, y)
    p = x.shape[1]
    return np.corrcoef(x, y, rowvar=False)[:p, p:]


@pytest.fixture
def data():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((40, 30))
    x[:, 5] = x[:, 4] + 0.2 * rng.standard_normal(40)
    x[:, 20] = -x[:, 3] + 0.2 * rng.standard_normal(40)
    y = rng.standard_normal((40, 17))
    y[:, 2] = x[:, 9] + 0.3 * rng.standard_normal(40)
    return x, y


@pytest.mark.parametrize("device,dtype", DEVICES)
@pytest.mark.parametrize("method", ["pearson", "spearman"])
@pytest.mark.parametrize("self_", [True, False])
@pytest.mark.parametrize("absolute", [True, False])
@pytest.mark.parametrize("tile", [1, 7, 2048])
def test_threshold_pairs(data, device, dtype, method, self_, absolute, tile):
    x, y = data
    yy = None if self_ else y
    r = ref_cor(x, x if self_ else y, method)
    s = np.abs(r) if absolute else r
    want = s >= 0.3
    if self_:
        want &= np.triu(np.ones_like(want), 1).astype(bool)
    edge = np.abs(s - 0.3) < 1e-4                  # either way in float32
    got = ws.cor_pairs(x, yy, threshold=0.3, method=method,
                       absolute=absolute, tile=tile, device=device,
                       dtype=dtype)
    have = np.zeros_like(want)
    have[got.i, got.j] = True
    assert np.array_equal(have & ~edge, want & ~edge)
    np.testing.assert_allclose(got.r, r[got.i, got.j], atol=tol(dtype))
    assert np.all(np.lexsort((got.i, got.j)) == np.arange(len(got)))
    if self_:
        assert np.all(got.i < got.j)


@pytest.mark.parametrize("device,dtype", DEVICES)
@pytest.mark.parametrize("self_", [True, False])
@pytest.mark.parametrize("absolute", [True, False])
@pytest.mark.parametrize("k,tile", [(1, 2048), (3, 5), (3, 1), (40, 8)])
def test_top_pairs(data, device, dtype, self_, absolute, k, tile):
    x, y = data
    r = ref_cor(x, x if self_ else y, "pearson")
    s = np.abs(r) if absolute else r.copy()
    if self_:
        np.fill_diagonal(s, -np.inf)
    got = ws.cor_pairs(x, None if self_ else y, top=k, absolute=absolute,
                       tile=tile, device=device, dtype=dtype)
    kk = min(k, x.shape[1] - self_)
    assert np.array_equal(np.bincount(got.j, minlength=s.shape[1]),
                          np.full(s.shape[1], kk))
    for j in range(s.shape[1]):
        sel = got.j == j
        gs = np.abs(got.r[sel]) if absolute else got.r[sel]
        np.testing.assert_allclose(gs, np.sort(s[:, j])[::-1][:kk],
                                   atol=tol(dtype))
        if self_:
            assert not np.any(got.i[sel] == j)


def test_pairs_test_and_refusals(data):
    x, _ = data
    got = ws.cor_pairs(x, threshold=0.5, test=True, device="cpu",
                       dtype="float64")
    pr = stats.pearsonr(x[:, got.i[0]], x[:, got.j[0]])
    assert got.p_value[0] == pytest.approx(pr.pvalue, rel=1e-8)
    x2 = x.copy()
    x2[:, 3] = 1.0
    every = ws.cor_pairs(x2, threshold=0, device="cpu", dtype="float64")
    assert not np.any((every.i == 3) | (every.j == 3))
    assert len(every) == 29 * 28 // 2
    with pytest.raises(ValueError, match="exactly one"):
        ws.cor_pairs(x)
    with pytest.raises(ValueError, match="more than 3"):
        ws.cor_pairs(x, threshold=0, limit=3, device="cpu")
    with pytest.raises(ValueError, match="missing"):
        x2[0, 0] = np.nan
        ws.cor_pairs(x2, threshold=0.5)


def ref_cor_perm(x, y, B, seed):
    self_ = y is None
    yy = x if self_ else y
    perms = _permutations(x.shape[0], B, seed).numpy()
    r0 = ref_cor(x, yy, "pearson")
    if self_:
        np.fill_diagonal(r0, np.nan)
    counts = np.zeros_like(r0)
    maxima = np.empty(B)
    for b in range(B):
        rb = np.abs(ref_cor(x, yy[perms[b]], "pearson"))
        if self_:
            np.fill_diagonal(rb, 0)
        counts += rb >= np.abs(r0) * (1 - 1e-10)
        maxima[b] = rb.max()
    adj = np.array([(np.sum(maxima >= a * (1 - 1e-10)) + 1) / (B + 1)
                    for a in np.abs(r0).ravel()]).reshape(r0.shape)
    p = (counts + 1) / (B + 1)
    p[np.isnan(r0)] = np.nan
    adj[np.isnan(r0)] = np.nan
    return r0, p, adj


@pytest.mark.parametrize("device,dtype", DEVICES)
@pytest.mark.parametrize("self_", [True, False])
def test_cor_perm(data, device, dtype, self_):
    x, y = data
    x, y = np.exp(x[:12, :9]), np.exp(y[:12, :7])      # small, skewed
    yy = None if self_ else y
    r0, p, adj = ref_cor_perm(x, yy, 300, seed=5)
    got = ws.cor_perm(x, yy, B=300, seed=5, batch=64, device=device,
                      dtype=dtype)
    np.testing.assert_allclose(got.r, r0, atol=tol(dtype))
    slack = 0 if dtype == "float64" else 3 / 301
    assert np.nanmax(np.abs(got.p_value - p)) <= slack + 1e-12
    assert np.nanmax(np.abs(got.p_adjusted - adj)) <= slack + 1e-12
    assert np.all(np.isnan(got.p_value) == np.isnan(p))


def ref_ttest_perm(x, in1, B, seed, equal_var):
    t0 = stats.ttest_ind(x[in1], x[~in1], equal_var=equal_var).statistic
    perms = _permutations(x.shape[0], B, seed).numpy()
    tb = np.array([stats.ttest_ind(x[in1[pm]], x[~in1[pm]],
                                   equal_var=equal_var).statistic
                   for pm in perms])
    bar = np.abs(t0) * (1 - 1e-10)
    p = ((np.abs(tb) >= bar).sum(axis=0) + 1) / (B + 1)
    maxima = np.abs(tb).max(axis=1)
    adj = np.array([(np.sum(maxima >= b) + 1) / (B + 1) for b in bar])
    return p, adj


@pytest.mark.parametrize("device,dtype", DEVICES)
@pytest.mark.parametrize("equal_var", [False, True])
def test_ttest_perm(device, dtype, equal_var):
    rng = np.random.default_rng(2)
    x = rng.exponential(10, (11, 15)) + 100           # far from 0, skewed
    group = np.array(list("babbaababab"))
    in1 = group == "a"
    x[in1, 1] += 40
    got = ws.ttest_perm(x, group, equal_var=equal_var, B=200, seed=3,
                        batch=64, device=device, dtype=dtype)
    want = stats.ttest_ind(x[in1], x[~in1], equal_var=equal_var)
    np.testing.assert_allclose(got.statistic, want.statistic, rtol=1e-12)
    np.testing.assert_allclose(got.p_value, want.pvalue, rtol=1e-12)
    p, adj = ref_ttest_perm(x, in1, 200, 3, equal_var)
    slack = 0 if dtype == "float64" else 2 / 201
    assert np.max(np.abs(got.p_perm - p)) <= slack + 1e-12
    assert np.max(np.abs(got.p_adjusted - adj)) <= slack + 1e-12
    assert np.all(got.p_adjusted >= got.p_perm)


def test_rank_columns_matches_scipy():
    rng = np.random.default_rng(4)
    for n in (1, 2, 7, 64, 1000):
        x = np.column_stack([rng.integers(0, 5, n), rng.standard_normal(n),
                             np.round(rng.standard_normal(n), 1)])
        np.testing.assert_array_equal(
            ws._core.rank_columns(x.astype(float)),
            np.apply_along_axis(stats.rankdata, 0, x))


def test_mps_refuses_float64():
    with pytest.raises(ValueError, match="no float64"):
        ws._core.pick_dtype("float64", torch.device("mps"))
