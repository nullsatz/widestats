"""The rest of the interface, beyond the agreement tests.

Spearman permutation tests, t tests without permutations, input handling,
device and precision choice, and every documented refusal.
"""

import numpy as np
import pytest
import torch
from scipy import stats

import widestats as ws
from widestats import _core
from widestats.perm import _permutations


@pytest.fixture
def x():
    return np.random.default_rng(10).standard_normal((20, 6))


# --- results the main suite does not reach -------------------------------

def test_cor_perm_spearman_matches_the_same_permutations():
    rng = np.random.default_rng(11)
    x = rng.exponential(size=(10, 4))
    y = rng.exponential(size=(10, 3))
    B = 200
    got = ws.cor_perm(x, y, B=B, method="spearman", seed=7, device="cpu",
                      dtype="float64")

    def rank(m):
        return np.apply_along_axis(stats.rankdata, 0, m)

    rx, ry = rank(x), rank(y)
    r0 = np.corrcoef(rx, ry, rowvar=False)[:4, 4:]
    perms = _permutations(10, B, 7).numpy()
    counts = sum(np.abs(np.corrcoef(rx, ry[pm], rowvar=False)[:4, 4:])
                 >= np.abs(r0) * (1 - 1e-10) for pm in perms)
    np.testing.assert_allclose(got.r, r0, atol=1e-12)
    np.testing.assert_array_equal(got.p_value, (counts + 1) / (B + 1))
    assert got.permutations == B


def test_ttest_without_permutations_is_scipys():
    rng = np.random.default_rng(12)
    x = rng.standard_normal((9, 5))
    group = [1, 1, 1, 1, 2, 2, 2, 2, 2]
    got = ws.ttest_perm(x, group)
    # Welch's by default, as R's t.test (SciPy's own default is pooled)
    want = stats.ttest_ind(x[:4], x[4:], equal_var=False)
    np.testing.assert_allclose(got.statistic, want.statistic, rtol=1e-12)
    np.testing.assert_allclose(got.df, want.df, rtol=1e-12)
    np.testing.assert_allclose(got.mean1, x[:4].mean(axis=0))
    np.testing.assert_allclose(got.mean2, x[4:].mean(axis=0))
    assert got.p_perm is None and got.p_adjusted is None


def test_same_seed_same_p_values_any_batch(x):
    a = ws.cor_perm(x, B=100, seed=3, batch=7, device="cpu", dtype="float64")
    b = ws.cor_perm(x, B=100, seed=3, batch=100, device="cpu",
                    dtype="float64")
    np.testing.assert_array_equal(a.p_value, b.p_value)
    np.testing.assert_array_equal(a.p_adjusted, b.p_adjusted)
    # with no seed, a fresh one each call: still valid p-values
    c = ws.cor_perm(x, B=50, device="cpu")
    assert np.nanmin(c.p_value) >= 1 / 51


def test_cor_pairs_empty_result_and_frame(x):
    none = ws.cor_pairs(x, threshold=0.999, device="cpu")
    assert len(none) == 0 and none.i.dtype == np.int64
    # one variable with itself: no pair at all, not even a tile to compute
    alone = ws.cor_pairs(x[:, :1], threshold=0, device="cpu")
    assert len(alone) == 0 and alone.j.dtype == np.int64
    pytest.importorskip("pandas")
    some = ws.cor_pairs(x, top=2, test=True, device="cpu")
    frame = some.to_frame()
    assert list(frame.columns) == ["i", "j", "r", "t", "p_value"]
    assert len(frame) == len(some) == 2 * x.shape[1]
    assert list(ws.cor_pairs(x, top=1, device="cpu").to_frame().columns) == \
        ["i", "j", "r"]


def test_a_perfect_correlation_tests_as_certain():
    v = np.arange(10.0)
    got = ws.cor_pairs(np.column_stack([v, 2 * v + 1]), threshold=0.5,
                       test=True, device="cpu", dtype="float64")
    assert got.r[0] == 1.0
    assert np.isinf(got.t[0]) and got.p_value[0] == 0.0


# --- inputs ---------------------------------------------------------------

def test_inputs_tensor_vector_list(x):
    want = ws.cor_pairs(x, top=1, device="cpu", dtype="float64")
    got = ws.cor_pairs(torch.from_numpy(x), top=1, device="cpu",
                       dtype="float64")
    np.testing.assert_array_equal(got.i, want.i)
    vec = ws.cor_pairs(x, x[:, 2], top=1, device="cpu")     # vector = 1 var
    assert vec.i[0] == 2 and len(vec) == 1
    assert len(ws.cor_pairs(x.tolist(), top=1, device="cpu")) == x.shape[1]


def test_as_matrix_refuses_more_than_two_dimensions():
    with pytest.raises(ValueError, match="matrix"):
        _core.as_matrix(np.zeros((2, 2, 2)))


# --- device and precision ------------------------------------------------

@pytest.mark.parametrize("cuda,mps,want", [(True, True, "cuda"),
                                           (False, True, "mps"),
                                           (False, False, "cpu")])
def test_pick_device_prefers_cuda_then_mps(monkeypatch, cuda, mps, want):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    monkeypatch.setattr(_core, "device_works", lambda kind: True)
    assert _core.pick_device().type == want


def test_pick_device_skips_a_reported_gpu_that_does_not_work(monkeypatch):
    # as on a CI virtual machine: MPS reported, unable to allocate
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(_core, "device_works", lambda kind: kind != "mps")
    assert _core.pick_device().type == "cpu"


def test_device_works():
    assert _core.device_works("cpu")
    # a device type this build of PyTorch cannot use at all
    assert not _core.device_works("xla")


@pytest.mark.parametrize("given,want", [(None, torch.float32),
                                        ("single", torch.float32),
                                        ("double", torch.float64),
                                        (torch.float64, torch.float64)])
def test_pick_dtype_names(given, want):
    assert _core.pick_dtype(given, torch.device("cpu")) == want


def test_pick_dtype_refuses_others():
    with pytest.raises(ValueError, match="float32 or float64"):
        _core.pick_dtype("float16", torch.device("cpu"))


# --- refusals ---------------------------------------------------------------

@pytest.mark.parametrize("kwargs,match", [
    ({"top": 0}, "positive"),
    ({"top": 1.5}, "positive"),
    ({"top": 1, "method": "kendall"}, "pearson"),
])
def test_cor_pairs_refuses(x, kwargs, match):
    with pytest.raises(ValueError, match=match):
        ws.cor_pairs(x, **kwargs)


def test_cor_pairs_refuses_mismatched_and_tiny(x):
    with pytest.raises(ValueError, match="rows"):
        ws.cor_pairs(x, x[:5], threshold=0.5)
    with pytest.raises(ValueError, match="3 rows"):
        ws.cor_pairs(x[:2], threshold=0.5)


@pytest.mark.parametrize("kwargs,match", [
    ({"B": 0}, "positive"),
    ({"B": 2.5}, "positive"),
    ({"method": "kendall"}, "pearson"),
])
def test_cor_perm_refuses(x, kwargs, match):
    with pytest.raises(ValueError, match=match):
        ws.cor_perm(x, **kwargs)


def test_cor_perm_refuses_mismatched_and_tiny(x):
    with pytest.raises(ValueError, match="rows"):
        ws.cor_perm(x, x[:5])
    with pytest.raises(ValueError, match="3 rows"):
        ws.cor_perm(x[:2])


@pytest.mark.parametrize("group,kwargs,match", [
    ([1] * 10 + [2] * 9, {}, "one entry per row"),
    ([1] * 7 + [2] * 7 + [3] * 6, {}, "exactly two levels"),
    ([1] + [2] * 19, {}, "at least 2"),
    ([1] * 10 + [2] * 10, {"B": -1}, "non-negative"),
])
def test_ttest_perm_refuses(x, group, kwargs, match):
    with pytest.raises(ValueError, match=match):
        ws.ttest_perm(x, group, **kwargs)


# --- regressions from the code review --------------------------------------

@pytest.mark.filterwarnings("ignore:Precision loss:RuntimeWarning")
def test_ttest_perm_extreme_separation_in_float32():
    # the groups differ by ~1e5 within-group sds: float32 cannot resolve the
    # within-group variance, which once made p_perm NaN
    rng = np.random.default_rng(20)
    g = np.repeat([0, 1], 10)
    x = (g + 1e-5 * rng.standard_normal(20))[:, None]
    x = np.column_stack([x, rng.standard_normal((20, 3))])
    res = ws.ttest_perm(x, g, B=999, seed=1, device="cpu", dtype="float32")
    assert np.isfinite(res.p_perm).all()
    assert res.p_perm[0] <= 3 / 1000 and res.p_adjusted[0] <= 3 / 1000
    # the null features are unaffected
    assert np.all(res.p_perm[1:] > 0.01)


@pytest.mark.filterwarnings("ignore:Precision loss:RuntimeWarning")
def test_ttest_perm_constant_within_groups():
    g = np.repeat([0, 1], 6)
    x = np.column_stack([g * 1.0, np.random.default_rng(21).normal(size=12)])
    res = ws.ttest_perm(x, g, B=500, seed=2, device="cpu", dtype="float64")
    assert np.isinf(res.statistic[0]) and res.p_value[0] == 0.0
    # only the labellings that separate the groups as completely reach it:
    # the observed one and its mirror, 2 of choose(12, 6) = 924
    assert np.isfinite(res.p_perm[0]) and res.p_perm[0] < 0.02
    assert res.p_adjusted[0] < 0.02


@pytest.mark.filterwarnings("ignore:Precision loss:RuntimeWarning")
def test_ttest_perm_constant_feature_has_no_test():
    x = np.column_stack([np.full(10, 3.0),
                         np.random.default_rng(22).normal(size=10)])
    res = ws.ttest_perm(x, np.repeat([0, 1], 5), B=50, seed=3, device="cpu")
    assert np.isnan(res.statistic[0])
    assert np.isnan(res.p_perm[0]) and np.isnan(res.p_adjusted[0])
    assert np.isfinite(res.p_perm[1])


def test_unseeded_runs_leave_the_global_generator_alone(x):
    torch.manual_seed(123)
    want = torch.rand(3)
    torch.manual_seed(123)
    ws.cor_perm(x, B=5, device="cpu")
    ws.ttest_perm(x, [0] * 10 + [1] * 10, B=5, device="cpu")
    torch.testing.assert_close(torch.rand(3), want)


def test_numpy_integer_seeds(x):
    a = ws.cor_perm(x, B=20, seed=np.int64(3), device="cpu", dtype="float64")
    b = ws.cor_perm(x, B=20, seed=3, device="cpu", dtype="float64")
    np.testing.assert_array_equal(a.p_value, b.p_value)


def test_permutations_are_permutations_for_large_n():
    perms = _permutations(50_000, 3, seed=4).numpy()
    for pm in perms:
        assert np.array_equal(np.sort(pm), np.arange(50_000))


def test_refuses_mixed_labels_and_infinities(x):
    mixed = np.array([1, "a"] * 10, dtype=object)
    with pytest.raises(ValueError, match="comparable"):
        ws.ttest_perm(x, mixed)
    y = x.copy()
    y[0, 0] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        ws.cor_pairs(y, threshold=0.5)


@pytest.mark.filterwarnings("ignore:Precision loss:RuntimeWarning")
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("n_each", [4, 5])
def test_ttest_perm_equal_groups_count_the_swap(dtype, n_each):
    # With equal groups, the swap of the observed labelling has the same
    # |t|; for strongly separated features it once fell a rounding error
    # short, halving p-values. Exact counts come from SciPy (two-pass
    # variances) on the same permutations.
    rng = np.random.default_rng(30)
    g = np.repeat([0, 1], n_each)
    # ten features at each separation: whether a swap rounds short varies
    seps = np.repeat([30, 1e3, 1e4, 1e5, 1e7], 10)
    x = np.column_stack([g * s + rng.standard_normal(2 * n_each)
                         for s in seps])
    B = 1000
    got = ws.ttest_perm(x, g, B=B, seed=6, device="cpu", dtype=dtype)
    t0 = stats.ttest_ind(x[g == 0], x[g == 1], equal_var=False).statistic
    perms = _permutations(2 * n_each, B, 6).numpy()
    tb = np.array([stats.ttest_ind(x[g[pm] == 0], x[g[pm] == 1],
                                   equal_var=False).statistic for pm in perms])
    want = ((np.abs(tb) >= np.abs(t0) * (1 - 1e-10)).sum(axis=0) + 1) / (B + 1)
    np.testing.assert_array_equal(got.p_perm, want)
    assert np.all(got.p_adjusted >= got.p_perm)


def test_refuses_fractional_seed(x):
    with pytest.raises(ValueError, match="whole number"):
        ws.cor_perm(x, B=5, seed=3.7)


@pytest.mark.filterwarnings("ignore:Precision loss:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_ttest_perm_exactly_constant_groups_in_maxt(dtype):
    # A 0/1 feature is constant within both groups under some labellings
    # (true t infinite); float32 once left it a rounding error of variance,
    # a finite t, so it missed a co-tested feature's infinite t in maxT.
    # the reviewer's construction, drawn in the same order
    rng = np.random.default_rng(0)
    n = 12
    g = np.repeat([0, 1], 6)
    rng.standard_normal(n)
    binary = (rng.random(n) < 0.5) * 1.0
    rng.standard_normal(n)
    separated = np.r_[np.zeros(6), np.ones(6)] * 1e6 + rng.standard_normal(n)
    x = np.column_stack([binary, separated])
    B = 2000
    got = ws.ttest_perm(x, g, B=B, seed=4, device="cpu", dtype=dtype)
    in1 = g == 0
    t0 = np.abs(stats.ttest_ind(x[in1], x[~in1], equal_var=False).statistic)
    perms = _permutations(n, B, 4).numpy()
    tb = np.abs(np.array([stats.ttest_ind(x[in1[pm]], x[~in1[pm]],
                                          equal_var=False).statistic
                          for pm in perms]))
    tb = np.where(np.isnan(tb), 0, tb)
    bar = t0 * (1 - 1e-10)
    want_p = ((tb >= bar).sum(axis=0) + 1) / (B + 1)
    want_adj = ((tb.max(axis=1)[:, None] >= bar).sum(axis=0) + 1) / (B + 1)
    if dtype == "float64":
        np.testing.assert_array_equal(got.p_perm, want_p)
        np.testing.assert_array_equal(got.p_adjusted, want_adj)
    else:
        # the separated feature, the regression, is exact
        assert got.p_perm[1] == want_p[1]
        assert got.p_adjusted[1] == want_adj[1]
        # the 0/1 feature's |t| = 1.1180340 is tied, to 7 digits, with the
        # separated feature's under some labellings; float32 cannot tell
        # them apart and counts them as reaching it: conservative
        assert np.all(got.p_perm >= want_p)
        assert np.all(got.p_adjusted >= want_adj)


def test_float64_solve_works():
    # the CPU can always; an Apple GPU cannot (no float64); CUDA can
    assert _core.float64_solve_works("cpu")
    if torch.backends.mps.is_available():
        assert not _core.float64_solve_works("mps")
    if torch.cuda.is_available():
        assert _core.float64_solve_works("cuda")
    assert not _core.float64_solve_works("xla")
