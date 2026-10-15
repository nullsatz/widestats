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
    assert _core.pick_device().type == want


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
