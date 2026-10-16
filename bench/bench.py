"""Benchmark widestats on a GPU against standard CPU tools.

For every function and size this times

- widestats on the GPU (float32, and float64 where it matters),
- widestats on the CPU (the same algorithm, PyTorch on the CPU), so that
  what the GPU buys is separate from what the algorithm buys, and
- the CPU tools people use for the job: NumPy on a multithreaded BLAS,
  statsmodels, CorALS.

Before a baseline's time is recorded, its answer is compared with the timed
widestats GPU result (and the widestats CPU result), within what float32
allows. Every implementation is timed the same way: a run, and if it took
under 10 s a warm-up's worth of further runs, reporting the median of three
(a single run for anything slower). GPU timings synchronise the device.
Results go to ``<out>/results.csv`` and machine details to
``<out>/machine.txt``; ``bench/report.py`` makes the table.

    python bench/bench.py [--out bench/results] [--quick] [--only f,g]
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import torch

import widestats as ws
from widestats.mi import bspline_weights
from widestats.perm import _permutations

GPU = "cuda" if torch.cuda.is_available() else None
ROWS = []


def sync():
    """Wait for the GPU, so that a timing includes its work."""
    if GPU:
        torch.cuda.synchronize()


def measure(f):
    """Time `f()` the same way for every implementation.

    One run; if it took under 10 s, three more, and the median of those
    (so a fast implementation is timed warm, as it would be in use).
    Returns the time and the result of the last run.
    """
    t = time.perf_counter()
    out = f()
    sync()
    first = time.perf_counter() - t
    if first >= 10:
        return first, out
    times = []
    for _ in range(3):
        t = time.perf_counter()
        out = f()
        sync()
        times.append(time.perf_counter() - t)
    return float(np.median(times)), out


def record(fn, size, what, seconds, note=""):
    """Keep one timing and print it."""
    ROWS.append(dict(function=fn, size=size, implementation=what,
                     seconds=seconds, note=note))
    print(f"{fn:12s} {size:38s} {what:28s} {seconds:10.3f} s  {note}",
          flush=True)


def widestats_cases(fn, size, call, dtypes=("float32",)):
    """Time widestats on the GPU (each dtype) and on the CPU in float64.

    Returns the results: the GPU's in the first dtype, and the CPU's.
    """
    gpu = None
    for dt in dtypes:
        if GPU:
            t, out = measure(lambda dt=dt: call(device=GPU, dtype=dt))
            record(fn, size, f"widestats GPU {dt}", t)
            gpu = out if gpu is None else gpu
    t, cpu = measure(lambda: call(device="cpu", dtype="float64"))
    record(fn, size, "widestats CPU float64", t)
    return (gpu if gpu is not None else cpu), cpu


# --- cor_pairs ----------------------------------------------------------------

def _modules(n, p, seed):
    """Random variables with a few hundred correlated groups among them."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    for _ in range(p // 100):
        c = rng.choice(p, 5, replace=False)
        x[:, c] = x[:, c[:1]] + 0.5 * rng.standard_normal((n, 5))
    return x


def _same_pairs(a, b, r, threshold, eps=1e-4):
    """Pair sets equal, but for pairs within float32 error of the cut."""
    edge = {(i, j) for i, j in a ^ b if abs(abs(r[i, j]) - threshold) < eps}
    return (a ^ b) <= edge


def bench_cor_pairs(quick):
    """Every pair with |r| >= 0.8 among p variables of 100 samples."""
    n, cut = 100, 0.8
    for p in ([5000] if quick else [5000, 20000, 50000]):
        x = _modules(n, p, 1)
        size = f"{p:,} variables x {n} samples"
        gpu, cpu = widestats_cases(
            "cor_pairs", size,
            lambda p=p, x=x, **kw: ws.cor_pairs(x, threshold=cut, **kw))
        exact = set(zip(cpu.i, cpu.j, strict=True))
        if p <= 20000:                       # 50,000^2 doubles: 20 GB
            def full(x=x):
                r = np.corrcoef(x, rowvar=False)
                i, j = np.nonzero(np.triu(np.abs(r) >= cut, 1))
                return set(zip(i, j, strict=True)), r
            t, (pairs, r) = measure(full)
            assert pairs == exact
            assert _same_pairs(pairs, set(zip(gpu.i, gpu.j, strict=True)),
                               r, cut)
            record("cor_pairs", size, "NumPy corrcoef + search", t)
        from corals.correlation.topk.default import cor_topk
        # CorALS returns a fraction k of all p x p entries, the diagonal
        # and both orders of each pair included, filtered by the threshold:
        # asked for enough to hold every pair over the cut. Its default
        # (n_jobs=None) was faster than n_jobs=8 in a trial.
        frac = 1.05 * (p + 2 * len(exact)) / p**2
        t, res = measure(lambda x=x, frac=frac: cor_topk(
            x, k=frac, threshold=cut, correlation_type="pearson"))
        ii, jj = (np.asarray(v) for v in res[1])
        off = ii != jj
        assert set(zip(np.minimum(ii, jj)[off], np.maximum(ii, jj)[off],
                       strict=True)) == exact
        record("cor_pairs", size, "CorALS cor_topk", t)


# --- cor_perm -------------------------------------------------------------------

def numpy_cor_perm(x, y, B, seed):
    """Run the same permutation test in NumPy: one product per permutation.

    Returns the permutation p-values.
    """
    def unit(m):
        m = m - m.mean(0)
        return m / np.sqrt((m * m).sum(0))
    zx, zy = unit(x), unit(y)
    bar = np.abs(zx.T @ zy) * (1 - 1e-10)
    counts = np.zeros(bar.shape)
    for pm in _permutations(len(x), B, seed).numpy():
        counts += np.abs(zx.T @ zy[pm]) >= bar
    return (counts + 1) / (B + 1)


def bench_cor_perm(quick):
    """Permutation p-values of every correlation, x by y, 50 samples."""
    n, B = 50, 1000
    for p in ([1000] if quick else [1000, 3000]):
        rng = np.random.default_rng(2)
        x, y = rng.exponential(size=(n, p)), rng.exponential(size=(n, p))
        size = f"{p:,} x {p:,} variables, {B:,} perms"
        gpu, cpu = widestats_cases(
            "cor_perm", size,
            lambda x=x, y=y, **kw: ws.cor_perm(x, y, B=B, seed=1, **kw))
        t, pv = measure(lambda x=x, y=y: numpy_cor_perm(x, y, B, 1))
        assert np.array_equal(pv, cpu.p_value)
        assert np.max(np.abs(pv - gpu.p_value)) <= 3 / (B + 1)
        record("cor_perm", size, "NumPy BLAS loop", t)


# --- ttest_perm -----------------------------------------------------------------

def numpy_ttest_perm(x, in1, B, seed):
    """Run Welch's t of every column under each permutation in NumPy.

    Vectorised over the columns, one permutation at a time. Returns the
    permutation p-values.
    """
    def welch(lab):
        a, b = x[lab], x[~lab]
        va, vb = a.var(0, ddof=1) / len(a), b.var(0, ddof=1) / len(b)
        return (a.mean(0) - b.mean(0)) / np.sqrt(va + vb)
    bar = np.abs(welch(in1)) * (1 - 1e-10)
    counts = np.zeros(x.shape[1])
    for pm in _permutations(len(x), B, seed).numpy():
        counts += np.abs(welch(in1[pm])) >= bar
    return (counts + 1) / (B + 1)


def bench_ttest_perm(quick):
    """Welch's t test of every gene, with permutations."""
    genes, n = 20000, 100
    for B in ([1000] if quick else [1000, 10000]):
        rng = np.random.default_rng(3)
        x = rng.standard_normal((n, genes))
        group = np.repeat([0, 1], n // 2)
        size = f"{genes:,} genes x {n} samples, {B:,} perms"
        gpu, cpu = widestats_cases(
            "ttest_perm", size,
            lambda x=x, B=B, g=group, **kw: ws.ttest_perm(x, g, B=B, seed=1,
                                                          **kw))
        if B <= 1000:
            t, pv = measure(lambda x=x, g=group, B=B:
                            numpy_ttest_perm(x, g == 0, B, 1))
            # equal groups: the swap of a labelling ties exactly in widestats
            # (canonical labellings) but within rounding in NumPy
            assert np.max(np.abs(pv - cpu.p_perm)) <= 2 / (B + 1)
            assert np.max(np.abs(pv - gpu.p_perm)) <= 3 / (B + 1)
            record("ttest_perm", size, "NumPy vectorised loop", t)


# --- granger --------------------------------------------------------------------

def statsmodels_granger(x, lag):
    """Run statsmodels' Granger test for every ordered pair.

    Returns the ssr F statistics, causes by effects.
    """
    from statsmodels.tsa.stattools import grangercausalitytests
    p = x.shape[1]
    f = np.full((p, p), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(p):
            for j in range(p):
                if i != j:
                    res = grangercausalitytests(x[:, [j, i]], maxlag=[lag])
                    f[i, j] = res[lag][0]["ssr_ftest"][0]
    return f


def bench_granger(quick):
    """Granger test of every ordered pair of random walks."""
    T, lag = 1000, 2
    for p in ([100] if quick else [100, 300, 1000]):
        rng = np.random.default_rng(4)
        x = rng.standard_normal((T, p)).cumsum(axis=0)
        size = f"{p:,} series x {T:,} times, lag {lag}"
        gpu, cpu = widestats_cases(
            "granger", size,
            lambda x=x, **kw: ws.granger(x, lag=lag, **kw),
            dtypes=("float64", "float32"))
        if p <= 300:
            t, f = measure(lambda x=x: statsmodels_granger(x, lag))
            np.testing.assert_allclose(gpu.f, f, rtol=1e-6)
            np.testing.assert_allclose(cpu.f, f, rtol=1e-6)
            record("granger", size, "statsmodels, pair by pair", t,
                   f"{p * (p - 1):,} pairs")


# --- mutual_info ----------------------------------------------------------------

def numpy_mutual_info(x, bins, order, tile=256):
    """Compute the same B-spline estimator in NumPy: weights, then GEMMs.

    Returns every pair's mutual information, in bits.
    """
    n, p = x.shape
    w = bspline_weights(x, bins, order).reshape(n, p * bins)
    mi = np.empty((p, p))
    for i0 in range(0, p, tile):
        for j0 in range(0, p, tile):
            i1, j1 = min(i0 + tile, p), min(j0 + tile, p)
            jt = (w[:, i0 * bins:i1 * bins].T @ w[:, j0 * bins:j1 * bins]) / n
            jt = jt.reshape(i1 - i0, bins, j1 - j0, bins)
            rows = jt.sum(3, keepdims=True)
            cols = jt.sum(1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                terms = np.where(jt > 0, jt * (np.log2(jt) - np.log2(rows)
                                               - np.log2(cols)), 0)
            mi[i0:i1, j0:j1] = terms.sum((1, 3))
    return np.maximum(mi, 0)


def bench_mutual_info(quick):
    """B-spline mutual information of every pair, 1,000 samples."""
    n = 1000
    for p in ([1000] if quick else [1000, 3000, 6000]):
        rng = np.random.default_rng(5)
        x = rng.standard_normal((n, p))
        size = f"{p:,} variables x {n:,} samples"
        gpu, cpu = widestats_cases(
            "mutual_info", size, lambda x=x, **kw: ws.mutual_info(x, **kw))
        t, mi = measure(lambda x=x: numpy_mutual_info(x, 10, 3))
        np.testing.assert_allclose(cpu, mi, atol=1e-10)
        np.testing.assert_allclose(gpu, mi, atol=1e-4)
        record("mutual_info", size, "NumPy, same algorithm", t)


def main():
    """Run the benchmarks and write the results."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/results")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gpu_name = torch.cuda.get_device_name() if GPU else "none"
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next(ln.split(":", 1)[1].strip() for ln in f
                       if ln.startswith("model name"))
    except OSError:
        pass
    threads = torch.get_num_threads()
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True,
                                check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = os.environ.get("WIDESTATS_COMMIT", "unknown")
    print(f"GPU {gpu_name}; CPU {cpu}, {threads} threads; torch "
          f"{torch.__version__}, numpy {np.__version__}; widestats {commit}",
          flush=True)
    for name, f in [("cor_pairs", bench_cor_pairs),
                    ("cor_perm", bench_cor_perm),
                    ("ttest_perm", bench_ttest_perm),
                    ("granger", bench_granger),
                    ("mutual_info", bench_mutual_info)]:
        if not a.only or name in a.only.split(","):
            f(a.quick)
            if GPU:
                torch.cuda.empty_cache()
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ROWS[0]))
        w.writeheader()
        w.writerows(ROWS)
    import scipy
    import statsmodels
    try:
        import corals
        corals_v = getattr(corals, "__version__", "installed")
    except ImportError:
        corals_v = "not installed"
    with open(out / "machine.txt", "w") as f:
        f.write(f"GPU: {gpu_name}\nCPU: {cpu}, {threads} threads\n"
                f"widestats {commit}; torch {torch.__version__}, numpy "
                f"{np.__version__}, scipy {scipy.__version__}, statsmodels "
                f"{statsmodels.__version__}, corals {corals_v}\n")


if __name__ == "__main__":
    main()
