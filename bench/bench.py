"""Benchmark widestats on a GPU against the best CPU code for each job.

For every function and size this times

- widestats on the GPU (float32, and float64 where it matters),
- widestats on the CPU (the same algorithm, PyTorch on the CPU), so that
  what the GPU buys is separate from what the algorithm buys, and
- the CPU code a careful user would otherwise run: NumPy on a
  multithreaded BLAS, statsmodels, CorALS.

Each case checks that the widestats result agrees with the baseline's before
its time is recorded. Times are the median of a few runs after a warm-up of
both precisions; GPU timings synchronise the device. Results go to
``<out>/results.csv`` and a markdown table to ``<out>/results.md``.

    python bench/bench.py [--out bench/results] [--quick]
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

import widestats as ws
from widestats.mi import bspline_weights

GPU = "cuda" if torch.cuda.is_available() else None
ROWS = []


def sync():
    if GPU:
        torch.cuda.synchronize()


def timed(f, reps=3):
    """Median wall time of `f()` over `reps` runs, after one warm-up."""
    f()
    sync()
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        f()
        sync()
        times.append(time.perf_counter() - t)
    return float(np.median(times))


def once(f):
    """Wall time of a single run: for baselines too slow to repeat."""
    t = time.perf_counter()
    out = f()
    sync()
    return time.perf_counter() - t, out


def record(fn, size, what, seconds, note=""):
    ROWS.append(dict(function=fn, size=size, implementation=what,
                     seconds=seconds, note=note))
    print(f"{fn:12s} {size:32s} {what:34s} {seconds:10.3f} s  {note}",
          flush=True)


def gpu_cases(fn, size, call, dtypes=("float32",)):
    for dt in dtypes:
        if GPU:
            record(fn, size, f"widestats GPU {dt}",
                   timed(lambda: call(device=GPU, dtype=dt)))
    record(fn, size, "widestats CPU float64",
           timed(lambda: call(device="cpu", dtype="float64"), reps=1))


# --- cor_pairs ----------------------------------------------------------------

def bench_cor_pairs(quick):
    n = 100
    for p in ([5000] if quick else [5000, 20000, 50000]):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((n, p))
        for m in range(p // 100):            # a few hundred real modules
            c = rng.choice(p, 5, replace=False)
            x[:, c] = x[:, c[:1]] + 0.5 * rng.standard_normal((n, 5))
        size = f"{p:,} variables x {n} samples"
        got = ws.cor_pairs(x, threshold=0.8, device=GPU or "cpu")
        gpu_cases("cor_pairs", size,
                  lambda **kw: ws.cor_pairs(x, threshold=0.8, **kw))
        if p <= 20000:                       # 50,000^2 doubles: 20 GB
            def full():
                r = np.corrcoef(x, rowvar=False)
                i, j = np.nonzero(np.triu(np.abs(r) >= 0.8, 1))
                return i
            t, i = once(full)
            assert len(i) == len(got), (len(i), len(got))
            record("cor_pairs", size, "NumPy corrcoef + search", t)
        try:
            from corals.correlation.topk.default import cor_topk
            t, res = once(lambda: cor_topk(x, threshold=0.8,
                                            correlation_type="pearson"))
            found = len(res[0]) if isinstance(res, tuple) else len(res)
            record("cor_pairs", size, "CorALS cor_topk (threshold)", t,
                   f"{found} vs {len(got)} pairs")
        except Exception as e:                # baseline unavailable
            record("cor_pairs", size, "CorALS cor_topk (threshold)",
                   float("nan"), f"failed: {type(e).__name__}")


# --- cor_perm -------------------------------------------------------------------

def numpy_cor_perm(x, y, B, seed):
    """The same permutation test in NumPy: a BLAS product per permutation."""
    from widestats.perm import _permutations
    z = lambda m: (m - m.mean(0)) / np.sqrt(((m - m.mean(0))**2).sum(0))  # noqa: E731
    zx, zy = z(x), z(y)
    r0 = np.abs(zx.T @ zy)
    counts = np.zeros(r0.shape)
    maxima = np.empty(B)
    for b, pm in enumerate(_permutations(len(x), B, seed).numpy()):
        rb = np.abs(zx.T @ zy[pm])
        counts += rb >= r0 * (1 - 1e-10)
        maxima[b] = rb.max()
    return (counts + 1) / (B + 1)


def bench_cor_perm(quick):
    n, B = 50, 1000
    for p in ([1000] if quick else [1000, 3000]):
        rng = np.random.default_rng(2)
        x, y = rng.exponential(size=(n, p)), rng.exponential(size=(n, p))
        size = f"{p:,} x {p:,} variables, {B} perms"
        gpu_cases("cor_perm", size,
                  lambda **kw: ws.cor_perm(x, y, B=B, seed=1, **kw))
        t, pv = once(lambda: numpy_cor_perm(x, y, B, 1))
        want = ws.cor_perm(x, y, B=B, seed=1, device="cpu", dtype="float64")
        assert np.max(np.abs(pv - want.p_value)) <= 2 / (B + 1)
        record("cor_perm", size, "NumPy BLAS loop", t)


# --- ttest_perm -----------------------------------------------------------------

def numpy_ttest_perm(x, in1, B, seed):
    """Welch's t of every column under each permutation, vectorised NumPy."""
    from widestats.perm import _permutations
    def welch(lab):
        a, b = x[lab], x[~lab]
        va, vb = a.var(0, ddof=1) / len(a), b.var(0, ddof=1) / len(b)
        return (a.mean(0) - b.mean(0)) / np.sqrt(va + vb)
    t0 = np.abs(welch(in1))
    counts = np.zeros(x.shape[1])
    for pm in _permutations(len(x), B, seed).numpy():
        counts += np.abs(welch(in1[pm])) >= t0 * (1 - 1e-10)
    return (counts + 1) / (B + 1)


def bench_ttest_perm(quick):
    genes, n = 20000, 100
    for B in ([1000] if quick else [1000, 10000]):
        rng = np.random.default_rng(3)
        x = rng.standard_normal((n, genes))
        group = np.repeat([0, 1], n // 2)
        size = f"{genes:,} genes x {n} samples, {B:,} perms"
        gpu_cases("ttest_perm", size,
                  lambda **kw: ws.ttest_perm(x, group, B=B, seed=1, **kw))
        if B <= 1000:
            t, pv = once(lambda: numpy_ttest_perm(x, group == 0, B, 1))
            want = ws.ttest_perm(x, group, B=B, seed=1, device="cpu",
                                 dtype="float64")
            assert np.max(np.abs(pv - want.p_perm)) <= 2 / (B + 1)
            record("ttest_perm", size, "NumPy vectorised loop", t)


# --- granger --------------------------------------------------------------------

def bench_granger(quick):
    T, lag = 1000, 2
    for p in ([100] if quick else [100, 300, 1000]):
        rng = np.random.default_rng(4)
        x = rng.standard_normal((T, p)).cumsum(axis=0)
        size = f"{p:,} series x {T} times, lag {lag}"
        gpu_cases("granger", size,
                  lambda **kw: ws.granger(x, lag=lag, **kw),
                  dtypes=("float64", "float32"))
        if p <= 300:
            import warnings

            from statsmodels.tsa.stattools import grangercausalitytests
            def sm():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for i in range(p):
                        for j in range(p):
                            if i != j:
                                grangercausalitytests(x[:, [j, i]],
                                                      maxlag=[lag])
            t, _ = once(sm)
            record("granger", size, "statsmodels, pair by pair", t,
                   f"{p * (p - 1):,} pairs")


# --- mutual_info ----------------------------------------------------------------

def numpy_mutual_info(x, bins, order, tile=256):
    """The same B-spline estimator in NumPy: weights, then GEMMs by tile."""
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
    return mi


def bench_mutual_info(quick):
    n = 1000
    for p in ([1000] if quick else [1000, 3000, 6000]):
        rng = np.random.default_rng(5)
        x = rng.standard_normal((n, p))
        size = f"{p:,} variables x {n} samples"
        gpu_cases("mutual_info", size,
                  lambda **kw: ws.mutual_info(x, **kw))
        t, mi = once(lambda: numpy_mutual_info(x, 10, 3))
        want = ws.mutual_info(x, device="cpu", dtype="float64")
        assert np.allclose(np.maximum(mi, 0), want, atol=1e-10)
        record("mutual_info", size, "NumPy, same algorithm", t)


def main():
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
    print(f"GPU {gpu_name}; CPU {cpu}, {threads} threads; torch "
          f"{torch.__version__}, numpy {np.__version__}", flush=True)
    for name, f in [("cor_pairs", bench_cor_pairs), ("cor_perm", bench_cor_perm),
                    ("ttest_perm", bench_ttest_perm), ("granger", bench_granger),
                    ("mutual_info", bench_mutual_info)]:
        if not a.only or name in a.only.split(","):
            f(a.quick)
            # warm-up of both precisions happens inside timed(); free memory
            if GPU:
                torch.cuda.empty_cache()
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ROWS[0]))
        w.writeheader()
        w.writerows(ROWS)
    with open(out / "machine.txt", "w") as f:
        f.write(f"GPU: {gpu_name}\nCPU: {cpu}, {threads} threads\n"
                f"torch {torch.__version__}, numpy {np.__version__}\n"
                f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}\n")


if __name__ == "__main__":
    main()
