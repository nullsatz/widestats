# widestats

GPU statistics for wide data, on PyTorch: tens of thousands of variables,
where the matrix of every pair is the problem.

```python
import widestats as ws

# every pair of genes with |r| >= 0.8, without the 10 GB correlation matrix
pairs = ws.cor_pairs(expr, threshold=0.8)          # expr: samples x genes
pairs = ws.cor_pairs(expr, top=10)                 # each gene's 10 strongest

# permutation p-values of every correlation, and maxT-adjusted ones
res = ws.cor_perm(x, y, B=1000, seed=1)

# Welch t test of every gene between two groups, with permutation and
# maxT p-values
res = ws.ttest_perm(expr, group, B=10_000, seed=1)
```

It runs on NVIDIA and AMD GPUs (`cuda`), Apple GPUs (`mps`, float32 only)
and the CPU, choosing the first available; pass `device=` and
`dtype="float64"` to choose. Inputs are NumPy arrays (or anything
`numpy.asarray` takes, or torch tensors), samples in rows; results are NumPy.

## How fast

On one NVIDIA RTX A5500 (a workstation card), against the fastest CPU code
we know of for each job, on 8 cores of a Xeon Gold 6336Y with a
multithreaded BLAS. Every widestats result was checked against the
baseline's answer before its time counted.

| job | size | fastest CPU code | widestats on the GPU | |
|---|---|---|---|---|
| every pair with \|r\| ≥ 0.8 | 50,000 variables | 6.4 min ([CorALS](https://github.com/mgbckr/corals-python); NumPy needs a 20 GB matrix) | 0.17 s | **2,272×** |
| every pair with \|r\| ≥ 0.8 | 20,000 variables | 7.0 s (NumPy `corrcoef` + search) | 0.038 s | **184×** |
| t test of every gene, 1,000 permutations, maxT | 20,000 genes × 100 samples | 7.4 s (vectorised NumPy) | 0.052 s | **142×** |
| permutation test of every correlation, maxT | 3,000 × 3,000 variables, 1,000 permutations | 62 s (NumPy BLAS loop) | 0.91 s | **68×** |
| B-spline mutual information, every pair | 6,000 variables × 1,000 samples | 73 s (the same algorithm in NumPy) | 1.0 s | **72×** |
| Granger test of every ordered pair | 300 series × 1,000 times | 2.4 min (statsmodels, pair by pair) | 0.024 s | **5,934×** |

Where the speed comes from differs. For the correlations, permutation tests
and mutual information it is mostly the GPU: widestats on the same 8 CPU
cores is 2.5-19× faster than the baseline, on the GPU 51-184× (and over
2,000× against CorALS). For Granger it is the algorithm: one batched
computation instead of 89,700 separate regressions, which is as fast on the
CPU (0.027 s) as on the GPU; for Granger, use whichever you have.

The full results, with every size and the widestats CPU times, are in
[`bench/results/results.md`](bench/results/results.md); `python
bench/bench.py` reproduces them on your hardware (`--quick` for a short
run) and `python bench/profile.py` shows where the time goes.

## What it does, and why

| function | does | as |
|---|---|---|
| `cor_pairs` | the strongly correlated pairs, every pair over a threshold or each variable's top k, Pearson or Spearman | `np.corrcoef` then a search, without forming the matrix |
| `cor_perm` | permutation p-values of every correlation, and Westfall-Young maxT adjusted ones | the same permutations through `np.corrcoef` |
| `ttest_perm` | Welch's or Student's t test of every feature between two groups, with permutation and maxT p-values | `scipy.stats.ttest_ind`, and the same permutations through it |
| `granger` | Granger causality F tests for every ordered pair of time series | statsmodels' `grangercausalitytests` (`ssr_ftest`), pair by pair |
| `mutual_info` | B-spline mutual information (Daub et al. 2004) of every pair, which detects any dependence, not only correlation | the estimator from the paper, pair by pair |

These are the functions of the R package
[gputools](https://github.com/nullsatz/gputools) that Python lacks on a GPU:
an eQTL scan is [tensorQTL](https://github.com/broadinstitute/tensorqtl)'s
job, and k-means and PCA are [cuML](https://github.com/rapidsai/cuml)'s.

- **`cor_pairs`** computes the correlation matrix a tile at a time on the
  device (a matrix product of standardised columns) and keeps only what
  passes, so memory goes with the tile and the answer, not with the number of
  pairs.
- **`cor_perm`** relies on shuffling rows leaving standardised columns
  standardised: each permutation's correlations are one matrix product, and
  a batch of permutations one batched product.
- **`ttest_perm`** gets every feature's group sums and sums of squares under
  every relabelling from two matrix products, `G'x` and `G'x²`, `G` the
  samples-by-labellings indicator matrix. The observed t tests are SciPy's,
  in float64 on the CPU.

p-values are `(1 + b) / (1 + B)` (Phipson & Smyth 2010); the maxT adjustment
(Westfall & Young 1993) controls the family-wise error rate. Permutations
are drawn from `seed` on the CPU, so a seed gives the same ones on any
device.

## Status

Early: every function agrees with an independent reference (NumPy, SciPy,
statsmodels, or the published method written out; see `tests/`), on the
CPU, Apple GPUs and CUDA; not yet benchmarked or released. Next:
benchmarks against the best CPU code, and a guide to the data sizes where
the GPU pays off.

## Installing

```sh
pip install -e ".[test]"  # from a checkout, for now
pytest                    # tests and docstring examples, on the CPU and any GPU
```

## Developing

- Docstrings follow the NumPy convention, and their examples run as tests.
  To read the API documentation:

  ```sh
  pip install -e ".[docs]"
  pdoc --docformat numpy widestats
  ```

- Before a pull request: `ruff check src tests`, and `pytest --cov`
  (coverage is at 100%). CI runs both on Linux (Python 3.10 to 3.13) and
  macOS, on the CPU; GPUs are tested by hand.

## License

GPL (>= 3), as gputools.
