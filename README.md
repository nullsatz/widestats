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

On one NVIDIA RTX A5500 (a workstation card), against the standard CPU
tools for each job on 8 cores of a Xeon Gold 6336Y with a multithreaded
BLAS. Each tool's answer was checked against the timed widestats result
(within float32 rounding) before its time counted.

| job | size | fastest standard CPU tool | widestats GPU | vs the tool | vs widestats on the CPU |
|---|---|---|---|---|---|
| every pair with \|r\| ≥ 0.8 | 50,000 variables × 100 samples | 6.9 s (NumPy, the matrix a tile at a time) | 0.19 s | **37×** | 22× |
| every pair with \|r\| ≥ 0.8 | 20,000 variables × 100 samples | 1.7 s (NumPy, tiled) | 0.039 s | **44×** | 16× |
| t test of every gene, 1,000 permutations, maxT | 20,000 genes × 100 samples | 11 s (vectorised NumPy) | 0.064 s | **169×** | 5× |
| permutation test of every correlation, maxT | 3,000 × 3,000 variables, 1,000 permutations | 63 s (NumPy, a BLAS product per permutation) | 0.95 s | **66×** | 18× |
| B-spline mutual information, every pair | 6,000 variables × 1,000 samples | 75 s (the same algorithm in NumPy and SciPy) | 1.1 s | **71×** | 32× |
| Granger test of every ordered pair | 300 series × 1,000 times | 2.4 min (statsmodels, pair by pair) | 0.45 s | **329×** | 1× |

The last column separates what the GPU buys from what the algorithm buys.
widestats on the same 8 CPU cores is already 2-31× faster than the tools,
by doing each job as a few large matrix products; the GPU adds another
5-32×. For Granger the gain is all the algorithm, one batched computation
instead of 89,700 separate regressions, done in float64 for accuracy,
which a workstation GPU runs no faster than the CPU (0.49 s). For the
strongest pairs, the dedicated CPU tool CorALS took 6.5 min at 50,000
variables: it is built to save memory rather than time, and plain NumPy in
tiles is faster.

Timings of the same code varied by up to about 40% between runs on this
shared machine, so read the ratios as approximate. The full results, every
size included, are in [`bench/results/results.md`](bench/results/results.md);
`python bench/bench.py` reproduces them on your hardware (`--quick` for a
short run) and `python bench/profile.py` shows where the time goes.

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
