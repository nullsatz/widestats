# widestats

Statistics for wide data on a GPU, built on PyTorch.

Wide data has far more variables than samples: 20,000 genes measured in 100
patients, 1,000 time series at 1,000 time points. Many questions about such
data are about *every pair* of variables, and the number of pairs grows with
the square of the number of variables. widestats answers those questions
with a few large matrix products on the GPU, in seconds instead of
minutes, and without building the full matrix of pairs in memory.

It is a Python port of the author's R package
[gputools](https://github.com/nullsatz/gputools), of the functions that
Python did not yet have on a GPU.

## Installing

```sh
pip install widestats
```

To work on it from a checkout, `pip install -e ".[test]"` and run `pytest`
for the tests and docstring examples.

widestats needs PyTorch 2.2 or later, NumPy and SciPy. It runs on NVIDIA
and AMD GPUs (`cuda`), Apple GPUs (`mps`) and the CPU, and uses the first of
those it finds; pass `device=` to choose.

## Using it

```python
import numpy as np
import widestats as ws

expr = np.load("expression.npy")      # samples x genes
group = np.load("group.npy")          # 0 or 1 per sample

# every pair of genes with |r| >= 0.8, without the 10 GB correlation matrix
pairs = ws.cor_pairs(expr, threshold=0.8)
pairs = ws.cor_pairs(expr, top=10)    # or each gene's 10 strongest partners
pairs.to_frame()                      # i, j, r as a pandas DataFrame

# Welch's t test of every gene between the two groups, with permutation
# p-values and Westfall-Young maxT adjusted ones
res = ws.ttest_perm(expr, group, B=10_000, seed=1)
res.statistic, res.p_value, res.p_adjusted

# permutation p-values for the correlation of every gene with every
# clinical variable
res = ws.cor_perm(expr, clinical, B=1000, seed=1)

# Granger causality of every ordered pair of time series
res = ws.granger(series, lag=2)       # series: times x series
res.f, res.p_value

# B-spline mutual information of every pair, which also finds
# nonlinear dependence
mi = ws.mutual_info(expr)
```

Inputs are NumPy arrays (or anything `numpy.asarray` accepts, or torch
tensors) with samples in rows. Results come back as NumPy arrays, or as
small result objects holding several of them. Every argument and result
is described in the [API documentation](https://nullsatz.github.io/widestats/).

| function | computes | agrees with |
|---|---|---|
| `cor_pairs` | the strongly correlated pairs: every pair above a threshold, or each variable's top k; Pearson or Spearman | `np.corrcoef` followed by a search |
| `cor_perm` | permutation p-values for every correlation, and maxT-adjusted ones | the same permutations through `np.corrcoef` |
| `ttest_perm` | Welch's or Student's t test of every feature between two groups, with permutation and maxT p-values | `scipy.stats.ttest_ind`, and the same permutations through it |
| `granger` | Granger causality F tests for every ordered pair of time series | statsmodels' `grangercausalitytests` (`ssr_ftest`), pair by pair |
| `mutual_info` | B-spline mutual information (Daub et al. 2004) of every pair | the estimator from the paper, pair by pair |

Every function is tested against the reference in the last column, on the
CPU, Apple GPUs and CUDA (see `tests/`). Permutation p-values are
`(1 + b) / (1 + B)` (Phipson & Smyth 2010), and the maxT adjustment
(Westfall & Young 1993) controls the family-wise error rate. Permutations
are drawn from `seed` on the CPU, so a seed gives the same ones on every
device.

For the pair problems that are not here: an eQTL scan is
[tensorQTL](https://github.com/broadinstitute/tensorqtl)'s job, and k-means
and PCA are [cuML](https://github.com/rapidsai/cuml)'s.

## How fast

Measured on one NVIDIA RTX A5500, a workstation card, against the fastest
standard CPU tool for each job running on 8 cores of a Xeon Gold 6336Y with
a multithreaded BLAS. Each tool's answer was checked against widestats'
before its time counted.

| job | size | fastest standard CPU tool | widestats GPU | speedup |
|---|---|---|---|---|
| every pair with \|r\| ≥ 0.8 | 50,000 variables × 100 samples | 8.4 s (NumPy, a tile at a time) | 0.17 s | **51×** |
| every pair with \|r\| ≥ 0.8 | 20,000 variables × 100 samples | 1.3 s (NumPy, a tile at a time) | 0.038 s | **34×** |
| t test of every gene, 1,000 permutations, maxT | 20,000 genes × 100 samples | 11 s (vectorized NumPy) | 0.064 s | **165×** |
| permutation test of every correlation, maxT | 3,000 × 3,000 variables, 1,000 permutations | 73 s (NumPy, one BLAS product per permutation) | 0.97 s | **75×** |
| B-spline mutual information, every pair | 6,000 variables × 1,000 samples | 90 s (the same algorithm in NumPy and SciPy) | 1.1 s | **85×** |
| Granger test of every ordered pair | 300 series × 1,000 times | 2.4 min (statsmodels, pair by pair) | 0.060 s | **2,400×** |

Not all of the speedup is the GPU. Run on the same 8 CPU cores, widestats
is still 2–25× faster than the standard tools, because each job is a few
large matrix products instead of a loop over pairs or permutations. The
GPU multiplies that by a further 7–29×.

Granger gets almost nothing from the GPU. Its 2,400× is the algorithm: one
batched computation in place of 89,700 separate regressions. It runs in
float64, which a workstation GPU does little faster than a CPU, so the
GPU takes 0.060 s and the CPU 0.073 s. At this size both are fixed
overhead rather than arithmetic; at 1,000 series the GPU is 1.4× the CPU.

CorALS, the CPU tool dedicated to finding the strongest pairs, took 6.5
minutes at 50,000 variables. It is built to save memory rather than time;
plain NumPy in tiles is faster.

Timings varied by up to about 40% between runs on this shared machine, so
read the ratios as approximate. The full results, including the CPU
widestats times and more sizes, are in
[`bench/results/results.md`](bench/results/results.md). To reproduce them
on your own hardware, run `python bench/bench.py` (`--quick` for a short
run); `python bench/profile.py` shows where the time goes.

## How it works

- **`cor_pairs`** computes the correlation matrix a tile at a time on the
  device, as a matrix product of standardized columns, and keeps only the
  pairs that pass. Memory scales with the tile and the answer, not with the
  number of pairs.
- **`cor_perm`** uses the fact that shuffling rows leaves standardized
  columns standardized: each permutation's correlations are one matrix
  product, and a batch of permutations one batched product.
- **`ttest_perm`** gets every feature's group sums and sums of squares under
  every relabelling from two matrix products, `G'x` and `G'x²`, where `G`
  is the samples-by-labellings indicator matrix. The observed t tests are
  SciPy's, in float64 on the CPU.
- **`granger`** fits the restricted and unrestricted regressions of every
  pair at once. The effect's own lags are projected out first, so each
  pair's F test needs only a small system in the cause's lags, solved
  batched over all pairs, in float64. For most pairs that system comes
  from a few small products; the pairs where that form would lose
  accuracy to cancellation (a cause almost a function of the effect's
  past, or nearly collinear lags) are detected and recomputed the slow,
  exact way.
- **`mutual_info`** spreads each value over neighboring bins with B-spline
  weights, so one matrix product of the weight matrices gives the joint
  distribution of every pair, and the mutual information of each is read
  from its block.

## Precision and devices

Computation is in float32 by default; pass `dtype="float64"` for double precision on CUDA or
the CPU. Apple GPUs have no float64. `granger` always computes in float64,
because the regressions it solves are sensitive to rounding.

## Developing

- Docstrings follow the NumPy convention, and their examples run as tests.
  The [API documentation](https://nullsatz.github.io/widestats/) is built
  from them with pdoc on every push to main; to preview it locally:

  ```sh
  pip install -e ".[docs]"
  pdoc --docformat numpy widestats '!widestats.granger'
  ```

- Before a pull request: `ruff check src tests`, and `pytest --cov`
  (coverage is at 100%). CI runs both on Linux (Python 3.10 to 3.13) and
  macOS, on the CPU; GPUs are tested by hand.

## License

GPL (>= 3), as gputools.
