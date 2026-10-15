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

## What it does, and why

| function | does | as |
|---|---|---|
| `cor_pairs` | the strongly correlated pairs, every pair over a threshold or each variable's top k, Pearson or Spearman | `np.corrcoef` then a search, without forming the matrix |
| `cor_perm` | permutation p-values of every correlation, and Westfall-Young maxT adjusted ones | the same permutations through `np.corrcoef` |
| `ttest_perm` | Welch's or Student's t test of every feature between two groups, with permutation and maxT p-values | `scipy.stats.ttest_ind`, and the same permutations through it |

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

Early: the three functions above agree with NumPy and SciPy (see `tests/`);
not yet benchmarked or released. Next: Granger causality tests for every pair
of series and B-spline mutual information (Daub et al. 2004), then
benchmarks against the best CPU code, and a guide to the data sizes where
the GPU pays off.

## Installing

```sh
pip install -e .          # from a checkout, for now
pytest                    # CPU, and any GPU present
```

## License

GPL (>= 3), as gputools.
